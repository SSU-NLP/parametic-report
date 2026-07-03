import json
import math
import os
import re
from pathlib import Path

import torch

from parametic_studio.device import dtype_for, resolve_device

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(?!.*lora_)(.+)$")  # lora_* params never enter spot/probe grids


class ModelSession:
    """A live inference session: model resident, manual token-by-token decode.

    Manual greedy loop (not TextIteratorStreamer) so each step can be stopped
    and, later, have attention captured. ponytail: greedy + eos + stop only.
    """

    @classmethod
    def from_pretrained(cls, model_id, device="auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = resolve_device(device)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype_for(dev), attn_implementation="eager"
        )
        return cls(model, tok, dev, model_id=model_id)

    def __init__(self, model, tokenizer, device, model_id=None):
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.device = device
        self.model_id = model_id  # None (tests/anonymous) → no disk persistence
        self._stop = False
        self.last_raw = None  # per-layer [heads, kv] of the last decoded step (for drilldown)
        self._acts = {}       # (layer, module) -> last-token output norm
        self.regions = {}         # LRU cache: name -> region masks (lazy; loaded from disk on demand, cap 2)
        self.region_grids = {}    # name -> [L][M] importance grid (lazy from meta; None for legacy/cell regions)
        self._region_files = {}   # name -> Path(.pt) — every saved region (masks NOT loaded at boot)
        self._region_meta = {}    # name -> {"count": int, "grid": [[..]]|None} from the .meta.json sidecar
        self._region_order = []   # load order for LRU eviction of self.regions
        d = self._region_dir()
        if d and d.exists():
            for f in d.glob("*.pt"):  # scan only — masks (~0.36GB each) stay on disk until get_region
                self._region_files[f.stem] = f
                meta = f.with_suffix(".meta.json")
                if not meta.exists():
                    self._migrate_meta(f, meta)  # one-time: legacy .pt with no sidecar → compute + write, then free
                self._region_meta[f.stem] = json.loads(meta.read_text())
                self.region_grids[f.stem] = self._region_meta[f.stem].get("grid")
        self.modules_probed = self._register_activation_hooks()

    def _migrate_meta(self, pt_path, meta_path):
        """Legacy .pt without a sidecar: load once → count/grid → write meta → drop the mask (1-time at boot)."""
        blob = torch.load(pt_path, map_location="cpu")
        if isinstance(blob, dict) and "masks" in blob:  # v2 format: masks + importance grid
            masks, grid = blob["masks"], blob.get("grid")
        else:  # v1: plain mask dict
            masks, grid = blob, None
        count = sum(int(v.sum()) for v in masks.values())
        meta_path.write_text(json.dumps({"count": count, "grid": grid}))
        del blob, masks

    def free_memory(self):
        """Release transient device memory after heavy ops (the mps caching allocator hoards freed blocks)."""
        self.model.zero_grad(set_to_none=True)
        if self.device.type == "mps":
            torch.mps.empty_cache()

    def _region_dir(self):
        if not self.model_id:
            return None
        root = os.environ.get("PARAMETIC_STUDIO_HOME", os.path.expanduser("~/.parametic_studio"))
        return Path(root) / "regions" / self.model_id.replace("/", "__")

    def _register_activation_hooks(self):
        names = ["self_attn", "mlp"]
        self._hook_handles = []
        for i, layer in enumerate(self.model.model.layers):
            for name in names:
                self._hook_handles.append(getattr(layer, name).register_forward_hook(self._act_hook(i, name)))
        return names

    def close(self):
        # remove hooks so the model (and this session) can be GC'd / freed.
        for h in getattr(self, "_hook_handles", []):
            h.remove()
        self._hook_handles = []

    def _act_hook(self, i, name):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out  # [b, seq, hidden]
            self._acts[(i, name)] = float(h[0, -1].detach().norm())
        return hook

    def _modules(self):
        return [m.group(2) for name, _ in self.model.named_parameters()
                if (m := _LAYER_RE.match(name)) and int(m.group(1)) == 0]

    def spot_step(self, example, acc, n):
        """Accumulate |grad×param| for one example into acc (mutated); return running grid normalized by n.

        Split out of compute_spot so the API can stream one step at a time off the event loop.
        """
        ids = torch.tensor([self.tok.encode(example)], device=self.device)
        self.model.zero_grad(set_to_none=True)
        self.model(ids, labels=ids).loss.backward()
        for name, p in self.model.named_parameters():
            m = _LAYER_RE.match(name)
            if m and p.grad is not None:
                key = (int(m.group(1)), m.group(2))
                acc[key] = acc.get(key, 0.0) + float((p.grad * p.data).abs().sum())
        self.model.zero_grad(set_to_none=True)
        L, modules = len(self.model.model.layers), self._modules()
        grid = [[acc.get((l, mod), 0.0) / n for mod in modules] for l in range(L)]
        return {"layers": L, "modules": modules, "grid": grid}

    def compute_spot(self, examples):
        """Coding-Spot importance: accumulate |grad×param| over examples, reduced to [layer, module]."""
        acc, last = {}, None
        for i, text in enumerate(examples):
            last = self.spot_step(text, acc, i + 1)
        if last is None:  # no examples → all-zero grid
            L, modules = len(self.model.model.layers), self._modules()
            last = {"layers": L, "modules": modules, "grid": [[0.0] * len(modules) for _ in range(L)]}
        return last

    # ---- knob (B1): Locate × Edit × Evaluate. A Region is {param_name: bool mask}. ----

    def locate_cell(self, layer, module):
        """Locate: a whole per-layer weight, e.g. (0, 'mlp.gate_proj.weight'). Masks live on CPU."""
        name = f"model.layers.{layer}.{module}"
        p = dict(self.model.named_parameters())[name]
        return {name: torch.ones(p.shape, dtype=torch.bool)}

    def locate_spot(self, examples, topk=0.05, return_grid=False, progress=None):
        """Locate: top-k% mask *within each layer param* by accumulated |grad×param|
        (matches scripts/create_approx_spot_masks.py — per-param, not a global threshold).
        return_grid=True also returns the [L][M] importance cell grid (what the spot view shows).
        progress(i, total): called after each example's backward+accumulate (i is 1-based)."""
        if not examples:
            return ({}, []) if return_grid else {}
        acc = {}
        total = len(examples)
        for i, text in enumerate(examples):
            ids = torch.tensor([self.tok.encode(text)], device=self.device)
            self.model.zero_grad(set_to_none=True)
            self.model(ids, labels=ids).loss.backward()
            for name, p in self.model.named_parameters():
                if _LAYER_RE.match(name) and p.grad is not None:
                    a = (p.grad * p.data).abs()
                    acc[name] = a if name not in acc else acc[name] + a
            if progress:
                progress(i + 1, total)
        self.model.zero_grad(set_to_none=True)
        # ponytail: per-param exact top-k via topk indices — no tie over-selection, no global flatten copy.
        # acc holds one importance tensor per layer param (peak ≈ layer-param bytes); stream from disk if models grow.
        region = {}
        for name, score in acc.items():
            flat = score.flatten()
            count = int(topk * flat.numel())
            if count <= 0:
                continue
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[torch.topk(flat, count, largest=True).indices] = True
            region[name] = mask.reshape(score.shape).cpu()  # masks live on CPU; ops move them per use
        grid = None
        if return_grid:  # per-cell importance sums — same numbers the spot view heatmap shows
            L, modules = len(self.model.model.layers), self._modules()
            n = len(examples)
            cell = {}
            for name, score in acc.items():
                m = _LAYER_RE.match(name)
                cell[(int(m.group(1)), m.group(2))] = float(score.sum()) / n
            grid = [[cell.get((l, mod), 0.0) for mod in modules] for l in range(L)]
        del acc
        self.free_memory()  # backward leftovers + acc tensors — don't let the allocator hoard them
        return (region, grid) if return_grid else region

    def tensor_list(self):
        """Metadata of every parameter (HF safetensors-viewer style): name, shape, dtype."""
        return [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype).removeprefix("torch.")}
                for n, p in self.model.named_parameters()]

    def _cache_region(self, name, region):
        """Insert into the LRU mask cache (cap 2) — evict the oldest *reloadable* region past the cap.
        Memory-only regions (no model_id → no backing .pt) are never evicted: there's nowhere to reload them."""
        if name in self.regions:
            self._region_order.remove(name)
        self.regions[name] = region
        self._region_order.append(name)
        evictable = [n for n in self._region_order if n in self._region_files]
        while len(evictable) > 2:  # cap 2: region masks are big; hold at most two disk-backed resident
            victim = evictable.pop(0)
            self._region_order.remove(victim)
            self.regions.pop(victim, None)

    def get_region(self, name):
        """Return a saved region's masks, loading from disk (v1/v2) on cache miss. LRU-cached (cap 2)."""
        if name in self.regions:
            return self.regions[name]
        blob = torch.load(self._region_files[name], map_location="cpu")
        region = blob["masks"] if isinstance(blob, dict) and "masks" in blob else blob  # v2 | v1
        self._cache_region(name, region)
        return region

    def region_meta(self):
        """Saved-region list without loading masks: [{name, count}] from the meta sidecars."""
        return [{"name": n, "count": m.get("count", 0)} for n, m in self._region_meta.items()]

    def save_region(self, name, region, grid=None):
        """Locate: keep a named mask (+ its importance grid, for faithful viewing) in the workspace and on disk."""
        self._cache_region(name, region)
        self.region_grids[name] = grid
        count = sum(int(v.sum()) for v in region.values())
        self._region_meta[name] = {"count": count, "grid": grid}
        d = self._region_dir()
        if d:
            d.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^\w.-]", "_", name)
            pt = d / f"{safe}.pt"
            # ponytail: bool dump ≈ 1 byte/param (~0.4GB for a 0.5B full region) — pack indices if disk matters.
            torch.save({"masks": {k: v.cpu() for k, v in region.items()}, "grid": grid}, pt)
            (d / f"{safe}.meta.json").write_text(json.dumps({"count": count, "grid": grid}))
            self._region_files[name] = pt

    def delete_region(self, name):
        """Remove a saved region from the workspace and disk (no-op if unknown)."""
        self.regions.pop(name, None)
        if name in self._region_order:
            self._region_order.remove(name)
        self.region_grids.pop(name, None)
        self._region_meta.pop(name, None)
        self._region_files.pop(name, None)
        d = self._region_dir()
        if d:
            safe = re.sub(r"[^\w.-]", "_", name)
            (d / f"{safe}.pt").unlink(missing_ok=True)
            (d / f"{safe}.meta.json").unlink(missing_ok=True)

    def region_grid(self, name):
        """Visualize a saved region: selection-fraction grid + the importance grid captured at save time."""
        region = self.get_region(name)
        L, modules = len(self.model.model.layers), self._modules()
        return {"layers": L, "modules": modules, "grid": self._cell_grid(region, L, modules),
                "importance": self.region_grids.get(name),
                "count": sum(int(v.sum()) for v in region.values())}

    def _cell_grid(self, region, L, modules):
        grid = [[0.0] * len(modules) for _ in range(L)]
        for pname, mask in region.items():
            m = _LAYER_RE.match(pname)
            if m and m.group(2) in modules:
                grid[int(m.group(1))][modules.index(m.group(2))] = float(mask.sum()) / mask.numel()
        return grid

    def region_compare(self, names):
        """Cross-dataset spot comparison: per-region cell grids, all-intersection grid, pairwise Jaccard."""
        from itertools import combinations
        L, modules = len(self.model.model.layers), self._modules()
        regions = {n: self.get_region(n) for n in names}  # local — may exceed LRU cap for the compare only
        # display grids: prefer the saved importance heatmap (what the spot view showed) — the
        # selection-fraction grid is ~uniform by construction (per-param top-k%).
        grids = {n: self.region_grids.get(n) or self._cell_grid(r, L, modules) for n, r in regions.items()}
        kinds = {n: ("importance" if self.region_grids.get(n) else "fraction") for n in names}
        jaccard = {}
        for a, b in combinations(names, 2):
            inter = union = 0
            for p in set(regions[a]) | set(regions[b]):
                ma, mb = regions[a].get(p), regions[b].get(p)
                if ma is None or mb is None:
                    union += int((ma if mb is None else mb).sum())
                else:
                    inter += int((ma & mb).sum())
                    union += int((ma | mb).sum())
            jaccard[f"{a}|{b}"] = inter / union if union else 0.0
        inter_region = {}
        for p in set.intersection(*(set(r) for r in regions.values())) if regions else set():
            acc = regions[names[0]][p]
            for n in names[1:]:
                acc = acc & regions[n][p]
            inter_region[p] = acc
        inter_grid = self._cell_grid(inter_region, L, modules)
        # lift = observed ∩-fraction / expected under independence (product of per-region fractions).
        # top-k% spots make raw fractions near-uniform and tiny — lift is the readable signal.
        frac = {n: self._cell_grid(r, L, modules) for n, r in regions.items()}
        lift = [[0.0] * len(modules) for _ in range(L)]
        for l in range(L):
            for c in range(len(modules)):
                expected = 1.0
                for n in names:
                    expected *= frac[n][l][c]
                lift[l][c] = inter_grid[l][c] / expected if expected > 0 else 0.0
        return {"layers": L, "modules": modules, "grids": grids, "kinds": kinds,
                "intersection": inter_grid, "intersection_lift": lift, "jaccard": jaccard}

    def intervene(self, region, op="scale", alpha=0.0, key="default"):
        """Edit: reversibly modify selected weights under `key`. op = scale|zero|mean|random.

        Multiple keys coexist (a mixing board of knobs). Re-using a key re-applies from the
        original weights. ponytail: assumes knobs target disjoint regions (per-cell); overlapping
        regions get last-write-wins on restore.
        """
        active = getattr(self, "_active", {})
        if key in active:
            self._restore(active.pop(key))  # re-adjust: undo this knob first, apply from original
        params = dict(self.model.named_parameters())
        backup = {}
        with torch.no_grad():
            for name, cmask in region.items():
                p = params[name]
                mask = cmask.to(p.device)
                sel = p.data[mask]
                backup[name] = sel.clone()  # only selected values → memory ∝ |region|
                if op == "scale":
                    p.data[mask] = sel * alpha
                elif op == "zero":
                    p.data[mask] = 0.0
                elif op == "mean":
                    p.data[mask] = sel.mean()
                elif op == "random":
                    p.data[mask] = torch.randn_like(sel) * sel.std() + sel.mean()
                else:
                    raise ValueError(f"unknown op: {op}")
        active[key] = (region, backup, op, alpha)
        self._active = active

    def _restore(self, entry):
        region, backup = entry[0], entry[1]
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, mask in region.items():
                params[name].data[mask.to(params[name].device)] = backup[name]

    def suspend(self):
        """A/B: restore original weights but keep knob entries (resume re-applies)."""
        for entry in getattr(self, "_active", {}).values():
            self._restore(entry)

    def resume(self):
        """A/B: re-apply all suspended knobs from the original weights.
        ponytail: op='random' redraws on resume — a new sample, not the identical one."""
        for key, (region, _b, op, alpha) in list(getattr(self, "_active", {}).items()):
            self.intervene(region, op, alpha, key=key)

    def clear(self, key=None):
        """Edit: restore weights. key → that knob only; None → all knobs (no-op if none)."""
        active = getattr(self, "_active", {})
        if key is None:
            for entry in active.values():  # disjoint regions → restore order-independent
                self._restore(entry)
            self._active = {}
        elif key in active:
            self._restore(active.pop(key))

    # ---- region-aware training: full | spot-freeze | spot-only | lora. Reversible via reset_training. ----

    def train_steps(self, examples, mode="full", steps=50, lr=1e-4, region=None, lora_dim=8):
        """Generator: one AdamW step per next(), yields {"step", "loss"}. stop() ends early.

        spot-freeze = gradients blocked inside region (train everything *except* the spot).
        spot-only   = only region weights train (element-masked). lora = base frozen, adapters train.
        """
        if getattr(self, "_trained", None) is not None:
            raise RuntimeError("reset_training first")  # ponytail: no stacked trainings in v1
        self.clear()  # invariant: knob backups are always relative to the current base weights
        self._stop = False
        params = dict(self.model.named_parameters())
        hooks = []
        if mode == "lora":
            from parametic_studio.kernel.lora import convert_to_lora
            convert_to_lora(self.model, lora_dim)
            for n, p in self.model.named_parameters():
                p.requires_grad_("lora_" in n)     # adapters are the *only* trainables (embed/norm/lm_head too)
            self._trained = ("lora", None)         # base weights never change → nothing to back up
        elif mode == "spot-freeze":
            for name, mask in region.items():
                dm = mask.to(params[name].device)  # hook runs on the grad's device
                hooks.append(params[name].register_hook(lambda g, m=dm: g.masked_fill(m, 0)))
            self._trained = (mode, {n: p.detach().to("cpu", copy=True) for n, p in params.items()})
        elif mode == "spot-only":
            for n, p in params.items():
                p.requires_grad_(n in region)
            for name, mask in region.items():
                dm = mask.to(params[name].device)
                hooks.append(params[name].register_hook(lambda g, m=dm: g * m))
            self._trained = (mode, (region, {n: params[n].data[m.to(params[n].device)].clone() for n, m in region.items()}))
        else:  # full
            self._trained = ("full", {n: p.detach().to("cpu", copy=True) for n, p in params.items()})
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        # weight_decay must be 0: decoupled decay moves weights even where the gradient is masked to zero,
        # which would silently violate spot-freeze/spot-only region guarantees.
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
        try:
            for step in range(steps):
                if self._stop:
                    return
                ids = torch.tensor([self.tok.encode(examples[step % len(examples)])], device=self.device)
                loss = self.model(ids, labels=ids).loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)  # bf16 NaN insurance
                opt.step()
                yield {"step": step, "loss": float(loss.detach())}
        finally:
            for h in hooks:
                h.remove()  # leaked hooks would silently corrupt every later spot map
            self.model.zero_grad(set_to_none=True)
            del opt
            for p in self.model.parameters():
                p.requires_grad_(True)  # locate_spot needs grads everywhere
            if self.device.type == "mps":
                torch.mps.empty_cache()  # release optimizer state

    def reset_training(self):
        """Restore pre-training weights (no-op if not trained)."""
        trained = getattr(self, "_trained", None)
        if trained is None:
            return
        self.clear()  # same invariant as train_steps
        mode, payload = trained
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            if mode == "lora":
                from parametic_studio.kernel.lora import remove_lora
                remove_lora(self.model)  # base params were frozen all along → bit-exact
            elif mode == "spot-only":
                region, backup = payload
                for n, m in region.items():
                    params[n].data[m.to(params[n].device)] = backup[n]
            else:
                for n, p in params.items():
                    p.data.copy_(payload[n].to(self.device))
        self._trained = None

    def ppl(self, examples):
        """Evaluate: mean-loss perplexity over examples (quantifies an intervention's damage)."""
        losses = []
        with torch.no_grad():
            for text in examples:
                ids = torch.tensor([self.tok.encode(text)], device=self.device)
                losses.append(float(self.model(ids, labels=ids).loss))
        return math.exp(sum(losses) / max(1, len(losses)))

    def stop(self):
        self._stop = True

    def drilldown(self, layer):
        return self.last_raw[layer]  # [heads, kv] of the last step

    def generate_text(self, prompt, max_tokens, probes=("attention",), temperature=0.0):
        tmpl = getattr(self.tok, "apply_chat_template", None)
        if tmpl and getattr(self.tok, "chat_template", None):
            prompt = tmpl([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = torch.tensor([self.tok.encode(prompt)])
        yield from self.generate(ids, max_tokens, probes, temperature=temperature)

    def complete_code(self, prompt, max_tokens=512, temperature=0.0, stops=None):
        """Continue a raw code `prompt` (no chat template) and return the completion text,
        cut at the first stop sequence. Uses the plain decode loop so the current knob/damage
        state applies verbatim. ponytail: reuse generate(), collect text, truncate."""
        from parametic_studio.kernel.humaneval import truncate_completion, STOP_SEQUENCES
        stops = STOP_SEQUENCES if stops is None else stops
        ids = torch.tensor([self.tok.encode(prompt)])
        out = "".join(ev["text"] for ev in self.generate(ids, max_tokens, probes=(), temperature=temperature))
        return truncate_completion(out, stops)

    def generate(self, input_ids, max_tokens, probes=("attention",), temperature=0.0):
        self._stop = False
        ids = input_ids.to(self.device)
        eos = self.tok.eos_token_id
        want_attn = "attention" in probes
        want_logit = "logitlens" in probes
        for step in range(max_tokens):
            if self._stop:
                return
            with torch.no_grad():
                # ponytail: no KV cache, full recompute each step — O(n²) but fine for demo lengths.
                out = self.model(ids, output_attentions=want_attn, output_hidden_states=want_logit)
            logits = out.logits[0, -1]
            if temperature and temperature > 0:  # temperature sampling; 0 = greedy
                nxt = int(torch.multinomial(torch.softmax(logits.float() / temperature, -1), 1))
            else:
                nxt = int(logits.argmax())
            if nxt == eos:
                return
            event = {"step": step, "token_id": nxt, "text": self.tok.decode([nxt])}
            if want_logit:
                # decode each layer's residual through the final norm + lm_head → top-1 prediction.
                norm, head = self.model.model.norm, self.model.lm_head
                with torch.no_grad():
                    rows = []
                    for h in out.hidden_states[1:]:  # [1:] skips the embedding layer
                        p = torch.softmax(head(norm(h[0, -1])).float(), -1)
                        tid = int(p.argmax())
                        rows.append({"token": self.tok.decode([tid]), "prob": float(p[tid])})
                event["logit"] = rows
            if want_attn:
                self.last_raw = [a[0, :, -1, :].float() for a in out.attentions]  # last query per-head (drilldown)
                if step == 0:
                    # prefill: emit every prompt query row (0..P-1) so the matrix is the full
                    # causal square N×N, not just the generated-token band. (no KV cache → full attn available)
                    P = ids.shape[1]
                    event["attn_rows"] = [torch.stack([a[0, :, q, : q + 1].mean(0).float() for a in out.attentions]) for q in range(P)]
                else:
                    event["attn_rows"] = [torch.stack([r.mean(0) for r in self.last_raw])]  # one new row [L, kv]
            if "activation" in probes:
                L = len(self.model.model.layers)
                event["act"] = torch.tensor(
                    [[self._acts[(l, m)] for m in self.modules_probed] for l in range(L)]
                )  # [L, M]
            ids = torch.cat([ids, torch.tensor([[nxt]], device=self.device)], dim=1)
            yield event
