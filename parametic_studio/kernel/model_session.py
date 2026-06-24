import re

import torch

from parametic_studio.device import dtype_for, resolve_device

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


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
        return cls(model, tok, dev)

    def __init__(self, model, tokenizer, device):
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.device = device
        self._stop = False
        self.last_raw = None  # per-layer [heads, kv] of the last decoded step (for drilldown)
        self._acts = {}       # (layer, module) -> last-token output norm
        self.modules_probed = self._register_activation_hooks()

    def _register_activation_hooks(self):
        names = ["self_attn", "mlp"]
        for i, layer in enumerate(self.model.model.layers):
            for name in names:
                getattr(layer, name).register_forward_hook(self._act_hook(i, name))
        return names

    def _act_hook(self, i, name):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out  # [b, seq, hidden]
            self._acts[(i, name)] = float(h[0, -1].detach().norm())
        return hook

    def compute_spot(self, examples):
        """Coding-Spot importance: accumulate |grad×param| over examples, reduced to [layer, module]."""
        L = len(self.model.model.layers)
        modules = [m.group(2) for name, _ in self.model.named_parameters()
                   if (m := _LAYER_RE.match(name)) and int(m.group(1)) == 0]
        acc = {}
        for text in examples:
            ids = torch.tensor([self.tok.encode(text)], device=self.device)
            self.model.zero_grad(set_to_none=True)
            self.model(ids, labels=ids).loss.backward()
            for name, p in self.model.named_parameters():
                m = _LAYER_RE.match(name)
                if m and p.grad is not None:
                    key = (int(m.group(1)), m.group(2))
                    acc[key] = acc.get(key, 0.0) + float((p.grad * p.data).abs().sum())
        self.model.zero_grad(set_to_none=True)
        n = max(1, len(examples))
        grid = [[acc.get((l, mod), 0.0) / n for mod in modules] for l in range(L)]
        return {"layers": L, "modules": modules, "grid": grid}

    def stop(self):
        self._stop = True

    def drilldown(self, layer):
        return self.last_raw[layer]  # [heads, kv] of the last step

    def generate_text(self, prompt, max_tokens, probes=("attention",)):
        tmpl = getattr(self.tok, "apply_chat_template", None)
        if tmpl and getattr(self.tok, "chat_template", None):
            prompt = tmpl([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = torch.tensor([self.tok.encode(prompt)])
        yield from self.generate(ids, max_tokens, probes)

    def generate(self, input_ids, max_tokens, probes=("attention",)):
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
            nxt = int(out.logits[0, -1].argmax())
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
