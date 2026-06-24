# Activations View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live **Activations** view (layer×module output-norm heatmap) end-to-end, introducing the **probe-subscription** mechanism so the kernel only computes the probes a view needs.

**Architecture:** `ModelSession.generate` gains a `probes` set; it gates the expensive `output_attentions` and assembles an `act` frame from forward-hook norms only when requested. WS emits a separate `activation` event. The frontend tabs become functional (Attention ｜ Activations), each rendering its probe's frame; the generate request subscribes to both probes.

**Tech Stack:** Python 3.11 (pyenv `python`), torch (mps), transformers 4.49 (eager), FastAPI WS, React+Vite+TS.

## Global Constraints

- Python interpreter: `python` (pyenv shim; has torch+mps). Run tests with `python -m pytest`.
- Models load with `attn_implementation="eager"`; mps dtype = **bf16** (fp16 overflows → garbage).
- TDD: tests first, tiny eager Llama on CPU (`cfg._attn_implementation = "eager"`). No GPU in tests.
- ponytail: minimal code; no KV cache (full recompute is fine for demo lengths); reuse existing helpers.
- **Commits only when the user asks** (CLAUDE.md). Each task ends by running the studio suite green as a checkpoint; do NOT `git commit` unless the user requests it.
- Visual language (`STUDIO_UI_SPEC §3`): opencode dark, JetBrains Mono, **blue (`#007AFF`) = activity** (attention/activation). Cells via the existing `cellColor` ramp.
- Probe names are exact strings: `"attention"`, `"activation"`. Module probe names: `"self_attn"`, `"mlp"`.

---

### Task 1: Probe subscription in `generate`

Gate attention computation behind a `probes` set so callers opt in. Default keeps current behavior.

**Files:**
- Modify: `parametic_studio/kernel/model_session.py` (`generate`, `generate_text`)
- Test: `tests/studio/test_model_session.py`

**Interfaces:**
- Produces: `ModelSession.generate(input_ids, max_tokens, probes=("attention",)) -> Iterator[dict]`. Event dict always has `step,token_id,text`; has `attn` (Tensor `[L,kv]`) only when `"attention" in probes`. `generate_text(prompt, max_tokens, probes=("attention",))` forwards `probes`.

- [ ] **Step 1: Write the failing test**

```python
def test_generate_omits_attn_when_not_subscribed():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    evs = list(s.generate(PROMPT, max_tokens=2, probes=()))
    assert [e["step"] for e in evs] == [0, 1]
    assert all("attn" not in e for e in evs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/studio/test_model_session.py::test_generate_omits_attn_when_not_subscribed -q`
Expected: FAIL — `generate()` got an unexpected keyword argument `probes` (or `attn` present).

- [ ] **Step 3: Write minimal implementation**

In `model_session.py`, change `generate_text` and `generate` signatures and gate attention:

```python
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
        for step in range(max_tokens):
            if self._stop:
                return
            with torch.no_grad():
                out = self.model(ids, output_attentions=want_attn)  # ponytail: no KV cache
            nxt = int(out.logits[0, -1].argmax())
            if nxt == eos:
                return
            event = {"step": step, "token_id": nxt, "text": self.tok.decode([nxt])}
            if want_attn:
                self.last_raw = [a[0, :, -1, :].float() for a in out.attentions]  # per layer [heads, kv]
                event["attn"] = torch.stack([r.mean(0) for r in self.last_raw])    # [L, kv]
            ids = torch.cat([ids, torch.tensor([[nxt]], device=self.device)], dim=1)
            yield event
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/studio/test_model_session.py -q`
Expected: PASS (new test + existing attention/eos/stop/generate_text tests still green — default `probes=("attention",)` preserves `attn`).

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/studio/ -q` → all green. (No commit unless asked.)

---

### Task 2: Activation probe (forward-hook output norms)

Capture each decoder layer's `self_attn` and `mlp` output norm for the current token; assemble an `[L, M]` frame when subscribed.

**Files:**
- Modify: `parametic_studio/kernel/model_session.py` (`__init__`, new `_register_activation_hooks`/`_act_hook`, `generate`)
- Test: `tests/studio/test_model_session.py`

**Interfaces:**
- Produces: event dict has `act` (Tensor `[L, M]`, M = len of `self.modules_probed`, values = last-token output norms ≥ 0) when `"activation" in probes`. `self.modules_probed == ["self_attn", "mlp"]`.

- [ ] **Step 1: Write the failing test**

```python
def test_generate_yields_activation_frame():
    s = ModelSession(_tiny(), _Tok(eos=-1), torch.device("cpu"))
    evs = list(s.generate(PROMPT, max_tokens=3, probes=("activation",)))
    L = 2  # tiny model layers
    for e in evs:
        assert "attn" not in e          # not subscribed
        assert e["act"].shape == (L, 2)  # [L, modules: self_attn, mlp]
        assert (e["act"] >= 0).all()     # norms are non-negative
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/studio/test_model_session.py::test_generate_yields_activation_frame -q`
Expected: FAIL — `KeyError: 'act'`.

- [ ] **Step 3: Write minimal implementation**

In `__init__`, after `self._stop = False`, register hooks:

```python
        self.last_raw = None
        self._acts = {}                                  # (layer, module) -> last-token norm
        self.modules_probed = self._register_activation_hooks()
```

Add the hook methods (anywhere in the class):

```python
    def _register_activation_hooks(self):
        names = ["self_attn", "mlp"]
        for i, layer in enumerate(self.model.model.layers):
            for name in names:
                getattr(layer, name).register_forward_hook(self._act_hook(i, name))
        return names

    def _act_hook(self, i, name):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out   # [b, seq, hidden]
            self._acts[(i, name)] = float(h[0, -1].norm())
        return hook
```

In `generate`, after the `want_attn` block (before `ids = torch.cat...`), add:

```python
            if "activation" in probes:
                L = len(self.model.model.layers)
                event["act"] = torch.tensor(
                    [[self._acts[(l, m)] for m in self.modules_probed] for l in range(L)]
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/studio/test_model_session.py -q`
Expected: PASS (new activation test + all existing).

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/studio/ -q` → all green.

---

### Task 3: WS `activation` event + `probes` request param

The WS `generate` message carries `probes`; the handler forwards them and emits an `activation` event per step when present.

**Files:**
- Modify: `parametic_studio/api.py` (`_run_generation`)
- Test: `tests/studio/test_ws.py`

**Interfaces:**
- Consumes: `ModelSession.generate_text(prompt, max_tokens, probes)`.
- Produces: WS server→client `{"type":"activation","run":0,"step":int,"shape":[L,M],"data":[[...]]}`. `generate` client message accepts optional `"probes": ["attention","activation"]` (default `["attention"]`).

- [ ] **Step 1: Write the failing test**

```python
def test_ws_streams_activation_when_subscribed():
    api.SESSION = _tiny_session()
    with TestClient(api.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "generate", "prompt": "hi", "max_tokens": 4, "probes": ["activation"]})
        acts = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "done":
                break
            if msg["type"] == "activation":
                acts.append(msg)
    assert [a["step"] for a in acts] == [0, 1, 2, 3]
    assert acts[0]["shape"] == [2, 2]   # [L, modules]
    assert len(acts[0]["data"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/studio/test_ws.py::test_ws_streams_activation_when_subscribed -q`
Expected: FAIL — no `activation` messages (loop ends at `done` with empty `acts`).

- [ ] **Step 3: Write minimal implementation**

In `api.py` `_run_generation`, read probes and emit the activation event. Replace the generation loop body:

```python
    session = SESSION
    max_tokens = msg.get("max_tokens", 256)
    probes = msg.get("probes", ["attention"])
    stopped = asyncio.Event()
    # ... listen_stop unchanged ...
    listener = asyncio.create_task(listen_stop())
    count = 0
    for ev in session.generate_text(msg["prompt"], max_tokens, probes=probes):
        await websocket.send_json(
            {"type": "token", "run": 0, "step": ev["step"], "token_id": ev["token_id"], "text": ev["text"]}
        )
        if "attn" in ev:
            await websocket.send_json(
                {"type": "attention", "run": 0, "step": ev["step"],
                 "shape": list(ev["attn"].shape), "data": ev["attn"].cpu().tolist()}
            )
        if "act" in ev:
            await websocket.send_json(
                {"type": "activation", "run": 0, "step": ev["step"],
                 "shape": list(ev["act"].shape), "data": ev["act"].cpu().tolist()}
            )
        count += 1
    # ... listener cancel + done unchanged ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/studio/ -q`
Expected: PASS (new activation WS test + existing generate/stop/drilldown/attention tests; default `probes=["attention"]` preserves attention streaming).

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/ -q` → all green (studio + platform).

---

### Task 4: Frontend — functional tabs + Activations view

Make the canvas tabs switch the rendered view; add the Activations heatmap (layer×module, blue). The generate request subscribes to both probes so both tabs have data without re-running.

**Files:**
- Modify: `studio_web/src/App.tsx`

**Interfaces:**
- Consumes: WS `attention` and `activation` events.
- Produces: `activeView: 'attention' | 'activations'` state; `act: number[][] | null` (latest activation frame).

- [ ] **Step 1: Add activation state + probe subscription**

In `App()`, add state and request both probes. Change the `attention`/new `activation` handling and the generate message:

```tsx
  const [act, setAct] = useState<number[][] | null>(null)
  const [activeView, setActiveView] = useState<'attention' | 'activations'>('attention')
```

In `send()`: reset `setAct(null)`, send `probes`, and handle the activation message:

```tsx
    sock.onopen = () => sock.send(JSON.stringify({ type: 'generate', prompt, max_tokens: 64, probes: ['attention', 'activation'] }))
    sock.onmessage = (e) => {
      const m = JSON.parse(e.data)
      if (m.type === 'token') { setOutput((o) => o + m.text); setCount((c) => c + 1) }
      else if (m.type === 'attention') setFrames((f) => [...f, m.data])
      else if (m.type === 'activation') setAct(m.data)
      else if (m.type === 'perhead') setPerhead({ layer: m.layer, data: m.data })
      else if (m.type === 'done') setBusy(false)
    }
```

- [ ] **Step 2: Make the tab bar switch views**

Replace the decorative `TABS.map(...)` tab bar with clickable tabs bound to `activeView` (keep `spot map`/`tensor` disabled):

```tsx
          <div style={{ display: 'flex', borderBottom: '1px solid var(--line-strong)', background: 'var(--bg-1)' }}>
            {([['attention', 'attention'], ['activations', 'activations']] as const).map(([id, label]) => (
              <span key={id} onClick={() => setActiveView(id)}
                style={{ padding: '6px 14px', cursor: 'pointer',
                  color: activeView === id ? 'var(--text-0)' : 'var(--text-2)',
                  background: activeView === id ? 'var(--bg-2)' : 'transparent',
                  borderBottom: activeView === id ? '2px solid var(--accent)' : '2px solid transparent' }}>{label}</span>
            ))}
            <span style={{ padding: '6px 14px', color: 'var(--text-2)', opacity: 0.5 }}>spot map</span>
            <span style={{ padding: '6px 14px', color: 'var(--text-2)', opacity: 0.5 }}>tensor</span>
          </div>
```

- [ ] **Step 3: Render the active view**

In the canvas body, wrap the existing attention rendering in `activeView === 'attention'` and add the activations branch (reuses the `Grid` component; M columns = self_attn, mlp):

```tsx
          <div style={{ flex: 1, overflow: 'auto', padding: '12px 14px' }}>
            {activeView === 'attention' && (<>
              {/* ...existing attention strip + triangle + per-head JSX unchanged... */}
            </>)}
            {activeView === 'activations' && (
              act ? (<>
                <div style={{ color: 'var(--text-1)', marginBottom: 8 }}>
                  layer × module · output norm · {act.length}×{act[0].length} <span style={{ color: 'var(--text-2)' }}>(self_attn · mlp)</span>
                </div>
                <Grid rows={act} cols={act[0].length} rowH={7} />
              </>) : <span style={{ color: 'var(--text-2)' }}>[ activations appear here while generating ]</span>
            )}
          </div>
```

- [ ] **Step 4: Verify end-to-end in the browser preview**

1. Ensure the kernel runs the new code: stop the old server (`ps ax | grep parametic_studio.api | grep -v grep | awk '{print $1}' | while read p; do kill "$p"; done`), then start it in background (`python -m parametic_studio.api`), wait for `:8000` to answer `/docs` with 200.
2. `preview_start` the `studio_web` server (reuse if running). Click the chat send button; wait ~10s for generation.
3. `preview_snapshot` / `preview_screenshot`: confirm Attention tab shows the heatmap, click the `activations` tab → an L×2 blue heatmap appears (`layer × module · output norm`).
4. `preview_console_logs level=error` → expect none.

- [ ] **Step 5: Checkpoint**

Run: `python -m pytest tests/ -q` → all green (backend unaffected). Frontend verified via preview screenshot. (No commit unless asked.)

---

## Self-Review

- **Spec coverage** (`STUDIO_VIEW_ARCH.md`): probe-subscription (§3 probe) → Task 1; Activations view (§5 catalog, `activation` probe, blue) → Tasks 2–4; view-module switching seed (§3 view registry, minimal) → Task 4. Model tabs, dataset source, Spot/Logit/Tensor views, C split, knobs → **out of scope for this plan** (separate milestones per §8).
- **Placeholder scan:** none — every code step shows full code; preview steps name exact MCP tools.
- **Type consistency:** `probes` is a set/list of exact strings `"attention"`/`"activation"`; event keys `attn`/`act`; WS types `attention`/`activation`; `modules_probed == ["self_attn","mlp"]` → activation shape `[L,2]` used consistently in Tasks 2–4.

## Execution Handoff

Scoped to the Activations milestone. Next milestones (own plans): Logit lens, Tensor inspector, Spot map (+ dataset source), model tabs (multi-session), C free-split shell, knob/diff.
