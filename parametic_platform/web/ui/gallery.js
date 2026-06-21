// Gallery route (#/): big title, a collapsible new-analysis launcher, and a
// responsive grid of analysis cards. Card click → Report (#/a/:id).
import { useState } from "preact/hooks";
import { html, Brand, StatusPill, Modal } from "./common.js";
import { catalogLabel, navigate } from "../store.js";
import { createAnalysis, resolveModel, registerModel } from "../api.js";
import { shortId, fmtRatio } from "../format.js";
import { useDamageRatio } from "../spotdata.js";

// Operator "Add model" flow: enter an HF id → resolve a compatibility preview →
// register. Gated by the server capability flag (catalog.caps.model_registration).
function AddModel({ catalog, onRegistered }) {
  const [open, setOpen] = useState(false);
  const [hf, setHf] = useState("");
  const [revision, setRevision] = useState("main");
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const close = () => { setOpen(false); setErr(null); setPreview(null); setHf(""); setRevision("main"); };

  async function resolve() {
    setBusy(true); setErr(null); setPreview(null);
    try { setPreview(await resolveModel({ hf_model_id: hf.trim(), revision })); }
    catch (e) { setErr(e.message); }
    finally { setBusy(false); }
  }

  async function register() {
    setBusy(true); setErr(null);
    try {
      const res = await registerModel({ hf_model_id: hf.trim(), revision });
      await (catalog.refresh && catalog.refresh());
      onRegistered && onRegistered(res);
      close();
    } catch (e) { setErr(e.message); }
    finally { setBusy(false); }
  }

  const compat = preview && preview.compatibility;
  const spec = preview && preview.model_spec;
  const canRegister = compat && compat.status !== "unsupported";

  return html`
    <button class="btn ghost" onClick=${() => setOpen(true)}>＋ Add model</button>
    ${open ? html`
      <${Modal} title="Add a Hugging Face model" onClose=${close}>
        <div class="form-grid">
          <label><span>HF model id</span>
            <input value=${hf} placeholder="org/Model-Name" onInput=${(e) => setHf(e.target.value)} /></label>
          <label><span>Revision</span>
            <input value=${revision} placeholder="main" onInput=${(e) => setRevision(e.target.value)} /></label>
        </div>
        <div class="modal-actions">
          <button class="btn" disabled=${busy || !hf.trim()} onClick=${resolve}>${busy && !preview ? "Resolving…" : "Resolve"}</button>
        </div>

        ${compat ? html`
          <div class="resolve-card">
            <div class="resolve-head">
              <span class="compat-badge ${compat.status}">${compat.status}</span>
              <code>${compat.model_type || "?"}</code>
              <span class="resolve-params">${compat.estimated_params_human || ""}</span>
            </div>
            ${spec ? html`<div class="resolve-spec">
              id <code>${spec.id}</code> · tensors <code>${spec.expected_tensors}</code>${compat.expected_tensors_estimated ? html` <span class="muted">(est.)</span>` : null}
            </div>` : null}
            ${(compat.notes || []).map((n) => html`<p class="resolve-note">${n}</p>`)}
          </div>` : null}

        ${err ? html`<p class="launcher-err">${err}</p>` : null}
        <div class="modal-actions">
          <button class="btn ghost" onClick=${close}>Cancel</button>
          <button class="btn run" disabled=${busy || !canRegister} title=${canRegister ? "" : "Resolve a supported model first"} onClick=${register}>${busy && preview ? "Registering…" : "Register"}</button>
        </div>
      </${Modal}>` : null}`;
}

// New-analysis launcher: a primary button that opens the request form in a modal.
function Launcher({ catalog, onSubmitted }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [form, setForm] = useState({ model_id: "", area_id: "", mode: "", k: "0.01" });

  // Default the selects to the first catalog entry once it loads.
  const model_id = form.model_id || catalog.models[0]?.id || "";
  const area_id = form.area_id || catalog.areas[0]?.id || "";
  const mode = form.mode || catalog.modes[0]?.id || "";

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));
  const close = () => { setOpen(false); setErr(null); };

  async function submit() {
    setBusy(true); setErr(null);
    try {
      const row = await createAnalysis({ model_id, area_id, mode, k: form.k });
      onSubmitted && onSubmitted(row);
      close();
    } catch (e) { setErr(e.message); }
    finally { setBusy(false); }
  }

  const opt = (items) => items.map((it) => html`<option value=${it.id}>${it.display_name || it.id}</option>`);

  return html`
    <button class="btn run" onClick=${() => setOpen(true)}>＋ New analysis</button>
    ${open ? html`
      <${Modal} title="New analysis" onClose=${close}>
        <div class="form-grid">
          <label><span>Model</span>
            <select value=${model_id} onChange=${set("model_id")}>${opt(catalog.models)}</select></label>
          <label><span>Area</span>
            <select value=${area_id} onChange=${set("area_id")}>${opt(catalog.areas)}</select></label>
          <label><span>Mode</span>
            <select value=${mode} onChange=${set("mode")}>${opt(catalog.modes)}</select></label>
          <label><span>Top-k</span>
            <input inputmode="decimal" value=${form.k} placeholder="0.01" onInput=${set("k")} /></label>
        </div>
        ${err ? html`<p class="launcher-err">${err}</p>` : null}
        <div class="modal-actions">
          <button class="btn ghost" onClick=${close}>Cancel</button>
          <button class="btn run" disabled=${busy} onClick=${submit}>${busy ? "Running…" : "Run analysis"}</button>
        </div>
      </${Modal}>` : null}`;
}

function AnalysisCard({ row, catalog }) {
  const model = catalogLabel(catalog.models, row.model_id);
  const area = catalogLabel(catalog.areas, row.area_id);
  const mode = catalogLabel(catalog.modes, row.mode);
  const ok = row.status === "succeeded";
  const failed = row.status === "failed";
  const ratio = useDamageRatio(row.request_id, ok);

  return html`
    <button class="card ${row.status}" onClick=${() => navigate(`/a/${row.request_id}`)}>
      <div class="card-top">
        <${StatusPill} status=${row.status} />
        <code class="card-id" title=${row.request_id}>${shortId(row.request_id)}</code>
      </div>
      <div class="card-title">${model}</div>
      <div class="card-meta">${area} · ${mode}</div>
      ${ok ? html`
        <div class="card-metric">
          <span class="card-metric-label">PPL collapse</span>
          ${ratio != null
            ? html`<span class="card-metric-val">×${fmtRatio(ratio)}</span>`
            : html`<span class="card-metric-cue">open report →</span>`}
        </div>` : null}
      ${failed ? html`<p class="card-err">${row.error || "failed"}</p>` : null}
      ${!ok && !failed ? html`<p class="card-pending">in progress…</p>` : null}
    </button>`;
}

export function Gallery({ catalog, rows, error, refresh }) {
  return html`
    <header class="app-topbar">
      <a class="brand-link" href="/">
        <${Brand} />
      </a>
    </header>

    <div class="gallery">
      ${catalog.ok === false ? html`<p class="banner bad">API unavailable — ${catalog.error}</p>` : null}

      <div class="gallery-toolbar">
        <h1 class="gallery-title">Analyses
          <span class="toolbar-count">${rows.length}</span></h1>
        <div class="toolbar-actions">
          ${catalog.caps?.model_registration ? html`<${AddModel} catalog=${catalog} onRegistered=${catalog.refresh} />` : null}
          <${Launcher} catalog=${catalog} onSubmitted=${refresh} />
        </div>
      </div>

      ${error ? html`<p class="banner bad">${error}</p>` : null}
      ${rows.length === 0 ? html`<p class="empty">No analyses yet. Run one to start.</p>` : html`
        <div class="card-grid">
          ${rows.map((row) => html`<${AnalysisCard} key=${row.request_id} row=${row} catalog=${catalog} />`)}
        </div>`}
    </div>`;
}
