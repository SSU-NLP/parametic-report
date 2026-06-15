// Gallery route (#/): big title, a collapsible new-analysis launcher, and a
// responsive grid of analysis cards. Card click → Report (#/a/:id).
import { useState } from "preact/hooks";
import { html, Brand, StatusPill } from "./common.js";
import { catalogLabel, navigate } from "../store.js";
import { createAnalysis } from "../api.js";
import { shortId, fmtRatio } from "../format.js";
import { useDamageRatio } from "../spotdata.js";

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

  async function submit() {
    setBusy(true); setErr(null);
    try {
      const row = await createAnalysis({ model_id, area_id, mode, k: form.k });
      onSubmitted && onSubmitted(row);
      setOpen(false);
    } catch (e) { setErr(e.message); }
    finally { setBusy(false); }
  }

  if (!open) {
    return html`
      <div class="launcher closed">
        <button class="btn ghost" onClick=${() => setOpen(true)}>＋ New analysis</button>
      </div>`;
  }

  const opt = (items) => items.map((it) => html`<option value=${it.id}>${it.display_name || it.id}</option>`);

  return html`
    <div class="launcher open">
      <label><span>Model</span>
        <select value=${model_id} onChange=${set("model_id")}>${opt(catalog.models)}</select></label>
      <label><span>Area</span>
        <select value=${area_id} onChange=${set("area_id")}>${opt(catalog.areas)}</select></label>
      <label><span>Mode</span>
        <select value=${mode} onChange=${set("mode")}>${opt(catalog.modes)}</select></label>
      <label class="k"><span>Top-k</span>
        <input inputmode="decimal" value=${form.k} placeholder="0.01" onInput=${set("k")} /></label>
      <div class="launcher-actions">
        <button class="btn run" disabled=${busy} onClick=${submit}>${busy ? "Running…" : "Run analysis"}</button>
        <button class="btn ghost small" onClick=${() => setOpen(false)}>Cancel</button>
      </div>
      ${err ? html`<p class="launcher-err">${err}</p>` : null}
    </div>`;
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
    <div class="gallery">
      <header class="gallery-head">
        <${Brand} tagline="Coding Spot discovery" />
        <h1>LLM Coding Spot</h1>
        <p class="lede">The tiny set of parameters that, when zeroed, collapses coding ability.
          Pick a run to see where the spot lives, what it is, whether it is stable, and what removing it does.</p>
        ${catalog.ok === false ? html`<p class="banner bad">API unavailable — ${catalog.error}</p>` : null}
      </header>

      <${Launcher} catalog=${catalog} onSubmitted=${refresh} />

      ${error ? html`<p class="banner bad">${error}</p>` : null}
      ${rows.length === 0 ? html`<p class="empty">No analyses yet. Run one to start.</p>` : html`
        <div class="card-grid">
          ${rows.map((row) => html`<${AnalysisCard} key=${row.request_id} row=${row} catalog=${catalog} />`)}
        </div>`}
    </div>`;
}
