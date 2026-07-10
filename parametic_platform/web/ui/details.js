// Reproducibility (spec) and raw-artifacts tab panels. Ported from the original
// app.js renderSpecCard / renderRawArtifacts so the data contract is identical.
import { html } from "./common.js";
import { escapeHtml } from "../format.js";

// The reproducibility spec: how this run was produced + the pipeline stages.
export function SpecPanel({ spec }) {
  if (!spec || !spec.analysis) {
    return html`<div class="panel"><p class="muted-note">No spec available for this run.</p></div>`;
  }
  const a = spec.analysis;
  const pairs = [
    ["Model", a.model && (a.model.hf_model_id || a.model.id)],
    ["Language", a.area && a.area.language],
    ["Mode", a.mode && a.mode.id],
    ["Top-k", a.k],
    ["Seeds", a.mode && (a.mode.seeds || []).join(", ")],
    ["Samples", a.mode && a.mode.sample_size],
    ["Method", "grad × parameter"],
    ["Pipeline", a.pipeline_version],
  ].filter((p) => p[1] != null && p[1] !== "");
  const stages = (spec.stages || []).map((s) => s.name);

  return html`
    <div class="panel">
      <p class="panel-desc">Everything needed to reproduce this run — same model, area, mode, and k
        hash to the same cache key.</p>
      <dl class="spec-list">
        ${pairs.map((p) => html`<div><dt>${p[0]}</dt><dd>${String(p[1])}</dd></div>`)}
      </dl>
      ${stages.length ? html`
        <div class="spec-stages">${stages.join(" → ")}</div>` : null}
    </div>`;
}

const KIND_ORDER = ["report", "spec", "metric", "table", "figure", "log", "artifact"];

// Every output file, grouped by kind, plus masks summarized by region.
export function ArtifactPanel({ items, masks }) {
  items = items || [];
  masks = masks || [];
  const groups = {};
  for (const it of items) (groups[it.kind] = groups[it.kind] || []).push(it);
  const ordered = KIND_ORDER.filter((k) => groups[k]);
  const maskParams = masks.reduce((sum, m) => sum + (m.count || 0), 0);
  const total = items.length + maskParams;

  if (!items.length && !masks.length) {
    return html`<div class="panel"><p class="muted-note">No artifacts for this run.</p></div>`;
  }

  return html`
    <div class="panel">
      <p class="panel-desc">Full transparency — every output file is downloadable.
        ${total ? html`<span class="art-count">${total} files</span>` : null}</p>
      <ul class="artifact-list">
        ${ordered.map((kind) => html`
          <li class="art-group"><span class="art-kind">${kind}</span></li>
          ${groups[kind].map((it) => html`
            <li><a href=${encodeURI(it.url)} target="_blank" rel="noreferrer">${it.path}</a></li>`)}
        `)}
        ${masks.map((m) => html`
          <li class="mask-row">${m.path} — ${m.count} tensors (the spot)</li>`)}
      </ul>
    </div>`;
}
