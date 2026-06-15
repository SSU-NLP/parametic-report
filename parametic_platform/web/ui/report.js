// Report route (#/a/:id). Stage 2: back bar + hero (punchline + reserved K%
// scatter slot) + first-class failed/running/queued states. The four-act spot
// story, reproducibility, and artifacts arrive in later stages.
import { html, StatusPill, Spinner } from "./common.js";
import { navigate, catalogLabel } from "../store.js";
import { shortId, fmtCompact, fmtRatio } from "../format.js";
import { deriveHero } from "../spotdata.js";
import { SpotStory } from "./acts.js";
import { TensorScatter } from "./viz.js";

// Title/meta resolved from spec when present, else from the list row labels.
function reportMeta(spec, listRow, catalog) {
  const a = (spec && spec.analysis) || {};
  const model = (a.model && a.model.display_name)
    || (listRow ? catalogLabel(catalog.models, listRow.model_id) : "Analysis");
  const lang = (a.area && a.area.language)
    || (listRow ? catalogLabel(catalog.areas, listRow.area_id) : "");
  const k = a.k != null ? a.k : null;
  return { model, lang, k };
}

function ScatterPlaceholder() {
  return html`
    <div class="scatter-slot" role="img" aria-label="K-percent parameter scatter — coming soon">
      <div class="scatter-axes">
        <span class="axis-y">gradient ↑</span>
        <span class="axis-x">value →</span>
      </div>
      <div class="scatter-center">
        <span class="scatter-ic" aria-hidden="true">⋰</span>
        <p class="scatter-title">K%-slider parameter scatter</p>
        <p class="scatter-sub">Hero slot reserved — needs a per-parameter dump (follow-up run)</p>
      </div>
    </div>`;
}

function Hero({ meta, spot }) {
  const hero = deriveHero(spot && spot.ppl);
  const matrix = spot && spot.matrix;
  const kLabel = meta.k != null ? `top-${meta.k}` : "the spot";
  const lang = meta.lang || "code";

  return html`
    <section class="hero">
      <h1 class="hero-h1">Where the ${lang} coding spot lives</h1>
      ${hero ? html`
        <p class="hero-punch">
          Zeroing the ${kLabel} coding spot collapses ${lang} modeling — PPL${" "}
          <b>${fmtCompact(hero.origPpl)}</b> → <b class="bad">${fmtCompact(hero.spotPpl)}</b>
          <span class="hero-x">×${fmtRatio(hero.ratio)}</span>
        </p>
        ${hero.zeroedParams ? html`
          <p class="hero-sub">${Number(hero.zeroedParams).toLocaleString()} params
            (${hero.maskTensors} tensors) removed</p>` : null}
      ` : html`<p class="hero-sub">Damage metrics unavailable for this run.</p>`}
      ${matrix
        ? html`<div class="hero-scatter"><${TensorScatter} data=${matrix}
            foot=${`Aggregated to ${matrix.points.length} tensors (layer × module). The per-parameter scatter (docs/image.png) drops in here once that dump exists.`} /></div>`
        : html`<${ScatterPlaceholder} />`}
    </section>`;
}

// Pull the failing stage name out of the error breadcrumb.
function failingStage(error) {
  const m = /stage '([^']+)'/.exec(error || "");
  return m ? m[1] : null;
}

function FailedState({ row, spec }) {
  const stage = failingStage(row.error);
  const stages = (spec && spec.stages) || [];
  return html`
    <section class="state-card bad">
      <div class="state-head"><span class="state-ic" aria-hidden="true">!</span>
        <div><h2>This analysis failed</h2>
          ${stage ? html`<p>Failed at stage <code>${stage}</code>.</p>` : null}</div>
      </div>
      <pre class="state-error">${row.error || "Unknown error."}</pre>
      ${stages.length ? html`
        <div class="stage-track">
          ${stages.map((s) => html`
            <span class="stage ${s.name === stage ? "failed" : ""}">${s.name}</span>`)}
        </div>` : null}
    </section>`;
}

function PendingState({ status }) {
  return html`
    <section class="state-card warn">
      <div class="state-head"><${Spinner} label=${`Analysis is ${status}…`} /></div>
      <p class="state-note">Results appear here automatically when it finishes (polling every 5s).</p>
    </section>`;
}

export function Report({ id, analysis, listRow, catalog }) {
  const { row, spec, spot, loading, error } = analysis;
  const status = (row && row.status) || (listRow && listRow.status);
  const meta = reportMeta(spec, listRow, catalog);
  const titleBits = [meta.model, meta.lang, meta.k != null ? `top-${meta.k}` : null].filter(Boolean);

  return html`
    <div class="report">
      <div class="report-bar">
        <button class="back" onClick=${() => navigate("/")}>← All analyses</button>
        <span class="report-bar-title">${titleBits.length ? titleBits.join(" · ") : shortId(id)}</span>
        ${status ? html`<${StatusPill} status=${status} />` : null}
      </div>

      <div class="report-body">
        ${loading && !row ? html`<${Spinner} label="Loading analysis…" />` : null}
        ${error ? html`<p class="banner bad">${error}</p>` : null}
        ${row && status === "succeeded" ? html`
          <${Hero} meta=${meta} spot=${spot} />
          <${SpotStory} spot=${spot} />
        ` : null}
        ${row && status === "failed" ? html`<${FailedState} row=${row} spec=${spec} />` : null}
        ${row && (status === "queued" || status === "running") ? html`<${PendingState} status=${status} />` : null}
      </div>
    </div>`;
}
