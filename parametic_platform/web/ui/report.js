// Report route (#/a/:id). Stage 2: back bar + hero (punchline + reserved K%
// scatter slot) + first-class failed/running/queued states. The four-act spot
// story, reproducibility, and artifacts arrive in later stages.
import { useState } from "preact/hooks";
import { html, StatusPill, Spinner, TabBar } from "./common.js";
import { navigate, catalogLabel } from "../store.js";
import { shortId, fmtCompact, fmtRatio } from "../format.js";
import { deriveHero } from "../spotdata.js";
import { STORY_TABS, StoryPanel } from "./acts.js";
import { SpecPanel, ArtifactPanel } from "./details.js";
import { SpotAtlas } from "./viz.js";

// Story acts + reproducibility + artifacts, all in one tab bar.
const REPORT_TABS = [...STORY_TABS, { id: "repro", label: "Reproducibility" }, { id: "artifacts", label: "Artifacts" }];

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

function AtlasPlaceholder() {
  return html`
    <div class="scatter-slot" role="img" aria-label="Spot map unavailable">
      <div class="scatter-center">
        <span class="scatter-ic" aria-hidden="true">⊞</span>
        <p class="scatter-title">Spot map unavailable</p>
        <p class="scatter-sub">This run did not emit a per-tensor parameter summary.</p>
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
      <h1 class="hero-h1">The ${lang} spot</h1>
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
        ? html`<div class="hero-atlas"><${SpotAtlas} data=${matrix}
            foot=${`Aggregated to ${matrix.points.length} tensors (layer × module). A per-parameter view (docs/image.png) can replace this once that dump exists.`} /></div>`
        : html`<${AtlasPlaceholder} />`}
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

// Succeeded report: hero pinned on top, the rest behind a single tab bar.
function SucceededReport({ analysis, meta }) {
  const { artifacts, masks, spec, spot } = analysis;
  const [tab, setTab] = useState(REPORT_TABS[0].id);
  const panel = tab === "repro" ? html`<${SpecPanel} spec=${spec} />`
    : tab === "artifacts" ? html`<${ArtifactPanel} items=${artifacts} masks=${masks} />`
    : html`<${StoryPanel} id=${tab} spot=${spot} />`;
  return html`
    <${Hero} meta=${meta} spot=${spot} />
    <${TabBar} tabs=${REPORT_TABS} active=${tab} onSelect=${setTab} />
    <div class="tab-panel">${panel}</div>`;
}

export function Report({ id, analysis, listRow, catalog }) {
  const { row, spec, loading, error } = analysis;
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
        ${row && status === "succeeded" ? html`<${SucceededReport} analysis=${analysis} meta=${meta} />` : null}
        ${row && status === "failed" ? html`<${FailedState} row=${row} spec=${spec} />` : null}
        ${row && (status === "queued" || status === "running") ? html`<${PendingState} status=${status} />` : null}
      </div>
    </div>`;
}
