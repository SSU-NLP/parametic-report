// The four-act "spot story" for a succeeded run: where the spot lives, what it
// is, whether it is stable, and what removing it does. Pure presentation over
// the derivations in spotdata.js. Figures open a zoom overlay via onZoom.
import { html } from "./common.js";
import { fmtCompact } from "../format.js";
import { figure, deriveCausal } from "../spotdata.js";
import { SpotHeatmap, DepthProfile, ModuleConcentration } from "./viz.js";

function Figure({ item, caption, onZoom }) {
  if (!item) return null;
  const cap = caption || item.path.split("/").pop();
  const url = encodeURI(item.url);
  return html`
    <figure class="figure">
      <img src=${url} alt=${cap} loading="lazy" onClick=${() => onZoom(url, cap)} />
      <figcaption>${cap}</figcaption>
    </figure>`;
}

function Act({ num, title, desc, children }) {
  return html`
    <section class="act">
      <div class="act-head">
        <span class="act-num">${num}</span>
        <div><h2>${title}</h2><p>${desc}</p></div>
      </div>
      <div class="act-body">${children}</div>
    </section>`;
}

function CausalChart({ ppl }) {
  const rows = deriveCausal(ppl);
  if (!rows.length) return html`<p class="muted-note">No damage metrics available.</p>`;
  return html`
    <div class="damage-chart">
      ${rows.map((r) => html`
        <div class="dmg-row ${r.isSpot ? "spot" : ""}">
          <span class="dmg-name">${r.name}</span>
          <span class="dmg-bar"><span style=${`width:${r.width.toFixed(1)}%`}></span></span>
          <span class="dmg-val">PPL ${fmtCompact(r.ppl)}</span>
        </div>`)}
    </div>
    <p class="dmg-note">Equal-size random/bottom regions barely move perplexity —
      only the discovered spot is causally responsible for coding.</p>`;
}

export function SpotStory({ items, spot, onZoom }) {
  const ppl = spot && spot.ppl;
  const csv = spot && spot.csv;
  const matrix = spot && spot.matrix;
  return html`
    <div class="spot-story">
      <${Act} num="1" title="Where it lives"
        desc="Importance across all 28 layers × 7 weight modules, drawn live from the run — the three MLP columns are the spine, and it is densest in the earliest layers.">
        ${matrix ? html`
          <${SpotHeatmap} data=${matrix} />
          <div class="act-subhead">Same signal, by depth</div>
          <${DepthProfile} data=${matrix} />
        ` : html`<p class="muted-note">Per-tensor matrix unavailable for this run.</p>`}
      </${Act}>

      <${Act} num="2" title="What it is"
        desc="Share of importance by module type. Roughly three-quarters of the spot lives in the MLP feed-forward weights; attention contributes mostly through o_proj.">
        <${ModuleConcentration} csv=${csv} />
      </${Act}>

      <${Act} num="3" title="Is it stable?"
        desc="Two independent calibration seeds. Agreement means the spot is real signal, not noise.">
        <div class="fig-grid">
          <${Figure} item=${figure(items, "seedAgreement")} caption="Seed agreement" onZoom=${onZoom} />
          <${Figure} item=${figure(items, "seedDisagreement")} caption="Seed disagreement" onZoom=${onZoom} />
        </div>
      </${Act}>

      <${Act} num="4" title="What removing it does"
        desc="Zero the spot vs. an equal-size random/bottom region, then measure code perplexity. Only the spot breaks coding — the causal payoff.">
        <${CausalChart} ppl=${ppl} />
      </${Act}>
    </div>`;
}

export function ZoomOverlay({ zoom, onClose }) {
  if (!zoom) return null;
  return html`
    <div class="zoom-overlay open" onClick=${onClose}>
      <img src=${zoom.url} alt=${zoom.caption || ""} />
      <div class="zoom-cap">${zoom.caption || ""}</div>
    </div>`;
}
