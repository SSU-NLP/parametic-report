// The spot story for a succeeded run: where the spot lives, what it is, and what
// removing it does. Pure presentation over the derivations in spotdata.js — all
// drawn in-browser from the run's CSV/metrics (no static figures).
import { html } from "./common.js";
import { fmtCompact } from "../format.js";
import { deriveCausal } from "../spotdata.js";
import { SpotHeatmap, DepthProfile, ModuleConcentration } from "./viz.js";

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

export function SpotStory({ spot }) {
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

      <${Act} num="3" title="What removing it does"
        desc="Zero the spot vs. an equal-size random/bottom region, then measure code perplexity. Only the spot breaks coding — the causal payoff.">
        <${CausalChart} ppl=${ppl} />
      </${Act}>
    </div>`;
}
