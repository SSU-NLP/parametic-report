// The spot story for a succeeded run, served as tab panels: where the spot
// lives, what it is, and what removing it does. Pure presentation over the
// derivations in spotdata.js — all drawn in-browser (no static figures).
import { html } from "./common.js";
import { fmtCompact } from "../format.js";
import { deriveCausal } from "../spotdata.js";
import { SpotHeatmap, DepthProfile, ModuleConcentration } from "./viz.js";

export const STORY_TABS = [
  { id: "where", label: "Where it lives" },
  { id: "what", label: "What it is" },
  { id: "damage", label: "What removing it does" },
];

const DESC = {
  where: "Importance across all 28 layers × 7 weight modules, drawn live from the run — the three MLP columns are the spine, and it is densest in the earliest layers.",
  what: "Share of importance by module type. Roughly three-quarters of the spot lives in the MLP feed-forward weights; attention contributes mostly through o_proj.",
  damage: "Zero the spot vs. an equal-size random/bottom region, then measure code perplexity. Only the spot breaks coding — the causal payoff.",
};

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

// One story act rendered as a tab panel (header line + its in-browser viz).
export function StoryPanel({ id, spot }) {
  const matrix = spot && spot.matrix;
  const csv = spot && spot.csv;
  const ppl = spot && spot.ppl;

  let body;
  if (id === "where") {
    body = matrix
      ? html`
        <${SpotHeatmap} data=${matrix} />
        <div class="act-subhead">Same signal, by depth</div>
        <${DepthProfile} data=${matrix} />`
      : html`<p class="muted-note">Per-tensor matrix unavailable for this run.</p>`;
  } else if (id === "what") {
    body = html`<${ModuleConcentration} csv=${csv} />`;
  } else {
    body = html`<${CausalChart} ppl=${ppl} />`;
  }

  return html`
    <div class="panel">
      <p class="panel-desc">${DESC[id]}</p>
      ${body}
    </div>`;
}
