// In-browser spot visualizations — the live replacement for the static PNG
// atlases. All drawn from spotdata's parsed CSV matrix (no images): a layer×
// module heatmap, a depth profile, an interactive tensor scatter (the honest
// stand-in for the future per-parameter hero), and a module concentration bar.
import { useState } from "preact/hooks";
import { html } from "./common.js";
import { deriveModules } from "../spotdata.js";

const lerp = (a, b, t) => a + (b - a) * t;

// Light → blue → deep-blue ramp by normalized importance.
function heat(v, max) {
  const t = Math.pow((v || 0) / (max || 1), 0.72);
  let r, g, b;
  if (t < 0.5) { const u = t / 0.5; r = lerp(234, 45, u); g = lerp(241, 108, u); b = lerp(252, 223, u); }
  else { const u = (t - 0.5) / 0.5; r = lerp(45, 21, u); g = lerp(108, 53, u); b = lerp(223, 111, u); }
  return `rgb(${r | 0},${g | 0},${b | 0})`;
}

// A — the model's anatomy: importance per (layer × module).
export function SpotHeatmap({ data }) {
  if (!data) return null;
  const head = ["q", "k", "v", "o", null, "gate", "up", "down"];
  const order = [0, 1, 2, 3, -1, 4, 5, 6]; // -1 = gap between attention and MLP
  const last = data.layers.length - 1;
  return html`
    <div class="heatmap">
      <div class="hm-legend"><span>low</span><i class="hm-scale"></i><span>high</span></div>
      <div class="hm-cols">
        ${head.map((c) => html`<div class=${c ? "hm-col" : "hm-gap"}>${c || ""}</div>`)}
      </div>
      ${data.layers.map((L, i) => html`
        <div class="hm-line">
          <div class="hm-rlab">${(L % 4 === 0 || L === data.layers[last]) ? "L" + L : ""}</div>
          <div class="hm-row">
            ${order.map((j) => j < 0
              ? html`<div class="hm-gap"></div>`
              : html`<div class="hm-cell"
                  title=${`layer ${L} · ${data.modules[j]} · ${data.cells[i][j].toFixed(2)}`}
                  style=${`background:${heat(data.cells[i][j], data.cellMax)}`}></div>`)}
          </div>
        </div>`)}
      <div class="hm-axis"><span>attention</span><span>MLP</span></div>
    </div>`;
}

// B — where it sits in depth: total importance per layer.
export function DepthProfile({ data }) {
  if (!data) return null;
  const W = 900, H = 220, pl = 40, pr = 14, pt = 14, pb = 28, iw = W - pl - pr, ih = H - pt - pb;
  const n = data.profile.length, mx = data.profileMax;
  const xf = (i) => pl + iw * i / (n - 1), yf = (v) => pt + ih * (1 - v / mx);
  let d = `M${xf(0)},${yf(data.profile[0])}`;
  data.profile.forEach((v, i) => { if (i) d += `L${xf(i)},${yf(v)}`; });
  const area = `${d}L${xf(n - 1)},${pt + ih}L${xf(0)},${pt + ih}Z`;
  const grid = [0, 1, 2, 3].map((g) => ({ y: pt + ih * g / 3, v: (mx * (1 - g / 3)).toFixed(0) }));
  const ticks = [0, 7, 14, 21, n - 1];
  return html`
    <svg class="chart" viewBox=${`0 0 ${W} ${H}`} role="img" aria-label="Importance by layer depth">
      ${grid.map((g) => html`
        <line class="gl" x1=${pl} y1=${g.y} x2=${W - pr} y2=${g.y} />
        <text class="ax" x=${pl - 6} y=${g.y + 3} text-anchor="end">${g.v}</text>`)}
      <path d=${area} fill="var(--blue)" opacity="0.09" />
      <path d=${d} fill="none" stroke="var(--blue)" stroke-width="2.5" stroke-linejoin="round" />
      ${data.profile.map((v, i) => html`
        <circle cx=${xf(i)} cy=${yf(v)} r=${i === 0 || i === n - 1 ? 3.5 : 2.4} fill="var(--blue)">
          <title>${`layer ${i} · ${v.toFixed(1)}`}</title></circle>`)}
      ${ticks.map((i) => html`<text class="ax" x=${xf(i)} y=${H - 9} text-anchor="middle">L${i}</text>`)}
    </svg>`;
}

// D — the spot as a field: every tensor at (depth, importance), sized by params,
// with a top-K% threshold slider. The interaction the per-parameter hero reuses.
export function TensorScatter({ data, foot }) {
  const [k, setK] = useState(20);
  if (!data || !data.points.length) return null;
  const W = 920, H = 300, pl = 42, pr = 16, pt = 14, pb = 28, iw = W - pl - pr, ih = H - pt - pb;
  const last = data.layers.length - 1;
  const ymax = data.pointMax * 1.05;
  const xf = (L) => pl + iw * L / last, yf = (v) => pt + ih * (1 - v / ymax);
  const rf = (s) => 3 + 6 * Math.sqrt((s || 0) / (data.selMax || 1));
  const sorted = [...data.points].sort((a, b) => b.v - a.v);
  const nOn = Math.max(1, Math.round(data.points.length * k / 100));
  const thr = sorted[nOn - 1].v;
  const grid = [0, 1, 2, 3];
  const ticks = [0, 7, 14, 21, last];
  return html`
    <div class="scatter">
      <div class="sc-top">
        <div class="sc-leg">
          <span><i class="sw blue"></i>MLP</span>
          <span><i class="sw gray"></i>attention</span>
        </div>
        <label class="sc-slider">top
          <input type="range" min="2" max="60" step="1" value=${k}
            onInput=${(e) => setK(+e.target.value)} aria-label="top K percent" />
          <b>${k}%</b>
        </label>
      </div>
      <svg class="chart" viewBox=${`0 0 ${W} ${H}`} role="img" aria-label="Tensor importance scatter">
        ${grid.map((g) => { const y = pt + ih * g / 3; return html`
          <line class="gl" x1=${pl} y1=${y} x2=${W - pr} y2=${y} />
          <text class="ax" x=${pl - 6} y=${y + 3} text-anchor="end">${(ymax * (1 - g / 3)).toFixed(1)}</text>`; })}
        ${ticks.map((L) => html`<text class="ax" x=${xf(L)} y=${H - 9} text-anchor="middle">L${L}</text>`)}
        ${data.points.map((p) => {
          const on = p.v >= thr;
          const fill = on ? (p.mlp ? "var(--blue)" : "#8a97a6") : "#dde3ea";
          return html`<circle cx=${xf(p.L)} cy=${yf(p.v)} r=${rf(p.sel)} fill=${fill} fill-opacity=${on ? 0.9 : 0.5}>
            <title>${`layer ${p.L} · ${p.m} · imp ${p.v.toFixed(2)} · ${p.sel.toLocaleString()} params`}</title></circle>`;
        })}
      </svg>
      <p class="sc-read"><b>${nOn.toLocaleString()}</b> of ${data.points.length} tensors above the top-${k}% importance cut · threshold ≥ ${thr.toFixed(2)}</p>
      ${foot ? html`<p class="sc-foot">${foot}</p>` : null}
    </div>`;
}

// C — what carries it: module-type share as a stacked bar + ranking.
const palette = (i) => i < 3 ? ["#15356f", "#2d6cdf", "#5b8be8"][i]
  : i < 7 ? ["#9bb6ea", "#b9ccf0", "#cdd9f2", "#dde6f6"][i - 3] : "#e7edf6";

export function ModuleConcentration({ csv }) {
  const mods = deriveModules(csv, 9);
  if (!mods.length) return null;
  const max = mods[0].share || 1;
  return html`
    <div class="concentration">
      <div class="stack">
        ${mods.map((m, i) => html`<span class="seg" title=${`${m.module} · ${m.share.toFixed(1)}%`}
          style=${`width:${m.share}%;background:${palette(i)}`}></span>`)}
      </div>
      <div class="module-rank">
        ${mods.slice(0, 7).map((m, i) => html`
          <div class="mod-row">
            <span class="mod-name" title=${m.module}>${m.module}</span>
            <span class="mod-bar"><span style=${`width:${(m.share / max * 100).toFixed(0)}%;background:${palette(i)}`}></span></span>
            <span class="mod-val">${m.share.toFixed(1)}%</span>
          </div>`)}
      </div>
    </div>`;
}
