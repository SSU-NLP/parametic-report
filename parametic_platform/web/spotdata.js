// Derivations over a succeeded run's artifacts — the shared "spot story" data
// layer used by the gallery card metric, the report hero, and the acts. Keeps
// the figure-matching + ppl/module parsing in one place (ported from app.js).
import { useState, useEffect } from "preact/hooks";
import { api, apiText, getArtifacts } from "./api.js";
import { findItem, parseCsv } from "./format.js";

// Figure lookups are k-agnostic (match by name substring).
export const FIGURES = {
  maskAtlas: "spot_mask_atlas",
  importanceAtlas: "spot_importance_atlas",
  layerModule: "layer_module_importance_sum",
  bubbleMap: "importance_bubble_map",
  seedAgreement: "seed_agreement_atlas",
  seedDisagreement: "seed_disagreement_atlas",
};

export const figure = (items, key) => findItem(items, FIGURES[key]);

// The PPL damage rows: original / code (spot) / bottom / random controls.
export function findPplItem(items) {
  return (items || []).find((i) => i.kind === "metric" && i.path.includes("ppl_damage"));
}
export function findModuleCsvItem(items) {
  return (items || []).find((i) => i.kind === "table" && i.path.includes("module_summary"));
}

// The headline: zeroing the spot multiplies PPL by `ratio`.
export function deriveHero(ppl) {
  if (!Array.isArray(ppl)) return null;
  const orig = ppl.find((r) => r.model === "original");
  const spot = ppl.find((r) => String(r.model).startsWith("code"));
  if (!orig || !spot) return null;
  return {
    origPpl: orig.ppl,
    spotPpl: spot.ppl,
    ratio: spot.ppl / orig.ppl,
    zeroedParams: spot.zeroed_params,
    maskTensors: spot.mask_tensors,
  };
}

// Causal chart rows with a log-scaled bar width and human labels.
export function deriveCausal(ppl) {
  if (!Array.isArray(ppl)) return [];
  const label = (m) =>
    m === "original" ? "Original (intact)"
      : String(m).startsWith("code") ? "Spot removed"
      : String(m).startsWith("bottom") ? "Bottom region (control)"
      : String(m).startsWith("random") ? "Random region (control)"
      : m;
  const maxLog = Math.max(...ppl.map((r) => Math.log10(Math.max(r.ppl, 1))), 1);
  return ppl.map((r) => ({
    name: label(r.model),
    ppl: r.ppl,
    isSpot: String(r.model).startsWith("code"),
    width: (Math.log10(Math.max(r.ppl, 1)) / maxLog) * 100,
  }));
}

// Top weight modules by importance share (Act 2 ranking bars).
export function deriveModules(csvText, limit = 8) {
  let rows;
  try { rows = parseCsv(csvText).filter((r) => r.module); } catch { return []; }
  rows.sort((a, b) => parseFloat(b.importance_share) - parseFloat(a.importance_share));
  return rows.slice(0, limit).map((r) => ({
    module: r.module,
    share: (parseFloat(r.importance_share) || 0) * 100,
  }));
}

// Fetch metric JSON + module CSV for a succeeded run's artifact list.
export async function loadSpotData(items) {
  const pplItem = findPplItem(items);
  const csvItem = findModuleCsvItem(items);
  let ppl = null, csv = null;
  try { if (pplItem) ppl = await api(pplItem.url); } catch { ppl = null; }
  try { if (csvItem) csv = await apiText(csvItem.url); } catch { csv = null; }
  return { ppl, csv };
}

// Card-level hook: fetch just the collapse ratio for a succeeded row.
export function useDamageRatio(id, enabled) {
  const [ratio, setRatio] = useState(null);
  useEffect(() => {
    if (!enabled || !id) return;
    let alive = true;
    (async () => {
      try {
        const { items } = await getArtifacts(id);
        const item = findPplItem(items);
        if (!item) return;
        const hero = deriveHero(await api(item.url));
        if (hero && alive) setRatio(hero.ratio);
      } catch { /* leave ratio null */ }
    })();
    return () => { alive = false; };
  }, [id, enabled]);
  return ratio;
}
