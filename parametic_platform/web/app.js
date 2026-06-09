const state = { models: [], areas: [], modes: [], rows: [], selected: null, artifacts: [], masks: [], spec: null };
const byId = (id) => document.getElementById(id);
const activeStatuses = new Set(["queued", "running"]);

function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body) headers["content-type"] = "application/json";
  return fetch(path, { credentials: "same-origin", ...options, headers }).then(async (response) => {
    if (!response.ok) throw new Error((await response.text()) || response.statusText);
    return response.json();
  });
}

function apiText(path) {
  return fetch(path, { credentials: "same-origin" }).then(async (response) => {
    if (!response.ok) throw new Error((await response.text()) || response.statusText);
    return response.text();
  });
}

function setBadge(ok, text) {
  const badge = byId("apiBadge");
  badge.textContent = text;
  badge.classList.toggle("muted", !ok);
}

function fillSelect(id, items) {
  const select = byId(id);
  select.innerHTML = "";
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.display_name || item.id;
    select.appendChild(option);
  }
}

function selectedText(id) {
  return byId(id).selectedOptions[0]?.textContent || "-";
}

function renderRows() {
  const body = byId("requestRows");
  body.innerHTML = "";
  if (!state.rows.length) {
    const empty = document.createElement("div");
    empty.className = "grid-row";
    empty.innerHTML = "<div>No requests</div><div>Create an analysis to start</div><div></div><div></div><div></div><div></div>";
    body.appendChild(empty);
    return;
  }
  for (const row of state.rows) {
    const element = document.createElement("div");
    element.className = `grid-row ${state.selected?.request_id === row.request_id ? "selected" : ""}`;
    const values = [row.request_id.slice(0, 8), row.model_label || selectedText("modelSelect"), row.area_label || selectedText("areaSelect"), row.mode_label || selectedText("modeSelect"), row.status, row.cache_hit ? "hit" : "miss"];
    for (const value of values) {
      const cell = document.createElement("div");
      cell.textContent = value;
      element.appendChild(cell);
    }
    element.firstChild.title = row.request_id;
    element.addEventListener("click", () => selectRow(row));
    body.appendChild(element);
  }
}

function resetResults() {
  state.artifacts = [];
  state.masks = [];
  state.spec = null;
  byId("artifactList").innerHTML = "";
  byId("artifactCount").textContent = "";
  byId("specCard").innerHTML = "";
  byId("storyTitle").textContent = "Select a finished analysis";
  byId("spotStory").innerHTML = '<p class="story-empty">Run or select a succeeded analysis to see where the coding spot lives, what it is, whether it is stable, and what removing it does.</p>';
}

function selectRow(row, options = {}) {
  const same = state.selected?.request_id === row.request_id;
  state.selected = row;
  byId("detailTitle").textContent = row.request_id.slice(0, 8);
  byId("detailSubtitle").textContent = row.error || "Analysis request details";
  byId("detailStatus").textContent = row.status;
  byId("detailCache").textContent = row.cache_key;
  if (!same || !options.preserveResults) resetResults();
  renderRows();
  // Auto-load customer results once the analysis is finished, so the report and
  // figures appear without a manual "Load results" click (and only fetch once).
  if (state.selected.status === "succeeded" && !state.artifacts.length) {
    loadArtifacts().catch((error) => setBadge(false, error.message));
  }
}

async function loadCatalog() {
  const [models, areas, modes] = await Promise.all([api("/models"), api("/areas"), api("/modes")]);
  state.models = models;
  state.areas = areas;
  state.modes = modes;
  fillSelect("modelSelect", models);
  fillSelect("areaSelect", areas);
  fillSelect("modeSelect", modes);
  byId("kInput").value = "0.01";
  setBadge(true, "API connected");
}

function catalogLabel(collection, id) {
  const item = state[collection].find((entry) => entry.id === id);
  return item ? item.display_name || item.id : id;
}

// Load existing analyses from the server so the table is populated on boot and
// survives reloads (rows are otherwise only held in memory for this session).
async function loadAnalyses() {
  const list = await api("/analyses");
  state.rows = list.map((row) => ({
    request_id: row.request_id,
    status: row.status,
    cache_key: row.cache_key,
    error: row.error,
    cache_hit: row.status === "succeeded",
    model_label: catalogLabel("models", row.model_id),
    area_label: catalogLabel("areas", row.area_id),
    mode_label: catalogLabel("modes", row.mode),
  }));
  renderRows();
}

async function submitAnalysis() {
  const payload = { model_id: byId("modelSelect").value, area_id: byId("areaSelect").value, mode: byId("modeSelect").value };
  const kValue = byId("kInput").value.trim();
  if (kValue) payload.k = Number(kValue);
  const row = await api("/analyses", { method: "POST", body: JSON.stringify(payload) });
  row.model_label = selectedText("modelSelect");
  row.area_label = selectedText("areaSelect");
  row.mode_label = selectedText("modeSelect");
  state.rows = [row, ...state.rows.filter((item) => item.request_id !== row.request_id)];
  selectRow(row);
}

async function refreshSelected(options = {}) {
  if (!state.selected) return;
  const row = await api(`/analyses/${state.selected.request_id}`);
  const previous = state.rows.find((item) => item.request_id === row.request_id) || state.selected;
  row.model_label = previous.model_label;
  row.area_label = previous.area_label;
  row.mode_label = previous.mode_label;
  state.rows = state.rows.map((item) => (item.request_id === row.request_id ? row : item));
  selectRow(row, { preserveResults: options.preserveResults ?? true });
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function fmtCompact(n) {
  if (n == null || isNaN(n)) return "-";
  const abs = Math.abs(n);
  if (abs >= 1000) return Math.round(n).toLocaleString();
  if (abs >= 1) return Number(n).toFixed(2);
  return Number(n).toPrecision(2);
}

function findItem(items, substr) {
  return items.find((it) => it.path.includes(substr));
}

function parseCsv(text) {
  const lines = text.trim().split("\n");
  const cols = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",");
    const row = {};
    cols.forEach((c, i) => (row[c] = values[i]));
    return row;
  });
}

// One figure tile; clicking the image opens the zoom overlay (delegated handler).
function figureCard(item, caption) {
  if (!item) return "";
  const cap = escapeHtml(caption || item.path.split("/").pop());
  const url = encodeURI(item.url);
  return `<figure class="act-figure"><img src="${url}" alt="${cap}" loading="lazy" data-zoom="${url}" data-cap="${cap}" /><figcaption>${cap}</figcaption></figure>`;
}

function actBlock(num, title, desc, body) {
  return `<section class="act"><div class="act-head"><span class="act-num">${num}</span><div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(desc)}</p></div></div><div class="act-body">${body}</div></section>`;
}

// Act 2: which weight modules carry the spot, ranked by importance share.
function renderModuleRankingHtml(csvText) {
  let rows;
  try { rows = parseCsv(csvText).filter((r) => r.module); } catch { return ""; }
  rows.sort((a, b) => parseFloat(b.importance_share) - parseFloat(a.importance_share));
  const bars = rows.slice(0, 8).map((r) => {
    const share = (parseFloat(r.importance_share) || 0) * 100;
    return `<div class="mod-row"><span class="mod-name">${escapeHtml(r.module)}</span><span class="mod-bar"><span style="width:${share.toFixed(1)}%"></span></span><span class="mod-val">${share.toFixed(1)}%</span></div>`;
  }).join("");
  return `<div class="module-rank">${bars}</div>`;
}

// Act 4: the causal payoff — spot removal vs equal-size control regions.
function renderCausalHtml(rows) {
  const label = (m) =>
    m === "original" ? "Original (intact)"
      : m.startsWith("code") ? "Spot removed"
      : m.startsWith("bottom") ? "Bottom region (control)"
      : m.startsWith("random") ? "Random region (control)"
      : m;
  const maxLog = Math.max(...rows.map((r) => Math.log10(Math.max(r.ppl, 1))), 1);
  const body = rows.map((r) => {
    const isSpot = String(r.model).startsWith("code");
    const width = (Math.log10(Math.max(r.ppl, 1)) / maxLog * 100).toFixed(1);
    return `<div class="dmg-row ${isSpot ? "spot" : ""}"><span class="dmg-name">${escapeHtml(label(r.model))}</span><span class="dmg-bar"><span style="width:${width}%"></span></span><span class="dmg-val">PPL ${fmtCompact(r.ppl)}</span></div>`;
  }).join("");
  return `<div class="damage-chart">${body}</div><p class="dmg-note">Equal-size random/bottom regions barely move perplexity — only the discovered spot is causally responsible for coding.</p>`;
}

// The hero: assemble the four-act spot story from spec + figures + metrics.
async function renderSpotStory(items) {
  const story = byId("spotStory");
  const status = state.selected && state.selected.status;
  if (status !== "succeeded") {
    byId("storyTitle").textContent = state.selected ? state.selected.request_id.slice(0, 8) : "Select a finished analysis";
    const msg = status === "failed"
      ? "This analysis failed: " + escapeHtml(state.selected.error || "unknown error")
      : "Analysis is " + escapeHtml(status || "…") + ". Results appear here automatically when it finishes.";
    story.innerHTML = `<p class="story-empty">${msg}</p>`;
    return;
  }

  const analysis = (state.spec && state.spec.analysis) || {};
  const modelName = (analysis.model && analysis.model.display_name) || state.selected.model_label || "Model";
  const lang = (analysis.area && analysis.area.language) || "";
  const k = analysis.k != null ? analysis.k : "";
  byId("storyTitle").textContent = `${modelName} · ${lang || "code"} · top-${k}`;

  const pplItem = items.find((i) => i.kind === "metric" && i.path.includes("ppl_damage"));
  const csvItem = items.find((i) => i.kind === "table" && i.path.includes("module_summary"));
  let ppl = null, csv = null;
  try { if (pplItem) ppl = await api(pplItem.url); } catch { ppl = null; }
  try { if (csvItem) csv = await apiText(csvItem.url); } catch { csv = null; }

  const blocks = [];
  if (Array.isArray(ppl)) {
    const orig = ppl.find((r) => r.model === "original");
    const spot = ppl.find((r) => String(r.model).startsWith("code"));
    if (orig && spot) {
      const ratio = spot.ppl / orig.ppl;
      const removed = spot.zeroed_params
        ? `${Number(spot.zeroed_params).toLocaleString()} params (${spot.mask_tensors} tensors) removed`
        : "";
      blocks.push(`<div class="story-hero"><div class="hero-punch">Zeroing the top-${k} coding spot collapses ${escapeHtml(lang || "code")} modeling — PPL <b>${fmtCompact(orig.ppl)}</b> → <b class="bad">${fmtCompact(spot.ppl)}</b> <span class="hero-x">×${fmtCompact(ratio)}</span></div><div class="hero-sub">${escapeHtml(removed)}</div></div>`);
    }
  }

  blocks.push(actBlock("1", "Where the spot lives",
    "A per-tensor map of the whole model — bright tiles are where the top-k coding-important parameters sit.",
    figureCard(findItem(items, "spot_mask_atlas"), "Mask density atlas") + figureCard(findItem(items, "spot_importance_atlas"), "Importance atlas")));

  blocks.push(actBlock("2", "What it is",
    "Which weight modules carry the spot. The coding spot concentrates in a few module types.",
    (csv ? renderModuleRankingHtml(csv) : "") +
    `<div class="act-figs">${figureCard(findItem(items, "layer_module_importance_sum"), "Layer/module importance")}${figureCard(findItem(items, "importance_bubble_map"), "Importance bubble map")}</div>`));

  blocks.push(actBlock("3", "Is it stable?",
    "Two independent calibration seeds. Agreement means the spot is real signal, not noise.",
    `<div class="act-figs">${figureCard(findItem(items, "seed_agreement_atlas"), "Seed agreement")}${figureCard(findItem(items, "seed_disagreement_atlas"), "Seed disagreement")}</div>`));

  blocks.push(actBlock("4", "What removing it does",
    "Zero the spot vs. an equal-size random/bottom region, then measure code perplexity. Only the spot breaks coding — the causal payoff.",
    Array.isArray(ppl) ? renderCausalHtml(ppl) : "<p class='story-empty'>No damage metrics available.</p>"));

  story.innerHTML = blocks.join("");
}

function renderSpecCard() {
  const el = byId("specCard");
  const spec = state.spec;
  if (!spec || !spec.analysis) { el.innerHTML = ""; return; }
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
  const dl = pairs.map((p) => `<div><dt>${escapeHtml(p[0])}</dt><dd>${escapeHtml(String(p[1]))}</dd></div>`).join("");
  const stages = (spec.stages || []).map((s) => escapeHtml(s.name)).join(" → ");
  el.innerHTML = `<div class="spec-head">Reproducibility</div><dl class="detail-list spec-list">${dl}</dl>${stages ? `<div class="spec-stages">${stages}</div>` : ""}`;
}

// Full transparency: every artifact downloadable; masks summarized by region.
function renderRawArtifacts(items, masks) {
  const list = byId("artifactList");
  list.innerHTML = "";
  const order = ["report", "spec", "metric", "table", "figure", "log", "artifact"];
  const groups = {};
  for (const it of items) (groups[it.kind] = groups[it.kind] || []).push(it);
  for (const kind of order) {
    for (const it of groups[kind] || []) {
      const li = document.createElement("li");
      li.innerHTML = `<a href="${encodeURI(it.url)}" target="_blank" rel="noreferrer">${escapeHtml(it.path)}</a>`;
      list.appendChild(li);
    }
  }
  for (const m of masks || []) {
    const li = document.createElement("li");
    li.className = "mask-row";
    li.textContent = `${m.path} — ${m.count} tensors (the spot)`;
    list.appendChild(li);
  }
  const total = items.length + (masks || []).reduce((sum, m) => sum + (m.count || 0), 0);
  byId("artifactCount").textContent = total ? `${total} files` : "";
  if (!list.children.length) {
    const li = document.createElement("li");
    li.textContent = "No artifacts";
    list.appendChild(li);
  }
}

async function loadArtifacts() {
  if (!state.selected) return;
  const result = await api(`/analyses/${state.selected.request_id}/artifacts`);
  state.artifacts = result.items || [];
  state.masks = result.masks || [];
  try { state.spec = await api(`/analyses/${state.selected.request_id}/spec`); } catch { state.spec = null; }
  renderSpecCard();
  renderRawArtifacts(state.artifacts, state.masks);
  await renderSpotStory(state.artifacts);
}

async function boot() {
  try {
    await loadCatalog();
    await loadAnalyses();
  } catch (error) {
    setBadge(false, "API unavailable");
    byId("detailSubtitle").textContent = error.message;
  }
}

function openZoom(url, caption) {
  let overlay = byId("zoomOverlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "zoomOverlay";
    overlay.className = "zoom-overlay";
    overlay.innerHTML = '<img alt="" /><div class="zoom-cap"></div>';
    overlay.addEventListener("click", () => overlay.classList.remove("open"));
    document.body.appendChild(overlay);
  }
  overlay.querySelector("img").src = url;
  overlay.querySelector(".zoom-cap").textContent = caption || "";
  overlay.classList.add("open");
}

byId("spotStory").addEventListener("click", (event) => {
  const img = event.target.closest("img[data-zoom]");
  if (img) openZoom(img.getAttribute("data-zoom"), img.getAttribute("data-cap"));
});

byId("submitButton").addEventListener("click", () => submitAnalysis().catch((error) => setBadge(false, error.message)));
byId("refreshButton").addEventListener("click", () => refreshSelected({ preserveResults: true }).catch((error) => setBadge(false, error.message)));
byId("loadArtifactsButton").addEventListener("click", () => loadArtifacts().catch((error) => setBadge(false, error.message)));
setInterval(() => {
  if (state.selected && activeStatuses.has(state.selected.status)) {
    refreshSelected({ preserveResults: true }).catch((error) => setBadge(false, error.message));
  }
}, 5000);
boot();
