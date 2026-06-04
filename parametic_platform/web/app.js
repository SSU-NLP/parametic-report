const state = { models: [], areas: [], modes: [], rows: [], selected: null, artifacts: [] };
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
  byId("artifactList").innerHTML = "";
  byId("reportText").textContent = "No report loaded";
  byId("metricBox").textContent = "No metrics loaded";
  byId("figureGrid").textContent = "No figures loaded";
}

function selectRow(row, options = {}) {
  const same = state.selected?.request_id === row.request_id;
  state.selected = row;
  byId("detailTitle").textContent = row.request_id.slice(0, 8);
  byId("detailSubtitle").textContent = row.error || "Analysis request details";
  byId("detailStatus").textContent = row.status;
  byId("detailCache").textContent = row.cache_key;
  byId("detailStorage").textContent = "Managed";
  if (!same || !options.preserveResults) resetResults();
  renderRows();
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

function renderArtifactList(items) {
  const list = byId("artifactList");
  list.innerHTML = "";
  for (const item of items) {
    const element = document.createElement("li");
    element.textContent = item.path;
    list.appendChild(element);
  }
  if (!list.children.length) {
    const element = document.createElement("li");
    element.textContent = "No customer-visible artifacts available";
    list.appendChild(element);
  }
}

function renderMetricTable(value) {
  const box = byId("metricBox");
  box.innerHTML = "";
  if (!Array.isArray(value) || !value.length || typeof value[0] !== "object") {
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(value, null, 2);
    box.appendChild(pre);
    return;
  }
  const columns = [...new Set(value.flatMap((row) => Object.keys(row)))];
  const table = document.createElement("table");
  const head = document.createElement("tr");
  for (const column of columns) {
    const th = document.createElement("th");
    th.textContent = column;
    head.appendChild(th);
  }
  table.appendChild(head);
  for (const row of value) {
    const tr = document.createElement("tr");
    for (const column of columns) {
      const td = document.createElement("td");
      const cell = row[column];
      td.textContent = typeof cell === "number" ? cell.toPrecision(5) : String(cell ?? "");
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }
  box.appendChild(table);
}

async function renderArtifacts(items) {
  const report = items.find((item) => item.kind === "report");
  const metrics = items.filter((item) => item.kind === "metric");
  const figures = items.filter((item) => item.kind === "figure");

  byId("reportText").textContent = report ? await apiText(report.url) : "No report available";
  if (metrics.length) {
    const metricValue = await api(metrics[0].url);
    renderMetricTable(metricValue);
  } else {
    byId("metricBox").textContent = "No metrics available";
  }

  const figureGrid = byId("figureGrid");
  figureGrid.innerHTML = "";
  for (const figure of figures) {
    const link = document.createElement("a");
    link.href = figure.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    const image = document.createElement("img");
    image.src = figure.url;
    image.alt = figure.path;
    const caption = document.createElement("span");
    caption.textContent = figure.path.split("/").pop();
    link.appendChild(image);
    link.appendChild(caption);
    figureGrid.appendChild(link);
  }
  if (!figures.length) figureGrid.textContent = "No figures available";
}

async function loadArtifacts() {
  if (!state.selected) return;
  const result = await api(`/analyses/${state.selected.request_id}/artifacts`);
  const items = result.items || (result.artifacts || []).map((path) => ({ path, kind: "artifact", url: `/analyses/${state.selected.request_id}/artifacts/${path}` }));
  state.artifacts = items;
  renderArtifactList(items);
  await renderArtifacts(items);
}

async function boot() {
  try {
    await loadCatalog();
    renderRows();
  } catch (error) {
    setBadge(false, "API unavailable");
    byId("detailSubtitle").textContent = error.message;
  }
}

byId("submitButton").addEventListener("click", () => submitAnalysis().catch((error) => setBadge(false, error.message)));
byId("refreshButton").addEventListener("click", () => refreshSelected({ preserveResults: true }).catch((error) => setBadge(false, error.message)));
byId("loadArtifactsButton").addEventListener("click", () => loadArtifacts().catch((error) => setBadge(false, error.message)));
setInterval(() => {
  if (state.selected && activeStatuses.has(state.selected.status)) {
    refreshSelected({ preserveResults: true }).catch((error) => setBadge(false, error.message));
  }
}, 5000);
boot();
