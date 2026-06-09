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

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function renderInline(text) {
  // text is already HTML-escaped; apply a minimal, safe inline markdown subset.
  return text
    .replace(/`([^`]+)`/g, (_, code) => `<code>${code}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, (_, bold) => `<strong>${bold}</strong>`)
    .replace(/_([^_]+)_/g, (_, em) => `<em>${em}</em>`);
}

// Minimal markdown renderer for our own generated report.md. Relative image
// paths are rewritten to the authenticated artifact endpoint so figures embedded
// in the report render inline alongside the customer view.
function renderReportMarkdown(markdown, artifactBase) {
  const html = [];
  let listOpen = false;
  const closeList = () => {
    if (listOpen) {
      html.push("</ul>");
      listOpen = false;
    }
  };
  for (const rawLine of markdown.split("\n")) {
    const line = rawLine.trimEnd();
    const image = line.match(/^!\[([^\]]*)\]\(([^)]+)\)\s*$/);
    if (image) {
      closeList();
      const alt = escapeHtml(image[1]);
      const src = /^https?:\/\//.test(image[2]) ? image[2] : artifactBase + image[2].replace(/^\.?\//, "");
      html.push(`<figure class="report-figure"><img src="${encodeURI(src)}" alt="${alt}" loading="lazy" /></figure>`);
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeList();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInline(escapeHtml(heading[2]))}</h${level}>`);
      continue;
    }
    const item = line.match(/^[-*]\s+(.*)$/);
    if (item) {
      if (!listOpen) {
        html.push("<ul>");
        listOpen = true;
      }
      html.push(`<li>${renderInline(escapeHtml(item[1]))}</li>`);
      continue;
    }
    if (!line) {
      closeList();
      continue;
    }
    closeList();
    html.push(`<p>${renderInline(escapeHtml(line))}</p>`);
  }
  closeList();
  return html.join("\n");
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

  const reportBox = byId("reportText");
  if (report) {
    const markdown = await apiText(report.url);
    const artifactBase = `/analyses/${state.selected.request_id}/artifacts/`;
    reportBox.innerHTML = renderReportMarkdown(markdown, artifactBase);
  } else {
    reportBox.textContent = "No report available";
  }
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
    await loadAnalyses();
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
