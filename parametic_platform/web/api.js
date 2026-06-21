// Thin fetch layer for the Parametic API. All endpoints sit behind basic auth
// (the browser caches it after the first authed request) and return JSON unless
// noted. Lifted from the original app.js helpers.

export function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body) headers["content-type"] = "application/json";
  return fetch(path, { credentials: "same-origin", ...options, headers }).then(async (response) => {
    if (!response.ok) throw new Error((await response.text()) || response.statusText);
    return response.json();
  });
}

export function apiText(path) {
  return fetch(path, { credentials: "same-origin" }).then(async (response) => {
    if (!response.ok) throw new Error((await response.text()) || response.statusText);
    return response.text();
  });
}

export const getModels = () => api("/models");
export const getAreas = () => api("/areas");
export const getModes = () => api("/modes");
export const getCapabilities = () => api("/capabilities");

// Operator model registration (gated server-side by PARAMETIC_ALLOW_MODEL_REGISTRATION).
export const resolveModel = ({ hf_model_id, revision }) =>
  api("/models/resolve", { method: "POST", body: JSON.stringify({ hf_model_id, revision: revision || "main" }) });
export const registerModel = ({ hf_model_id, revision }) =>
  api("/models/register", { method: "POST", body: JSON.stringify({ hf_model_id, revision: revision || "main" }) });
export const getAnalyses = (limit) => api(`/analyses${limit ? `?limit=${limit}` : ""}`);
export const getAnalysis = (id) => api(`/analyses/${id}`);
export const getArtifacts = (id) => api(`/analyses/${id}/artifacts`);
export const getSpec = (id) => api(`/analyses/${id}/spec`);

export function createAnalysis({ model_id, area_id, mode, k }) {
  const payload = { model_id, area_id, mode };
  if (k != null && k !== "") payload.k = Number(k);
  return api("/analyses", { method: "POST", body: JSON.stringify(payload) });
}
