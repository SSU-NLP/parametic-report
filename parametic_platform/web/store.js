// App state, hash router, and data loaders. Keeps the original 5s polling of
// active (queued|running) analyses. The backend is untouched; this only reads
// the documented endpoints in api.js.
import { useState, useEffect, useCallback } from "preact/hooks";
import * as API from "./api.js";

export const ACTIVE_STATUSES = new Set(["queued", "running"]);

// Parse "#/" (gallery) or "#/a/<id>" (report) from the URL hash.
export function parseHash() {
  const hash = (location.hash || "#/").replace(/^#/, "");
  const m = hash.match(/^\/a\/([^/]+)/);
  if (m) return { name: "report", id: decodeURIComponent(m[1]) };
  return { name: "gallery", id: null };
}

export function navigate(route) {
  location.hash = route;
}

// Reactive route — re-renders on hashchange.
export function useRoute() {
  const [route, setRoute] = useState(parseHash());
  useEffect(() => {
    const onHash = () => setRoute(parseHash());
    addEventListener("hashchange", onHash);
    return () => removeEventListener("hashchange", onHash);
  }, []);
  return route;
}

export function catalogLabel(collection, id) {
  const item = (collection || []).find((entry) => entry.id === id);
  return item ? item.display_name || item.id : id;
}

// Loads the catalog once and exposes it for selectors + row labels.
export function useCatalog() {
  const [catalog, setCatalog] = useState({ models: [], areas: [], modes: [], ok: null, error: null });
  useEffect(() => {
    Promise.all([API.getModels(), API.getAreas(), API.getModes()])
      .then(([models, areas, modes]) => setCatalog({ models, areas, modes, ok: true, error: null }))
      .catch((error) => setCatalog((c) => ({ ...c, ok: false, error: error.message })));
  }, []);
  return catalog;
}

// Loads the analyses list and polls active rows every 5s.
export function useAnalyses() {
  const [rows, setRows] = useState([]);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setRows(await API.getAnalyses());
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    const id = setInterval(() => {
      // Only poll if something is still active.
      if (rows.some((r) => ACTIVE_STATUSES.has(r.status))) refresh();
    }, 5000);
    return () => clearInterval(id);
  }, [rows, refresh]);

  return { rows, error, refresh };
}

// Loads one analysis row + its artifacts + spec, polling while active.
export function useAnalysis(id) {
  const [state, setState] = useState({ row: null, artifacts: [], masks: [], spec: null, loading: true, error: null });

  const load = useCallback(async () => {
    if (!id) return;
    try {
      const row = await API.getAnalysis(id);
      let artifacts = [], masks = [], spec = null;
      if (row.status === "succeeded") {
        const res = await API.getArtifacts(id);
        artifacts = res.items || [];
        masks = res.masks || [];
        try { spec = await API.getSpec(id); } catch { spec = null; }
      }
      setState({ row, artifacts, masks, spec, loading: false, error: null });
    } catch (e) {
      setState((s) => ({ ...s, loading: false, error: e.message }));
    }
  }, [id]);

  useEffect(() => { setState((s) => ({ ...s, loading: true })); load(); }, [load]);

  useEffect(() => {
    if (!state.row || !ACTIVE_STATUSES.has(state.row.status)) return;
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [state.row, load]);

  return state;
}
