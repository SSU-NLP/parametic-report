// Entry point. Wires the hash router to the two routes and mounts the app.
// Buildless: loaded as <script type="module"> with an importmap in index.html.
import { render } from "htm/preact";
import { html } from "./ui/common.js";
import { useRoute, useCatalog, useAnalyses, useAnalysis } from "./store.js";
import { Gallery } from "./ui/gallery.js";
import { Report } from "./ui/report.js";

function App() {
  const route = useRoute();
  const catalog = useCatalog();
  const { rows, error, refresh } = useAnalyses();
  // Always called (hooks rule); the loader no-ops when id is null.
  const analysis = useAnalysis(route.name === "report" ? route.id : null);

  if (route.name === "report") {
    // The single-analysis endpoint omits model/area/mode — carry them from the
    // list row so the title resolves even before the spec loads.
    const listRow = rows.find((r) => r.request_id === route.id) || null;
    return html`<${Report} id=${route.id} analysis=${analysis} listRow=${listRow} catalog=${catalog} />`;
  }
  return html`<${Gallery} catalog=${catalog} rows=${rows} error=${error} refresh=${refresh} />`;
}

render(html`<${App} />`, document.getElementById("app"));
