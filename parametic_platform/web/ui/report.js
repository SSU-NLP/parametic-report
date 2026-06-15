// Report route (#/a/:id). Stage 1 = shell only (back bar + status). The hero,
// four-act spot story, reproducibility, and artifacts land in later stages.
import { html, Brand, StatusPill, Spinner } from "./common.js";
import { navigate, catalogLabel } from "../store.js";
import { shortId } from "../format.js";

export function Report({ id, analysis, listRow, catalog }) {
  const { row, loading, error } = analysis;
  const status = (row && row.status) || (listRow && listRow.status);
  const model = listRow ? catalogLabel(catalog.models, listRow.model_id) : "";
  const area = listRow ? catalogLabel(catalog.areas, listRow.area_id) : "";

  return html`
    <div class="report">
      <div class="report-bar">
        <button class="back" onClick=${() => navigate("/")}>← All analyses</button>
        <span class="report-bar-title">${listRow ? html`${model} · ${area}` : shortId(id)}</span>
        ${status ? html`<${StatusPill} status=${status} />` : null}
      </div>

      <div class="report-body">
        ${loading ? html`<${Spinner} label="Loading analysis…" />` : null}
        ${error ? html`<p class="banner bad">${error}</p>` : null}
        ${row ? html`<p class="placeholder">Report view — coming in the next stage.</p>` : null}
      </div>
    </div>`;
}
