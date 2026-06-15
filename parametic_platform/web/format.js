// Pure formatting / parsing helpers — no DOM, no Preact. Ported from the
// original app.js so the data contract handling stays identical.

export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}

// Compact number for perplexity / param counts.
export function fmtCompact(n) {
  if (n == null || isNaN(n)) return "-";
  const abs = Math.abs(n);
  if (abs >= 1000) return Math.round(n).toLocaleString();
  if (abs >= 1) return Number(n).toFixed(2);
  return Number(n).toPrecision(2);
}

// "×66,901" style ratio used as the headline collapse factor.
export function fmtRatio(n) {
  if (n == null || isNaN(n)) return "-";
  return Math.round(n).toLocaleString();
}

export function parseCsv(text) {
  const lines = text.trim().split("\n");
  const cols = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",");
    const row = {};
    cols.forEach((c, i) => (row[c] = values[i]));
    return row;
  });
}

// Match an artifact by path substring (k-agnostic figure lookup).
export function findItem(items, substr) {
  return (items || []).find((it) => it.path.includes(substr));
}

// Short request id for chips / labels.
export function shortId(id) {
  return id ? String(id).slice(0, 8) : "-";
}
