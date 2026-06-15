// Shared UI atoms. `html` is htm bound to Preact's h via the htm/preact build,
// so every component shares one Preact instance (hooks work across modules).
import { html } from "htm/preact";

export { html };

const STATUS_META = {
  succeeded: { label: "succeeded", cls: "ok", icon: "✓" },
  failed: { label: "failed", cls: "bad", icon: "!" },
  running: { label: "running", cls: "warn", icon: "•" },
  queued: { label: "queued", cls: "warn", icon: "•" },
};

export function StatusPill({ status }) {
  const m = STATUS_META[status] || { label: status || "—", cls: "muted", icon: "" };
  return html`<span class="pill ${m.cls}">${m.icon ? html`<i>${m.icon}</i>` : null}${m.label}</span>`;
}

export function Spinner({ label }) {
  return html`<div class="loading"><span class="spin" aria-hidden="true"></span>${label || "Loading…"}</div>`;
}

// Brand lockup used in both routes.
export function Brand({ tagline }) {
  return html`
    <div class="brand">
      <span class="brand-mark">P</span>
      <span class="brand-name">Parametic</span>
      ${tagline ? html`<span class="brand-sub">${tagline}</span>` : null}
    </div>`;
}
