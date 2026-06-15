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

// Underline-style tab bar. `tabs` = [{id, label}], `active` = id.
export function TabBar({ tabs, active, onSelect }) {
  return html`
    <div class="tab-bar" role="tablist">
      ${tabs.map((t) => html`
        <button class="tab-btn ${t.id === active ? "active" : ""}" role="tab"
          aria-selected=${t.id === active} onClick=${() => onSelect(t.id)}>${t.label}</button>`)}
    </div>`;
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
