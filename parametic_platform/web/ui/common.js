// Shared UI atoms. `html` is htm bound to Preact's h via the htm/preact build,
// so every component shares one Preact instance (hooks work across modules).
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";

export { html };

// Centered modal dialog. Closes on backdrop click, the ✕ button, or Escape.
export function Modal({ title, onClose, children }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { removeEventListener("keydown", onKey); document.body.style.overflow = prev; };
  }, [onClose]);
  return html`
    <div class="modal-backdrop" onClick=${onClose}>
      <div class="modal" role="dialog" aria-modal="true" aria-label=${title}
        onClick=${(e) => e.stopPropagation()}>
        <div class="modal-head">
          <h2>${title}</h2>
          <button class="modal-x" aria-label="Close" onClick=${onClose}>✕</button>
        </div>
        <div class="modal-body">${children}</div>
      </div>
    </div>`;
}

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

// Brand lockup — the "Parametic Report" serif wordmark (matches the logo).
export function Brand({ tagline }) {
  return html`
    <div class="brand">
      <span class="brand-name">Parametic Report</span>
      ${tagline ? html`<span class="brand-sub">${tagline}</span>` : null}
    </div>`;
}
