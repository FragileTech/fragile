// Tooltip help: every element carrying a `data-help` attribute gets a small
// "?" icon; hovering, focusing or tapping it shows the text in one shared
// floating tooltip. Dependency-free, so it deploys with the rest of web/.

let tooltip = null;
let pinnedIcon = null; // icon whose tooltip was opened by click/tap
let listenersInstalled = false;

function ensureTooltip() {
  if (tooltip) return tooltip;
  tooltip = document.createElement("div");
  tooltip.id = "tooltip";
  tooltip.setAttribute("role", "tooltip");
  tooltip.hidden = true;
  document.body.appendChild(tooltip);
  return tooltip;
}

function setText(text) {
  tooltip.textContent = "";
  for (const para of text.split("\n")) {
    if (!para.trim()) continue;
    const p = document.createElement("p");
    p.textContent = para;
    tooltip.appendChild(p);
  }
}

function place(icon) {
  const r = icon.getBoundingClientRect();
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  // Measure at the final width first, then decide where it fits.
  tooltip.style.left = "0px";
  tooltip.style.top = "0px";
  const tw = tooltip.offsetWidth;
  const th = tooltip.offsetHeight;
  let left = r.left + r.width / 2 - tw / 2;
  left = Math.max(8, Math.min(left, vw - tw - 8));
  let top = r.bottom + 6;
  if (top + th > vh - 8) top = Math.max(8, r.top - th - 6);
  tooltip.style.left = `${Math.round(left)}px`;
  tooltip.style.top = `${Math.round(top)}px`;
}

function show(icon, pinned) {
  ensureTooltip();
  setText(icon.dataset.help);
  tooltip.hidden = false;
  tooltip.classList.toggle("pinned", !!pinned);
  place(icon);
  if (pinned) pinnedIcon = icon;
}

function hide(force) {
  if (!tooltip || tooltip.hidden) return;
  if (pinnedIcon && !force) return;
  tooltip.hidden = true;
  tooltip.classList.remove("pinned");
  pinnedIcon = null;
}

function makeIcon(text) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "help-icon";
  btn.textContent = "?";
  btn.setAttribute("aria-label", "Help");
  btn.dataset.help = text;
  btn.addEventListener("mouseenter", () => { if (!pinnedIcon) show(btn, false); });
  btn.addEventListener("mouseleave", () => hide(false));
  btn.addEventListener("focus", () => { if (!pinnedIcon) show(btn, false); });
  btn.addEventListener("blur", () => hide(false));
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (pinnedIcon === btn) hide(true);
    else show(btn, true);
  });
  return btn;
}

/**
 * Attach a "?" help icon to every `[data-help]` element under `root`.
 * The icon is inserted right after the element's first text node (label /
 * heading text) so it sits inline with the caption, before any input.
 */
export function initHelp(root = document) {
  ensureTooltip();
  for (const el of root.querySelectorAll("[data-help]")) {
    if (el.classList.contains("help-icon")) continue;
    if (el.querySelector(":scope > .help-icon")) continue;
    const icon = makeIcon(el.dataset.help);
    icon.setAttribute("aria-describedby", "tooltip");
    // Place after the caption (text plus inline value spans), i.e. right
    // before the first control / block child; append when there is no
    // caption text (e.g. a bare button group) or no control.
    const hasCaption = [...el.childNodes].some(
      (n) => n.nodeType === Node.TEXT_NODE && n.textContent.trim(),
    );
    const control = hasCaption
      ? el.querySelector(":scope > :is(input, select, textarea, canvas, div, p, button)")
      : null;
    if (control) control.before(icon);
    else el.appendChild(icon);
  }

  if (!listenersInstalled) {
    document.addEventListener("click", (e) => {
      if (pinnedIcon && !tooltip.contains(e.target)) hide(true);
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") hide(true);
    });
    window.addEventListener("scroll", () => { if (pinnedIcon) place(pinnedIcon); }, true);
    window.addEventListener("resize", () => { if (pinnedIcon) place(pinnedIcon); });
    listenersInstalled = true;
  }
}
