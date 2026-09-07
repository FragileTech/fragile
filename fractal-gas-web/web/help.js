// Tooltip help: captions get a small "?" button; native controls show help
// directly on hover/focus so their content and activation remain intact.
// Dependency-free, so it deploys with the rest of web/.

let tooltip = null;
let pinnedIcon = null; // icon whose tooltip was opened by click/tap
let listenersInstalled = false;
const initialized = new WeakSet();
let labelId = 0;

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
  btn.addEventListener("mouseenter", () => {
    if (!pinnedIcon) show(btn, false);
  });
  btn.addEventListener("mouseleave", () => hide(false));
  btn.addEventListener("focus", () => {
    if (!pinnedIcon) show(btn, false);
  });
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
 * Attach help to every `[data-help]` element under `root`.
 * The icon is inserted right after the element's first text node (label /
 * heading text) so it sits inline with the caption, before any input.
 */
export function initHelp(root = document) {
  ensureTooltip();
  for (const el of root.querySelectorAll("[data-help]")) {
    if (el.classList.contains("help-icon")) continue;
    // Native controls cannot contain interactive help buttons. In particular,
    // select/textarea/input content is owned by the browser, and a nested
    // button can interfere with activation and focus in Firefox.
    if (el.matches("button, input, select, textarea, a")) {
      if (initialized.has(el)) continue;
      initialized.add(el);
      const describedBy = new Set(
        (el.getAttribute("aria-describedby") || "")
          .split(/\s+/)
          .filter(Boolean),
      );
      describedBy.add("tooltip");
      el.setAttribute("aria-describedby", [...describedBy].join(" "));
      el.addEventListener("mouseenter", () => {
        if (!pinnedIcon) show(el, false);
      });
      el.addEventListener("mouseleave", () => hide(false));
      el.addEventListener("focus", () => {
        if (!pinnedIcon) show(el, false);
      });
      el.addEventListener("blur", () => hide(false));
      el.addEventListener("pointerdown", () => hide(true));
      el.addEventListener("keydown", () => hide(true));
      continue;
    }
    if (el.querySelector(":scope > .help-icon")) continue;
    // Preserve an implicit label's original control before inserting another
    // labelable element (the help button) ahead of it.
    if (el.matches("label") && !el.hasAttribute("for")) {
      const target = el.control;
      if (target) {
        if (!target.id) {
          let id;
          do {
            id = `help-control-${++labelId}`;
          } while (document.getElementById(id));
          target.id = id;
        }
        el.htmlFor = target.id;
      }
    }
    const icon = makeIcon(el.dataset.help);
    icon.setAttribute("aria-describedby", "tooltip");
    // Place after the caption (text plus inline value spans), i.e. right
    // before the first control / block child; append when there is no
    // caption text (e.g. a bare button group) or no control.
    const hasCaption = [...el.childNodes].some(
      (n) => n.nodeType === Node.TEXT_NODE && n.textContent.trim(),
    );
    const control = hasCaption
      ? el.querySelector(
          ":scope > :is(input, select, textarea, canvas, div, p, button)",
        )
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
    window.addEventListener(
      "scroll",
      () => {
        if (pinnedIcon) place(pinnedIcon);
      },
      true,
    );
    window.addEventListener("resize", () => {
      if (pinnedIcon) place(pinnedIcon);
    });
    listenersInstalled = true;
  }
}
