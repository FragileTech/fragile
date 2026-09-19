// Reuse the live controls so responsive changes never replace simulation state.
export function mountMobileWorkspace() {
  const $ = (id) => document.getElementById(id);
  const body = document.body;
  const main = document.querySelector("main");
  const toolbar = document.querySelector(".workspace-toolbar");
  const stage = document.querySelector(".stage");
  const mobile = matchMedia(
    "(max-width: 779px), (pointer: coarse) and (max-height: 500px)",
  );
  const make = (tag, id, html) => {
    const node = document.createElement(tag);
    node.id = id;
    node.innerHTML = html;
    return node;
  };
  const menuButton = make("button", "mobile-menu-toggle", "Menu");
  const menu = make("section", "mobile-menu", "");
  const tools = make("section", "mobile-tools", "");
  const bar = make(
    "div",
    "mobile-action-bar",
    '<button id="mobile-tools-toggle">Tools</button><button id="mobile-replay-toggle">Replay</button>',
  );
  toolbar.append(menuButton);
  main.append(menu, tools);
  stage.append(bar);
  const panels = {
    menu,
    tools,
    settings: document.querySelector(".controls"),
    inspector: $("selection-inspector"),
    replay: document.querySelector(".timeline-dock"),
    editor: $("editor"),
  };
  const triggers = {
    menu: menuButton,
    tools: $("mobile-tools-toggle"),
    replay: $("mobile-replay-toggle"),
    settings: $("toggle-settings"),
    inspector: $("toggle-inspector"),
    editor: $("mode-edit"),
  };
  let active = null;
  let returnFocus = null;
  const closeButtons = [];
  for (const [key, panel] of Object.entries(panels)) {
    panel.id ||= `mobile-${key}-panel`;
    panel.dataset.mobilePanel = key;
    const close = make(
      "button",
      `mobile-close-${key}`,
      `Close ${key === "replay" ? "replay" : key}`,
    );
    close.className = "mobile-panel-close";
    close.onclick = () => show(null);
    panel.prepend(close);
    closeButtons.push(close);
    triggers[key].setAttribute("aria-controls", panel.id);
  }
  function show(key, focus = true) {
    if (!mobile.matches) return;
    const previousFocus = returnFocus;
    const previousPanel = active;
    active = key;
    body.dataset.mobilePanel = key || "";
    body.classList.toggle("settings-open", key === "settings");
    body.classList.toggle("inspector-open", key === "inspector");
    for (const [name, panel] of Object.entries(panels)) {
      panel.inert = name !== key;
      triggers[name].setAttribute("aria-expanded", String(name === key));
    }
    if (key) {
      returnFocus = document.activeElement;
      if (focus) panels[key].querySelector(".mobile-panel-close").focus();
    } else if (focus) {
      // A nested tool may now be hidden. Return to its visible entry point.
      const target =
        previousFocus?.checkVisibility() && !previousFocus.closest("[inert]")
          ? previousFocus
          : triggers[
              previousPanel === "settings" || previousPanel === "menu"
                ? "menu"
                : "tools"
            ];
      target?.focus();
    }
  }
  for (const key of ["menu", "tools", "replay"])
    triggers[key].onclick = () => show(active === key ? null : key);
  // Capture the existing drawer buttons only in the compact layout.
  for (const key of ["settings", "inspector"])
    triggers[key].addEventListener(
      "click",
      (event) => {
        if (!mobile.matches) return;
        event.stopImmediatePropagation();
        show(active === key ? null : key);
      },
      true,
    );
  $("close-inspector").addEventListener("click", () => {
    if (mobile.matches) show(null);
  });
  for (const id of ["choose-task", "save-menu"])
    $(id).addEventListener(
      "click",
      () => {
        if (mobile.matches) show(null, false);
      },
      true,
    );
  document.addEventListener("keydown", (event) => {
    if (
      mobile.matches &&
      active &&
      event.key === "Escape" &&
      !document.querySelector("dialog[open]")
    ) {
      event.preventDefault();
      show(null);
    }
  });
  // Driving and editing can open panels through the existing mode handlers.
  new MutationObserver(() => {
    if (!mobile.matches) return;
    if (body.classList.contains("inspector-open") && active !== "inspector")
      show("inspector");
  }).observe(body, { attributes: true, attributeFilter: ["class"] });
  new MutationObserver(() => {
    if (!mobile.matches) return;
    if (!panels.editor.hidden && active !== "editor") show("editor");
    else if (panels.editor.hidden && active === "editor") show(null);
  }).observe(panels.editor, { attributes: true, attributeFilter: ["hidden"] });
  const moves = [
    [$("choose-task"), menu],
    [$("toggle-settings"), menu],
    [$("save-menu"), menu],
    [document.querySelector(".mode-toolbar"), tools],
    [document.querySelector(".stage-toolbar"), tools],
    [$("playback-status"), bar],
  ].map(([node, destination]) => {
    const anchor = document.createComment("desktop control position");
    node.before(anchor);
    return { node, destination, anchor };
  });
  const sync = () => {
    body.classList.toggle("mobile-workspace", mobile.matches);
    if (mobile.matches) {
      for (const { node, destination } of moves) destination.append(node);
      show(null, false);
    } else {
      for (const { node, anchor } of moves) anchor.after(node);
      active = null;
      delete body.dataset.mobilePanel;
      body.classList.remove("settings-open", "inspector-open");
      for (const panel of Object.values(panels)) panel.inert = false;
      for (const trigger of Object.values(triggers))
        trigger.setAttribute("aria-expanded", "false");
    }
    menu.hidden =
      tools.hidden =
      bar.hidden =
      menuButton.hidden =
        !mobile.matches;
    for (const close of closeButtons) close.hidden = !mobile.matches;
  };
  mobile.addEventListener("change", sync);
  sync();
}
