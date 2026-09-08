import { readPreference, savePreference } from "./workspace-state.js";
const $ = (id) => document.getElementById(id);
function element(tag, cls, html = "") {
  const node = document.createElement(tag);
  node.className = cls;
  node.innerHTML = html;
  return node;
}
export function mountWorkspace() {
  document.body.classList.add("workspace");
  const controls = document.querySelector(".controls");
  const stage = document.querySelector(".stage");
  const toolbar = element(
    "div",
    "workspace-toolbar",
    `<button id="choose-task">Tasks</button><button id="toggle-settings" aria-expanded="false">Settings</button><div id="toolbar-environment"></div><div id="transport"></div><span id="execution-status" role="status">Preparing…</span><button id="save-menu">Save / Open</button>`,
  );
  document.querySelector("main").before(toolbar);
  document
    .querySelector(".masthead")
    .append(document.querySelector(".masthead nav"));
  $("toolbar-environment").append($("scenario").closest("label"));
  $("transport").append(document.querySelector(".run-controls"));
  $("reset").textContent = "Restart";
  const panels = {};
  const tabs = element("nav", "settings-tabs", "");
  tabs.setAttribute("aria-label", "Settings sections");
  for (const name of ["Setup", "Controller", "Rewards", "View"]) {
    const key = name.toLowerCase();
    const button = element("button", "", name);
    button.id = `tab-${key}`;
    button.setAttribute("aria-controls", `settings-${key}`);
    const panel = element("section", "settings-panel");
    panel.id = `settings-${key}`;
    panel.setAttribute("aria-label", name);
    panels[key] = panel;
    button.onclick = () => {
      for (const [id, p] of Object.entries(panels)) {
        p.hidden = id !== key;
        $(`tab-${id}`).setAttribute("aria-pressed", String(id === key));
      }
      savePreference("panel", key);
    };
    tabs.append(button);
  }
  let group = "setup";
  for (const node of [...controls.children]) {
    if (node.matches("h1,.intro,.section-number,.controls-footer")) {
      node.hidden = true;
      continue;
    }
    if (node.id === "reward-settings") {
      panels.rewards.append(node);
      continue;
    }
    if (node.matches(".keyboard-toggle,.keyboard-hint")) {
      node.hidden = true;
      continue;
    }
    if (node.matches(".control-heading"))
      group = node.textContent.includes("CONTROLLER") ? "controller" : "view";
    panels[group].append(node);
  }
  controls.prepend(tabs);
  controls.append(...Object.values(panels));
  const physics = element("details", "", "<summary>World physics</summary>");
  physics.id = "world-physics";
  physics.append($("flight-controls"), $("rock-controls"), $("problem-settings"));
  panels.setup.append(physics);
  for (const details of controls.querySelectorAll("details")) {
    const key =
      details.id || details.querySelector("summary")?.textContent.trim();
    details.open = readPreference(`expanded.${key}`, false);
    details.addEventListener("toggle", () =>
      savePreference(`expanded.${key}`, details.open),
    );
  }
  ($("tab-" + readPreference("panel", "setup")) || $("tab-setup")).click();
  panels.view.prepend(document.querySelector(".visual-style-control"));
  const pending = element(
    "div",
    "pending-settings",
    `<details><summary id="pending-count">No pending changes</summary><ul id="pending-diff"></ul></details><button id="apply-configuration" class="primary">Apply and restart</button><button id="discard-configuration">Discard changes</button>`,
  );
  pending.id = "pending-settings";
  pending.hidden = true;
  controls.append(pending);
  const modebar = element(
    "div",
    "mode-toolbar",
    `<div role="group" aria-label="Interaction mode"><button id="mode-inspect" aria-pressed="true">Inspect</button><button id="mode-edit" aria-pressed="false">Edit</button><button id="mode-drive" aria-pressed="false">Drive</button></div><span id="playback-status">Live</span><span id="mode-hint">Click a vehicle to inspect it · Drag to pan</span><button id="toggle-inspector" aria-expanded="false">Inspector</button>`,
  );
  stage.prepend(modebar);
  const inspector = element(
    "aside",
    "selection-inspector",
    `<div class="inspector-heading"><h2>Inspector</h2><button id="close-inspector" aria-label="Close inspector">×</button></div><p id="selected-body">Select a vehicle in the world.</p><dl id="selection-details"></dl><section id="drive-controls" hidden><h3>Drive controls</h3><p id="drive-hint">Select a controlled vehicle.</p><div id="drive-keys"></div><div id="drive-channels"></div><button id="drive-step">Apply action · 1 frame</button></section><details id="decision-inspector"><summary>Selected decision</summary><pre id="decision-details">No decision selected.</pre></details><details id="technical-diagnostics"><summary>Diagnostics</summary><div id="technical-metrics"></div></details>`,
  );
  inspector.id = "selection-inspector";
  document.querySelector("main").append(inspector);
  const metrics = document.querySelector(".telemetry");
  for (const child of [...metrics.children])
    if (
      !child.contains($("time")) &&
      !child.contains($("score")) &&
      !child.contains($("used"))
    )
      $("technical-metrics").append(child);
  metrics.prepend($("score").parentElement);
  $("used").previousElementSibling.textContent = "Planning progress";
  $("score").previousElementSibling.textContent = "Task progress";
  $("time").previousElementSibling.textContent = "Simulation";
  document.querySelector(".legend").innerHTML = '<span><i class="green"></i>High reward</span><span><i class="rose"></i>Terminal</span><span><i class="violet"></i>Tethered / alternative</span>';
  const dock = element(
    "section",
    "timeline-dock",
    `<div class="timeline-tabs" role="group" aria-label="Timeline view"><button id="timeline-motion" aria-pressed="true">World motion</button><button id="timeline-decisions" aria-pressed="false">Planner decisions</button><button id="timeline-events">Events & notes</button></div><div id="motion-workspace"></div><div id="decision-workspace" hidden></div><div id="event-workspace" hidden></div>`,
  );
  stage.append(dock);
  $("motion-workspace").append(document.querySelector(".motion-panel"));
  $("decision-workspace").append(
    document.querySelector(".history-bar"),
    document.querySelector(".replay-bar"),
  );
  $("event-workspace").append(
    $("motion-events"),
    $("event-note"),
    $("add-event"),
  );
  for (const kind of ["motion", "decisions"])
    $(`timeline-${kind}`).onclick = () => {
      $("motion-workspace").hidden = kind !== "motion";
      $("decision-workspace").hidden = kind !== "decisions";
      $("timeline-motion").setAttribute(
        "aria-pressed",
        String(kind === "motion"),
      );
      $("timeline-decisions").setAttribute(
        "aria-pressed",
        String(kind === "decisions"),
      );
      document.dispatchEvent(new CustomEvent("lab-timeline", { detail: kind }));
    };
  $("timeline-events").onclick = () => {
    $("event-workspace").hidden = !$("event-workspace").hidden;
  };
  $("motion-resume").textContent = "Create run from this frame";
  $("motion-live").textContent = "Return to live";
  const files = element(
    "dialog",
    "files-dialog",
    `<form method="dialog"><h2>Save / Open</h2><button class="dialog-close" aria-label="Close Save / Open">×</button></form><label class="field">Run name<input id="run-name" maxlength="120"></label><output id="save-status" role="status">Saved runs stay on this device.</output><div id="recording-files"><h3>Recordings</h3><p>Save motion and recorded decisions locally, or export a portable file.</p></div><div id="scene-files"><h3>Scene</h3><p>World configuration as JSON.</p></div><div id="snapshot-files"><h3>World snapshot</h3><p>Restore a physical world state; the planner starts fresh.</p></div><div id="checkpoint-files"><h3>Planner checkpoint</h3><p>Preserve the exact controller search state.</p></div>`,
  );
  files.id = "files-dialog";
  document.body.append(files);
  const saveSummary = element("span", "save-summary", "Device recording on");
  toolbar.append(saveSummary);
  new MutationObserver(
    () => (saveSummary.textContent = $("save-status").textContent),
  ).observe($("save-status"), {
    childList: true,
    subtree: true,
    characterData: true,
  });
  for (const id of ["flush-run", "open-library", "export-run", "import-run"])
    $("recording-files").append($(id));
  for (const id of ["export-scene", "import-scene"])
    $("scene-files").append($(id));
  for (const id of ["save-state", "load-state"])
    $("snapshot-files").append($(id));
  for (const id of ["save-checkpoint", "load-checkpoint"])
    $("checkpoint-files").append($(id));
  $("persistent-recording").checked = true;
  $("persistent-recording").closest("label").hidden = true;
  document.querySelector(".masthead nav").classList.add("workspace-links");
  $("technical-diagnostics").append(
    $("inspect-physics").closest("label"),
    $("performance-readout"),
    $("physics-readout"),
  );
  document.querySelector(".storage-controls").hidden = true;
  $("save-menu").onclick = () => files.showModal();
  for (const id of ["toggle-settings", "toggle-inspector", "close-inspector"])
    $(id).onclick = () => {
      const key = id === "toggle-settings" ? "settings-open" : "inspector-open";
      document.body.classList.toggle(key);
      $(id === "toggle-settings" ? id : "toggle-inspector").setAttribute(
        "aria-expanded",
        String(document.body.classList.contains(key)),
      );
    };
  const chooser = element(
    "dialog",
    "task-chooser",
    `<form method="dialog"><h2>What would you like to explore?</h2><button class="dialog-close" aria-label="Close task chooser">×</button></form><p>Choose a world, watch a controller plan, then explore its decisions.</p><div id="task-cards"></div><button id="chooser-library">Open saved run</button><button id="chooser-explore">Explore the workspace</button>`,
  );
  chooser.id = "task-chooser";
  document.body.append(chooser);
  $("choose-task").onclick = () => chooser.showModal();
  $("chooser-explore").onclick = () => {
    savePreference("onboarded", true);
    chooser.close();
  };
  $("chooser-library").onclick = () => {
    chooser.close();
    $("open-library").click();
  };
  const intro = element(
    "div",
    "workspace-intro",
    `<span><strong>1.</strong> Run the controller. <strong>2.</strong> Click a vehicle to inspect it. <strong>3.</strong> Pause and explore the timeline.</span><button id="dismiss-intro" aria-label="Dismiss introduction">×</button>`,
  );
  intro.id = "workspace-intro";
  stage.querySelector("#viewport").append(intro);
  intro.hidden = readPreference("intro-dismissed", false);
  $("dismiss-intro").onclick = () => {
    intro.hidden = true;
    savePreference("intro-dismissed", true);
  };
  // Existing numeric properties precede JSON; JSON remains fully available.
  const editor = $("editor");
  const json = element(
    "details",
    "entity-json",
    "<summary>Advanced entity JSON</summary>",
  );
  json.append($("entity").closest("label"), $("apply-entity").parentElement);
  editor.append(json);
  const note = editor.querySelector("p");
  note.textContent =
    "Edit a draft. Drag to move; Shift-click to select several objects. Apply and restart commits all edits together.";
  editor.prepend(
    element(
      "div",
      "editor-actions",
      `<button id="apply-editor" class="primary">Apply and restart</button><button id="discard-editor">Discard</button>`,
    ),
  );
  const leave = element(
    "dialog",
    "",
    `<h2>Keep your scene edits?</h2><p>Your scene has changes that have not been applied.</p><button id="leave-apply">Apply and restart</button><button id="leave-discard">Discard</button><button id="leave-cancel">Cancel</button>`,
  );
  leave.id = "leave-editor";
  document.body.append(leave);
  const failure = element(
    "dialog",
    "",
    `<h2>Run could not be saved</h2><p id="save-error-message"></p><p>The current run is still available. Export a copy before continuing without a device save.</p><button id="save-retry">Retry</button><button id="save-export">Export</button><button id="save-cancel">Cancel</button><button id="save-discard">Continue without saving</button>`,
  );
  failure.id = "save-failure";
  document.body.append(failure);
  document.querySelectorAll("dialog").forEach((dialog) =>
    new MutationObserver(() => {
      if (dialog.open) document.dispatchEvent(new Event("lab-modal"));
    }).observe(dialog, { attributes: true, attributeFilter: ["open"] }),
  );
  return {
    async tasks(entries, choose) {
      const watch = {
        harvest: "Watch the rocket attach to a rock and tow it to the refinery.",
        ants: "Watch harvesters collect drops, fill their tanks, and return to unload.",
        tandem: "Watch two vehicles coordinate their movement through the gates.",
        mining: "Watch vehicles cooperate to haul a heavy rock.",
        rocket: "Watch the planner explore alternative routes through the arena.",
        racing: "Watch the kart steer through checkpoints and complete a lap.",
      };
      for (const entry of entries) {
        const response = await fetch(`./scenarios/${entry.id}.json`);
        if (!response.ok) continue;
        const scene = await response.json();
        const button = element("button", "task-card");
        const title = document.createElement("strong");
        title.textContent = entry.label;
        const description = document.createElement("span");
        description.textContent =
          watch[entry.id] || scene.description ||
          "Explore this world and inspect the controller’s decisions.";
        const detail = document.createElement("small");
        detail.textContent = `${scene.bodies?.filter((b) => b.controlled || b.agent_type).length || scene.bodies?.length || 1} vehicles · Default FMC configuration`;
        button.append(title, description, detail);
        button.onclick = async () => {
          savePreference("onboarded", true);
          chooser.close();
          await choose(entry.id);
        };
        $("task-cards").append(button);
      }
      if (!readPreference("onboarded", false)) chooser.showModal();
    },
  };
}
