import { EntityProperties, actionSliders } from "./entity-properties.js";
import { resolveAgentTypes, resolveBodies } from "./agent-types.js";

// Scene-editing state and undo history are independent of simulation and replay.
export function createSceneEditor({
  renderer,
  getState,
  getInfo,
  getChannels,
  applyAction,
  isReady,
  loadScene,
  stop,
  status,
  error,
  download,
  upload,
  slug,
  onWorldClick,
}) {
  const $ = (id) => document.getElementById(id),
    copy = (value) => structuredClone(value);
  let currentScene,
    selected,
    selections = [],
    action,
    drag,
    polygon = [],
    tetherSource,
    undo = [],
    redo = [];
  const properties = new EntityProperties($("entity-properties"));
  function resolvedSelection() {
    return selected?.key === "bodies"
      ? {
          angle: 0,
          mass: 1,
          radius: 0.5,
          thrust: 12,
          torque: 8,
          drag: 0.15,
          angular_drag: 2,
          ...resolveBodies(currentScene)[selected.i],
        }
      : selected
        ? currentScene[selected.key][selected.i]
        : undefined;
  }
  function commitScene(scene) {
    undo.push(copy(currentScene));
    if (undo.length > 40) undo.shift();
    redo = [];
    loadScene(scene);
  }
  $("edit").onclick = () => {
    stop();
    $("editor").hidden = !$("editor").hidden;
    document.body.classList.toggle("editing", !$("editor").hidden);
  };
  $("close-editor").onclick = () => {
    $("editor").hidden = true;
    document.body.classList.remove("editing");
  };
  $("tool").onchange = () => {
    polygon = [];
    tetherSource = undefined;
    $("hint").textContent = ["holes", "boundary"].includes($("tool").value)
      ? "CLICK VERTICES · FINISH POLYGON TO COMPILE"
      : "CLICK TO SELECT OR PLACE · DRAG TO MOVE";
  };
  function nearest(point) {
    let best,
      distance = Infinity;
    for (const key of ["bodies", "pickups", "bases", "gates", "gravity"])
      for (const [i, entity] of (currentScene[key] || []).entries()) {
        const pos =
          key === "bodies" && getState()
            ? [getState()[8 + i], getState()[8 + getInfo()[1] + i]]
            : entity.position;
        const d = Math.hypot(point[0] - pos[0], point[1] - pos[1]);
        if (d < Math.max(1.5, entity.radius || 1) && d < distance) {
          distance = d;
          best = { key, i, pos };
        }
      }
    return best;
  }
  function showEntity() {
    $("selection-name").textContent = selected
      ? `${selected.key} / ${selected.i}${selections.length > 1 ? ` · ${selections.length} selected` : ""}`
      : "None";
    $("entity").value = selected
      ? JSON.stringify(currentScene[selected.key][selected.i], null, 2)
      : "";
    properties.show(resolvedSelection());
    renderer.selectMany(selections.map((s) => s.pos));
  }
  $("world").addEventListener("pointerdown", (event) => {
    if (!isReady() || event.button !== 0 || event.altKey) return;
    const point = renderer.worldPoint(event);
    if (!point) return;
    if ($("editor").hidden) {
      onWorldClick(point);
      return;
    }
    const tool = $("tool").value;
    if (tool === "select") {
      const hit = nearest(point);
      if (event.shiftKey && hit) {
        const index = selections.findIndex(
          (s) => s.key === hit.key && s.i === hit.i,
        );
        if (index >= 0) selections.splice(index, 1);
        else selections.push(hit);
      } else if (!hit) selections = [];
      else if (!selections.some((s) => s.key === hit.key && s.i === hit.i))
        selections = [hit];
      selected = selections.at(-1);
      drag = hit ? { point, moved: false } : undefined;
      showEntity();
      return;
    }
    if (tool === "holes" || tool === "boundary") {
      polygon.push(point.map((v) => +v.toFixed(3)));
      status(
        `${polygon.length} polygon vertices · click Finish polygon when ready`,
      );
      return;
    }
    const next = copy(currentScene);
    if (tool === "tether") {
      const hit = nearest(point);
      if (hit?.key !== "bodies") return;
      if (tetherSource === undefined) {
        tetherSource = hit.i;
        status("Now click the second body.");
        return;
      }
      if (hit.i === tetherSource) {
        status("Choose a different body.");
        return;
      }
      (next.tethers ||= []).push({
        a: tetherSource,
        b: hit.i,
        rest_length: Math.hypot(
          ...point.map((v, k) => v - next.bodies[tetherSource].position[k]),
        ),
        stiffness: 25,
        damping: 6,
      });
      tetherSource = undefined;
      commitScene(next);
      return;
    }
    const position = point.map((v) => +v.toFixed(3));
    let entity = { position };
    const key = tool === "cargo" ? "bodies" : tool;
    if (tool === "bodies" && $("agent-type").value)
      entity = { position, agent_type: $("agent-type").value };
    else if (tool === "bodies")
      entity = {
        position,
        radius: 0.65,
        controlled: true,
        mass: 1,
        thrust: 16,
        torque: 3,
      };
    else if (tool === "cargo")
      entity = {
        position,
        cargo: true,
        mass: 3,
        vertices: [
          [1, 0],
          [0.5, 0.86],
          [-0.5, 0.86],
          [-1, 0],
          [-0.5, -0.86],
          [0.5, -0.86],
        ],
      };
    else if (tool === "gravity")
      entity = { position, strength: 25, softening: 3 };
    else entity.radius = tool === "pickups" ? 0.4 : 2.5;
    (next[key] ||= []).push(entity);
    commitScene(next);
  });
  $("world").addEventListener("pointermove", (event) => {
    if (!drag || !selected) return;
    const point = renderer.worldPoint(event);
    if (!point) return;
    drag.moved =
      drag.moved || Math.hypot(...point.map((v, k) => v - drag.point[k])) > 0.1;
    drag.end = point;
    renderer.selectMany(
      selections.map((s) => s.pos.map((v, k) => v + point[k] - drag.point[k])),
    );
  });
  window.addEventListener("pointerup", () => {
    if (drag?.moved && selected) {
      const next = copy(currentScene);
      for (const entity of selections)
        next[entity.key][entity.i].position = entity.pos.map(
          (v, k) => +(v + drag.end[k] - drag.point[k]).toFixed(3),
        );
      commitScene(next);
    }
    drag = undefined;
  });
  $("finish-polygon").onclick = () => {
    if (polygon.length < 3) {
      status("A polygon needs at least three vertices.");
      return;
    }
    const next = copy(currentScene);
    if ($("tool").value === "boundary") next.boundary = polygon;
    else (next.holes ||= []).push(polygon);
    polygon = [];
    commitScene(next);
  };
  $("apply-entity").onclick = () => {
    try {
      if (!selected) return;
      const next = copy(currentScene);
      next[selected.key][selected.i] = JSON.parse($("entity").value);
      commitScene(next);
    } catch (e) {
      error(e);
    }
  };
  $("apply-properties").onclick = () => {
    try {
      if (!selected) return;
      const next = copy(currentScene);
      next[selected.key][selected.i] = properties.apply(resolvedSelection());
      commitScene(next);
    } catch (e) {
      error(e);
    }
  };
  $("duplicate-entities").onclick = () => {
    if (!selections.length) return;
    const next = copy(currentScene),
      mapping = new Map();
    for (const entity of selections) {
      const clone = copy(next[entity.key][entity.i]);
      clone.position = entity.pos.map((v) => v + 2);
      if (entity.key === "bodies") mapping.set(entity.i, next.bodies.length);
      next[entity.key].push(clone);
    }
    for (const tether of currentScene.tethers || [])
      if (mapping.has(tether.a) && mapping.has(tether.b))
        (next.tethers ||= []).push({
          ...tether,
          a: mapping.get(tether.a),
          b: mapping.get(tether.b),
        });
    commitScene(next);
  };
  $("save-template").onclick = () => {
    try {
      if (selected?.key !== "bodies")
        throw new Error("Select a body to save an agent template");
      const name = $("template-name").value.trim();
      if (
        !name ||
        name.length > 80 ||
        ["__proto__", "constructor", "prototype"].includes(name)
      )
        throw new Error("Enter a template name (1–80 characters)");
      const next = copy(currentScene);
      if (Object.hasOwn(next.agent_types || {}, name))
        throw new Error("A template with that name already exists");
      const { agent_type, position, velocity, angle, visual, ...physics } =
        resolveBodies(currentScene)[selected.i];
      (next.agent_types ||= {})[name] = {
        label: name,
        physics,
        visual: visual || {},
      };
      commitScene(next);
      status(`Saved agent type: ${name}`);
    } catch (e) {
      error(e);
    }
  };
  $("apply-action").onclick = () => {
    if (action && isReady()) applyAction(action.slice());
  };
  $("delete-entity").onclick = () => {
    if (!selections.length) return;
    const next = copy(currentScene),
      removed = new Set(
        selections.filter((s) => s.key === "bodies").map((s) => s.i),
      );
    const remap = new Map();
    let count = 0;
    next.bodies.forEach((_, i) => {
      if (!removed.has(i)) remap.set(i, count++);
    });
    for (const key of new Set(selections.map((s) => s.key))) {
      const indices = new Set(
        selections.filter((s) => s.key === key).map((s) => s.i),
      );
      next[key] = next[key].filter((_, i) => !indices.has(i));
    }
    next.tethers = (next.tethers || [])
      .filter((t) => !removed.has(t.a) && !removed.has(t.b))
      .map((t) => ({ ...t, a: remap.get(t.a), b: remap.get(t.b) }));
    commitScene(next);
  };
  $("undo").onclick = () => {
    if (undo.length) {
      redo.push(copy(currentScene));
      loadScene(undo.pop());
    }
  };
  $("redo").onclick = () => {
    if (redo.length) {
      undo.push(copy(currentScene));
      loadScene(redo.pop());
    }
  };
  $("export-scene").onclick = () =>
    download(JSON.stringify(currentScene, null, 2), `${slug()}.json`);
  $("import-scene").onclick = () =>
    upload(".json", async (file) => commitScene(JSON.parse(await file.text())));
  $("edit-json").onclick = () => {
    $("scene-json").value = JSON.stringify(currentScene, null, 2);
    $("json-error").textContent = "";
    $("json-dialog").showModal();
  };
  $("apply-json").onclick = () => {
    try {
      const next = JSON.parse($("scene-json").value);
      if (!next.bodies?.length)
        throw new Error("Scene needs at least one body");
      commitScene(next);
      $("json-dialog").close();
    } catch (e) {
      $("json-error").textContent = e.message;
    }
  };

  return {
    get selection() {
      return selected;
    },
    refreshChannels() {
      action = actionSliders($("action-channels"), getChannels());
    },
    clearHistory() {
      undo = [];
      redo = [];
    },
    setScene(scene) {
      currentScene = copy(scene);
      selected = drag = tetherSource = undefined;
      polygon = [];
      selections = [];
      properties.show();
      const types = resolveAgentTypes(scene.agent_types);
      $("agent-type").replaceChildren(
        ...Array.from(
          types,
          ([name, def]) => new Option(def.label || name, name),
        ),
      );
      if (!types.size) $("agent-type").add(new Option("Default agent", ""));
      $("selection-name").textContent = "None";
      $("entity").value = "";
    },
  };
}
