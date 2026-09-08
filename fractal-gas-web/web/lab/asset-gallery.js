import * as T from "./vendor/three.module.js";
import { assetModel } from "./visuals/assets.js";
import { animatedParts, animateAgent } from "./visuals/registry.js";
import { laboratoryEnvironment, contactShadow } from "./visuals/lighting.js";
import { disposeGroup } from "./visuals/resources.js";
import { stylePalette } from "./visuals/style-palette.js";
import { labStyle } from "./visual-style.js";
import { installStyleControls } from "./style-controls.js";
import { labAnimations } from "./animations.js";
import { installAnimationControls } from "./animation-controls.js";
import { worldCatalog } from "./visuals/world-catalog.js";
import { animateWorld } from "./visuals/world.js";
import { actionLayout, createActionBinding } from "./actions.js";
import { resolveAgentTypes } from "./agent-types.js";
import { ActionEffects } from "./visuals/action-effects.js";
import {
  labActionGuides,
  installActionGuideControls,
} from "./action-guides.js";

const catalogResponse = await fetch("./agent-catalog.json");
if (!catalogResponse.ok) throw new Error("Could not load actuator catalog");
const agentCatalog = resolveAgentTypes(await catalogResponse.json());

const canvas = document.getElementById("asset-world");
const renderer = new T.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = T.ACESFilmicToneMapping;
const world = new T.Scene();
const camera = new T.OrthographicCamera(-1.5, 1.5, 1, -1, 0.01, 50);
const cameraOffset = new T.Vector3();
camera.up.set(0, 0, 1);
const holder = new T.Group();
world.add(holder);
const key = new T.DirectionalLight(0xe4efff, 3);
key.position.set(3, -4, 5);
world.add(key, new T.HemisphereLight(0xdcecff, 0x33261e, 2));
const rim = new T.DirectionalLight(0x99aaff, 2);
rim.position.set(-2, 2, 3);
world.add(rim);
let environment,
  model,
  parts = [],
  yaw = 0,
  pitch = 0.58,
  view = "hero";
let center = new T.Vector3(0, 0, 0.4),
  modelSpan = 2;
const select = document.getElementById("vehicle");
const playButton = document.getElementById("animate-parts");
const variant = document.getElementById("actuator-variant");
const sliderPanel = document.getElementById("action-sliders");
const guideReadout = document.getElementById("action-guide-readout");
let binding,
  actionEffects,
  previewLayer,
  actionValues,
  actionChannels = [];
const rememberedActions = new Map();
const sliderControls = [];
let previewPlaying = false,
  previewTime = 0,
  lastTime;
const motion = {
  time: 0,
  idleTime: 0,
  speed: 0,
  thrust: 0,
  steer: 0,
  enabled: false,
  playing: false,
  wheelTravel: 0,
};
function resetPreview() {
  previewTime = 0;
  lastTime = undefined;
  motion.time =
    motion.idleTime =
    motion.speed =
    motion.thrust =
    motion.steer =
    motion.wheelTravel =
      0;
  motion.enabled = motion.playing = false;
  animateAgent(parts, motion);
  if (model) animateWorld(model, 0, { enabled: false });
  refreshActionPresentation();
}
function refreshActionPresentation() {
  if (!binding) return;
  motion.commands = binding.sample(actionValues);
  animateAgent(parts, motion);
  actionEffects?.update();
  guideReadout.textContent =
    actionEffects?.updateGuides(0, labActionGuides.enabled) || "";
  guideReadout.hidden = !labActionGuides.enabled;
}
function prepareActuator(style) {
  actionEffects?.dispose();
  actionEffects = previewLayer = binding = undefined;
  motion.commands = undefined;
  sliderPanel.replaceChildren();
  sliderControls.length = 0;
  const typeName = select.value === "rocket" ? variant.value : select.value;
  const definition = agentCatalog.get(typeName);
  document.getElementById("action-preview").hidden = !definition;
  document.getElementById("actuator-variant-label").hidden =
    select.value !== "rocket";
  if (!definition) {
    guideReadout.hidden = true;
    return;
  }
  const body = { ...definition.physics, visual: { ...definition.visual } };
  actionChannels = actionLayout({ bodies: [body] });
  actionValues = rememberedActions.get(typeName);
  if (!actionValues) {
    actionValues = new Float32Array(actionChannels.length);
    rememberedActions.set(typeName, actionValues);
  }
  binding = createActionBinding(actionChannels, body, 0);
  motion.commands = binding.sample(actionValues);
  previewLayer = {
    models: [model],
    controlled: [0],
    commands: [motion.commands],
    bodies: [body],
    presentations: [parts.pose],
    inView: [true],
    active: [true],
  };
  actionEffects = new ActionEffects(previewLayer, holder, style);
  document.getElementById("actuator-kind").textContent = {
    vector: "Thrust and torque",
    kart: "Drive, steering and brake",
    holonomic: "Planar force and torque",
    thrusters: "Independent thruster commands",
  }[body.actuator.kind];
  actionChannels.forEach((channel, index) => {
    const label = document.createElement("label");
    const name = channel.name.startsWith("thruster_")
      ? `Thruster ${Number(channel.name.slice(9)) + 1}`
      : channel.name.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
    const title = document.createElement("span");
    title.textContent = name;
    const value = document.createElement("output");
    const input = document.createElement("input");
    input.type = "range";
    input.id = `preview-${channel.name}`;
    input.min = channel.low;
    input.max = channel.high;
    input.step = 0.01;
    input.value = actionValues[index];
    input.setAttribute("aria-label", name);
    value.htmlFor = input.id;
    const update = () => {
      actionValues[index] = Number(input.value);
      value.textContent = Number(input.value).toFixed(2);
      input.setAttribute("aria-valuetext", value.textContent);
    };
    update();
    input.addEventListener("input", () => {
      update();
      refreshActionPresentation();
    });
    sliderControls.push({ input, update });
    label.append(title, value, input);
    sliderPanel.append(label);
  });
  refreshActionPresentation();
}
variant.addEventListener("change", () => {
  if (model) prepareActuator(labStyle.current);
});
for (const [id, maximum] of [
  ["action-neutral", false],
  ["action-max", true],
])
  document.getElementById(id).addEventListener("click", () => {
    sliderControls.forEach(({ input, update }, index) => {
      input.value = maximum ? actionChannels[index].high : 0;
      update();
    });
    refreshActionPresentation();
  });
labActionGuides.subscribe(refreshActionPresentation);
function syncPlayback() {
  playButton.disabled = !labAnimations.enabled;
  playButton.textContent = previewPlaying
    ? "Pause animation"
    : "Play animation";
  playButton.setAttribute("aria-pressed", String(previewPlaying));
  playButton.title = labAnimations.enabled
    ? "Preview this asset’s motion"
    : "Enable Animations above to preview motion";
}
playButton.addEventListener("click", () => {
  previewPlaying = !previewPlaying;
  lastTime = undefined;
  syncPlayback();
});
labAnimations.subscribe(() => {
  if (!labAnimations.enabled) {
    previewPlaying = false;
    resetPreview();
  }
  syncPlayback();
});
document.addEventListener("visibilitychange", () => {
  lastTime = undefined;
});
const groups = new Map();
for (const [kind, spec] of Object.entries(worldCatalog)) {
  if (!groups.has(spec.family)) {
    const group = document.createElement("optgroup");
    group.label = spec.family.replaceAll("-", " ");
    groups.set(spec.family, group);
    select.append(group);
  }
  const option = document.createElement("option");
  option.value = kind;
  option.textContent = kind
    .replaceAll("-", " ")
    .replace(/^./, (c) => c.toUpperCase());
  groups.get(spec.family).append(option);
}
const requested = new URLSearchParams(location.search).get("asset");
if ([...select.options].some((option) => option.value === requested))
  select.value = requested;
const descriptions = {
  refinery:
    "Shared unloading apron, receiving hopper, twin processing tanks and transfer machinery. Full vehicles discharge gradually over two simulation seconds.",
  rocket:
    "Twin engine pods, a closed canopy, swept stabilizers, and independent animated exhausts.",
  kart: "Four exposed wheels, front steering, an open cockpit, and a chassis built around its power unit.",
  drone:
    "Four protected survey rotors surround a central optical instrument and compact hull.",
  harvester:
    "Six wheels carry the raised cab and ore hopper. The toothed mineral intake turns independently.",
};
function resize() {
  const { width, height } = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(width, height, false);
  const aspect = width / height,
    span = modelSpan * 1.24;
  camera.left = (-span * Math.max(1, aspect)) / 2;
  camera.right = -camera.left;
  camera.top = (span * Math.max(1, 1 / aspect)) / 2;
  camera.bottom = -camera.top;
  camera.updateProjectionMatrix();
}
function poseCamera() {
  const angle =
    view === "side" ? 0 : view === "top" ? Math.PI / 2 - 0.0001 : pitch;
  const turn = view === "hero" ? yaw + 0.65 : 0;
  camera.position
    .copy(center)
    .add(
      cameraOffset.set(
        modelSpan * 1.5 * Math.cos(angle) * Math.sin(turn),
        -modelSpan * 1.5 * Math.cos(angle) * Math.cos(turn),
        modelSpan * 1.5 * Math.sin(angle),
      ),
    );
  camera.lookAt(center);
}
function prepare(style) {
  const lod = document.getElementById("asset-lod").value;
  const next = assetModel(style, select.value, lod);
  if (!next) throw new Error("The selected asset has not loaded");
  const nextEnvironment = laboratoryEnvironment(renderer, style);
  return {
    cancel() {
      nextEnvironment.dispose();
      disposeGroup(next);
    },
    commit() {
      actionEffects?.dispose();
      actionEffects = binding = undefined;
      disposeGroup(holder);
      environment?.dispose();
      environment = nextEnvironment;
      world.environment = environment.texture;
      world.background = new T.Color(stylePalette[style].background);
      key.color.setHex(style === "steampunk" ? 0xffdfb4 : 0xe4efff);
      rim.color.setHex(style === "steampunk" ? 0xe4ad68 : 0x99aaff);
      model = next;
      holder.add(model, contactShadow());
      const bounds = new T.Box3().setFromObject(model),
        size = bounds.getSize(new T.Vector3());
      center = bounds.getCenter(new T.Vector3());
      modelSpan = Math.max(size.x, size.y, size.z);
      const spec = worldCatalog[select.value];
      if (spec) {
        const box = new T.Box3(
          new T.Vector3(-spec.size[0] / 2, -spec.size[1] / 2, 0.025),
          new T.Vector3(
            spec.size[0] / 2,
            spec.size[1] / 2,
            spec.size[2] + 0.025,
          ),
        );
        const helper = new T.Box3Helper(box, 0xf2c575);
        helper.name = "Shared asset envelope";
        helper.visible = document.getElementById("show-envelope").checked;
        holder.add(helper);
      }
      parts = animatedParts(model, { kind: select.value, style });
      prepareActuator(style);
      resetPreview();
      const name = select.selectedOptions[0].text;
      document.getElementById("asset-title").textContent = name;
      document.getElementById("asset-description").textContent = spec
        ? `Shared envelope: ${spec.size.join(" × ")} units. Both styles use the same scene-defined collisions. The GLB download contains all 42 individually addressable world assets.`
        : descriptions[select.value];
      const concept = spec
        ? `./concepts/${style}/world/${spec.family}.png`
        : `./concepts/${style}/${select.value}.png`;
      const reference = document.getElementById("concept-image");
      reference.hidden = true;
      reference.onload = () => {
        reference.hidden = false;
      };
      reference.src = concept;
      reference.alt = `${style} ${name} concept sheet`;
      document.getElementById("concept-link").href = concept;
      document.getElementById("download-glb").href =
        `./assets/${style}/${spec ? "world" : select.value}-${lod}.glb`;
      document.getElementById("download-glb").textContent = spec
        ? "Download world GLB collection"
        : "Download 3D model";
      document.getElementById("download-blend").href =
        `./assets/sources/${style}/${spec ? "world" : select.value}.blend`;
      resize();
    },
  };
}
labStyle.subscribe(prepare);
select.addEventListener("change", () => {
  if (labStyle.ready) prepare(labStyle.current).commit();
});
document.getElementById("asset-lod").addEventListener("change", () => {
  if (labStyle.ready) prepare(labStyle.current).commit();
});
document.getElementById("show-envelope").addEventListener("change", (event) => {
  const box = holder.getObjectByName("Shared asset envelope");
  if (box) box.visible = event.target.checked;
});
for (const button of document.querySelectorAll("[data-view]"))
  button.addEventListener("click", () => {
    view = button.dataset.view;
    if (view !== "hero") {
      previewPlaying = false;
      resetPreview();
      syncPlayback();
    }
    for (const other of document.querySelectorAll("[data-view]"))
      other.setAttribute("aria-pressed", String(other === button));
  });
let drag;
canvas.addEventListener("pointerdown", (event) => {
  drag = [event.clientX, event.clientY];
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!drag) return;
  view = "hero";
  yaw -= (event.clientX - drag[0]) * 0.008;
  pitch = T.MathUtils.clamp(
    pitch + (event.clientY - drag[1]) * 0.006,
    0.04,
    1.5,
  );
  drag = [event.clientX, event.clientY];
});
canvas.addEventListener("pointerup", () => {
  drag = null;
});
canvas.addEventListener("lostpointercapture", () => {
  drag = null;
});
new ResizeObserver(resize).observe(canvas.parentElement);
function animate(time) {
  requestAnimationFrame(animate);
  if (document.hidden) return;
  if (previewPlaying && labAnimations.enabled) {
    const dt =
      lastTime === undefined ? 0 : Math.min((time - lastTime) / 1000, 0.05);
    previewTime += dt;
    motion.time = motion.idleTime = previewTime;
    // Actions come only from sliders; playback advances decorative clocks.
    motion.speed = 0;
    motion.enabled = motion.playing = true;
    animateAgent(parts, motion);
    if (model) animateWorld(model, previewTime);
    actionEffects?.update();
  }
  lastTime = time;
  poseCamera();
  renderer.render(world, camera);
}
installStyleControls();
installAnimationControls();
installActionGuideControls();
syncPlayback();
requestAnimationFrame(animate);
