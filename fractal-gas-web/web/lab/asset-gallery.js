import * as T from "./vendor/three.module.js";
import { assetModel } from "./visuals/assets.js";
import { animatedParts, animateAgent } from "./visuals/registry.js";
import { laboratoryEnvironment, contactShadow } from "./visuals/lighting.js";
import { disposeGroup } from "./visuals/resources.js";
import { stylePalette } from "./visuals/style-palette.js";
import { labStyle } from "./visual-style.js";
import { installStyleControls } from "./style-controls.js";
import { worldCatalog } from "./visuals/world-catalog.js";
import { animateWorld } from "./visuals/world.js";

const canvas = document.getElementById("asset-world");
const renderer = new T.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = T.ACESFilmicToneMapping;
const world = new T.Scene();
const camera = new T.OrthographicCamera(-1.5, 1.5, 1, -1, 0.01, 50);
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
  refinery: "Shared unloading apron, receiving hopper, twin processing tanks and transfer machinery. Full vehicles discharge gradually over two simulation seconds.",
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
      new T.Vector3(
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
      parts = animatedParts(model);
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
  const moving = document.getElementById("animate-parts").checked;
  animateAgent(parts, {
    time: moving ? time / 1000 : 0,
    speed: moving ? 0.4 : 0,
    thrust: moving ? 0.6 : 0,
    steer: moving ? Math.sin(time / 2000) * 0.6 : 0,
  });
  if (model) animateWorld(model, moving ? time / 1000 : 0);
  poseCamera();
  renderer.render(world, camera);
}
installStyleControls();
requestAnimationFrame(animate);
