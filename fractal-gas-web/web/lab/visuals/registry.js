import * as T from "../vendor/three.module.js";
import { shipModel, kartModel, droneModel } from "./vehicles.js";
import { harvesterModel } from "./harvester.js";
import { metal, glow } from "./primitives.js";
import { assetModel } from "./assets.js";

// Register a factory once. Factories return a Three Group; hierarchy is unrestricted.
const factories = new Map();
export function registerAgentModel(name, factory) {
  if (
    typeof name !== "string" ||
    !name ||
    factories.has(name) ||
    typeof factory !== "function"
  )
    throw new Error(`Invalid or duplicate model: ${name}`);
  factories.set(name, factory);
}
export function createAgentModel(
  visual = {},
  color,
  { style, lod = "high" } = {},
) {
  if (
    visual.scale != null &&
    (!Number.isFinite(visual.scale) || visual.scale <= 0 || visual.scale > 100)
  )
    throw new Error("Visual scale must be in (0, 100]");
  const factory = factories.get(visual.model ?? "rocket");
  if (!factory) throw new Error(`Unknown agent model: ${visual.model}`);
  if (style) {
    const asset = assetModel(
      style,
      visual.model ?? "rocket",
      lod,
      visual.color ?? color,
    );
    if (asset) return asset;
  }
  return factory(visual.color ?? color, visual);
}
// JSON-defined kits allow new silhouettes without changes to the engine or renderer.
registerAgentModel("kit", (color, { parts = [] }) => {
  if (!Array.isArray(parts) || !parts.length || parts.length > 128)
    throw new Error("Model kits need 1–128 parts");
  const group = new T.Group();
  for (const part of parts) {
    const size = part.size || [1, 1, 1];
    const dimensions =
      part.shape === "box" ? 3 : part.shape === "sphere" ? 1 : 2;
    if (
      !Array.isArray(size) ||
      size.length < dimensions ||
      size.length > 3 ||
      !size.every((v) => Number.isFinite(v) && v > 0 && v <= 100)
    )
      throw new Error("Invalid kit dimensions");
    for (const field of ["position", "rotation"]) {
      const vector = part[field] ?? [0, 0, 0];
      if (
        !Array.isArray(vector) ||
        vector.length !== 3 ||
        !vector.every(Number.isFinite)
      )
        throw new Error(`Invalid kit ${field}`);
    }
    if (
      part.motion != null &&
      !["thrust", "steer", "wheel", "rotor"].includes(part.motion)
    )
      throw new Error(`Unknown kit motion: ${part.motion}`);
    let geometry;
    switch (part.shape) {
      case "box":
        geometry = new T.BoxGeometry(...size);
        break;
      case "sphere":
        geometry = new T.SphereGeometry(size[0], 16, 10);
        break;
      case "cylinder":
        geometry = new T.CylinderGeometry(size[0], size[0], size[1], 16);
        break;
      case "cone":
        geometry = new T.ConeGeometry(size[0], size[1], 16);
        break;
      case "ring":
        geometry = new T.TorusGeometry(size[0], size[1], 6, 24);
        break;
      default:
        throw new Error(`Unknown kit shape: ${part.shape}`);
    }
    const mesh = new T.Mesh(
      geometry,
      (part.emissive ? glow : metal)(part.color ?? color),
    );
    mesh.position.set(...(part.position || [0, 0, 0]));
    mesh.rotation.set(...(part.rotation || [0, 0, 0]));
    mesh.userData.motion = part.motion;
    group.add(mesh);
  }
  return group;
});
registerAgentModel("rocket", shipModel);
registerAgentModel("kart", kartModel);
registerAgentModel("drone", droneModel);
registerAgentModel("harvester", harvesterModel);

function animationVisible(binding) {
  for (const ancestor of binding.ancestors) if (!ancestor.visible) return false;
  return true;
}
function animationAncestors(part, model) {
  const ancestors = [];
  for (let p = part.parent; p && p !== model; p = p.parent) ancestors.push(p);
  return ancestors;
}

export function animatedParts(
  model,
  { kind = model.userData.assetModel, style = model.userData.assetStyle } = {},
) {
  const parts = [];
  const airborne = kind === "drone" || kind === "rocket";
  if (airborne) {
    const presentation = new T.Group();
    presentation.name = "Vehicle cosmetic presentation";
    for (const child of [...model.children])
      if (!/marker|shadow/i.test(child.name)) presentation.add(child);
    model.add(presentation);
    parts.pose = presentation;
  } else parts.pose = new T.Object3D();
  model.traverse((part) => {
    if (part.userData.motion)
      parts.push({
        part,
        rotation: part.rotation.clone(),
        scale: part.scale.clone(),
        ancestors: animationAncestors(part, model),
        intake: /intake/i.test(part.name),
      });
  });
  // Cache independent child branches once. Wheels and their steering carriers
  // stay planted; markers and caller-owned attachments keep authoritative poses.
  parts.presentation = [];
  parts.kind = kind;
  parts.style = style;
  function bind(node) {
    if (/marker|shadow/i.test(node.name)) return;
    if (node.userData.motion) return;
    if (node.isMesh) {
      parts.presentation.push({
        part: node,
        position: node.position.clone(),
        rotation: node.rotation.clone(),
        ancestors: animationAncestors(node, model),
        engine:
          /engine|chimney|cab/i.test(node.name) ||
          (node.position.x > 0.25 && node.position.z > 0.5),
      });
      return;
    }
    for (const child of node.children) bind(child);
  }
  if (!airborne) bind(model);
  return parts;
}
// Animation derives from simulation time and state: scrubbing is deterministic.
export function animateAgent(
  parts,
  {
    time = 0,
    speed = 0,
    thrust = 0,
    steer = 0,
    modelScale = 1,
    enabled = true,
    idleTime,
    playing = true,
    wheelTravel,
    mechanicalTime = time,
    throttle = thrust,
    brake = 0,
  },
) {
  for (const binding of parts) {
    if (enabled && idleTime != null && !animationVisible(binding)) continue;
    const { part, rotation, scale } = binding;
    part.rotation.copy(rotation);
    part.scale.copy(scale);
    part.visible = true;
    switch (part.userData.motion) {
      case "thrust":
        part.visible = enabled && thrust > 0.01;
        part.scale.x = enabled
          ? scale.x *
            (0.35 +
              thrust *
                (0.85 +
                  0.1 *
                    Math.sin(
                      (idleTime ?? time) *
                        (parts.style === "steampunk" ? 23 : 45),
                    )))
          : scale.x;
        break;
      case "steer":
        part.rotation.z += steer * 0.35;
        break;
      case "wheel":
        part.rotation.y += enabled
          ? (wheelTravel ?? time * speed) /
            ((part.userData.wheelRadius || 0.215) * modelScale)
          : 0;
        if (enabled && idleTime != null && binding.intake)
          part.rotation.y +=
            mechanicalTime * (parts.style === "steampunk" ? 1.4 : 2);
        break;
      case "rotor":
        part.rotation.z += enabled ? (idleTime ?? time) * 32 : 0;
        break;
    }
  }
  if (
    parts.presentation &&
    ((enabled && idleTime != null) || parts.cosmeticActive)
  ) {
    const active = enabled && idleTime != null;
    parts.cosmeticActive = active;
    const clock = idleTime ?? 0;
    const heavy = parts.style === "steampunk";
    const drone = parts.kind === "drone",
      rocket = parts.kind === "rocket";
    const airborne = drone || rocket;
    const ground = parts.kind === "kart" || parts.kind === "harvester";
    const weight = parts.kind === "harvester" ? 0.4 : 1;
    const bank = active
      ? airborne
        ? -steer * (drone ? 0.07 : 0.045)
        : ground
          ? -steer * Math.min(1, Math.abs(speed)) * 0.012 * weight
          : 0
      : 0;
    const pitch = active
      ? airborne
        ? thrust * 0.025
        : ground
          ? (-throttle + brake * 1.4) * 0.009 * weight
          : 0
      : 0;
    const lift = active
      ? airborne
        ? Math.sin(clock * (heavy ? 1.8 : 2.4)) * (drone ? 0.022 : 0.006)
        : parts.kind === "kart"
          ? Math.sin(clock * (heavy ? 8 : 12)) * 0.0015 - thrust * 0.004
          : 0
      : 0;
    parts.pose.position.z = lift;
    parts.pose.rotation.set(bank, pitch, 0);
    for (const binding of parts.presentation) {
      if (active && !animationVisible(binding)) continue;
      const { part, position, rotation } = binding;
      part.position.copy(position);
      part.rotation.copy(rotation);
      part.position.z += lift + bank * position.y - pitch * position.x;
      if (active && parts.kind === "harvester" && binding.engine)
        part.position.z += Math.sin(clock * (heavy ? 9 : 14)) * 0.0015;
      part.rotation.x += bank;
      part.rotation.y += pitch;
    }
  }
}
