import * as T from "../vendor/three.module.js";
import { shipModel, kartModel, droneModel } from "./vehicles.js";
import { harvesterModel } from "./harvester.js";
import { metal, glow } from "./primitives.js";

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
export function createAgentModel(visual = {}, color) {
  if (
    visual.scale != null &&
    (!Number.isFinite(visual.scale) || visual.scale <= 0 || visual.scale > 100)
  )
    throw new Error("Visual scale must be in (0, 100]");
  const factory = factories.get(visual.model ?? "rocket");
  if (!factory) throw new Error(`Unknown agent model: ${visual.model}`);
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

export function animatedParts(model) {
  const parts = [];
  model.traverse((part) => {
    if (part.userData.motion)
      parts.push({
        part,
        rotation: part.rotation.clone(),
        scale: part.scale.clone(),
      });
  });
  return parts;
}
// Animation derives from simulation time and state: scrubbing is deterministic.
export function animateAgent(parts, { time, speed, thrust, steer }) {
  for (const { part, rotation, scale } of parts) {
    part.rotation.copy(rotation);
    part.scale.copy(scale);
    part.visible = true;
    switch (part.userData.motion) {
      case "thrust":
        part.visible = thrust > 0.01;
        part.scale.x =
          scale.x * (0.35 + thrust * (0.85 + 0.1 * Math.sin(time * 45)));
        break;
      case "steer":
        part.rotation.z += steer * 0.35;
        break;
      case "wheel":
        part.rotation.y += (time * speed) / 0.215;
        break;
      case "rotor":
        part.rotation.z += time * 32;
        break;
    }
  }
}
