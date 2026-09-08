import { resolveBodies } from "./agent-types.js";
import { inside, clearance } from "./ants-scene.js";

export function rockOptions(scene) {
  if (scene.task !== "harvest") return null;
  const count = resolveBodies(scene).filter((body) => body.cargo).length;
  return count
    ? {
        scale: scene.rock_options?.scale ?? 1,
        count,
        weight: scene.rock_options?.weight ?? 1,
      }
    : null;
}
const radius = (body) =>
  body.vertices?.length
    ? Math.max(...body.vertices.map(([x, y]) => Math.hypot(x, y)))
    : (body.radius ?? 0.5);

// Scale the collision hull itself: rendering and physics share these dimensions.
export function configureRocks(template, { scale, count, weight, stiffness }) {
  const previous = rockOptions(template);
  if (!previous) throw new RangeError("This scene has no mining rocks");
  if (!Number.isFinite(scale) || scale < 0.1 || scale > 2)
    throw new RangeError(
      "Rock size must be from 0.1 to 2 times the original size",
    );
  if (!Number.isInteger(count) || count < 1 || count > 20)
    throw new RangeError("Rock count must be an integer from 1 to 20");
  const nextWeight = weight ?? previous.weight;
  if (!Number.isFinite(nextWeight) || nextWeight < 0.01 || nextWeight > 10)
    throw new RangeError(
      "Rock weight must be from 0.01 to 10 times the original weight",
    );
  if (
    stiffness !== undefined &&
    (!Number.isFinite(stiffness) || stiffness < 0 || stiffness > 1000000)
  )
    throw new RangeError("Hook stiffness must be between 0 and 1000000 N/m");
  const scene = structuredClone(template);
  if (stiffness !== undefined)
    for (const tether of scene.tethers || []) tether.stiffness = stiffness;
  scene.boundary ||= [
    [0, 0],
    [scene.size[0], 0],
    [...scene.size],
    [0, scene.size[1]],
  ];
  const resolved = resolveBodies(template);
  const rocks = resolved.filter((body) => body.cargo);
  const mapping = new Map();
  const keptIndices = [];
  let kept = 0;
  // Keep agent_type references on vehicle instances. Flattening resolved
  // actuators here would shadow later per-agent action multiplier edits.
  scene.bodies = template.bodies
    .filter((body, i) => {
      if (resolved[i].cargo && kept++ >= count) return false;
      keptIndices.push(i);
      mapping.set(i, mapping.size);
      return true;
    })
    .map((body) => structuredClone(body));
  scene.tethers = (scene.tethers || [])
    .filter(
      ({ a, b, automatic }) =>
        mapping.has(a) && (b === -1 || mapping.has(b) || automatic),
    )
    .map((t) => ({ ...t, a: mapping.get(t.a), b: mapping.get(t.b) ?? -1 }));
  for (let i = rocks.length; i < count; i++)
    scene.bodies.push(structuredClone(rocks[i % rocks.length]));
  const occupied = keptIndices
    .filter((i) => !resolved[i].cargo)
    .map((i) => structuredClone(resolved[i]));
  for (const body of scene.bodies.filter((body) => body.cargo)) {
    const ratio = scale / previous.scale;
    body.mass = (body.mass ?? 1) * (nextWeight / previous.weight);
    if (body.vertices?.length)
      body.vertices = body.vertices.map(([x, y]) => [x * ratio, y * ratio]);
    body.radius = radius(body) * (body.vertices?.length ? 1 : ratio);
    body.respawn = true;
    const r = radius(body);
    const clear = (p) =>
      inside(p, scene.boundary) &&
      clearance(p, scene.boundary) >= r + 0.1 &&
      (scene.holes || []).every(
        (ring) => !inside(p, ring) && clearance(p, ring) >= r + 0.1,
      ) &&
      (scene.bases || []).every(
        (base) =>
          Math.hypot(p[0] - base.position[0], p[1] - base.position[1]) >=
          r + base.radius + 0.1,
      ) &&
      occupied.every(
        (other) =>
          Math.hypot(p[0] - other.position[0], p[1] - other.position[1]) >=
          r + radius(other) + 0.1,
      );
    if (!clear(body.position)) {
      let position;
      for (let y = r + 0.1; y < scene.size[1] && !position; y += r + 0.2)
        for (let x = r + 0.1; x < scene.size[0]; x += r + 0.2)
          if (clear([x, y])) {
            position = [x, y];
            break;
          }
      if (!position)
        throw new RangeError(
          "Insufficient clear space: reduce rock size or count",
        );
      body.position = position;
    }
    occupied.push(body);
  }
  scene.rock_options = { ...scene.rock_options, scale, weight: nextWeight };
  return scene;
}
