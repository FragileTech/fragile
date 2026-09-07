import { resolveBodies } from "./agent-types.js";
import { inside, clearance } from "./ants-scene.js";

export const MAX_VEHICLES = 128;
export const vehicleCount = (scene) =>
  resolveBodies(scene).filter((body) => body.controlled).length;

function radius(body) {
  return body.vertices?.length
    ? Math.max(...body.vertices.map(([x, y]) => Math.hypot(x, y)))
    : (body.radius ?? 0.5);
}

// Preserve world objects and existing starts; only added vehicles need placement.
export function configureVehicleCount(template, count) {
  if (!Number.isInteger(count) || count < 1 || count > MAX_VEHICLES)
    throw new RangeError(
      `Vehicle count must be an integer from 1 to ${MAX_VEHICLES}`,
    );
  const scene = structuredClone(template);
  const resolved = resolveBodies(scene);
  const vehicles = resolved.flatMap((body, i) => (body.controlled ? [i] : []));
  if (!vehicles.length)
    throw new RangeError("This scene has no vehicles to copy");
  const mapping = new Map();
  scene.bodies = scene.bodies.filter((body, i) => {
    if (resolved[i].controlled && vehicles.indexOf(i) >= count) return false;
    mapping.set(i, mapping.size);
    return true;
  });
  if (scene.tethers)
    scene.tethers = scene.tethers
      .filter(({ a, b }) => mapping.has(a) && (b === -1 || mapping.has(b)))
      .map((tether) => ({
        ...tether,
        a: mapping.get(tether.a),
        b: tether.b === -1 ? -1 : mapping.get(tether.b),
      }));
  const occupied = resolved.filter((_, i) => mapping.has(i));
  const placements = new Map();
  for (let i = vehicles.length; i < count; i++) {
    const source = vehicles[i % vehicles.length];
    const body = structuredClone(template.bodies[source]);
    const r = radius(resolved[source]);
    let candidates = placements.get(source);
    if (!candidates) {
      candidates = [];
      const spacing = 2 * r + 0.2;
      for (let y = r + 0.1; y < scene.size[1]; y += spacing)
        for (let x = r + 0.1; x < scene.size[0]; x += spacing) {
          const point = [x, y];
          if (
            inside(point, scene.boundary) &&
            clearance(point, scene.boundary) >= r + 0.1 &&
            (scene.holes || []).every(
              (ring) =>
                !inside(point, ring) && clearance(point, ring) >= r + 0.1,
            )
          )
            candidates.push(point);
        }
      candidates.sort(
        (a, b) =>
          Math.hypot(a[0] - body.position[0], a[1] - body.position[1]) -
          Math.hypot(b[0] - body.position[0], b[1] - body.position[1]),
      );
      placements.set(source, candidates);
    }
    const position = candidates.find(([x, y]) =>
      occupied.every(
        (other) =>
          Math.hypot(x - other.position[0], y - other.position[1]) >=
          r + radius(other) + 0.1,
      ),
    );
    if (!position)
      throw new RangeError(
        "The arena has insufficient clear space for this vehicle count",
      );
    body.position = position;
    for (const tether of template.tethers || [])
      if (tether.a === source && tether.automatic)
        (scene.tethers ||= []).push({
          ...tether,
          a: scene.bodies.length,
          b: -1,
        });
    scene.bodies.push(body);
    occupied.push({ ...resolved[source], position: body.position });
  }
  return scene;
}
