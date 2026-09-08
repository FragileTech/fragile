import {
  resolveAgentTypes,
  resolveBodies,
  VEHICLE_TYPES,
} from "./agent-types.js";
import { inside, clearance, antsOptionsFromScene } from "./ants-scene.js";

export const MAX_VEHICLES = 128;
export const vehicleCount = (scene) =>
  resolveBodies(scene).filter((body) => body.controlled).length;

// Recognize tuned variants such as racing_kart without rewriting their defaults.
export function vehicleType(scene) {
  const vehicles = resolveBodies(scene).filter((body) => body.controlled);
  const type = vehicles[0]?.visual?.model;
  const kind = {
    rocket: "vector",
    drone: "holonomic",
    kart: "kart",
    harvester: "kart",
  }[type];
  if (
    VEHICLE_TYPES.includes(type) &&
    vehicles.every(
      (body) =>
        body.visual?.model === type &&
        (body.actuator?.kind || "vector") === kind,
    )
  )
    return type;
}

function updateDescription(scene) {
  const options = antsOptionsFromScene(scene);
  // Update the generated fleet prefix without overwriting an edited description
  // or its cargo/respawn instructions.
  if (options && typeof scene.description === "string") {
    const { agentType, count } = options;
    scene.description = scene.description.replace(
      /^\d+ (?:rockets?|drones?|karts?|harvesters?)\./,
      `${count} ${agentType}${count === 1 ? "" : "s"}.`,
    );
  }
  return scene;
}

// Physical defaults belong to the selected type; instance state and task roles
// survive a switch. Include engine defaults not normally present in the catalog.
const vehiclePhysics = [
  "radius",
  "vertices",
  "mass",
  "inertia",
  "drag",
  "angular_drag",
  "thrust",
  "torque",
  "restitution",
  "friction",
  "controlled",
  "flight_capable",
  "actuator",
];

export function configureVehicleType(template, agentType, catalog = {}) {
  if (!VEHICLE_TYPES.includes(agentType))
    throw new RangeError("Choose Rockets, Drones, Karts or Harvesters");
  const scene = structuredClone(template);
  const bodies = resolveBodies(scene);
  if (!bodies.some((body) => body.controlled))
    throw new RangeError("This scene has no vehicles to replace");
  // Imported scenes may omit unused types. Copy missing catalog definitions as
  // resolved data so exports remain portable, without changing existing types.
  const available = resolveAgentTypes(catalog);
  scene.agent_types ||= {};
  for (const name of VEHICLE_TYPES)
    if (!Object.hasOwn(scene.agent_types, name) && available.has(name)) {
      const { extends: parent, ...definition } = available.get(name);
      scene.agent_types[name] = structuredClone(definition);
    }
  const selected = resolveAgentTypes(scene.agent_types).get(agentType);
  if (!selected?.physics?.controlled)
    throw new RangeError(`Missing controlled vehicle type: ${agentType}`);
  scene.bodies = scene.bodies.map((body, i) => {
    if (!bodies[i].controlled) return body;
    const next = { ...bodies[i] };
    for (const key of vehiclePhysics) delete next[key];
    delete next.visual;
    next.agent_type = agentType;
    return next;
  });
  return updateDescription(scene);
}

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
  return updateDescription(scene);
}
