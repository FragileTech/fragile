import { VEHICLE_TYPES } from "./agent-types.js";

export const DEFAULT_ANTS_OPTIONS = { agentType: "harvester", count: 5 };
export const MAX_ANTS_VEHICLES = 128;

export function inside(point, ring) {
  let result = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [x, y] = ring[i],
      [px, py] = ring[j];
    if (
      y > point[1] !== py > point[1] &&
      point[0] < ((px - x) * (point[1] - y)) / (py - y) + x
    )
      result = !result;
  }
  return result;
}

export function clearance(point, ring) {
  return Math.min(
    ...ring.map((a, i) => {
      const b = ring[(i + 1) % ring.length];
      const dx = b[0] - a[0],
        dy = b[1] - a[1];
      const t = Math.max(
        0,
        Math.min(
          1,
          ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) /
            (dx * dx + dy * dy),
        ),
      );
      return Math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy);
    }),
  );
}

// Build only from the preset template; saved scenes already contain concrete bodies.
export function configureAntsScene(template, { agentType, count }) {
  if (!VEHICLE_TYPES.includes(agentType))
    throw new RangeError("Choose Rockets, Drones, Karts or Harvesters");
  if (!Number.isInteger(count) || count < 1 || count > MAX_ANTS_VEHICLES)
    throw new RangeError(
      `Vehicle count must be an integer from 1 to ${MAX_ANTS_VEHICLES}`,
    );
  const scene = structuredClone(template);
  const boundary = scene.boundary,
    holes = scene.holes || [];
  const candidates = [];
  for (let y = 4; y <= scene.size[1] - 4; y += 4)
    for (let x = 4; x <= scene.size[0] - 4; x += 4) {
      const point = [x, y];
      if (
        inside(point, boundary) &&
        clearance(point, boundary) >= 1.5 &&
        holes.every(
          (ring) => !inside(point, ring) && clearance(point, ring) >= 1.5,
        )
      )
        candidates.push(point);
    }
  if (candidates.length < count)
    throw new RangeError(
      "The arena has insufficient clear space for this vehicle count",
    );
  scene.bodies = Array.from({ length: count }, (_, i) => ({
    agent_type: agentType,
    position: candidates[Math.floor(((i + 0.5) * candidates.length) / count)],
    angle: (i % 6) - 3,
  }));
  const label = count === 1 ? agentType : `${agentType}s`;
  scene.description = `${count} ${label}. Fill 5-drop tanks and unload at the refinery over 2 simulation seconds. Drops return after 3 seconds.`;
  scene.respawn_seconds = 3;
  return scene;
}

export function antsOptionsFromScene(scene) {
  if (scene.name !== "Ants & drops" || scene.task !== "forage") return;
  const count = scene.bodies?.length,
    agentType = scene.bodies?.[0]?.agent_type;
  if (
    count >= 1 &&
    count <= MAX_ANTS_VEHICLES &&
    VEHICLE_TYPES.includes(agentType) &&
    scene.bodies.every((body) => body.agent_type === agentType)
  )
    return { agentType, count };
}
