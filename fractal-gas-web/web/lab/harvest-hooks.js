import { resolveBodies } from "./agent-types.js";

// Mirror native hook expansion for presentation only. Scene exports keep the
// original body/tether indices; recorded layouts select legacy presentation.
export function harvestPresentation(scene, info) {
  if (scene.task !== "harvest" || info[1] === scene.bodies.length) return scene;
  const next = structuredClone(scene);
  next.bodies = resolveBodies(scene);
  const cables = [];
  next.tethers = (scene.tethers || []).map((t) => {
    const vehicle = next.bodies[t.a];
    if (!vehicle.controlled) return { ...t };
    const hook = next.bodies.length;
    const radius = vehicle.vertices?.length
      ? Math.max(...vehicle.vertices.map(([x, y]) => Math.hypot(x, y)))
      : (vehicle.radius ?? 0.5);
    next.bodies.push({
      position: vehicle.position,
      radius: 0.2,
      mass: scene.hook_mass ?? 0.25,
      hook: true,
    });
    cables.push({
      ...t,
      b: hook,
      automatic: false,
      permanent: true,
      anchor_a: [-radius, 0],
    });
    return { ...t, a: hook, owner: t.a };
  });
  next.tethers.push(...cables);
  return next;
}

export function withHookMass(scene, mass) {
  if (!Number.isFinite(mass) || mass < 0.01 || mass > 100)
    throw new RangeError("Hook mass must be between 0.01 and 100");
  return { ...structuredClone(scene), hook_mass: mass };
}
