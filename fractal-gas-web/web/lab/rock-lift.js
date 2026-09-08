import { flightMode, resolveBodies } from "./agent-types.js";
import { rockOptions } from "./rock-scene.js";

// Static, upward-facing thrust budget for one rock. This is not a trajectory
// predictor: drag, local gravity wells, rope angle and collisions still matter.
export function rockLiftBudget(scene, weight = rockOptions(scene)?.weight) {
  const options = rockOptions(scene);
  const gravity = scene.environment?.downward_gravity ?? 9.81;
  if (!options || !flightMode(scene) || gravity <= 0) return null;
  const bodies = resolveBodies(scene);
  const haulers = [
    ...new Set(
      (scene.tethers || [])
        .filter(
          (t) =>
            (t.stiffness ?? 25) > 0 &&
            (t.break_force ?? 500) > 0 &&
            (t.automatic || bodies[t.b]?.cargo),
        )
        .map((t) => t.a),
    ),
  ]
    .map((i) => bodies[i])
    .filter((b) => b?.controlled);
  if (!haulers.length)
    return {
      unavailable: "Enable a tow hook with positive stiffness to lift a rock.",
    };
  const thrusts = haulers.map((body) => {
    const actuator = body.actuator || {};
    const multipliers = actuator.action_multipliers || {};
    if ((actuator.kind || "vector") === "vector")
      return (body.thrust ?? 12) * (multipliers.thrust ?? 1);
    if (actuator.kind === "holonomic")
      return (
        (body.thrust ?? 12) *
        Math.max(multipliers.force_x ?? 1, multipliers.force_y ?? 1)
      );
    return null;
  });
  if (thrusts.includes(null))
    return {
      unavailable:
        "Lift estimates are available for rocket and drone tow hooks.",
    };
  const baseMass = Math.max(
    ...bodies.filter((b) => b.cargo).map((b) => (b.mass ?? 1) / options.weight),
  );
  const thrust = thrusts.reduce((a, b) => a + b, 0);
  const hookMass = scene.hook_mass ?? 0.25;
  const vehicleMass = haulers.reduce(
    (sum, b) => sum + (b.mass ?? 1) + hookMass,
    0,
  );
  const load = (vehicleMass + baseMass * weight) * gravity;
  // Leave 20% of thrust in reserve, and size for the weakest solo hauler.
  // Round down onto the logarithmic weight slider's 0.01 tick grid.
  const soloMass = Math.min(
    ...haulers.map(
      (b, i) => (0.8 * thrusts[i]) / gravity - (b.mass ?? 1) - hookMass,
    ),
  );
  const limit = Math.min(10, soloMass / baseMass);
  const suggestedWeight =
    limit >= 0.01 ? 10 ** (Math.floor(Math.log10(limit) * 100) / 100) : null;
  return { thrust, load, canLift: thrust > load, suggestedWeight };
}
