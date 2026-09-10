import { resolveBodies } from "./agent-types.js";

// Scene compilation validates these fields. Keep the display's pair ordering
// and scalar fallback aligned with Scene::compile, including inherited agents.
export function formationPairs(scene, controlled) {
  if (scene.task !== "tandem") return [];
  controlled ??= resolveBodies(scene).flatMap((body, i) =>
    body.controlled ? [i] : [],
  );
  const overrides = new Map(
    (scene.formation_pairs || []).map((pair) => [
      `${Math.min(pair.a, pair.b)}:${Math.max(pair.a, pair.b)}`,
      pair.distance,
    ]),
  );
  const pairs = [];
  for (let i = 0; i < controlled.length; ++i)
    for (let j = i + 1; j < controlled.length; ++j) {
      const a = Math.min(controlled[i], controlled[j]);
      const b = Math.max(controlled[i], controlled[j]);
      pairs.push({
        a,
        b,
        target: overrides.get(`${a}:${b}`) ?? scene.formation_distance ?? 3,
      });
    }
  return pairs;
}

export function formationPairScore(target, actual) {
  return target / (target + Math.abs(target - actual));
}
