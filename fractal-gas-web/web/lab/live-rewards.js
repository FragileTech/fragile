import { NativeEngine } from "./native.js";
import {
  coefficientValues,
  rewardValues,
  withRewards,
} from "./reward-settings.js";

// Build replacements off to the side: the live world remains usable on failure.
export function prepareRewardEngines(
  engine,
  scene,
  values,
  coefficients,
  rows,
  root,
) {
  const normalized = withRewards(scene, rewardValues(values));
  // A historical snapshot may use omitted defaults; preserve its exact scene hash.
  const nextScene = root ? structuredClone(values) : normalized;
  const nextCoefficients = coefficientValues(coefficients);
  let next, predict;
  try {
    next = new NativeEngine(engine.m, nextScene);
    predict = new NativeEngine(engine.m, nextScene);
    if (next.info.some((v, i) => v !== engine.info[i]))
      throw new Error("Reward update changed the world layout");
    if (root) next.restore(root); // Keep native fingerprint validation.
    next.restoreRows(rows || engine.states());
    predict.restore(next.snapshot());
    return {
      engine: next,
      predict,
      scene: nextScene,
      coefficients: nextCoefficients,
    };
  } catch (error) {
    next?.dispose();
    predict?.dispose();
    throw error;
  }
}
