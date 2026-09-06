// Presentation metadata is independent of agents, physics tasks and planners.
// New scenes can select metric readouts without application-level task branches.
const metrics = new Map([
  ["reward", (m) => m[0]],
  ["deliveries", (m) => m[5]],
  ["pickups", (m) => m[6]],
  ["gates", (m) => m[7]],
]);
export function registerSceneMetric(id, read) {
  if (!id || metrics.has(id) || typeof read !== "function")
    throw new Error(`Invalid or duplicate scene metric: ${id}`);
  metrics.set(id, read);
}
const defaults = {
  forage: {
    task_label: "Swarm foraging",
    score: { metric: "pickups", label: "Food collected" },
  },
  tandem: {
    task_label: "Formation flight",
    score: { metric: "gates", label: "Gates crossed" },
  },
  navigation: {
    task_label: "Vehicle test range",
    score: { metric: "gates", label: "Gates crossed" },
  },
  harvest: {
    task_label: "Ore recovery",
    score: { metric: "deliveries", label: "Cargo deliveries" },
  },
};
export function scenePresentation(scene) {
  return {
    ...(defaults[scene.task] || defaults.navigation),
    ...scene.presentation,
  };
}
export function sceneReadout(scene, values) {
  const p = scenePresentation(scene),
    read = (spec) => {
      const value = metrics.get(spec.metric)?.(values);
      return Number.isFinite(value) ? value : 0;
    };
  const divisor = p.score.divisor ?? 1;
  if (!Number.isFinite(divisor) || divisor <= 0)
    throw new Error("Invalid score divisor");
  let label = p.score.label;
  if (p.progress) {
    const cycle = p.progress.cycle;
    if (!Number.isInteger(cycle) || cycle < 1)
      throw new Error("Invalid progress cycle");
    label += ` · ${p.progress.label} ${(read(p.progress) % cycle) + 1}/${cycle}`;
  }
  return { score: Math.floor(read(p.score) / divisor), label };
}
