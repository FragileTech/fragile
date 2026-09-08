import { frameInfo } from "./native.js";

// IOHanalyzer's documented custom CSV input. These are recorded checkpoints;
// no interpolation or extrapolation invents quality at unobserved budgets.
export function exportFixedBudgetCSV(recording) {
  const c = recording.config;
  const gasFields =
    c.algorithm === "gas"
      ? ["gas_tabu", "gas_local_search", "gas_local_evaluations"]
      : [];
  const columns = [
    "evaluations",
    "best",
    "function",
    "algorithm",
    "dimension",
    "run",
    "instance",
    "seed",
    "objective",
    "budget",
    "engine",
    "perturbation",
    "perturbation_std",
    ...gasFields,
  ];
  const rows = [columns];
  let previous = -1;
  for (const frame of recording.frames) {
    const info = frameInfo(frame);
    if (!Number.isFinite(info.best) || info.evaluations === previous) continue;
    previous = info.evaluations;
    rows.push([
      info.evaluations,
      info.best,
      c.benchmark,
      c.algorithm,
      c.dimensions,
      `${c.algorithm}-i${c.coco_instance ?? 1}-seed${c.seed}`,
      c.coco_instance ?? 1,
      c.seed,
      c.objective ?? "minimize",
      c.max_evaluations ?? 0,
      recording.engine,
      c.perturbation ?? "gaussian",
      c.perturbation === "gas_adaptive" ? "" : (c.perturbation_std ?? ""),
      ...gasFields.map((key) => c[key]),
    ]);
  }
  if (rows.length === 1)
    throw new Error("No finite objective checkpoints to export");
  const cell = (v) => `"${String(v ?? "").replaceAll('"', '""')}"`;
  return rows.map((r) => r.map(cell).join(",")).join("\n") + "\n";
}
