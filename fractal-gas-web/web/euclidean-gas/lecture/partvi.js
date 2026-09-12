import { metadata } from "./partvi-metadata.js";
const liveIds = new Set([
  2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 21, 32, 34, 35, 36, 39, 48,
  51, 52,
]);
const continuationIds = new Set([19, 22, 45]);
const select = (key, label, value, options) => ({
  key,
  label,
  type: "select",
  value,
  options: options.map((value) => ({ value, label: String(value) })),
});
function snapshot(result, tick, params) {
  const matrices = Object.entries(result.details)
    .filter(
      ([key, a]) =>
        /gram|kernel|transition|covariance|hessian|ricci|choi/.test(key) &&
        Array.isArray(a) &&
        a.length >= 4 &&
        a.length <= 256 &&
        Number.isInteger(Math.sqrt(a.length)) &&
        a.every(Number.isFinite),
    )
    .slice(0, 3)
    .map(([key, a]) => {
      const n = Math.sqrt(a.length);
      return {
        title: key.replaceAll("_", " "),
        matrix: Array.from({ length: n }, (_, i) =>
          a.slice(i * n, (i + 1) * n),
        ),
        xLabel: "Column",
        yLabel: "Row",
      };
    });
  return {
    step: tick,
    time: tick,
    charts: [
      ...result.plots.map((p) => ({
        title: p.title,
        xLabel: p.x_label,
        yLabel: p.y_label,
        series: p.series.map((s) => ({
          name: s.name,
          points: s.points,
          style:
            s.kind === "scatter"
              ? "scatter"
              : s.kind === "bars"
                ? "bars"
                : "line",
        })),
      })),
      ...matrices,
    ],
    metrics: result.metrics.map((m) => ({
      ...m,
      value: m.value === null ? "Unavailable" : m.value,
    })),
    message: [
      result.details.calculation_origin?.replaceAll("_", " "),
      result.model,
      ...result.notes,
    ]
      .filter(Boolean)
      .join(" · "),
    table: {
      columns: ["Calculation", "Value"],
      rows: Object.entries(result.details)
        .filter(
          ([k, v]) =>
            ![
              "request",
              "archive_steps",
              "schema_version",
              "precision",
            ].includes(k) &&
            v !== null &&
            typeof v !== "object",
        )
        .map(([k, v]) => [k, String(v)]),
    },
    result,
    done: tick >= 8,
  };
}
async function gasConfig(engine, seed, params, experiment) {
  const c = await engine.defaults();
  c.walkers = params.walkers || 32;
  c.dimensions = 3;
  c.benchmark = "quadratic";
  c.initial_lower = -1;
  c.initial_upper = 1;
  c.gas.seed = seed;
  c.gas.backend = "cpu";
  c.gas.precision = "f64";
  const boundary = params.engine_boundary || "unbounded";
  c.gas.boundary =
    boundary === "unbounded"
      ? { kind: boundary }
      : {
          kind: boundary,
          field: "positions",
          domain: { lower: [-4, -4, -4], upper: [4, 4, 4] },
        };
  c.gas.qft = {
    viscosity: {
      coefficient: Number(params.engine_viscosity ?? 1),
      bandwidth: 1,
      row_normalized: false,
    },
    innovation_shifts: [],
  };
  c.gas.kinetic = {
    integrator: {
      kind: "baoab",
      positions: "positions",
      velocities: "velocities",
      dt: Number(params.engine_dt ?? 0.04),
      friction: Number(params.engine_friction ?? 1),
    },
    noise: {
      innovation: params.engine_innovation || "gaussian",
      geometry: {
        kind: "isotropic",
        scale: { kind: "constant", values: [Math.sqrt(0.8)] },
      },
    },
  };
  for (const role of ["distance_donors", "cloning_donors"]) {
    c.gas[role].history_window = Number(params.engine_memory ?? 2);
    c.gas[role].kernel = { kind: "uniform" };
    c.gas[role].law = "independent";
    if (boundary === "periodic_box")
      c.gas[role].distance.periodic = c.gas.boundary.domain;
  }
  if ([39, 45, 48].includes(experiment)) {
    c.physics_metric = {
      epsilon: 1,
      temperature: 0.4,
      policy: "clipped",
      curvature: true,
      clipping_threshold: 1e-8,
    };
    c.gas.fitness.reward_standardizer = { kind: "global", sigma_min: 0.1 };
    c.gas.fitness.diversity_standardizer = { kind: "global", sigma_min: 0.1 };
    c.gas.fitness.distance_floor = 0.15;
  }
  return c;
}
async function createGas(engine, seed, params, experiment) {
  const gas = await engine.create(
    await gasConfig(engine, seed, params, experiment),
  );
  gas.start_recording(
    JSON.stringify({ max_steps: 128, max_bytes: 128 * 1024 * 1024 }),
  );
  const warmup = [3, 4, 5, 6, 12, 13, 32, 36].includes(experiment)
    ? 96
    : Math.max(4, Number(params.engine_memory ?? 2) + 1);
  for (let remaining = warmup; remaining > 0; remaining -= 16)
    await gas.step(Math.min(16, remaining));
  return gas;
}
export async function createModel(
  meta,
  { params, seed, engine, archive: imported },
) {
  let tick = 0,
    gas = null,
    result;
  if (liveIds.has(meta.experiment) && params.source === "recorded" && !imported)
    gas = await createGas(engine, seed, params, meta.experiment);
  const request = () => ({
    experiment: meta.experiment,
    parameters: {
      ...params,
      ...(meta.experiment === 45 && params.source !== "recorded"
        ? { replicas: params.reference_replicas }
        : {}),
      seed: (seed + tick) >>> 0,
    },
  });
  async function calculate() {
    const req = request();
    if (continuationIds.has(meta.experiment) && params.source === "recorded")
      result = await engine.qftRun(
        req,
        await gasConfig(engine, (seed + tick) >>> 0, params, meta.experiment),
      );
    else if (gas) result = await gas.qft(JSON.stringify(req));
    else result = await engine.qft(req, imported);
    if (
      meta.experiment === 39 &&
      params.backend &&
      params.backend !== "host_f64" &&
      result.details.fitness_jet
    ) {
      const request = {
        jets: [result.details.fitness_jet],
        epsilon: result.details.epsilon,
        policy: result.details.policy,
        threshold: result.details.clipping_threshold ?? 1e-8,
        backend: params.backend === "webgpu_f32" ? "wgpu" : "cpu",
        precision: params.backend.endsWith("f32") ? "f32" : "f64",
      };
      try {
        const batch = await engine.curvatureBatch(request);
        result.details.backend_calculation = batch;
        const measured = batch.curvatures[0].scalar;
        result.metrics.push({
          label: "Selected backend scalar curvature",
          value: measured,
          unit: params.backend,
        });
        result.notes.push(
          "Selected batch backend: " +
            batch.backend +
            " " +
            batch.precision +
            "; host eigensolver " +
            batch.host_eigensolver +
            ".",
        );
      } catch (error) {
        result.details.backend_calculation = {
          status: "unavailable",
          backend: params.backend,
          reason: error.message || String(error),
        };
        result.notes.push(
          "Requested backend calculation unavailable: " +
            (error.message || String(error)),
        );
      }
    }
  }
  try {
    await calculate();
  } catch (error) {
    gas?.free();
    throw error;
  }
  return {
    async step() {
      if (tick >= 8) return;
      if (gas) await gas.step(4);
      tick++;
      await calculate();
    },
    snapshot: () => {
      const view = snapshot(result, tick, params);
      if (imported) view.imported = true;
      if (gas) {
        const frame = gas.snapshot(),
          p = frame.population,
          positions = p.observations.fields.positions.values;
        view.scene = {
          title: "Recorded three-dimensional population",
          coordinateSystem: "spatial",
          message:
            "Coordinates and recipient generations come from the executed run.",
          nodes: Array.from({ length: p.generations.length }, (_, i) => ({
            id: String(i),
            owner: i,
            position: positions.slice(3 * i, 3 * i + 3),
            label: "Slot " + i + " · generation " + p.generations[i],
            layer: "walkers",
            color: "#60bdcf",
            detail: "Recorded step " + frame.step,
          })),
          edges: [],
          faces: [],
        };
      }
      return view;
    },
    ...(gas
      ? { archive: () => gas.archive(), checkpoint: () => gas.checkpoint() }
      : imported
        ? { archive: () => imported }
        : {}),
    dispose() {
      gas?.free();
      gas = null;
    },
  };
}
export async function importedQftModel(payload, engine) {
  const meta = metadata.find((d) => d.id === payload.id);
  if (!meta) throw new Error("Unknown QFT workbench");
  return createModel(meta, {
    params: payload.params,
    seed: payload.seed,
    engine,
    archive: payload.archive,
  });
}
export function importedResultsModel(payload) {
  const rows = payload.results;
  if (
    payload.schema !== "fragile-partvi-results-v1" ||
    !Array.isArray(rows) ||
    !rows.length ||
    rows.length > 4096
  )
    throw new Error("Invalid Part VI results");
  for (const r of rows) {
    if (
      !Number.isInteger(r.experiment) ||
      r.experiment < 1 ||
      r.experiment > 66 ||
      !Array.isArray(r.plots) ||
      !Array.isArray(r.metrics) ||
      !Array.isArray(r.notes) ||
      !r.notes.every((n) => typeof n === "string") ||
      typeof r.title !== "string" ||
      typeof r.model !== "string" ||
      !r.details ||
      Array.isArray(r.details) ||
      typeof r.details !== "object" ||
      !r.metrics.every(
        (m) =>
          typeof m.label === "string" &&
          (m.value === null || Number.isFinite(m.value)),
      )
    )
      throw new Error("Invalid native result");
    for (const p of r.plots)
      for (const s of p.series)
        for (const point of s.points)
          if (point.length !== 2 || !point.every(Number.isFinite))
            throw new Error("Nonfinite native result");
  }
  let cursor = 0;
  return {
    async step() {
      cursor = Math.min(cursor + 1, rows.length - 1);
    },
    snapshot() {
      return {
        ...snapshot(rows[cursor], cursor, {}),
        imported: true,
        done: cursor === rows.length - 1,
      };
    },
    dispose() {},
  };
}
export const demos = metadata.map((meta) => ({
  ...meta,
  kind: "Rust/WASM · " + meta.validation,
  controls: [
    ...(meta.experiment === 45
      ? [
          select("readout", "Metric observable", "material", [
            "material",
            "fixed_probe",
          ]),
        ]
      : []),
    ...(meta.experiment === 39
      ? [
          select("backend", "Curvature calculation", "host_f64", [
            "host_f64",
            "cpu_f64",
            "cpu_f32",
            "webgpu_f32",
          ]),
        ]
      : []),
    ...(liveIds.has(meta.experiment) || continuationIds.has(meta.experiment)
      ? [
          select("source", "Data source", "recorded", [
            "reference",
            "recorded",
          ]),
          select("walkers", "Recorded walkers", 32, [16, 32, 64]),
          select("engine_memory", "Engine donor history window", 2, [0, 2, 8]),
          select("engine_dt", "Engine time step", 0.04, [0.01, 0.04, 0.1]),
          select("engine_friction", "Engine friction", 1, [0, 0.5, 1, 2]),
          select(
            "engine_viscosity",
            "Engine viscous coefficient",
            1,
            [0, 0.5, 1, 2],
          ),
          select("engine_boundary", "Engine boundary (box ±4)", "unbounded", [
            "unbounded",
            "absorbing_box",
            ...([39, 45, 48].includes(meta.experiment) ? [] : ["periodic_box"]),
          ]),
          ...(meta.experiment === 19
            ? []
            : [
                select(
                  "engine_innovation",
                  "Engine innovation law",
                  "gaussian",
                  ["gaussian", "standardized_uniform"],
                ),
              ]),
        ]
      : []),
    ...(continuationIds.has(meta.experiment)
      ? [
          select(
            "replicas",
            "Engine replicas per independent group",
            16,
            meta.experiment === 45 ? [8, 16, 32] : [8, 16, 32, 64],
          ),
          select(
            "horizon",
            "Engine continuation steps",
            meta.experiment === 45 ? 1 : 4,
            meta.experiment === 45 ? [1, 2, 4] : [1, 2, 4, 8],
          ),
          ...(meta.experiment === 19
            ? [
                {
                  key: "theta",
                  label: "Engine source shift",
                  type: "range",
                  value: 0.25,
                  min: 0.01,
                  max: 1.5,
                  step: 0.01,
                },
              ]
            : []),
        ]
      : []),
    ...meta.controls.map((c) =>
      meta.experiment === 45 && c.key === "replicas"
        ? {
            ...c,
            key: "reference_replicas",
            label: "Reference ensemble samples",
          }
        : c,
    ),
  ],
  create: (ctx) => createModel(meta, ctx),
}));
