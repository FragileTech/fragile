// The objective catalog comes from the Rust engine (`benchmark_catalog()`): ids,
// names, groups, domains, dimension rules and parameter fields. Nothing about a
// benchmark is restated here.
export const benchmarkId = (benchmark) =>
  typeof benchmark === "string" ? benchmark : benchmark?.id;
export function catalogEntry(catalog, benchmark) {
  const id = benchmarkId(benchmark);
  return catalog?.benchmarks?.find((entry) => entry.id === id) || null;
}
// Current parameter values of a configured benchmark, defaults filled in.
export function benchmarkParameters(entry, benchmark) {
  const given = typeof benchmark === "object" && benchmark ? benchmark : {};
  return Object.fromEntries(
    (entry?.parameterFields || []).map((field) => [
      field.key,
      given[field.key] ?? field.default,
    ]),
  );
}
// The engine's encoding: a bare id, or {id, ...parameters} when it has parameters.
export function benchmarkValue(entry, parameters = {}) {
  const fields = entry.parameterFields || [];
  if (!fields.length) return entry.id;
  const value = { id: entry.id };
  for (const field of fields) {
    const n = Number(parameters[field.key] ?? field.default);
    if (
      !Number.isFinite(n) ||
      n < field.min ||
      n > field.max ||
      (field.kind === "integer" && !Number.isInteger(n))
    )
      throw new Error(
        `${field.label} must be ${field.kind === "integer" ? "an integer" : "a number"} in ${field.min}–${field.max}.`,
      );
    value[field.key] = n;
  }
  return value;
}
// Dimension the benchmark admits: fixed (2D functions), 3 × atoms, the nearest
// COCO dimension, or the requested one above the benchmark's minimum.
export function dimensionFor(entry, parameters, requested) {
  if (entry.dimension) return entry.dimension;
  if (entry.dimensionRule)
    return 3 * Number(parameters?.n_atoms ?? entry.parameters.n_atoms);
  if (entry.dimensions)
    return entry.dimensions.reduce((best, d) =>
      Math.abs(d - requested) < Math.abs(best - requested) ? d : best,
    );
  return Math.max(requested, entry.minDimension || 1);
}
export const dimensionLocked = (entry) =>
  !!(entry?.dimension || entry?.dimensionRule);
// Display facts about the configured objective. `objective` is the engine's
// resolved info for the run (instance minimum, molecule view); renderers and
// view code take bounds and the known minimum from here.
export function benchmarkInfo(config, catalog, objective) {
  const entry = catalogEntry(catalog, config.benchmark);
  const bounds = objective?.bounds || entry?.bounds || [-1, 1];
  return {
    id: benchmarkId(config.benchmark),
    low: bounds[0],
    high: bounds[1],
    minimum: objective ? objective.minimum : (entry?.minimum ?? null),
    minimizer: objective?.minimizer ?? null,
    label:
      objective?.name || entry?.name || String(benchmarkId(config.benchmark)),
    molecule: !!(objective?.molecule ?? entry?.molecule),
    stochastic: !!(objective?.stochastic ?? entry?.stochastic),
    objectiveExecution:
      objective?.objective_execution || entry?.objective_execution || "graph",
    gradientExecution:
      objective?.gradient_execution || entry?.gradient_execution || "graph",
    problemId: objective?.coco_problem_id || null,
  };
}
function checkDimension(entry, config) {
  const d = config.dimensions;
  if (!Number.isInteger(d) || d < 2 || d > 128)
    throw new Error(
      "The lab supports 2–128 dimensions; use the Rust API for other dimensions.",
    );
  const expected = dimensionFor(
    entry,
    benchmarkParameters(entry, config.benchmark),
    d,
  );
  if (expected !== d)
    throw new Error(
      entry.dimensions
        ? `${entry.name} is defined in ${entry.dimensions.join(", ")} dimensions.`
        : `${entry.name} requires ${expected} dimensions.`,
    );
}
export function validateEliteCount(count, walkers) {
  if (!Number.isInteger(count) || count < 0 || count > walkers)
    throw new Error(
      "Elite walkers must be an integer between 0 and the walker count.",
    );
  return count;
}
export function validateLabConfig(config, catalog) {
  const entry = config && catalogEntry(catalog, config.benchmark);
  if (!entry) throw new Error("Choose a benchmark supported by this lab.");
  checkDimension(entry, config);
  if (
    !Number.isInteger(config.walkers) ||
    config.walkers < 1 ||
    config.walkers > 16384
  )
    throw new Error("The lab supports 1–16384 walkers.");
  if (!config.gas || !["cpu", "wgpu"].includes(config.gas.backend))
    throw new Error("Select WASM CPU or WebGPU for this lab.");
  if (
    !Number.isInteger(config.gas.seed) ||
    config.gas.seed < 0 ||
    config.gas.seed > 4294967295
  )
    throw new Error("The lab seed must be an unsigned 32-bit integer.");
  validateEliteCount(config.gas.n_elite ?? 0, config.walkers);
  return config;
}
export function resolveConfig(base, values, catalog) {
  const config = structuredClone(base);
  const number = (name) => {
    const n = Number(values[name]);
    if (!Number.isFinite(n)) throw new Error(`Invalid ${name}`);
    return n;
  };
  const entry = catalogEntry(catalog, values.benchmark);
  if (!entry) throw new Error("Choose a benchmark supported by this lab.");
  config.benchmark = benchmarkValue(entry, values.parameters);
  config.dimensions = number("dimensions");
  config.walkers = number("walkers");
  const d = config.dimensions;
  checkDimension(entry, config);
  if (
    !Number.isInteger(config.walkers) ||
    config.walkers < 1 ||
    config.walkers > 16384
  )
    throw new Error("Use 1–16384 walkers in the browser lab.");
  // The closed-form physics metric and an independent potential are defined for
  // the lecture objectives only; they do not carry over to another benchmark.
  if (benchmarkId(base.benchmark) !== entry.id) {
    if (!entry.physics_metric) config.physics_metric = null;
    config.potential = null;
    config.reward_shift = [];
  }
  const [low, high] = entry.bounds;
  if (values["initial-box"] === "domain") {
    config.initial_lower = low;
    config.initial_upper = high;
  } else if (values["initial-box"] === "unit") {
    config.initial_lower = Math.max(low, -1);
    config.initial_upper = Math.min(high, 1);
  }
  const gas = config.gas;
  gas.n_elite = validateEliteCount(
    Number(values["n-elite"] ?? gas.n_elite ?? 0),
    config.walkers,
  );
  gas.seed = number("seed");
  gas.backend = values.backend;
  gas.precision = values.precision;
  if (!Number.isInteger(gas.seed) || gas.seed < 0 || gas.seed > 4294967295)
    throw new Error("The lab seed must be an unsigned 32-bit integer.");
  if (gas.backend === "wgpu" && gas.precision !== "f32")
    throw new Error("WebGPU requires f32. Select WASM CPU for f64.");
  const domain = { lower: Array(d).fill(low), upper: Array(d).fill(high) };
  gas.boundary =
    values.boundary === "unbounded"
      ? { kind: "unbounded" }
      : { kind: values.boundary, field: "positions", domain };
  const periodic =
    values.boundary === "periodic_box" ? structuredClone(domain) : null;
  let distance;
  if (values.distance === "cosine") {
    if (periodic)
      throw new Error(
        "Cosine does not define a periodic box distance. Select Euclidean or an unbounded domain.",
      );
    distance = { kind: "cosine", field: "positions", zero_tolerance: 0 };
  } else if (values.distance === "phase_space") {
    if (values.kinetic !== "baoab")
      throw new Error("The phase-space preset requires Langevin velocities.");
    distance = {
      kind: "phase_space",
      positions: "positions",
      velocities: "velocities",
      position_scale: 1,
      velocity_scale: 1,
      lambda: 1,
      periodic,
    };
  } else
    distance = {
      kind: "euclidean",
      field: "positions",
      scales: [],
      squared: false,
      periodic,
    };
  const module = (previous, law, count) => {
    const weighted = law === "gaussian" || law === "gaussian_greedy";
    const kernel = weighted
      ? values.distance === "cosine"
        ? { kind: "exponential", temperature: number("kernel-width") }
        : { kind: "gaussian", width: number("kernel-width") }
      : { kind: "uniform" };
    return {
      ...previous,
      distance: structuredClone(distance),
      kernel,
      law: ["gaussian", "uniform"].includes(law) ? "independent" : law,
      count,
      history_window: ["gaussian", "uniform"].includes(law)
        ? previous.history_window
        : 0,
    };
  };
  gas.distance_donors = module(
    gas.distance_donors,
    values["distance-law"],
    number("distance-count"),
  );
  gas.cloning_donors = module(gas.cloning_donors, values["clone-law"], 1);
  gas.reducer = { kind: values.reducer };
  const standardizer =
    values.standardizer === "local"
      ? {
          kind: "local",
          include_self: false,
          sigma_min: number("sigma-min"),
          distance,
          kernel: gas.distance_donors.kernel,
        }
      : { kind: "global", sigma_min: number("sigma-min") };
  const map =
    values["positive-map"] === "logistic"
      ? { kind: "logistic", amplitude: 2, floor: 1e-6 }
      : { kind: "legacy_asymmetric", floor: 1e-6 };
  Object.assign(gas.fitness, {
    direction: values.direction,
    reward_exponent: number("alpha"),
    diversity_exponent: number("beta"),
    reward_map: map,
    diversity_map: structuredClone(map),
    reward_standardizer: standardizer,
    diversity_standardizer: structuredClone(standardizer),
  });
  gas.kinetic.integrator =
    values.kinetic === "baoab"
      ? {
          kind: "baoab",
          positions: "positions",
          velocities: "velocities",
          dt: number("dt"),
          friction: number("friction"),
        }
      : {
          kind: values.kinetic,
          field: "positions",
          amplitude: number("amplitude"),
          ...(values.kinetic === "brownian" ? { dt: number("dt") } : {}),
        };
  const scale = number("noise-scale");
  let geometry;
  if (values["noise-geometry"] === "full") {
    if (d !== 2)
      throw new Error(
        "The lab full-factor preset is 2D. Import a configuration to use a different full matrix.",
      );
    geometry = {
      kind: "full",
      factor: { kind: "constant", values: [scale, 0, scale / 2, 1.5 * scale] },
    };
  } else if (values["noise-geometry"] === "diagonal")
    geometry = {
      kind: "diagonal",
      factor: {
        kind: "constant",
        values: Array.from({ length: d }, (_, i) => scale * (1 + i / d)),
      },
    };
  else if (values["noise-geometry"] === "low_rank")
    geometry = {
      kind: "low_rank",
      rank: 1,
      factor: { kind: "constant", values: Array(d).fill(scale) },
    };
  else
    geometry = {
      kind: "isotropic",
      scale: { kind: "constant", values: [scale] },
    };
  gas.kinetic.noise = { innovation: values.innovation, geometry };
  return config;
}
export function alive(state, includeTruncated = false) {
  return (
    !state.invalid &&
    !state.out_of_bounds &&
    !state.terminated &&
    (!state.truncated || includeTruncated)
  );
}
export function frameMetrics(frame, config) {
  const values = Array.from(frame.population.rewards.raw).filter(
    (v, i) =>
      Number.isFinite(v) &&
      alive(frame.population.validity[i], config.gas.include_truncated),
  );
  const best = values.length
    ? values.reduce((a, b) =>
        config.gas.fitness.direction === "minimize"
          ? Math.min(a, b)
          : Math.max(a, b),
      )
    : null;
  const mean = values.length
    ? values.reduce((a, b) => a + b, 0) / values.length
    : null;
  return {
    step: frame.step,
    evaluations: frame.reward_evaluations,
    best,
    mean,
    alive: values.length,
  };
}
