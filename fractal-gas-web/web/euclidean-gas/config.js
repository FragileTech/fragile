export const BENCHMARKS = {
  sphere: { label: "Sphere", bounds: [-1000, 1000] },
  rastrigin: { label: "Rastrigin", bounds: [-5.12, 5.12] },
  rosenbrock: { label: "Rosenbrock", bounds: [-10, 10] },
  styblinski_tang: { label: "Styblinski–Tang", bounds: [-5, 5] },
};
export function validateLabConfig(config) {
  if (!config || !Object.hasOwn(BENCHMARKS, config.benchmark))
    throw new Error("Choose a benchmark supported by this lab.");
  if (
    !Number.isInteger(config.dimensions) ||
    config.dimensions < 2 ||
    config.dimensions > 128
  )
    throw new Error(
      "The lab supports 2–128 dimensions; use the Rust API for other dimensions.",
    );
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
  return config;
}
export function resolveConfig(base, values) {
  const config = structuredClone(base);
  const number = (name) => {
    const n = Number(values[name]);
    if (!Number.isFinite(n)) throw new Error(`Invalid ${name}`);
    return n;
  };
  config.benchmark = values.benchmark;
  config.dimensions = number("dimensions");
  config.walkers = number("walkers");
  const d = config.dimensions;
  if (!Number.isInteger(d) || d < 2 || d > 128)
    throw new Error(
      "The lab supports 2–128 dimensions; the Rust API also supports 1D.",
    );
  if (
    !Number.isInteger(config.walkers) ||
    config.walkers < 1 ||
    config.walkers > 16384
  )
    throw new Error("Use 1–16384 walkers in the browser lab.");
  const gas = config.gas;
  gas.seed = number("seed");
  gas.backend = values.backend;
  gas.precision = values.precision;
  if (!Number.isInteger(gas.seed) || gas.seed < 0 || gas.seed > 4294967295)
    throw new Error("The lab seed must be an unsigned 32-bit integer.");
  if (gas.backend === "wgpu" && gas.precision !== "f32")
    throw new Error("WebGPU requires f32. Select WASM CPU for f64.");
  const domain = {
    lower: Array(d).fill(BENCHMARKS[config.benchmark].bounds[0]),
    upper: Array(d).fill(BENCHMARKS[config.benchmark].bounds[1]),
  };
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
