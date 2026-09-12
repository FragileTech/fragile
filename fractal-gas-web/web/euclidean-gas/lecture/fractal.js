import { metadata } from "./partv-metadata.js";
import { positions, line, scatter, covariance2 } from "./math.js";
const range = (key, label, value, min, max, step = 0.1) => ({
  key,
  label,
  type: "range",
  value,
  min,
  max,
  step,
});
const select = (key, label, value, options) => ({
  key,
  label,
  type: "select",
  value,
  options: options.map((v) => ({ value: v, label: String(v) })),
});
const m = (label, value, unit = "") => ({ label, value, unit });
const plot = (title, xLabel, yLabel, series, extra = {}) => ({
  title,
  xLabel,
  yLabel,
  series,
  ...extra,
});
const identity = [
    [1, 0],
    [0, 1],
  ],
  bounds = [-3, 3, -3, 3];
const eventKey = (e) =>
  [e.epoch, e.step, e.slot, e.generation, e.version].join(":");
function field(p, name = "positions") {
  const f = p.observations?.fields?.[name] || p.fields?.[name];
  if (!f) return [];
  const d = f.item_shape[0];
  return Array.from({ length: f.values.length / d }, (_, i) =>
    f.values.slice(i * d, (i + 1) * d),
  );
}
const colors = {
  cst: "#60bdcf",
  ancestry: "#edb35d",
  persistence: "#8b9ab2",
  ig_distance: "#b394e8",
  ig_cloning: "#e68da7",
  ia_distance: "#80caa5",
  ia_cloning: "#e28377",
  historical_distance: "#d2baed",
  historical_cloning: "#f3a3b7",
};
function graphView(archive, graph, mode) {
  const lookup = new Map();
  for (const a of archive.anchors)
    lookup.set(`${a.epoch}:${a.step}:${a.population.version}`, a.population);
  for (const s of archive.steps) {
    lookup.set(`${s.epoch}:${s.report.step - 1}:${s.before.version}`, s.before);
    lookup.set(
      `${s.epoch}:${s.report.step}:${s.final_population.version}`,
      s.final_population,
    );
  }
  // Keep all identities in archive; bound only the drawn time window.
  const latest = Math.max(...graph.nodes.map((n) => n.step), 0),
    recent = graph.nodes.filter((n) => n.step >= latest - 16);
  const nodes = recent.map((e) => {
    const p = field(lookup.get(`${e.epoch}:${e.step}:${e.version}`) || {});
    return {
      id: eventKey(e),
      owner: e.slot,
      position: [...(p[e.slot] || [0, 0]), e.step * 0.15],
      layer: "events",
      label: `step ${e.step} · slot ${e.slot} · generation ${e.generation}`,
      detail: `epoch ${e.epoch}, step ${e.step}, slot ${e.slot}, generation ${e.generation}, version ${e.version}`,
      color: colors.persistence,
    };
  });
  const ids = new Set(nodes.map((n) => n.id));
  const kinds =
    mode === 1
      ? ["cst", "ancestry", "persistence"]
      : mode === 17
        ? ["cst"]
        : Object.keys(colors);
  const edges = graph.edges
    .filter(
      (e) =>
        kinds.includes(e.kind) &&
        ids.has(eventKey(e.source)) &&
        ids.has(eventKey(e.target)),
    )
    .map((e) => ({
      source: eventKey(e.source),
      target: eventKey(e.target),
      layer: e.kind,
      color: colors[e.kind],
    }));
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  const faces =
    mode === 2
      ? (graph.triangles || []).slice(-128).flatMap((t) => {
          const ns = t.vertices.map((e) => nodeMap.get(eventKey(e)));
          return ns.every(Boolean)
            ? [
                {
                  vertices: ns.map((n) => n.position),
                  layer: "interaction triangles",
                  owner: t.vertices[0].slot,
                },
              ]
            : [];
        })
      : [];
  return {
    faces,
    title:
      mode === 17
        ? "Recorded CST and geometric light cones"
        : "Recorded events and their relationships",
    nodes,
    edges,
    message:
      "The vertical coordinate is iteration × 0.15. The scene shows the last 16 iterations; the archive preserves every recorded step.",
  };
}
function orderView(comparison) {
  const nodes = comparison.nodes.map((n) => ({
    id: eventKey(n.event),
    position: [...n.position, n.time],
    label: `step ${n.event.step} · slot ${n.event.slot}`,
    layer: "events",
    detail: `Physical time ${n.time}; epoch ${n.event.epoch}, generation ${n.event.generation}`,
  }));
  const edges = [
    ...comparison.cst_pairs.map(([a, b]) => ({
      source: nodes[a].id,
      target: nodes[b].id,
      layer: "CST reachability",
      color: colors.cst,
    })),
    ...comparison.lorentz_pairs.map(([a, b]) => ({
      source: nodes[a].id,
      target: nodes[b].id,
      layer: "geometric light cone",
      color: colors.ancestry,
    })),
  ];
  return {
    title: "CST reachability and physical light cones",
    nodes,
    edges,
    message: `Light-cone speed ${comparison.speed}; time per iteration ${comparison.dt}. Camera rotation does not change either relation.`,
  };
}
async function makeGas(engine, seed, n, params = {}) {
  const c = await engine.defaults();
  c.walkers = n;
  c.dimensions = 2;
  c.benchmark = "quadratic";
  c.initial_lower = -1.8;
  c.initial_upper = 1.8;
  c.gas.seed = seed;
  c.gas.precision = "f64";
  c.gas.backend = "cpu";
  c.gas.boundary = { kind: "unbounded" };
  for (const role of ["distance_donors", "cloning_donors"]) {
    c.gas[role].kernel = { kind: "uniform" };
    c.gas[role].law = "independent";
    c.gas[role].history_window = params.history || 0;
  }
  c.gas.kinetic = {
    integrator: {
      kind: "baoab",
      positions: "positions",
      velocities: "velocities",
      dt: params.dt || 0.04,
      friction: 1,
    },
    noise: {
      innovation: "gaussian",
      geometry: {
        kind: "isotropic",
        scale: { kind: "constant", values: [Math.sqrt(0.8)] },
      },
    },
  };
  if (params.neutral) {
    c.gas.fitness.reward_exponent = 0;
    c.gas.fitness.diversity_exponent = 0;
  }
  if (params.factor)
    c.gas.kinetic.noise.geometry = {
      kind: "full",
      factor: { kind: "constant", values: params.factor.flat() },
    };
  if (params.adaptive) {
    c.physics_metric = { epsilon: 1, temperature: 0.4, policy: "clipped" };
    c.gas.fitness.reward_standardizer = { kind: "global", sigma_min: 0.1 };
    c.gas.fitness.diversity_standardizer = { kind: "global", sigma_min: 0.1 };
    c.gas.fitness.distance_floor = 0.15;
  }
  const gas = await engine.create(c);
  if (["revival", "singleton"].includes(params.scenario)) {
    const alive = Array.from({ length: n }, (_, j) =>
      params.scenario === "singleton" ? j === 0 : j >= Math.ceil(n / 4),
    );
    await gas.set_population(
      JSON.stringify({ positions: positions(gas.snapshot()), alive }),
    );
  }
  gas.start_recording(
    JSON.stringify({ max_steps: 256, max_bytes: 128 * 1024 * 1024 }),
  );
  return gas;
}
function meshChart(mesh, points, title = "Metric Voronoi cells") {
  return plot(title, "x₁", "x₂", [scatter("Sites", points)], {
    xDomain: bounds.slice(0, 2),
    yDomain: bounds.slice(2),
    segments: mesh.cells.flatMap((c) =>
      c.vertices.map((p, i) => [p, c.vertices[(i + 1) % c.vertices.length]]),
    ),
  });
}
const tensorChart = (title, matrix) => ({
  title,
  matrix,
  rowLabels: matrix.map((_, i) => String(i + 1)),
  columnLabels: matrix.map((_, i) => String(i + 1)),
});
function scalarRows(data, prefix = "", out = []) {
  for (const [key, value] of Object.entries(data || {})) {
    if (typeof value === "number") out.push([prefix + key, value]);
    else if (value && typeof value === "object" && !Array.isArray(value))
      scalarRows(value, prefix + key + ".", out);
  }
  return out;
}
const analysisKinds = {
  3: "spin2",
  4: "transport",
  14: "integration",
  15: "kernel",
  16: "manufactured",
  18: "counts",
  19: "dimension",
  20: "curvature",
};
function analysisControls(i) {
  const controls = [
    select("samples", "Samples", 256, [64, 128, 256, 512, 1024]),
    select("replicas", "Independent replicas", 32, [8, 16, 32, 64]),
  ];
  if (i === 19)
    controls[0] = select("samples", "Samples", 256, [64, 128, 256, 512]);
  if ([14, 16, 20].includes(i))
    controls[0] = select(
      "samples",
      "Samples",
      256,
      [64, 256, 1024, 4096, 16384],
    );
  if ([15, 16, 20].includes(i))
    controls.push(range("bandwidth", "Bandwidth", 0.4, 0.1, 0.8, 0.05));
  if (i === 18)
    controls.push(
      select("count_model", "Sampling law", "poisson", ["poisson", "fixed"]),
    );
  if (i === 19)
    controls.push(
      select("spacetime_dimension", "Spacetime dimension", 3, [2, 3, 4]),
      range("density_contrast", "Density contrast", 0, 0, 0.9, 0.1),
    );
  if (i === 14)
    controls.push(
      range("density_contrast", "Density contrast", 0.5, 0, 0.9, 0.1),
    );
  if (i === 20)
    controls.push(
      range("curvature", "Spatial curvature K", 0.2, -0.5, 0.5, 0.05),
    );
  if ([3, 4].includes(i))
    return [
      range("phase", "Rotation / connection angle", 0.7, 0, 4 * Math.PI, 0.05),
    ];
  if (i === 15)
    return [
      range("bandwidth", "Manufactured-field bandwidth", 0.4, 0.1, 0.8, 0.05),
    ];
  return controls;
}
async function analysisModel(i, { params, seed, engine }) {
  let tick = 0,
    result,
    history = [];
  async function update() {
    result = await engine.analysis({
      kind: analysisKinds[i],
      seed: seed + tick,
      ...params,
      ...([3, 4].includes(i) ? { phase: params.phase + tick * 0.05 } : {}),
    });
    for (const v of result.metrics || [])
      if (typeof v.value === "number") {
        let s = history.find((s) => s.name === v.name);
        if (!s) {
          s = { name: v.name, points: [] };
          history.push(s);
        }
        s.points.push([tick, v.value]);
      }
  }
  await update();
  return {
    async step() {
      tick++;
      await update();
    },
    snapshot() {
      const series = (result.series || []).map((s) =>
        line(
          s.name,
          s.x.map((x, j) => [x, s.y[j]]),
        ),
      );
      const charts = [];
      const groups =
        i === 3
          ? [
              {
                title: "Continuous Spin(2) lift",
                x: "Rotation angle",
                y: "Spinor component",
                match: () => true,
              },
            ]
          : i === 16
            ? [
                {
                  title: "Wave operator estimate",
                  x: "Bandwidth ε",
                  y: "Operator value",
                  match: (n) => /operator versus|operator mean/.test(n),
                },
                {
                  title: "Deterministic truncation error",
                  x: "Bandwidth ε",
                  y: "Absolute bias",
                  match: (n) => n === "absolute bias",
                  log: true,
                },
                {
                  title: "Sampling error and total error",
                  x: "Bandwidth ε",
                  y: "Squared operator units",
                  match: (n) => /single-run variance|MSE/.test(n),
                  log: true,
                },
                {
                  title: "Points inside kernel support",
                  x: "Bandwidth ε",
                  y: "Mean points",
                  match: (n) => n === "mean support count",
                },
                {
                  title: "Empty support frequency",
                  x: "Bandwidth ε",
                  y: "Replica fraction",
                  match: (n) => /zero-support/.test(n),
                },
              ]
            : series.map((s) => ({
                title: s.name,
                x: /bandwidth|bias/.test(s.name)
                  ? "Bandwidth ε"
                  : /replicas|support counts/.test(s.name)
                    ? "Replica"
                    : /kernel/.test(s.name)
                      ? "Normalized time coordinate"
                      : "Replica / component",
                y: /dimension/.test(s.name)
                  ? "Dimension"
                  : /fraction/.test(s.name)
                    ? "Fraction"
                    : /curvature/.test(s.name)
                      ? "Scalar curvature"
                      : "Computed value",
                match: (n) => n === s.name,
              }));
      for (const group of groups) {
        const ss = series.filter((s) => group.match(s.name));
        if (ss.length)
          charts.push(
            plot(
              group.title,
              group.x,
              group.y,
              ss,
              i === 16 && group.title === "Wave operator estimate"
                ? {
                    segments: result.series
                      .find((s) => s.name === "operator versus bandwidth")
                      .x.map((x, j) => {
                        const mean = result.series.find(
                          (s) => s.name === "operator versus bandwidth",
                        ).y[j];
                        const se = result.series.find(
                          (s) => s.name === "Predicted standard error of mean",
                        ).y[j];
                        return [
                          [x, mean - 1.96 * se],
                          [x, mean + 1.96 * se],
                        ];
                      }),
                  }
                : group.log
                  ? { xScale: "log", yScale: "log" }
                  : {},
            ),
          );
      }
      if (result.points?.length)
        charts.push(
          plot("Independent spacetime sample", "x₁", "x₂", [
            scatter(
              "Events",
              result.points.map((p) => [p[1], p[2] || 0]),
            ),
          ]),
        );
      const primary =
        (i === 20
          ? result.metrics?.find(
              (v) => v.name === "kernel estimated scalar curvature",
            )
          : null) ||
        (result.metrics || []).find((v) => v.standard_error !== undefined) ||
        result.metrics?.[0];
      if (primary) {
        const measured = history.find((s) => s.name === primary.name);
        const ss = [measured];
        if (primary.reference !== undefined)
          ss.push(
            line(
              i === 20
                ? "Continuum curvature R = 2K"
                : "Analytical / quadrature prediction",
              measured.points.map(([x]) => [x, primary.reference]),
              { dashed: true },
            ),
          );
        if (i === 20) {
          const finite = result.metrics.find(
            (v) => v.name === "kernel quadrature finite-radius curvature",
          ).value;
          const predictedSE = result.metrics.find(
            (v) => v.name === "kernel predicted standard error of mean",
          ).value;
          ss.push(
            line(
              "Finite-bandwidth quadrature",
              measured.points.map(([x]) => [x, finite]),
              { dashed: true },
            ),
          );
          for (const sign of [-1, 1])
            ss.push(
              line(
                sign < 0
                  ? "Quadrature − 1.96 predicted SE"
                  : "Quadrature + 1.96 predicted SE",
                measured.points.map(([x]) => [
                  x,
                  finite + sign * 1.96 * predictedSE,
                ]),
                { dashed: true },
              ),
            );
        }
        charts.push(
          plot(
            primary.name,
            [3, 4, 15].includes(i) ? "Parameter step" : "Independent ensemble",
            "Measured value",
            ss,
            primary.standard_error !== undefined
              ? {
                  segments: [
                    [
                      [tick, primary.value - 1.96 * primary.standard_error],
                      [tick, primary.value + 1.96 * primary.standard_error],
                    ],
                  ],
                }
              : {},
          ),
        );
      }
      const nodes = (result.points || []).slice(0, 512).map((p, j) => ({
        id: String(j),
        label: "Reference event " + j,
        position: [p[1], p[2] || 0, p[0]],
        layer: "reference events",
      }));
      return {
        step: tick,
        time: tick,
        charts,
        metrics: (result.metrics || []).map((v) => m(v.name, v.value)),
        message:
          i === 16
            ? `${params.replicas} independent replicas; ${params.samples} samples each. This field has zero gradient at the query, so leading variance scales as 1/(N ε³). Deterministic variance stays positive even if every sampled support is empty. ±1.96-SE bars are pointwise scales, not guaranteed confidence intervals for sparse samples.`
            : i === 20
              ? `${params.replicas} independent replicas; ${params.samples} samples each. The sampling cube shrinks with ε, so leading curvature variance scales as 1/(N ε⁴). Finite-radius bias, predicted sampling uncertainty and empirical uncertainty are distinct.`
              : [3, 4, 15].includes(i)
                ? "Deterministic native calculation; controls change the field or connection. Numerical values and model conventions are included in the JSON export."
                : `${params.replicas} independent replicas, ${params.samples} target samples per replica. Error bars show a pointwise ±1.96-standard-error scale; sparse or skewed samples need not give 95% coverage. Model details are included in the JSON export.`,
        table: {
          columns: ["Measurement", "Value", "Reference", "Standard error"],
          rows: (result.metrics || []).map((v) => [
            v.name,
            v.value,
            v.reference ?? "",
            v.standard_error ?? "",
          ]),
        },
        ...(nodes.length
          ? {
              scene: {
                title:
                  params.spacetime_dimension === 4
                    ? "Reference spacetime sample (x₃ omitted)"
                    : "Reference spacetime sample",
                nodes,
              },
            }
          : {}),
        result,
        done: tick >= 31,
      };
    },
    dispose() {},
  };
}
function geometryControls(i) {
  const common = [
    select("walkers", "Walkers", 64, [16, 32, 64, 128]),
    range("anisotropy", "Metric anisotropy", 2, 1, 8, 0.25),
  ];
  if ([5, 13].includes(i))
    return [
      select("walkers", "Walkers", 64, [16, 32, 64]),
      range("query", "Query x₁", 0.3, -2, 2, 0.05),
      range("epsilon", "Metric regularization", 1, 0.1, 4, 0.1),
      range("distance_floor", "Distance regularization", 0.15, 0.05, 0.8, 0.05),
      select("policy", "Metric policy", "clipped", ["clipped", "strict"]),
      ...(i === 13
        ? [
            select("stratum", "Frozen stratum", "recorded", [
              "recorded",
              "new_companions",
              "half_alive",
            ]),
          ]
        : []),
    ];
  if (i === 6)
    return [
      select("source", "Metric source", "sampled_fitness", [
        "sampled_fitness",
        "constant_metric",
      ]),
      range("anisotropy", "Constant-metric anisotropy", 2, 1, 8, 0.25),
      select(
        "samples",
        "Independent innovations",
        1024,
        [128, 512, 1024, 4096],
      ),
      range("dt", "O-step duration", 0.04, 0.01, 0.2, 0.01),
    ];
  if (i === 11) {
    common[0] = select("walkers", "Walkers", 16, [8, 16, 32]);
    common.push(select("resolution", "Spacetime refinement", 4, [2, 4, 6, 8]));
  }
  if (i === 9) {
    common[0] = select("walkers", "Walkers", 64, [16, 32, 64]);
    common.push(
      select("geometry", "Distance geometry", "constant", [
        "constant",
        "variable",
      ]),
      select("resolution", "Geodesic grid refinement", 16, [8, 16, 24]),
      range("site_x", "Slot 0 x₁", 0.5, -2.5, 2.5, 0.05),
      range("site_y", "Slot 0 x₂", 0.5, -2.5, 2.5, 0.05),
    );
  }
  if (i === 8)
    common[0] = select("walkers", "Independent walkers", 128, [64, 128, 256]);
  if (i === 10)
    common.push(select("copying", "Copying", "active", ["active", "off"]));
  return common;
}
async function geometryModel(i, { params: p, seed, engine }) {
  let metric = [
      [p.anisotropy || 2, 0],
      [0, 1],
    ],
    n = p.walkers || 64;
  let tick = 0,
    gas,
    frame,
    mesh,
    result,
    reference,
    history = [],
    volumeFrames = [];
  if (i !== 6 || p.source === "sampled_fitness") {
    gas = await makeGas(engine, seed, n, {
      neutral: i === 8 || p.copying === "off",
      adaptive: i === 6,
      dt: p.dt,
      ...(i === 8
        ? {
            factor: [
              [Math.sqrt(0.8 / metric[0][0]), 0],
              [0, Math.sqrt(0.8)],
            ],
          }
        : {}),
    });
    frame = gas.snapshot();
    if (i === 6) frame = await gas.step(1);
  }
  if (i === 9) {
    const pts = positions(frame);
    pts[0] = [p.site_x, p.site_y];
    frame = await gas.set_population(JSON.stringify({ positions: pts }));
  }
  let queryPoints = frame ? positions(frame) : [];
  let companions = Array.from({ length: n }, (_, j) => (j + 1) % n);
  if ([5, 13].includes(i)) {
    frame = await gas.step(1);
    const record = gas.archive().steps[0];
    queryPoints = field(record.before);
    const batch = record.report.distance_companions;
    companions = Array.from(
      { length: n },
      (_, j) =>
        record.report.distance_sources[batch.indices[j * batch.count]]?.slot ??
        j,
    );
  }
  async function update() {
    if ([5, 13].includes(i)) {
      const q = [p.query + tick * 0.025, 0.25];
      const alive = Array.from({ length: n }, (_, j) =>
        p.stratum === "half_alive" ? j < n / 2 : true,
      );
      const chosen =
        p.stratum === "new_companions"
          ? companions.map((j) => (j + 1) % n)
          : p.stratum === "half_alive"
            ? companions.map((_, j) => (j + 1) % (n / 2))
            : companions;
      result = await engine.geometry({
        kind: "fitness",
        points: queryPoints,
        companions: chosen,
        alive,
        target: 0,
        query: q,
        objective: { kind: "quadratic", curvature: identity },
        sigma_min: 0.1,
        distance_floor: p.distance_floor,
        metric_epsilon: p.epsilon,
        metric_policy: "clipped",
      });
      if (p.policy === "strict") {
        const clipped = result.metric;
        try {
          result.metric = await engine.geometry({
            kind: "metric",
            hessian: result.hessian,
            epsilon: p.epsilon,
            policy: "strict",
          });
        } catch (error) {
          result.metric = null;
          result.strict_margin =
            Math.min(...clipped.hessian_eigenvalues) + p.epsilon;
          result.status = String(error);
        }
      }
      history.push([
        q[0],
        typeof result.fitness === "number"
          ? result.fitness
          : (result.jet?.value ?? result.value),
      ]);
    } else if (i === 6) {
      if (gas) {
        const evaluation = gas
          .archive()
          .steps.at(-1)
          .field_evaluations.find(
            (f) => f.field === "fitness_metric" && f.stage === "O",
          );
        if (!evaluation) throw new Error("Recorded O-stage metric is missing.");
        metric = [evaluation.values.slice(0, 2), evaluation.values.slice(2, 4)];
      }
      result = await engine.geometry({
        kind: "ou",
        metric,
        gamma: 1,
        temperature: 0.4,
        dt: p.dt,
        samples: p.samples,
        seed: seed + tick,
      });
    } else if (i === 8) {
      if (!reference)
        reference = await engine.geometry({
          kind: "harmonic",
          curvature: identity,
          metric,
          gamma: 1,
          temperature: 0.4,
          dt: 0.04,
          initial_half_width: 1.8,
          samples: n,
          steps: 128,
        });
      result = reference;
      // covariance2 uses the empirical-measure divisor N. Compare an unbiased
      // sample variance with the population covariance returned by Rust.
      history.push([
        tick * 0.04,
        (covariance2(positions(frame))[0][0] * n) / (n - 1),
      ]);
    } else {
      const points = positions(frame);
      mesh = await engine.geometry({ kind: "voronoi", points, bounds, metric });
      result = mesh;
      if (i === 9 && p.geometry === "variable") {
        const r = p.resolution;
        const metric_grid = Array.from({ length: r * r }, (_, j) => {
          const x = -3 + (6 * (j % r)) / (r - 1),
            y = -3 + (6 * Math.floor(j / r)) / (r - 1);
          return [
            [1 + (p.anisotropy * x * x) / 9, 0],
            [0, 1 + (p.anisotropy * y * y) / 9],
          ];
        });
        result = await engine.geometry({
          kind: "graph_distance",
          points,
          bounds,
          resolution: r,
          metric_grid,
        });
      }
      history.push([tick, mesh.neighbors.length]);
      if (i === 11) {
        if (tick) {
          const recorded = gas.archive().steps.at(-1);
          const post = recorded.stages.find((s) => s.stage === "post_clone");
          volumeFrames = [
            { time: (tick - 1) * 0.04, points: field(recorded.before) },
            { time: (tick - 1) * 0.04, points: field(post) },
            { time: tick * 0.04, points },
          ];
          result = await engine.geometry({
            kind: "spacetime",
            frames: volumeFrames,
            bounds,
            resolution: p.resolution,
            metric,
          });
        } else volumeFrames = [{ time: 0, points }];
      }
      if (i === 12) {
        volumeFrames.push(points);
        if (volumeFrames.length > 8) volumeFrames.shift();
        result = await engine.geometry({
          kind: "triangulation",
          frames: volumeFrames,
          metric,
        });
      }
      if (i === 10 && tick) {
        const step = gas.archive().steps.at(-1),
          pre = field(step.before);
        const postStage =
          step.stages.find((s) => s.stage === "post_clone") ||
          step.stages.find((s) => s.stage.includes("post_clone"));
        if (!postStage)
          throw new Error(
            "Validated post-clone positions are unavailable; clone/kinetic interface changes cannot be separated.",
          );
        const post = field(postStage);
        const a = await engine.geometry({
            kind: "voronoi",
            points: pre,
            bounds,
            metric,
          }),
          b = await engine.geometry({
            kind: "voronoi",
            points: post,
            bounds,
            metric,
          });
        const diff = (x, y) => {
          const a = new Set(x.neighbors.map((e) => e.join(":"))),
            b = new Set(y.neighbors.map((e) => e.join(":")));
          return (
            [...a].filter((e) => !b.has(e)).length +
            [...b].filter((e) => !a.has(e)).length
          );
        };
        result = {
          ...mesh,
          copy_changed_interfaces: diff(a, b),
          kinetic_changed_interfaces: diff(b, mesh),
          post_clone_stage: postStage?.stage || "unavailable",
        };
      }
    }
  }
  await update();
  return {
    async step() {
      tick++;
      if (gas && ![5, 13].includes(i)) frame = await gas.step(1);
      await update();
    },
    snapshot() {
      const rows = scalarRows(result),
        charts = [],
        metrics = rows
          .filter(([k]) => !["rng_step", "rng_substep", "seed"].includes(k))
          .slice(0, 8)
          .map(([k, v]) => m(k.replaceAll("_", " ").replaceAll(".", " / "), v));
      let scene;
      if ([5, 13].includes(i)) {
        const f = result.jet || result,
          g = result.metric || {};
        if (f.hessian)
          charts.push(
            tensorChart(
              "Exact conditional teaching-profile Hessian",
              f.hessian,
            ),
          );
        if (g.metric) charts.push(tensorChart("Regularized metric", g.metric));
        charts.push(
          plot(
            "Conditional fitness along a frozen stratum",
            "Query x₁",
            "Fitness",
            [line("Exact jet value", history)],
          ),
        );
        charts.push(
          plot("Frozen population and query", "x₁", "x₂", [
            scatter("Frozen sites", queryPoints),
            scatter("Query", [[p.query + tick * 0.025, 0.25]]),
          ]),
        );
      } else if (i === 6) {
        if (result.cloud || result.sample_points)
          charts.push(
            plot("O-stage innovation cloud", "Δv₁", "Δv₂", [
              scatter("Rust innovations", result.cloud || result.sample_points),
            ]),
          );
        for (const key of [
          "predicted_covariance",
          "sample_covariance",
          "expected_covariance",
          "empirical_covariance",
        ])
          if (result[key])
            charts.push(tensorChart(key.replaceAll("_", " "), result[key]));
        const z = result.sample_covariance.map((row, j) =>
          row.map(
            (v, k) =>
              (v - result.expected_covariance[j][k]) /
              result.covariance_standard_error[j][k],
          ),
        );
        charts.push(
          tensorChart("Covariance discrepancy in standard errors", z),
        );
        metrics.push(
          m(
            "Largest covariance discrepancy",
            Math.max(...z.flat().map(Math.abs)),
            "SE",
          ),
        );
      } else if (i === 8) {
        charts.push(
          plot("Harmonic relaxation", "Time", "Position variance", [
            line("Unbiased measured x₁ variance", history),
            line(
              "Exact transient prediction",
              history.map(([t], j) => [
                t,
                reference.transient_covariances[j][0][0],
              ]),
            ),
            ...[-1, 1].map((sign) =>
              line(
                sign < 0 ? "Prediction − 1.96 SE" : "Prediction + 1.96 SE",
                history.map(([t], j) => [
                  t,
                  reference.transient_covariances[j][0][0] +
                    sign *
                      1.96 *
                      reference.transient_variance_standard_errors[j][0],
                ]),
                { dashed: true },
              ),
            ),
            line(
              "Discrete stationary prediction",
              history.map(([t]) => [t, reference.discrete_covariance[0][0]]),
            ),
            line(
              "Continuous stationary prediction",
              history.map(([t]) => [t, reference.continuous_covariance[0][0]]),
            ),
          ]),
        );
        metrics.push(
          m("Measured unbiased x₁ variance", history.at(-1)[1]),
          m(
            "Transient x₁ variance",
            reference.transient_covariances[tick][0][0],
          ),
          m(
            "Sample variance standard error",
            reference.transient_variance_standard_errors[tick][0],
          ),
          m(
            "Transient discrepancy",
            (history.at(-1)[1] - reference.transient_covariances[tick][0][0]) /
              reference.transient_variance_standard_errors[tick][0],
            "SE",
          ),
        );
        for (const key of ["discrete_covariance", "continuous_covariance"])
          if (reference[key])
            charts.push(tensorChart(key.replaceAll("_", " "), reference[key]));
      } else {
        if (i === 9 && p.geometry === "variable")
          charts.push({
            title:
              "Variable-metric nearest-site partition (refined graph distance)",
            matrix: Array.from({ length: p.resolution }, (_, j) =>
              result.owners.slice(j * p.resolution, (j + 1) * p.resolution),
            ),
            rowLabels: [],
            columnLabels: [],
          });
        else charts.push(meshChart(mesh, positions(frame)));
        if (i === 12) {
          const records = result.frames,
            last = records.at(-1);
          charts.push(
            plot("Delaunay maintenance work", "Recorded frame", "Operations", [
              line(
                "Vertex removals",
                records.map((f) => [f.frame, f.removed_vertices]),
              ),
              line(
                "Vertex insertions",
                records.map((f) => [f.frame, f.inserted_vertices]),
              ),
              line(
                "Independent rebuild insertions",
                records.map((f) => [f.frame, f.rebuild_insertions]),
              ),
            ]),
          );
          charts.push(
            plot("Interface activity", "Recorded frame", "Count", [
              line(
                "Unique changed interfaces",
                records.map((f) => [f.frame, f.unique_changed_interfaces]),
              ),
              line(
                "Changed interface incidences",
                records.map((f) => [f.frame, f.changed_interface_incidences]),
              ),
            ]),
          );
          metrics.push(
            m("Adjacency disagreements", last.edge_disagreements.length),
            m(
              "Euler V − E + F",
              last.euler_vertices_minus_edges_plus_bounded_faces,
            ),
            m("Slot search visits", last.slot_search_visits),
          );
          rows.push(...scalarRows(last, "latest."));
        }
        charts.push(
          plot(
            i === 9 && p.geometry === "variable"
              ? "Constant-metric comparison adjacency"
              : "Geometric adjacency activity",
            "Iteration",
            "Unique interfaces",
            [line("Current interfaces", history)],
          ),
        );
        charts.push(
          plot(
            i === 9 && p.geometry === "variable"
              ? "Constant-metric comparison volumes"
              : "Cell volumes",
            "Slot",
            "Geometric volume",
            [
              {
                name: "Computed volume",
                style: "bars",
                points:
                  i === 11 && result.geometric_slot_volumes
                    ? result.geometric_slot_volumes.map((v, j) => [j, v])
                    : mesh.cells.map((c) => [c.slot, c.geometric_area]),
              },
            ],
          ),
        );
        if (i === 11 && result.geometric_slot_volumes) {
          charts.at(-1).title = "Spacetime cell volumes";
          charts.at(-1).yLabel = "Geometric 3-volume";
        }
        if (i === 11) {
          scene = {
            title: "Computed spacetime cell partition",
            verticalScale: 25,
            nodes: volumeFrames.flatMap((f, phase) =>
              f.points.map((p, j) => ({
                id: `${tick}:${phase}:${j}`,
                owner: j,
                position: [...p, f.time],
                label: `time ${f.time.toFixed(2)} · slot ${j}`,
                layer: "sites",
              })),
            ),
            faces: [
              ...(result.boundary_faces || []).map((face) => ({
                owner: face.slot,
                layer: "cell surfaces",
                vertices: face.vertices,
              })),
              ...(result.jump_caps || []).flatMap((cap) =>
                ["before", "after"].flatMap((side) =>
                  cap[side].cells.map((cell) => ({
                    owner: cell.slot,
                    layer: side + " jump cap",
                    vertices: cell.vertices.map((p) => [...p, cap.time]),
                  })),
                ),
              ),
            ],
            message:
              "Select a site to isolate its cell. Jump caps mark instantaneous copying; vertical display scale does not change measured volume.",
          };
        }
      }
      if (!charts.length)
        charts.push(
          plot("Computed scalar diagnostics", "Index", "Value", [
            {
              name: "Native result",
              style: "bars",
              points: rows.map((r, j) => [j, r[1]]),
            },
          ]),
        );
      return {
        step: tick,
        time: gas ? Number(frame.step) * (i === 6 ? p.dt : 0.04) : tick,
        charts,
        metrics,
        message: [5, 13].includes(i)
          ? result.status ||
            "Conditional teaching profile on recorded coordinates: quadratic objective, global σmin=0.1, logistic map 2/(1+exp(−z))+10⁻⁶, unit channel exponents, position-only distance. Source coordinates and alive membership remain frozen."
          : i === 6
            ? gas
              ? "The metric is recorded at the actual BAOAB O stage for slot 0. Independent innovations test its conditional covariance with the full thermostat prefactor."
              : "Independent frozen O-step innovations with full thermostat prefactor."
            : i === 11
              ? "Cells share a tetrahedral grid. Coordinate volume and geometric volume are measured separately; refine the grid to resolve the moving interfaces."
              : i === 12
                ? "Incremental and rebuilt Delaunay graphs use the same sites. Co-circular configurations can admit different valid diagonals."
                : i === 10
                  ? "Neighbor changes compare the recorded pre-clone, validated post-clone, and final positions."
                  : i === 8
                    ? "Independent harmonic walkers use an unbiased sample variance. The exact transient starts from the uniform initial law; its pointwise ±1.96-SE bands include the initial fourth cumulant. Stationary limits are shown separately."
                    : i === 9 && p.geometry === "variable"
                      ? "The grid approximates geodesic distance. Refinement adds spatial samples and directions; the other plots provide a constant-metric comparison."
                      : "Cells partition the observation window. With a constant metric, geometric volume is coordinate area multiplied by the square root of the metric determinant.",
        table: { columns: ["Quantity", "Value"], rows },
        ...(scene ? { scene } : {}),
        result,
        done: tick >= ([5, 13].includes(i) ? 64 : 128),
      };
    },
    checkpoint: gas ? () => gas.checkpoint() : undefined,
    archive: gas ? () => gas.archive() : undefined,
    dispose() {
      gas?.free();
    },
  };
}
async function reconstructionModel(ctx) {
  const model = await analysisModel(3, ctx),
    gas = await makeGas(ctx.engine, ctx.seed, 64);
  let frame = await gas.step(1),
    reconstruction = gas.reconstruct(0, 1, "post_kinetic");
  return {
    async step() {
      await model.step();
      frame = await gas.step(1);
      reconstruction = gas.reconstruct(0, Number(frame.step), "post_kinetic");
    },
    snapshot() {
      const base = model.snapshot();
      const decoded = field(reconstruction);
      return {
        ...base,
        charts: [
          plot("Coordinates reconstructed from scalar addresses", "x₁", "x₂", [
            scatter("Decoded coordinates", decoded),
          ]),
          ...base.charts,
        ],
        metrics: [
          m("Reconstruction residual", reconstruction.max_absolute_residual),
          m("Indexed scalar components", reconstruction.scalar_count),
          ...base.metrics,
        ],
        reconstruction,
        message:
          "The position plot is decoded from the recorded scalar component addresses. The separate Spin(2) experiment uses a continuous lifted rotation.",
      };
    },
    checkpoint: () => gas.checkpoint(),
    archive: () => gas.archive(),
    dispose() {
      model.dispose();
      gas.free();
    },
  };
}
async function graphModel(i, { params, seed, engine }) {
  const gas = await makeGas(engine, seed, params.walkers, {
    history: params.history,
    scenario: params.scenario,
  });
  let tick = 0,
    frame = gas.snapshot(),
    archive = gas.archive(),
    graph = gas.fractal_set();
  let orders = i === 17 ? gas.compare_orders(params.speed, 0.04, 200) : null;
  function current() {
    archive = gas.archive();
    graph = gas.fractal_set();
    if (i === 17) orders = gas.compare_orders(params.speed, 0.04, 200);
  }
  return {
    reuse(request) {
      if (
        !["V-01", "V-02", "V-17"].includes(request.id) ||
        request.seed !== seed ||
        request.params.walkers !== params.walkers ||
        request.params.history !== params.history ||
        request.params.scenario !== params.scenario
      )
        return false;
      i = Number(request.id.split("-")[1]);
      params = request.params;
      orders = i === 17 ? gas.compare_orders(params.speed, 0.04, 200) : null;
      return true;
    },
    async step() {
      frame = await gas.step(1);
      tick++;
      current();
    },
    snapshot() {
      const counts = Object.entries(colors).map(([kind]) => [
        kind,
        graph.edges.filter((e) => e.kind === kind).length,
      ]);
      return {
        step: tick,
        time: tick * 0.04,
        charts: [
          ...(orders
            ? [
                plot("Comparable event pairs", "Relation", "Pairs", [
                  {
                    name: "Both orders",
                    style: "bars",
                    points: [[0, orders.both]],
                  },
                  {
                    name: "CST only",
                    style: "bars",
                    points: [[1, orders.cst_only]],
                  },
                  {
                    name: "Light cone only",
                    style: "bars",
                    points: [[2, orders.lorentz_only]],
                  },
                ]),
              ]
            : []),
          plot("Recorded walker positions", "x₁", "x₂", [
            scatter("Walkers", positions(frame)),
          ]),
          plot("Separate edge relations", "Relation index", "Edges", [
            {
              name: "Recorded edges",
              style: "bars",
              points: counts.map(([, v], j) => [j, v]),
            },
          ]),
        ],
        scene: orders ? orderView(orders) : graphView(archive, graph, i),
        metrics: [
          m("Recorded steps", archive.steps.length),
          m("Events", graph.nodes.length),
          m("Interaction triangles", graph.triangles?.length || 0),
          m("Unresolved earlier sources", graph.unresolved_sources.length),
          ...(orders
            ? [
                m("Both orders", orders.both),
                m("CST only", orders.cst_only),
                m("Light cone only", orders.lorentz_only),
              ]
            : []),
        ],
        table: { columns: ["Relation", "Edges"], rows: counts },
        message:
          "Every committed microstep is retained. Slot generation counts recipient replacements; donor identity is stored independently.",
        done: tick >= 256,
      };
    },
    checkpoint: () => gas.checkpoint(),
    archive: () => archive,
    dispose() {
      gas.free();
    },
  };
}
export async function importedArchiveModel(archive, engine) {
  const checked = await engine.inspectArchive(archive),
    graph = checked.graph;
  const final =
    archive.steps.at(-1)?.final_population || archive.anchors.at(-1).population;
  return {
    async step() {},
    snapshot() {
      return {
        step: archive.steps.length,
        time: 0,
        charts: [
          plot("Imported recorded positions", "x₁", "x₂", [
            scatter("Recorded walkers", field(final)),
          ]),
        ],
        scene: graphView(archive, graph, 1),
        metrics: [
          m("Validated steps", archive.steps.length),
          m("Events", graph.nodes.length),
          m("Unresolved earlier sources", graph.unresolved_sources.length),
        ],
        message:
          "Validated standalone archive. All recorded stages remain in the export; use the scene controls to inspect event identities.",
        done: true,
      };
    },
    archive: () => archive,
    dispose() {},
  };
}
export const demos = metadata.map((meta, j) => {
  const i = j + 1;
  return {
    ...meta,
    kind: [1, 2, 10, 11, 12, 17].includes(i)
      ? "WASM recording + geometry"
      : "Rust/WASM reference experiment",
    controls: analysisKinds[i]
      ? analysisControls(i)
      : [1, 2, 17].includes(i)
        ? [
            select("walkers", "Walkers", 64, [16, 32, 64]),
            select("history", "Donor history", 0, [0, 2, 4]),
            select("scenario", "Initial population", "ordinary", [
              "ordinary",
              "revival",
              "singleton",
            ]),
            ...(i === 17
              ? [range("speed", "Light-cone speed", 1, 0.25, 4, 0.25)]
              : []),
          ]
        : geometryControls(i),
    create: (ctx) =>
      i === 3
        ? reconstructionModel(ctx)
        : analysisKinds[i]
          ? analysisModel(i, ctx)
          : [1, 2, 17].includes(i)
            ? graphModel(i, ctx)
            : geometryModel(i, ctx),
  };
});
