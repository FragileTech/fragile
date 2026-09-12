import {
  avg,
  varOf,
  norm2,
  center,
  spread,
  random,
  transport,
  assignment,
  mv,
  mm,
  perron,
  killedKernel,
  qsd,
  reservoir,
  collision,
  geometricPartition,
  histogram,
  grid,
  gaussian,
  linearBAOAB,
  covarianceStep,
} from "./convergence-math.js";
const range = (key, label, value, min, max, step = 0.1) => ({
  key,
  label,
  type: "range",
  value,
  min,
  max,
  step,
});
const select = (key, label, value, values) => ({
  key,
  label,
  type: "select",
  value,
  options: values.map((v) =>
    typeof v === "object" ? v : { value: v, label: String(v) },
  ),
});
const curve = (name, points, extra = {}) => ({ name, points, ...extra });
const dots = (name, points, extra = {}) =>
  curve(name, points, { style: "points", ...extra });
const chart = (title, xLabel, yLabel, series, extra = {}) => ({
  title,
  xLabel,
  yLabel,
  series,
  ...extra,
});
const metric = (label, value, unit = "") => ({ label, value, unit });
const keep = (a, v) => {
  a.push(v);
  if (a.length > 400) a.shift();
};
const pts = (f) => {
  const p = f.population.observations.fields.positions;
  return Array.from({ length: p.rows }, (_, i) =>
    Array.from(p.values.slice(i * p.item_shape[0], (i + 1) * p.item_shape[0])),
  );
};
const model = (snapshot, step, dispose = () => {}) => ({
  snapshot,
  step,
  dispose,
});
function descriptor(
  id,
  title,
  kind,
  question,
  prediction,
  explanation,
  controls,
  create,
) {
  return {
    id,
    part: id.split("-")[0],
    title,
    kind,
    question,
    prediction,
    explanation,
    controls,
    create: async (args) =>
      create({
        ...args,
        params: {
          ...Object.fromEntries(controls.map((c) => [c.key, c.value])),
          ...args.params,
        },
      }),
  };
}
async function configuration(engine, seed, n = 64) {
  const c = await engine.defaults();
  c.benchmark = "sphere";
  c.walkers = n;
  c.dimensions = 2;
  c.initial_lower = -1;
  c.initial_upper = 1;
  c.gas.backend = "cpu";
  c.gas.precision = "f64";
  c.gas.seed = seed >>> 0;
  c.gas.boundary = { kind: "unbounded" };
  c.gas.fitness.diversity_exponent = 0;
  c.gas.distance_donors.kernel = { kind: "uniform" };
  c.gas.cloning_donors.kernel = { kind: "uniform" };
  return c;
}
async function frozenStep(engine, base, points, seed) {
  const c = structuredClone(base);
  c.gas.seed = seed >>> 0;
  const run = await engine.create(c);
  try {
    await run.set_population(JSON.stringify({ positions: points }));
    return await run.step(1);
  } finally {
    run.free();
  }
}

const keystone = descriptor(
  "II-01",
  "The Keystone chain as linked evidence",
  "WASM experiment",
  "Does geometric error produce an inward copying signal?",
  "With the reward optimum in the main cluster, high-error walkers tend to copy inward; moving the optimum can reverse the signal.",
  "Complete-linkage partition from def-unified-high-low-error-sets: diameter ≤ 2ε, minimum cluster size max(5,ceil(.05N)), and top 90% of valid between-cluster variance. Velocities are zero in this positional fixture.",
  [
    select("walkers", "Walkers", 64, [64, 128]),
    range("fraction", "Outlier fraction", 0.15, 0.05, 0.4, 0.05),
    range("separation", "Cluster separation", 2, 1, 4, 0.25),
    range("alpha", "Reward exponent", 1, 0, 3, 0.25),
    range("beta", "Diversity exponent", 0, 0, 3, 0.25),
    select("landscape", "Reward optimum", "main", [
      { value: "main", label: "Main cluster" },
      { value: "outliers", label: "Outlier cluster" },
    ]),
    range("epsilon", "Partition scale ε", 0.3, 0.1, 0.8, 0.1),
  ],
  async ({ params: p, seed, engine }) => {
    const r = random(seed),
      n = p.walkers,
      out = Math.round(n * p.fraction),
      peak = p.landscape === "outliers" ? p.separation : 0;
    const positions = Array.from({ length: n }, (_, i) => [
      (i < out ? p.separation : 0) + 0.08 * r.normal() - peak,
      0.08 * r.normal(),
    ]);
    const base = await configuration(engine, seed, n);
    base.gas.fitness.reward_exponent = p.alpha;
    base.gas.fitness.diversity_exponent = p.beta;
    base.gas.kinetic.integrator.amplitude = 0;
    const partition = geometricPartition(positions, p.epsilon),
      mu = center(positions),
      radius = positions.map((x) => norm2(x, mu));
    let count = 0,
      frame = null,
      stage = [0, 0, 0],
      changes = [[], [], []],
      history = [];
    return model(
      () => ({
        step: count,
        time: count,
        charts: [
          chart(
            "Frozen population and geometric partition",
            "x₁ (reward optimum = 0)",
            "x₂",
            [
              dots(
                "High error",
                positions.filter((_, i) => partition.high.has(i)),
              ),
              dots(
                "Remaining population",
                positions.filter((_, i) => !partition.high.has(i)),
              ),
            ],
          ),
          chart(
            "Radius and sampled fitness",
            "Squared radius about swarm center",
            "Fitness",
            [
              dots(
                "High error",
                frame
                  ? radius
                      .map((x, i) => [
                        x,
                        frame.report.pre_clone_fitness.fitness[i],
                      ])
                      .filter((_, i) => partition.high.has(i))
                  : [],
              ),
              dots(
                "Remaining",
                frame
                  ? radius
                      .map((x, i) => [
                        x,
                        frame.report.pre_clone_fitness.fitness[i],
                      ])
                      .filter((_, i) => !partition.high.has(i))
                  : [],
              ),
            ],
          ),
          chart(
            "Links averaged over frozen-state repetitions",
            "Category (1 high-error, 2 unfit, 3 overlap)",
            "Fraction",
            [
              curve(
                "Measured fraction",
                stage.map((v, i) => [i + 1, count ? v / count / n : 0]),
                { style: "bars" },
              ),
            ],
          ),
          chart(
            "Conditional replacement change",
            "Independent repetition",
            "Mean change of squared radius",
            [
              curve(
                "High-error group",
                history.map((x) => [x[0], x[1]]),
              ),
              curve(
                "Overlap group",
                history.map((x) => [x[0], x[2]]),
              ),
            ],
          ),
        ],
        metrics: [
          metric("Repetitions", count),
          metric("Clusters", partition.clusters.length),
          metric("Mean high-error Δradius²", avg(changes[0])),
          metric("Mean overlap Δradius²", avg(changes[2])),
        ],
        message:
          "Each repetition restores identical positions and draws independent WASM donor/clone decisions. Negative Δradius² means inward replacement. Unfit means below the measured swarm mean fitness.",
        done: count >= 160,
      }),
      async () => {
        frame = await frozenStep(
          engine,
          base,
          positions,
          seed + 104729 * (count + 1),
        );
        const fitness = frame.report.pre_clone_fitness.fitness,
          mean = avg(fitness),
          after = pts(frame);
        const high = positions.map((_, i) => partition.high.has(i)),
          unfit = fitness.map((x) => x < mean),
          overlap = high.map((v, i) => v && unfit[i]);
        [high, unfit, overlap].forEach((group, g) => {
          stage[g] += group.filter(Boolean).length;
          group.forEach((yes, i) => {
            if (yes) changes[g].push(norm2(after[i], mu) - radius[i]);
          });
        });
        count++;
        keep(history, [count, avg(changes[0]), avg(changes[2])]);
        if (changes[0].length > 40000)
          changes = changes.map((a) => a.slice(-20000));
      },
    );
  },
);

const drift = descriptor(
  "II-02",
  "Drift is an average over possible next steps",
  "WASM experiment",
  "Which initial spreads have negative average one-step drift?",
  "Averaging independent frozen-state updates reveals the drift; kinetic noise raises the residual floor.",
  "Each point is a real one-step WASM result from the same seeded shape at a selected scale. The affine line is an empirical least-squares fit, and the vertical bars are 95% normal intervals across independent updates.",
  [
    select("stage", "Operator", "clone", [
      { value: "clone", label: "Cloning only" },
      { value: "kinetic", label: "Kinetic only" },
      { value: "full", label: "Full update" },
    ]),
    range(
      "jitter",
      "Clone jitter / kinetic noise amplitude",
      0.05,
      0,
      0.3,
      0.01,
    ),
    select("replicates", "Replicates per spread", 24, [12, 24, 48]),
    select("walkers", "Walkers", 32, [32, 64]),
  ],
  async ({ params: p, seed, engine }) => {
    const base = await configuration(engine, seed, p.walkers);
    base.gas.kinetic.integrator.amplitude = p.stage === "clone" ? 0 : p.jitter;
    if (p.stage !== "kinetic") {
      base.gas.clone_transform = {
        position_field: "positions",
        jitter: structuredClone(base.gas.kinetic.noise),
        jitter_amplitude: p.jitter,
        velocity_field: null,
        restitution: null,
      };
    }
    if (p.stage === "kinetic") base.gas.fitness.reward_exponent = 0;
    const r = random(seed),
      shape = Array.from({ length: p.walkers }, () => [r.normal(), r.normal()]),
      scales = grid(0.1, 2, 10),
      groups = scales.map((s) => ({
        points: shape.map((x) => x.map((v) => s * v)),
        deltas: [],
      }));
    groups.forEach((g) => (g.v = spread(g.points)));
    let count = 0;
    const fit = () => {
      const rows = groups.filter((g) => g.deltas.length),
        x = rows.map((g) => g.v),
        y = rows.map((g) => avg(g.deltas)),
        xm = avg(x),
        ym = avg(y),
        den = x.reduce((s, v) => s + (v - xm) ** 2, 0),
        slope = den
          ? x.reduce((s, v, i) => s + (v - xm) * (y[i] - ym), 0) / den
          : 0;
      return { slope, intercept: ym - slope * xm };
    };
    return model(
      () => {
        const f = fit(),
          means = groups
            .filter((g) => g.deltas.length)
            .map((g) => [g.v, avg(g.deltas)]),
          segments = groups
            .filter((g) => g.deltas.length > 1)
            .map((g) => {
              const m = avg(g.deltas),
                se = 1.96 * Math.sqrt(varOf(g.deltas) / (g.deltas.length - 1));
              return [
                [g.v, m - se],
                [g.v, m + se],
              ];
            });
        return {
          step: count,
          time: count,
          charts: [
            chart(
              "One-step conditional drift",
              "Initial positional variance",
              "Δ variance",
              [
                dots(
                  "Individual updates",
                  groups.flatMap((g) => g.deltas.map((d) => [g.v, d])),
                ),
                curve("Conditional mean", means),
                curve(
                  "Empirical affine fit",
                  groups.map((g) => [g.v, f.slope * g.v + f.intercept]),
                  { dashed: true },
                ),
              ],
              { segments },
            ),
            chart(
              "Ensemble precision",
              "Initial variance",
              "Independent outcomes",
              [
                curve(
                  "Replicates",
                  groups.map((g) => [g.v, g.deltas.length]),
                  { style: "bars" },
                ),
              ],
            ),
          ],
          metrics: [
            metric("Independent updates", count),
            metric("Fitted slope", f.slope),
            metric("Fitted intercept", f.intercept),
            metric(
              "Fitted zero crossing",
              f.slope < 0 ? -f.intercept / f.slope : 0,
            ),
          ],
          message: `${p.stage} operator. Zero crossing is shown as 0 when the fitted slope is nonnegative. All states are alive and unbounded; variance is normalized by N.`,
          done: count >= groups.length * p.replicates,
        };
      },
      async () => {
        const g = groups[count % groups.length];
        if (count >= groups.length * p.replicates) return;
        const f = await frozenStep(
          engine,
          base,
          g.points,
          seed + 7919 * (count + 1),
        );
        g.deltas.push(spread(pts(f)) - g.v);
        count++;
      },
    );
  },
);

const matching = descriptor(
  "II-03",
  "Matching clouds is an optimization problem",
  "Mathematical model",
  "Can relabeling an identical cloud change its transport distance?",
  "Label pairing can cost a lot while optimal matching costs zero. Translation adds exactly the squared barycenter separation.",
  "Exact Hungarian assignment minimizes equal-mass squared cost. The phase-space fixture appends √λ times a synthetic velocity coordinate. The plotted connecting lines are the spatial projection of that assignment.",
  [
    select("walkers", "Equal masses", 8, [4, 8, 16]),
    range("translation", "Cloud translation", 0, 0, 3, 0.1),
    range("distortion", "Shape distortion", 0, 0, 1, 0.05),
    select("coupling", "Display coupling", "optimal", [
      { value: "optimal", label: "Optimal assignment" },
      { value: "labels", label: "Slot labels" },
      { value: "product", label: "Independent product" },
    ]),
    select("cost", "Coordinates", "position", [
      { value: "position", label: "Position" },
      { value: "phase", label: "Position + velocity" },
    ]),
    range("weight", "Velocity weight λ", 1, 0, 3, 0.1),
  ],
  async ({ params: p, seed }) => {
    const r = random(seed),
      a = Array.from({ length: p.walkers }, () => [
        r.normal(),
        r.normal(),
        r.normal(),
      ]);
    let step = 0;
    const state = () => {
      const angle = step * 0.035,
        b = a
          .map((_, i) => a[(i + 3) % a.length])
          .map((x) => [
            x[0] + p.translation,
            p.distortion * Math.sin(x[0] * 2 + angle) +
              x[1] * (1 + p.distortion),
            x[2],
          ]),
        cost = (x) =>
          p.cost === "phase"
            ? [x[0], x[1], Math.sqrt(p.weight) * x[2]]
            : x.slice(0, 2),
        aa = a.map(cost),
        bb = b.map(cost),
        t = transport(aa, bb),
        pairs =
          p.coupling === "labels"
            ? a.map((_, i) => [i, i])
            : p.coupling === "product"
              ? a.flatMap((_, i) => b.map((_, j) => [i, j]))
              : t.pairs.map((j, i) => [i, j]);
      return { b, t, pairs };
    };
    return model(
      () => {
        const { b, t, pairs } = state();
        return {
          step,
          time: step,
          charts: [
            chart(
              "Clouds and chosen coupling",
              "x₁",
              "x₂",
              [
                dots(
                  "Source",
                  a.map((x) => x.slice(0, 2)),
                ),
                dots(
                  "Target",
                  b.map((x) => x.slice(0, 2)),
                ),
              ],
              {
                segments: pairs.map(([i, j]) => [
                  a[i].slice(0, 2),
                  b[j].slice(0, 2),
                ]),
              },
            ),
            {
              title: "Squared assignment costs",
              matrix: t.costs,
              rowLabels: a.map((_, i) => `A${i}`),
              columnLabels: a.map((_, i) => `B${i}`),
            },
          ],
          metrics: [
            metric("Optimal W₂²", t.cost),
            metric("Label pairing cost", t.label),
            metric("Barycenter distance²", t.barycenter),
            metric("Centered W₂²", t.centered),
            metric(
              "Decomposition error",
              Math.abs(t.cost - t.barycenter - t.centered),
            ),
            metric("Product coupling cost", t.proxy + t.barycenter),
          ],
          message:
            "Permutation is fixed at a cyclic shift of three labels. Play changes the distortion phase while preserving the barycenter identity. Hungarian objective uses all selected coordinates.",
          done: step >= 200,
        };
      },
      async () => {
        step++;
      },
    );
  },
);

const proxy = descriptor(
  "II-04",
  "Compare the variance proxy with transport cost",
  "WASM experiment",
  "How tight is the variance upper envelope on actual cloud distance?",
  "Identical broad clouds have zero centered transport and a positive proxy. The centered optimum remains below the sum of variances.",
  "Two real WASM runs evolve a controlled equal-mass fixture; exact assignment is recomputed after every update. Product coupling cost equals the centered variance proxy; slot coupling is a separate feasible cost.",
  [
    select("walkers", "Walkers", 16, [16, 32, 64]),
    range("spread", "Initial spread", 1, 0.2, 2, 0.1),
    range("translation", "Initial translation", 1, 0, 3, 0.1),
    select("fixture", "Second cloud", "translated", [
      { value: "translated", label: "Translated copy" },
      { value: "independent", label: "Independent shape" },
    ]),
    range("noise", "Kinetic jump", 0.02, 0, 0.2, 0.01),
  ],
  async ({ params: p, seed, engine }) => {
    const r = random(seed),
      a = Array.from({ length: p.walkers }, () => [
        p.spread * r.normal(),
        p.spread * r.normal(),
      ]),
      b = a.map((x) =>
        p.fixture === "translated"
          ? [x[0] + p.translation, x[1]]
          : [p.spread * r.normal() + p.translation, p.spread * r.normal()],
      );
    const c = await configuration(engine, seed, p.walkers);
    c.gas.kinetic.integrator.amplitude = p.noise;
    const runs = [];
    try {
      for (let i = 0; i < 2; i++) {
        const d = structuredClone(c);
        d.gas.seed = (seed + i * 104729) >>> 0;
        const run = await engine.create(d);
        runs.push(run);
        await run.set_population(JSON.stringify({ positions: i ? b : a }));
      }
    } catch (e) {
      runs.forEach((g) => g.free());
      throw e;
    }
    let step = 0,
      clouds = [a, b],
      history = [];
    const record = () => {
      const t = transport(...clouds),
        ca = clouds.map((x) => {
          const m = center(x);
          return x.map((v) => v.map((z, j) => z - m[j]));
        });
      keep(history, [
        step,
        t.centered,
        t.proxy,
        avg(ca[0].map((x, i) => norm2(x, ca[1][i]))),
        t.barycenter,
      ]);
      return t;
    };
    let t = record();
    return model(
      () => ({
        step,
        time: step,
        charts: [
          chart(
            "Exact centered transport and variance envelope",
            "Full update",
            "Squared distance",
            [
              curve(
                "Optimal centered W₂²",
                history.map((x) => [x[0], x[1]]),
              ),
              curve(
                "Variance sum / product coupling",
                history.map((x) => [x[0], x[2]]),
              ),
              curve(
                "Centered slot coupling",
                history.map((x) => [x[0], x[3]]),
                { dashed: true },
              ),
            ],
          ),
          chart("Current clouds", "x₁", "x₂", [
            dots("Run A", clouds[0]),
            dots("Run B", clouds[1]),
          ]),
          chart("Center displacement", "Full update", "Barycenter distance²", [
            curve(
              "Centers",
              history.map((x) => [x[0], x[4]]),
            ),
          ]),
        ],
        metrics: [
          metric("Centered optimum", t.centered),
          metric("Variance proxy", t.proxy),
          metric("Envelope gap", t.proxy - t.centered),
        ],
        message:
          "Independent WASM random streams, sphere reward, unbounded domain. Every cloud contains N equal masses; the exact Hungarian solver uses all N points.",
        done: step >= 240,
      }),
      async () => {
        clouds = await Promise.all(runs.map(async (g) => pts(await g.step(1))));
        step++;
        t = record();
      },
      () => runs.forEach((g) => g.free()),
    );
  },
);

const ou = descriptor(
  "II-05",
  "Friction removes memory while noise restores variance",
  "Mathematical model",
  "Can the velocity mean decay while its variance increases?",
  "Mean velocity follows 2e⁻ᵞᵗ and variance approaches T. Under shared innovations, pair separation decays exactly.",
  "Exact discrete sampling of dV=−γVdt+L dB, with no force or cap. Fixed-temperature mode sets L=√(2γT); fixed-diffusion mode sets L=√(2T), so the thermal variance is T/γ.",
  [
    range("gamma", "Friction γ", 1, 0.1, 3, 0.1),
    range("temperature", "Temperature / diffusion parameter", 0.5, 0.1, 1, 0.1),
    select("h", "Timestep", 0.02, [0.005, 0.02, 0.05]),
    select("walkers", "Independent velocities", 256, [128, 256, 512]),
    select("mode", "Noise convention", "temperature", [
      { value: "temperature", label: "Fixed temperature" },
      { value: "diffusion", label: "Fixed diffusion L" },
    ]),
  ],
  async ({ params: p, seed }) => {
    const r = random(seed),
      eq = p.mode === "temperature" ? p.temperature : p.temperature / p.gamma,
      a = Math.exp(-p.gamma * p.h),
      s = Math.sqrt(eq * (1 - a * a));
    let v = Array(p.walkers).fill(2),
      w = Array(p.walkers).fill(-2),
      x = Array(p.walkers).fill(0),
      step = 0,
      h = [];
    const record = () => {
      const t = step * p.h;
      keep(h, [
        t,
        avg(v),
        varOf(v),
        varOf(x),
        avg(v.map((z, i) => Math.abs(z - w[i]))),
      ]);
    };
    record();
    return model(
      () => {
        const t = step * p.h;
        return {
          step,
          time: t,
          charts: [
            chart(
              "Memory of the initial mean",
              "Physical time",
              "Mean velocity",
              [
                curve(
                  "Sample mean",
                  h.map((z) => [z[0], z[1]]),
                ),
                curve(
                  "2 exp(−γt)",
                  h.map((z) => [z[0], 2 * Math.exp(-p.gamma * z[0])]),
                  { dashed: true },
                ),
              ],
            ),
            chart("Variance restored by noise", "Physical time", "Variance", [
              curve(
                "Velocity variance",
                h.map((z) => [z[0], z[2]]),
              ),
              curve(
                "Expected empirical variance",
                h.map((z) => [
                  z[0],
                  (1 - 1 / p.walkers) *
                    eq *
                    (1 - Math.exp(-2 * p.gamma * z[0])),
                ]),
                { dashed: true },
              ),
              curve(
                "Position variance (trapezoid integral)",
                h.map((z) => [z[0], z[3]]),
              ),
            ]),
            chart("Velocity distribution", "v", "Density", [
              curve("Measured", histogram(v, -5, 5)),
              curve(
                "Current Gaussian",
                grid(-5, 5).map((z) => [
                  z,
                  gaussian(
                    z,
                    2 * Math.exp(-p.gamma * t),
                    Math.sqrt(
                      Math.max(1e-6, eq * (1 - Math.exp(-2 * p.gamma * t))),
                    ),
                  ),
                ]),
                { dashed: true },
              ),
            ]),
            chart(
              "Shared-innovation coupling",
              "Physical time",
              "Mean absolute pair difference",
              [
                curve(
                  "Measured |v−w|",
                  h.map((z) => [z[0], z[4]]),
                ),
                curve(
                  "4 exp(−γt)",
                  h.map((z) => [z[0], 4 * Math.exp(-p.gamma * z[0])]),
                  { dashed: true },
                ),
              ],
            ),
          ],
          metrics: [
            metric("Thermal variance", eq),
            metric("Mean velocity", avg(v)),
            metric("Velocity variance", varOf(v)),
            metric("Sample size", p.walkers),
          ],
          message:
            "Every velocity gets an independent Gaussian innovation; paired v,w reuse that same innovation at each step. Position uses the trapezoid integral of sampled velocities.",
          done: step >= 400,
        };
      },
      async () => {
        for (let i = 0; i < v.length; i++) {
          const z = s * r.normal(),
            old = v[i];
          v[i] = a * old + z;
          w[i] = a * w[i] + z;
          x[i] += (p.h * (old + v[i])) / 2;
        }
        step++;
        record();
      },
    );
  },
);

const boundary = descriptor(
  "II-06",
  "Confinement and absorption act differently",
  "Mathematical model",
  "How do return, survival, and phase-space spreading change near a boundary?",
  "The harmonic force pulls the mean inward. One BAOAB noise kick has rank one in phase space; two kicks usually give rank two.",
  "Linear harmonic BAOAB reference, force −κx. Exact matrix propagation computes mean and covariance. A particle is absorbed when a sampled full-step position leaves (−b,b); endpoints are checked after every step.",
  [
    range("stiffness", "Confinement κ", 2, 0, 5, 0.25),
    range("box", "Absorbing half-width b", 1, 0.5, 2, 0.1),
    range("start", "Start as fraction of b", 0.8, 0, 0.98, 0.02),
    select("h", "BAOAB step", 0.1, [0.02, 0.05, 0.1, 0.2]),
    range("temperature", "Temperature", 1, 0.1, 2, 0.1),
    select("steps", "Transition block", 2, [1, 2, 4, 8]),
  ],
  async ({ params: p, seed }) => {
    const r = random(seed),
      { M, g } = linearBAOAB(p.h, 1, p.temperature, p.stiffness),
      starts = grid(0, 0.98 * p.box, 9),
      stats = starts.map((x) => ({ x, total: 0, survive: 0 }));
    let samples = [],
      count = 0,
      C = [
        [0, 0],
        [0, 0],
      ],
      mu = [p.start * p.box, 0];
    for (let k = 0; k < p.steps; k++) {
      C = covarianceStep(C, M, g);
      mu = mv(M, mu);
    }
    const det = C[0][0] * C[1][1] - C[0][1] ** 2,
      rank = det > 1e-12 ? 2 : C[0][0] + C[1][1] > 1e-14 ? 1 : 0;
    function draw(x) {
      let z = [x, 0],
        alive = true;
      for (let k = 0; k < p.steps; k++) {
        const noise = r.normal();
        z = mv(M, z).map((v, j) => v + g[j] * noise);
        if (Math.abs(z[0]) >= p.box) alive = false;
      }
      return { z, alive };
    }
    return model(
      () => ({
        step: count,
        time: p.steps * p.h,
        charts: [
          chart(
            "Transition support from selected start",
            "x",
            "v",
            [
              dots(
                "Surviving endpoints",
                samples.filter((z) => z.alive).map((z) => z.z),
              ),
              dots(
                "Absorbed paths: final endpoint",
                samples.filter((z) => !z.alive).map((z) => z.z),
              ),
              dots("Unkilled analytic mean", [mu]),
            ],
            {
              segments: [
                [
                  [p.box, -4],
                  [p.box, 4],
                ],
                [
                  [-p.box, -4],
                  [-p.box, 4],
                ],
              ],
            },
          ),
          chart(
            "Conditional Monte Carlo survival",
            "Start distance to right boundary",
            "Block survival probability",
            [
              curve(
                "Measured survival",
                stats
                  .filter((z) => z.total)
                  .map((z) => [p.box - z.x, z.survive / z.total])
                  .sort((a, b) => a[0] - b[0]),
              ),
            ],
            { yDomain: [0, 1] },
          ),
          {
            title: "Unkilled phase-space covariance",
            matrix: C,
            rowLabels: ["x", "v"],
            columnLabels: ["x", "v"],
          },
        ],
        metrics: [
          metric("Samples per starting state", count),
          metric("Noise support rank", rank),
          metric("Covariance determinant", det),
          metric("Mean inward displacement", p.start * p.box - mu[0]),
          metric(
            "Mean selected confining energy",
            avg(
              samples.map(
                (s) => 0.5 * p.stiffness * s.z[0] ** 2 + 0.5 * s.z[1] ** 2,
              ),
            ),
          ),
        ],
        message:
          "The covariance matrix describes the unabsorbed linear transition. Survival is measured separately with full-step killing. Plot retains the latest 400 selected-start samples.",
        done: count >= 1200,
      }),
      async () => {
        for (let j = 0; j < 8; j++) {
          keep(samples, draw(p.start * p.box));
          for (const row of stats) {
            row.total++;
            if (draw(row.x).alive) row.survive++;
          }
          count++;
        }
      },
    );
  },
);

const comparison = descriptor(
  "II-07",
  "Assemble component bounds in a comparison matrix",
  "Mathematical model",
  "Can coupled component inequalities produce a scalar contraction?",
  "When r<1, positive left weights turn the affine recurrence into an envelope approaching source/(1−r). Increasing the source raises the floor.",
  "Illustrative nonnegative coefficients, supplied by this fixture. Cloning acts first, so the full matrix is K C and the full source is K b_C+b_K. Perron left weights are computed by power iteration.",
  [
    range("transfer", "Clone x←v coupling", 0.12, 0, 1.5, 0.03),
    range("damping", "Kinetic diagonal damping", 0.6, 0.3, 0.95, 0.05),
    range("source", "Source per component", 0.03, 0, 0.2, 0.01),
  ],
  async ({ params: p }) => {
    const C = [
        [0.65, p.transfer, 0.03, 0.02],
        [0.02, 1.08, 0.02, 0.01],
        [0.01, 0.02, 1.04, 0.02],
        [0.05, 0.01, 0.02, 0.7],
      ],
      K = [
        [0.91, 0.12, 0.03, 0.01],
        [0.03, p.damping, 0.02, 0.01],
        [0.02, 0.05, p.damping, 0.02],
        [0.03, 0.01, 0.02, 0.85],
      ],
      A = mm(K, C),
      b = mv(K, Array(4).fill(p.source)).map((v) => v + p.source),
      { weights: w, r } = perron(A),
      wb = avg(w.map((v, i) => v * b[i])) * 4;
    let x = [2, 1, 1.5, 2.5],
      step = 0,
      envelope = x.reduce((s, v, i) => s + w[i] * v, 0),
      history = [];
    const record = () =>
      keep(history, [
        step,
        x.reduce((s, v, i) => s + w[i] * v, 0),
        envelope,
        ...x,
      ]);
    record();
    return model(
      () => ({
        step,
        time: step,
        charts: [
          {
            title: "Clone comparison C",
            matrix: C,
            rowLabels: ["Vx", "Vv", "Ev", "Wb"],
            columnLabels: ["Vx", "Vv", "Ev", "Wb"],
          },
          {
            title: "Kinetic comparison K",
            matrix: K,
            rowLabels: ["Vx", "Vv", "Ev", "Wb"],
            columnLabels: ["Vx", "Vv", "Ev", "Wb"],
          },
          {
            title: "Full comparison K C",
            matrix: A,
            rowLabels: ["Vx", "Vv", "Ev", "Wb"],
            columnLabels: ["Vx", "Vv", "Ev", "Wb"],
          },
          chart(
            "Weighted recurrence and scalar envelope",
            "Iteration",
            "Weighted value",
            [
              curve(
                "Exact illustrative affine recurrence",
                history.map((z) => [z[0], z[1]]),
              ),
              curve(
                "r y + weighted source",
                history.map((z) => [z[0], z[2]]),
                { dashed: true },
              ),
            ],
          ),
          chart(
            "Component recurrence",
            "Iteration",
            "Component value",
            ["Vx", "Vv", "Ev", "Wb"].map((name, i) =>
              curve(
                name,
                history.map((z) => [z[0], z[i + 3]]),
              ),
            ),
          ),
        ],
        metrics: [
          metric("Computed r", r),
          metric("Weighted source", wb),
          metric(
            "Predicted scalar floor",
            r < 1 ? wb / (1 - r) : "No finite contracting floor",
          ),
        ],
        table: {
          columns: ["Component", "Positive left weight", "(wᵀA)ⱼ", "r wⱼ"],
          rows: w.map((v, j) => [
            ["Vx", "Vv", "Ev", "Wb"][j],
            v,
            A.reduce((s, row, i) => s + w[i] * row[j], 0),
            r * v,
          ]),
        },
        message:
          r < 1
            ? "The coordinate inequalities hold with the displayed positive weights, giving contraction for this illustrative recurrence."
            : "The current comparison matrix has r ≥ 1. Reduce coupling or kinetic diagonal damping to recover a contracting envelope.",
        done: step >= 150 || envelope > 1e8,
      }),
      async () => {
        x = mv(A, x).map((v, i) => v + b[i]);
        envelope = r * envelope + wb;
        step++;
        record();
      },
    );
  },
);

const survival = descriptor(
  "II-08",
  "Survival and conditional equilibrium on separate axes",
  "Mathematical model",
  "What settles down while total surviving probability decreases?",
  "The normalized surviving law approaches the left eigenmeasure q; starting from q gives survival αⁿ and an unchanged conditional shape.",
  "Exact three-state substochastic kernel plus independent Monte Carlo replicas. States label whole-system configurations A/B/C. Cemetery is an additional absorbing outcome; this finite reference is not a one-walker QSD.",
  [
    range("kill", "Killing strength", 0.08, 0.01, 0.3, 0.01),
    select("initial", "Initial law", "A", [
      { value: "A", label: "State A" },
      { value: "C", label: "State C" },
      { value: "qsd", label: "Exact QSD" },
    ]),
    select("boundary", "Boundary convention", "killed", [
      { value: "killed", label: "Absorbing cemetery" },
      { value: "conservative", label: "Conservative" },
    ]),
    select("replicas", "Replicas", 512, [128, 512, 1024]),
  ],
  async ({ params: p, seed }) => {
    const r = random(seed),
      kernel = killedKernel(p.kill, p.boundary === "conservative"),
      q = qsd(kernel);
    let law =
        p.initial === "qsd"
          ? [...q.law]
          : p.initial === "A"
            ? [1, 0, 0]
            : [0, 0, 1],
      states = Array.from({ length: p.replicas }, () => {
        const u = r();
        return u < law[0] ? 0 : u < law[0] + law[1] ? 1 : 2;
      }),
      step = 0,
      history = [];
    const record = () => {
      const mass = law.reduce((a, b) => a + b, 0),
        alive = states.filter((x) => x >= 0),
        counts = [0, 1, 2].map((i) => alive.filter((x) => x === i).length);
      keep(history, [
        step,
        mass,
        alive.length / p.replicas,
        ...law.map((v) => v / Math.max(mass, 1e-300)),
        ...counts.map((v) => v / Math.max(1, alive.length)),
      ]);
    };
    record();
    return model(
      () => {
        const last = history.at(-1),
          alive = states.filter((x) => x >= 0).length;
        return {
          step,
          time: step,
          charts: [
            {
              title: "Whole-system killed kernel",
              matrix: kernel,
              rowLabels: ["A", "B", "C"],
              columnLabels: ["A", "B", "C"],
            },
            chart("Unnormalized survival", "Full block", "Probability", [
              curve(
                "Exact kernel",
                history.map((z) => [z[0], z[1]]),
              ),
              curve(
                "Monte Carlo replicas",
                history.map((z) => [z[0], z[2]]),
              ),
              ...(p.initial === "qsd"
                ? [
                    curve(
                      "αⁿ",
                      history.map((z) => [z[0], q.alpha ** z[0]]),
                      { dashed: true },
                    ),
                  ]
                : []),
            ]),
            chart(
              "Conditional shape among survivors",
              "State index",
              "Conditional probability",
              [
                curve(
                  "Exact conditioned law",
                  [0, 1, 2].map((i) => [i, last[i + 3]]),
                ),
                curve(
                  "Monte Carlo",
                  alive ? [0, 1, 2].map((i) => [i, last[i + 6]]) : [],
                ),
                curve(
                  "QSD eigenmeasure",
                  q.law.map((v, i) => [i, v]),
                  { dashed: true },
                ),
              ],
              { yDomain: [0, 1] },
            ),
            chart(
              "Replica state / cemetery",
              "Replica index",
              "State (−1 cemetery)",
              [
                dots(
                  "Current replicas",
                  states.map((x, i) => [i, x]),
                ),
              ],
            ),
          ],
          metrics: [
            metric("Surviving replicas", alive),
            metric("QSD survival eigenvalue α", q.alpha),
            metric(
              "Eigenmeasure residual",
              Math.max(
                ...mv(
                  kernel[0].map((_, j) => kernel.map((row) => row[j])),
                  q.law,
                ).map((v, i) => Math.abs(v - q.alpha * q.law[i])),
              ),
            ),
            metric(
              "Survival 95% Wilson half-width",
              (1.96 *
                Math.sqrt(
                  (last[2] * (1 - last[2])) / p.replicas +
                    1.96 ** 2 / (4 * p.replicas ** 2),
                )) /
                (1 + 1.96 ** 2 / p.replicas),
            ),
          ],
          message:
            "Conditional estimates use surviving replicas only. The exact kernel continues to show the conditional law even after the finite sample reaches cemetery.",
          done: step >= 160,
        };
      },
      async () => {
        law = kernel[0].map((_, j) =>
          law.reduce((s, v, i) => s + v * kernel[i][j], 0),
        );
        states = states.map((i) => {
          if (i < 0) return -1;
          const u = r();
          let c = 0;
          for (let j = 0; j < 3; j++) {
            c += kernel[i][j];
            if (u < c) return j;
          }
          return -1;
        });
        step++;
        record();
      },
    );
  },
);

const massLedger = descriptor(
  "III-01",
  "A population mass ledger for the forward equation",
  "Mathematical model",
  "Where does alive mass go during killing, revival, and copying?",
  "Internal copying cancels from the total mass equation. Killing and escaping offspring transfer alive mass to the dead reservoir; revival transfers it back.",
  "Exact constant-rate two-reservoir reference: m′=λ(1−m)−(c+a e)m. The integral of m is computed analytically to show cumulative fluxes with exact conservation.",
  [
    range("kill", "Killing rate c", 0.5, 0, 2, 0.1),
    range("revive", "Revival rate λ", 1, 0.1, 3, 0.1),
    range("attempt", "Internal replacement rate a", 1, 0, 5, 0.25),
    range("escape", "Offspring escape probability e", 0, 0, 0.3, 0.02),
    range("initial", "Initial alive mass", 0.3, 0.1, 1, 0.05),
  ],
  async ({ params: p }) => {
    let step = 0,
      history = [];
    const death = p.kill + p.attempt * p.escape,
      total = death + p.revive,
      eq = p.revive / total;
    const values = () => {
      const t = step * 0.05,
        m = reservoir(t, death, p.revive, p.initial),
        integral =
          eq * t + ((p.initial - eq) * -Math.expm1(-total * t)) / total;
      return {
        t,
        m,
        killed: p.kill * integral,
        escaped: p.attempt * p.escape * integral,
        revived: p.revive * (t - integral),
        internal: p.attempt * (1 - p.escape) * integral,
      };
    };
    const record = () => {
      const v = values();
      keep(history, [v.t, v.m, 1 - v.m, v.killed, v.escaped, v.revived]);
    };
    record();
    return model(
      () => {
        const v = values();
        return {
          step,
          time: v.t,
          charts: [
            chart(
              "Alive and dead reservoirs",
              "Physical time",
              "Mass",
              [
                curve(
                  "Alive mₐ",
                  history.map((x) => [x[0], x[1]]),
                ),
                curve(
                  "Dead m_d",
                  history.map((x) => [x[0], x[2]]),
                ),
                curve(
                  "Total",
                  history.map((x) => [x[0], x[1] + x[2]]),
                  { dashed: true },
                ),
                curve(
                  "Equilibrium alive mass",
                  history.map((x) => [x[0], eq]),
                  { dashed: true },
                ),
              ],
              { yDomain: [0, 1.05] },
            ),
            chart(
              "Integrated transfer ledger",
              "Physical time",
              "Cumulative transferred mass",
              [
                curve(
                  "Killing: alive → dead",
                  history.map((x) => [x[0], x[3]]),
                ),
                curve(
                  "Offspring escape: alive → dead",
                  history.map((x) => [x[0], x[4]]),
                ),
                curve(
                  "Revival: dead → alive",
                  history.map((x) => [x[0], x[5]]),
                ),
              ],
            ),
            chart("Spatial reference with fixed shape", "x", "Density", [
              curve(
                "Alive density f=mₐρ",
                grid(-3, 3).map((x) => [x, v.m * gaussian(x)]),
              ),
              curve(
                "Normalized shape ρ",
                grid(-3, 3).map((x) => [x, gaussian(x)]),
                { dashed: true },
              ),
            ]),
          ],
          metrics: [
            metric("Alive mass", v.m),
            metric("Dead mass", 1 - v.m),
            metric(
              "Ledger residual",
              v.m - (p.initial - v.killed - v.escaped + v.revived),
            ),
            metric("Mass-neutral internal replacements", v.internal),
            metric("Equilibrium mₐ", eq),
          ],
          table: {
            columns: ["Transfer", "Instantaneous rate", "Integrated mass"],
            rows: [
              ["Killing", p.kill * v.m, v.killed],
              ["Revival", p.revive * (1 - v.m), v.revived],
              ["Offspring escape", p.attempt * p.escape * v.m, v.escaped],
              [
                "Internal accepted copy (net zero)",
                p.attempt * (1 - p.escape) * v.m,
                v.internal,
              ],
            ],
          },
          message:
            "Finite-rate continuous reservoir; the spatial inset uses a fixed standard-normal shape to isolate mass accounting.",
          done: step >= 300,
        };
      },
      async () => {
        step++;
        record();
      },
    );
  },
);

const reaction = descriptor(
  "III-02",
  "Shrinking the timestep changes reaction frequency",
  "Mathematical model",
  "Does the same per-step probability describe the same physical reaction rate?",
  "Fixed probability produces rate p/h. The finite-rate probability 1−exp(−a h) gives an event rate approaching a as h decreases.",
  "Independent Bernoulli reference timelines cover T=5. Events flip a binary observable, whose exact expectation is ½[1−(1−2q)ⁿ]. Repetitions estimate event rates with uncertainty.",
  [
    range("probability", "Fixed probability p", 0.3, 0.1, 0.6, 0.1),
    range("rate", "Finite rate a", 2, 0.5, 5, 0.5),
    select("h", "Timeline timestep", 0.025, [0.1, 0.05, 0.025, 0.0125]),
  ],
  async ({ params: p, seed }) => {
    const r = random(seed),
      hs = [0.1, 0.05, 0.025, 0.0125],
      rows = hs.map((h) => ({ h, fixed: [], finite: [] }));
    let step = 0,
      events = [[], []],
      paths = [[], []];
    return model(
      () => ({
        step,
        time: 5,
        charts: [
          chart(
            "Refinement and observed reaction intensity",
            "h",
            "Events per physical time",
            [
              curve(
                "Measured fixed p",
                rows
                  .filter((z) => z.fixed.length)
                  .map((z) => [z.h, avg(z.fixed)]),
              ),
              curve(
                "p/h",
                rows.map((z) => [z.h, p.probability / z.h]),
                { dashed: true },
              ),
              curve(
                "Measured finite rate",
                rows
                  .filter((z) => z.finite.length)
                  .map((z) => [z.h, avg(z.finite)]),
              ),
              curve(
                "(1−exp(−a h))/h",
                rows.map((z) => [z.h, -Math.expm1(-p.rate * z.h) / z.h]),
                { dashed: true },
              ),
            ],
            { xScale: "log" },
          ),
          chart(
            "Most recent coupled-duration timelines",
            "Physical time",
            "Rule (0 fixed p, 1 finite rate)",
            [
              dots(
                "Fixed-p events",
                events[0].map((t) => [t, 0]),
              ),
              dots(
                "Finite-rate events",
                events[1].map((t) => [t, 1]),
              ),
            ],
          ),
          chart(
            "Binary observable after event flips",
            "Physical time",
            "State / expectation",
            [
              curve("Fixed p sample", paths[0]),
              curve("Finite rate sample", paths[1]),
              curve(
                "Fixed p exact expectation",
                grid(0, 5, 201).map((t) => [
                  t,
                  0.5 * (1 - (1 - 2 * p.probability) ** Math.floor(t / p.h)),
                ]),
                { dashed: true },
              ),
              curve(
                "Finite-rate exact expectation",
                grid(0, 5, 201).map((t) => [
                  t,
                  0.5 -
                    0.5 *
                      (1 - 2 * -Math.expm1(-p.rate * p.h)) **
                        Math.floor(t / p.h),
                ]),
                { dashed: true },
              ),
            ],
          ),
        ],
        metrics: [
          metric("Independent repetitions", step),
          metric("Physical duration", 5),
          metric("Selected step count", Math.round(5 / p.h)),
          metric("Limiting finite intensity", p.rate),
        ],
        table: {
          columns: ["h", "Fixed mean ± 1.96 SE", "Finite mean ± 1.96 SE"],
          rows: rows.map((z) => [
            z.h,
            `${avg(z.fixed).toFixed(3)} ± ${(1.96 * Math.sqrt(varOf(z.fixed) / Math.max(1, z.fixed.length - 1))).toFixed(3)}`,
            `${avg(z.finite).toFixed(3)} ± ${(1.96 * Math.sqrt(varOf(z.finite) / Math.max(1, z.finite.length - 1))).toFixed(3)}`,
          ]),
        },
        message:
          "Accepted events are counted per physical time. Each row uses independent repeated Bernoulli experiments; the two plotted timelines have the same physical duration.",
        done: step >= 200,
      }),
      async () => {
        for (const row of rows) {
          const counts = [0, 0],
            e = [[], []],
            path = [[[0, 0]], [[0, 0]]],
            qs = [p.probability, -Math.expm1(-p.rate * row.h)];
          for (let i = 1; i <= Math.round(5 / row.h); i++)
            for (let j = 0; j < 2; j++) {
              if (r() < qs[j]) {
                counts[j]++;
                e[j].push(i * row.h);
              }
              if (i % Math.max(1, Math.ceil(5 / row.h / 390)) === 0)
                path[j].push([i * row.h, counts[j] % 2]);
            }
          keep(row.fixed, counts[0] / 5);
          keep(row.finite, counts[1] / 5);
          if (row.h === p.h) {
            events = e;
            paths = path;
          }
        }
        step++;
      },
    );
  },
);

const chaos = descriptor(
  "III-03",
  "Watch the empirical law fluctuate between runs",
  "WASM experiment",
  "How do whole-cloud fluctuations change as N increases?",
  "Independent initial populations give concentrating empirical averages; a shared random initial displacement introduces visible cloud-to-cloud fluctuations.",
  "Independent WASM runs measure φ(x)=tanh(x₁) after a fixed number of full updates. The N⁻¹ comparison uses initial independent Monte Carlo variance; measured correlations include the actual copying dynamics.",
  [
    select("updates", "Observation update", 4, [1, 4, 12]),
    select("initial", "Initial population", "independent", [
      { value: "independent", label: "Independent Gaussian" },
      { value: "shared", label: "Shared random displacement" },
    ]),
    range("selection", "Reward exponent", 0, 0, 2, 0.25),
    range("noise", "Kinetic jump amplitude", 0.05, 0, 0.2, 0.01),
    select("replicas", "Replicas per N", 24, [12, 24, 48]),
  ],
  async ({ params: p, seed, engine }) => {
    const r = random(seed),
      sizes = [16, 32, 64, 128],
      rows = sizes.map((n) => ({ n, means: [], pairs: [] })),
      base = await configuration(engine, seed, 16);
    base.gas.fitness.reward_exponent = p.selection;
    base.gas.kinetic.integrator.amplitude = p.noise;
    let step = 0,
      cloud = [];
    const reference = Array.from({ length: 20000 }, () =>
        Math.tanh(r.normal()),
      ),
      sigma = varOf(reference);
    return model(
      () => {
        const last = rows.at(-1),
          bins = 6,
          joint = Array.from({ length: bins }, () => Array(bins).fill(0));
        for (const [a, b] of last.pairs) {
          const i = Math.min(bins - 1, Math.floor(((a + 1) * bins) / 2)),
            j = Math.min(bins - 1, Math.floor(((b + 1) * bins) / 2));
          joint[i][j]++;
        }
        const count = Math.max(1, last.pairs.length),
          mx = joint.map((row) => row.reduce((a, b) => a + b, 0) / count),
          my = joint[0].map(
            (_, j) => joint.reduce((s, row) => s + row[j], 0) / count,
          ),
          difference = joint.map((row, i) =>
            row.map((v, j) => v / count - mx[i] * my[j]),
          );
        return {
          step,
          time: p.updates,
          charts: [
            chart(
              "Across-run variance of empirical observable",
              "N",
              "Var[L_N tanh(x₁)]",
              [
                curve(
                  "Measured across independent runs",
                  rows
                    .filter((z) => z.means.length > 1)
                    .map((z) => [
                      z.n,
                      (varOf(z.means) * z.means.length) / (z.means.length - 1),
                    ]),
                ),
                curve(
                  "Independent initial σ²/N",
                  sizes.map((n) => [n, sigma / n]),
                  { dashed: true },
                ),
              ],
              { xScale: "log", yScale: "log" },
            ),
            chart(
              "Distribution of whole-cloud averages",
              "L_N φ",
              "Density",
              rows.map((z) => curve(`N=${z.n}`, histogram(z.means, -1, 1, 16))),
            ),
            chart("Latest real engine cloud", "x₁", "x₂", [
              dots("Walkers", cloud),
            ]),
            {
              title: "N=128 two-label joint minus marginal product",
              matrix: difference,
              rowLabels: grid(-1, 1, bins).map((x) => x.toFixed(1)),
              columnLabels: grid(-1, 1, bins).map((x) => x.toFixed(1)),
            },
          ],
          metrics: [
            metric("Independent completed runs", step),
            metric("N=128 replicas", last.means.length),
            metric(
              "N=128 joint-product L¹",
              difference.flat().reduce((s, v) => s + Math.abs(v), 0),
            ),
          ],
          message:
            "Uncertainty is across entire runs. Two labels are fixed slots 0 and 1; sparse joint histograms improve as more independent replicas finish. Observation time is in discrete full updates.",
          done: step >= sizes.length * p.replicas,
        };
      },
      async () => {
        if (step >= sizes.length * p.replicas) return;
        const row = rows[step % rows.length],
          config = structuredClone(base);
        config.walkers = row.n;
        config.gas.seed = (seed + 104729 * (step + 1)) >>> 0;
        const shared = p.initial === "shared" ? 1.5 * r.normal() : 0,
          positions = Array.from({ length: row.n }, () => [
            r.normal() + shared,
            r.normal(),
          ]);
        const gas = await engine.create(config);
        try {
          await gas.set_population(JSON.stringify({ positions }));
          cloud = pts(await gas.step(p.updates));
          const values = cloud.map((x) => Math.tanh(x[0]));
          row.means.push(avg(values));
          row.pairs.push(values.slice(0, 2));
        } finally {
          gas.free();
        }
        step++;
      },
    );
  },
);

const limits = descriptor(
  "III-04",
  "Two limits and a stationary residual",
  "Mathematical model",
  "Does a stationary numerical law annihilate the continuous generator?",
  "Population growth reduces sampling noise. Timestep refinement reduces the Euler stationary bias; the exact discrete and continuous residuals differ by an O(h) term.",
  "Matched independent OU reference dX=−Xdt+√(2T)dB, Euler X′=(1−h)X+√(2Th)Z. For φ=x², (P_hφ−φ)/h=(−2+h)x²+2T while Lφ=−2x²+2T.",
  [
    range("temperature", "OU temperature T", 1, 0.2, 2, 0.2),
    select("initial", "Initial law", "stationary", [
      { value: "stationary", label: "Euler stationary Gaussian" },
      { value: "point", label: "Point at x=2" },
    ]),
    select("test", "Test function", "square", [
      { value: "square", label: "x²" },
      { value: "linear", label: "x" },
      { value: "constant", label: "1" },
    ]),
  ],
  async ({ params: p, seed }) => {
    const r = random(seed),
      sizes = [64, 128, 256],
      hs = [0.2, 0.1, 0.05],
      cells = sizes.map((n) =>
        hs.map((h) => ({
          n,
          h,
          x: Array.from({ length: n }, () =>
            p.initial === "stationary"
              ? Math.sqrt(p.temperature / (1 - h / 2)) * r.normal()
              : 2,
          ),
        })),
      );
    let step = 0,
      history = [];
    const residual = (c, continuous) =>
      p.test === "constant"
        ? 0
        : p.test === "linear"
          ? -avg(c.x)
          : (-2 + (continuous ? 0 : c.h)) * avg(c.x.map((x) => x * x)) +
            2 * p.temperature;
    const observe = (c) =>
      p.test === "constant"
        ? 1
        : p.test === "linear"
          ? avg(c.x)
          : avg(c.x.map((x) => x * x));
    const record = () =>
      keep(history, [
        step * 0.2,
        ...cells.at(-1).flatMap((c) => [residual(c, false), residual(c, true)]),
      ]);
    record();
    return model(
      () => ({
        step,
        time: step * 0.2,
        charts: [
          {
            title: "Finite-step residual E[(P_hφ−φ)/h]",
            matrix: cells.map((row) => row.map((c) => residual(c, false))),
            rowLabels: sizes.map((n) => `N=${n}`),
            columnLabels: hs.map((h) => `h=${h}`),
          },
          {
            title: "Continuous generator residual E[Lφ]",
            matrix: cells.map((row) => row.map((c) => residual(c, true))),
            rowLabels: sizes.map((n) => `N=${n}`),
            columnLabels: hs.map((h) => `h=${h}`),
          },
          chart(
            "N=256 residuals across time",
            "Physical time",
            "Weak residual",
            hs.flatMap((h, j) => [
              curve(
                `Discrete h=${h}`,
                history.map((z) => [z[0], z[1 + 2 * j]]),
              ),
              curve(
                `Continuous h=${h}`,
                history.map((z) => [z[0], z[2 + 2 * j]]),
                { dashed: true },
              ),
            ]),
          ),
        ],
        metrics: [
          metric("Physical time", step * 0.2),
          metric("Conservative survival", 1),
          metric("Analytic discrete stationary residual", 0),
          metric(
            "Continuum stationary bias at h=.2",
            p.test === "square" ? (-p.temperature * 0.2) / (1 - 0.2 / 2) : 0,
          ),
        ],
        table: {
          columns: [
            "N",
            "h",
            "Empirical φ",
            "Discrete residual",
            "Continuous residual",
          ],
          rows: cells
            .flat()
            .map((c) => [
              c.n,
              c.h,
              observe(c),
              residual(c, false),
              residual(c, true),
            ]),
        },
        message:
          "All cells advance the same physical duration. Matrix rows change N and columns change h. Residuals use exact conditional expectations of the selected operator, averaged over the simulated population.",
        done: step >= 150,
      }),
      async () => {
        for (const c of cells.flat())
          for (let k = 0; k < Math.round(0.2 / c.h); k++)
            c.x = c.x.map(
              (x) =>
                (1 - c.h) * x + Math.sqrt(2 * p.temperature * c.h) * r.normal(),
            );
        step++;
        record();
      },
    );
  },
);

const equilibrium = descriptor(
  "III-05",
  "Watch an equilibrium profile emerge",
  "Mathematical model",
  "Why should fitness become constant on the stationary support?",
  "The positive local-replicator density approaches ρ* proportional to R^(αD/β), and its fitness becomes spatially constant.",
  "Positive periodic reward on the bounded interval [−π,π]. A conservative explicit quadrature evolution solves ∂tρ=ρ(F−∫ρF), F=R^α ρ^(−β/D). Adaptive timesteps preserve positivity; the reference uses the same grid normalization.",
  [
    range("ratio", "α / β", 1, 0.25, 2, 0.25),
    select("dimension", "Reference dimension D", 1, [1, 2]),
    range("contrast", "Reward contrast", 3, 1, 10, 0.5),
    select("initial", "Positive initial density", "uniform", [
      { value: "uniform", label: "Uniform" },
      { value: "tilted", label: "Tilted toward low reward" },
    ]),
    select("resolution", "Quadrature cells", 96, [48, 96, 192]),
  ],
  async ({ params: p }) => {
    const n = p.resolution,
      dx = (2 * Math.PI) / n,
      x = Array.from({ length: n }, (_, i) => -Math.PI + (i + 0.5) * dx),
      R = x.map((z) =>
        Math.exp((Math.log(p.contrast) * (1 + Math.cos(z))) / 2),
      ),
      normalize = (a) => {
        const sum = a.reduce((s, v) => s + v, 0) * dx;
        return a.map((v) => v / sum);
      },
      target = normalize(R.map((v) => v ** (p.ratio * p.dimension)));
    let rho = normalize(
        x.map((z) => (p.initial === "uniform" ? 1 : Math.exp(-Math.cos(z)))),
      ),
      time = 0,
      step = 0,
      history = [];
    const fitness = () =>
      R.map((v, i) => v ** p.ratio / rho[i] ** (1 / p.dimension));
    const measures = () => {
      const F = fitness(),
        m = F.reduce((s, v, i) => s + v * rho[i] * dx, 0),
        res = rho.map((v, i) => v * (F[i] - m)),
        error = rho.reduce((s, v, i) => s + Math.abs(v - target[i]) * dx, 0);
      return { F, m, res, error };
    };
    const record = () => {
      const m = measures();
      keep(history, [
        time,
        m.error,
        Math.sqrt(m.res.reduce((s, v) => s + v * v * dx, 0)),
      ]);
    };
    record();
    return model(
      () => {
        const m = measures();
        return {
          step,
          time,
          charts: [
            chart("Local equilibrium density", "x", "Density", [
              curve(
                "Evolving density",
                x.map((z, i) => [z, rho[i]]),
              ),
              curve(
                "R^(αD/β) normalized",
                x.map((z, i) => [z, target[i]]),
                { dashed: true },
              ),
            ]),
            chart("Positive reward and local fitness", "x", "Value", [
              curve(
                "Reward R",
                x.map((z, i) => [z, R[i]]),
              ),
              curve(
                "Fitness F",
                x.map((z, i) => [z, m.F[i]]),
              ),
              curve(
                "Mean fitness",
                x.map((z) => [z, m.m]),
                { dashed: true },
              ),
            ]),
            chart("Stationary replicator residual", "x", "ρ(F−mean F)", [
              curve(
                "Current residual",
                x.map((z, i) => [z, m.res[i]]),
              ),
            ]),
            chart(
              "Convergence to predicted balance",
              "Reference time",
              "Error",
              [
                curve(
                  "L¹ density error",
                  history.map((z) => [z[0], z[1]]),
                ),
                curve(
                  "Residual L² norm",
                  history.map((z) => [z[0], z[2]]),
                ),
              ],
            ),
          ],
          metrics: [
            metric(
              "Total quadrature mass",
              rho.reduce((s, v) => s + v * dx, 0),
            ),
            metric("L¹ profile error", m.error),
            metric("Fitness range", Math.max(...m.F) - Math.min(...m.F)),
            metric("Grid cells", n),
          ],
          message:
            "β=1; α is the selected ratio. This is the local replicator reference, with no transport. Change grid resolution to inspect quadrature sensitivity.",
          done: step >= 300,
        };
      },
      async () => {
        for (let k = 0; k < 4; k++) {
          const { F, m } = measures(),
            dt = Math.min(0.02, 0.2 / Math.max(...F));
          rho = normalize(rho.map((v, i) => v * (1 + dt * (F[i] - m))));
          time += dt;
        }
        step++;
        record();
      },
    );
  },
);

const diffusion = descriptor(
  "III-06",
  "A sine, a parabola, and a surviving shape",
  "Mathematical model",
  "Why do killed diffusion and source-balanced diffusion approach different shapes?",
  "The source-free conditional density approaches the principal sine. A constant source creates the stationary parabola s x(L−x)/(2D₀).",
  "Dirichlet heat equation on (0,L), solved by exactly evolved sine modes. The initial density is sin(πx/L)+.35sin(2πx/L)+.2sin(3πx/L). Constant source has sine coefficients 4s/(nπ) for odd n.",
  [
    select("length", "Interval length L", 1, [1, 2, 4]),
    range("diffusivity", "Diffusivity D₀", 0.5, 0.1, 1, 0.1),
    range("source", "Constant source s", 0.5, 0, 1, 0.1),
    select("modes", "Sine modes", 31, [15, 31, 63]),
  ],
  async ({ params: p }) => {
    const L = p.length,
      D = p.diffusivity,
      n = p.modes,
      x = grid(0, L, 121),
      modes = Array.from({ length: n }, (_, i) => i + 1),
      initial = (k) => (k === 1 ? 1 : k === 2 ? 0.35 : k === 3 ? 0.2 : 0),
      rate = (k) => D * ((k * Math.PI) / L) ** 2;
    let step = 0,
      time = 0,
      history = [];
    const coefficients = (source) =>
      modes.map(
        (k) =>
          initial(k) * Math.exp(-rate(k) * time) +
          ((k % 2 ? (4 * source) / (k * Math.PI) : 0) / rate(k)) *
            -Math.expm1(-rate(k) * time),
      );
    const profile = (c) =>
      x.map((z) =>
        c.reduce((s, v, j) => s + v * Math.sin(((j + 1) * Math.PI * z) / L), 0),
      );
    const mass = (c) =>
      c.reduce(
        (s, v, j) => s + ((j + 1) % 2 ? (2 * L * v) / ((j + 1) * Math.PI) : 0),
        0,
      );
    const state = () => {
      const a = coefficients(0),
        b = coefficients(p.source),
        killed = profile(a),
        fed = profile(b),
        m = mass(a),
        target = x.map(
          (z) => (Math.PI / (2 * L)) * Math.sin((Math.PI * z) / L),
        ),
        conditional = killed.map((v) => v / Math.max(m, 1e-300));
      return {
        killed,
        fed,
        m,
        fedMass: mass(b),
        target,
        conditional,
        error: conditional.reduce(
          (s, v, i) => s + (Math.abs(v - target[i]) * L) / (x.length - 1),
          0,
        ),
      };
    };
    const record = () => {
      const z = state();
      keep(history, [time, z.m, z.fedMass, z.error]);
    };
    record();
    return model(
      () => {
        const z = state();
        return {
          step,
          time,
          charts: [
            chart("Source-free absorbing diffusion", "x/L", "Density", [
              curve(
                "Unnormalized killed density",
                x.map((v, i) => [v / L, z.killed[i]]),
              ),
              curve(
                "Normalized surviving shape",
                x.map((v, i) => [v / L, z.conditional[i]]),
              ),
              curve(
                "Principal sine normalized",
                x.map((v, i) => [v / L, z.target[i]]),
                { dashed: true },
              ),
            ]),
            chart("Constant-source balance", "x/L", "Density", [
              curve(
                "Source-fed solution",
                x.map((v, i) => [v / L, z.fed[i]]),
              ),
              curve(
                "Stationary parabola",
                x.map((v) => [v / L, (p.source * v * (L - v)) / (2 * D)]),
                { dashed: true },
              ),
            ]),
            chart(
              "Mass distinguishes the equations",
              "Physical time",
              "Total mass",
              [
                curve(
                  "Killed mass",
                  history.map((v) => [v[0], v[1]]),
                ),
                curve(
                  "Source-fed mass",
                  history.map((v) => [v[0], v[2]]),
                ),
              ],
            ),
            chart(
              "Conditional profile convergence",
              "Physical time",
              "L¹ distance to sine",
              [
                curve(
                  "Conditional error",
                  history.map((v) => [v[0], v[3]]),
                ),
              ],
            ),
          ],
          metrics: [
            metric("Principal decay rate Dπ²/L²", rate(1)),
            metric("Source-free mass", z.m),
            metric("Source-fed mass", z.fedMass),
            metric("Parabola mass sL³/(12D)", (p.source * L ** 3) / (12 * D)),
            metric("Conditional L¹ error", z.error),
          ],
          message: `Exactly evolved ${n}-mode Dirichlet heat reference. Initial modes are represented exactly; changing mode count refines the constant-source expansion.`,
          done: step >= 240,
        };
      },
      async () => {
        time += (0.01 * L * L) / D;
        step++;
        record();
      },
    );
  },
);

const exchangeability = descriptor(
  "III-07",
  "Exchangeable walkers can move as one random object",
  "Mathematical model",
  "Does permutation symmetry force the empirical mean to concentrate?",
  "Both ensembles are exchangeable with N(0,1) marginals. Independent means have variance 1/N; shared-state means have variance ρ+(1−ρ)/N.",
  "Generate Xᵢ=√ρ Z+√(1−ρ)εᵢ with independent standard normals. A fresh uniform permutation changes slot labels; it leaves the empirical mean exactly unchanged.",
  [
    select("walkers", "Walkers N", 64, [4, 16, 64, 256]),
    range("correlation", "Shared-state strength ρ", 1, 0, 1, 0.1),
    select("relabel", "Label order", "permuted", [
      { value: "permuted", label: "Random permutation" },
      { value: "original", label: "Original" },
    ]),
  ],
  async ({ params: p, seed }) => {
    const r = random(seed);
    let step = 0,
      independent = [],
      shared = [],
      pairs = [],
      marginals = [],
      permutationError = 0,
      current = [];
    return model(
      () => ({
        step,
        time: step,
        charts: [
          chart(
            "Distribution of the empirical mean",
            "Mean over N walkers",
            "Density",
            [
              curve("Independent samples", histogram(independent, -3, 3, 36)),
              curve("Shared-state model", histogram(shared, -3, 3, 36)),
              curve(
                "Independent reference Gaussian",
                grid(-3, 3, 121).map((x) => [
                  x,
                  gaussian(x, 0, 1 / Math.sqrt(p.walkers)),
                ]),
                { dashed: true },
              ),
              curve(
                "Shared reference Gaussian",
                grid(-3, 3, 121).map((x) => [
                  x,
                  gaussian(
                    x,
                    0,
                    Math.sqrt(p.correlation + (1 - p.correlation) / p.walkers),
                  ),
                ]),
                { dashed: true },
              ),
            ],
          ),
          chart("Two fixed labels across independent runs", "X₁", "X₂", [
            dots("Joint samples", pairs),
          ]),
          chart("One-label marginal", "x", "Density", [
            curve("Sampled X₁", histogram(marginals, -4, 4)),
            curve(
              "N(0,1)",
              grid(-4, 4).map((x) => [x, gaussian(x)]),
              { dashed: true },
            ),
          ]),
          chart("Current labeled population", "Slot label", "Position", [
            dots(
              "Population",
              current.map((x, i) => [i, x]),
            ),
          ]),
        ],
        metrics: [
          metric("Whole-population repetitions", step),
          metric("Independent mean variance", varOf(independent)),
          metric("Shared mean variance", varOf(shared)),
          metric(
            "Predicted shared variance",
            p.correlation + (1 - p.correlation) / p.walkers,
          ),
          metric("Relabeling mean error", permutationError),
        ],
        message:
          "Histories retain 400 independent population draws. ρ=1 assigns the same random value to every walker; increasing N then leaves empirical-mean fluctuations intact.",
        done: step >= 400,
      }),
      async () => {
        for (let k = 0; k < 8; k++) {
          const z = r.normal(),
            a = Array.from({ length: p.walkers }, () => r.normal()),
            b = Array.from(
              { length: p.walkers },
              () =>
                Math.sqrt(p.correlation) * z +
                Math.sqrt(1 - p.correlation) * r.normal(),
            ),
            before = avg(b);
          if (p.relabel === "permuted")
            for (let j = b.length - 1; j > 0; j--) {
              const i = Math.floor(r() * (j + 1));
              [b[i], b[j]] = [b[j], b[i]];
            }
          permutationError = Math.max(
            permutationError,
            Math.abs(avg(b) - before),
          );
          keep(independent, avg(a));
          keep(shared, avg(b));
          keep(pairs, b.slice(0, 2));
          keep(marginals, b[0]);
          current = b;
          step++;
        }
      },
    );
  },
);

const finiteCorrection = descriptor(
  "III-08",
  "Sampling with replacement exposes the finite-N correction",
  "Mathematical model",
  "How often does a sampled tuple repeat a population label?",
  "Collision frequency agrees with 1−(N)ₖ/Nᵏ and stays below k(k−1)/(2N). Fixed k improves with N; larger tuples collide more often.",
  "Ordered samples are coupled by keeping each fresh with-replacement draw if unused, otherwise drawing uniformly from the remaining labels for the without-replacement sample. Covariance uses a bounded ±1 shared-state mixture; entropy uses B=1 and the chapter’s exact 4(H_N+½log2)/N bound.",
  [
    select(
      "walkers",
      "Population labels N",
      64,
      [8, 16, 32, 64, 128, 256, 512],
    ),
    range("tuple", "Tuple length k (capped at N)", 8, 1, 32, 1),
    range("correlation", "Off-diagonal correlation ρ", 0, 0, 1, 0.1),
    select("entropy", "Total entropy budget H_N", "constant", [
      { value: "constant", label: "1" },
      { value: "sqrt", label: "√N" },
      { value: "linear", label: "N" },
    ]),
  ],
  async ({ params: p, seed }) => {
    const r = random(seed),
      n = p.walkers,
      k = Math.min(n, p.tuple),
      maxK = Math.min(n, 32),
      theory = collision(n, k);
    let step = 0,
      hits = 0,
      replacement = [],
      distinct = [],
      history = [];
    const H = (N) =>
      p.entropy === "constant" ? 1 : p.entropy === "sqrt" ? Math.sqrt(N) : N;
    return model(
      () => ({
        step,
        time: step,
        charts: [
          chart(
            "Tuple collision correction",
            "Tuple length k",
            "Probability",
            [
              curve(
                "Exact 1−(N)ₖ/Nᵏ",
                Array.from({ length: maxK }, (_, i) => [
                  i + 1,
                  collision(n, i + 1),
                ]),
              ),
              curve(
                "Union bound",
                Array.from({ length: maxK }, (_, i) => [
                  i + 1,
                  Math.min(1, (i * (i + 1)) / (2 * n)),
                ]),
                { dashed: true },
              ),
              dots("Measured selected k", step ? [[k, hits / step]] : []),
            ],
            { yDomain: [0, 1] },
          ),
          chart(
            "Collision estimate across trials",
            "Independent tuple trials",
            "Collision probability",
            [
              curve("Measured", history),
              curve(
                "Exact selected correction",
                history.map((z) => [z[0], theory]),
                { dashed: true },
              ),
            ],
          ),
          chart("Coupled ordered samples", "Draw index", "Population label", [
            dots(
              "With replacement",
              replacement.map((v, i) => [i + 1, v]),
            ),
            dots(
              "Without replacement",
              distinct.map((v, i) => [i + 1, v]),
            ),
          ]),
          chart(
            "Covariance and entropy scaling",
            "Population N",
            "Variance / upper bound",
            [
              curve(
                "Diagonal 1/N",
                grid(8, 512, 64).map((N) => [N, 1 / N]),
              ),
              curve(
                "Off-diagonal (1−1/N)ρ",
                grid(8, 512, 64).map((N) => [N, (1 - 1 / N) * p.correlation]),
              ),
              curve(
                "Bounded mixture mean variance",
                grid(8, 512, 64).map((N) => [
                  N,
                  1 / N + (1 - 1 / N) * p.correlation,
                ]),
              ),
              curve(
                "Entropy concentration bound B=1",
                grid(8, 512, 64).map((N) => [
                  N,
                  (4 * (H(N) + 0.5 * Math.log(2))) / N,
                ]),
                { dashed: true },
              ),
            ],
            { xScale: "log" },
          ),
        ],
        metrics: [
          metric("Exact collision probability", theory),
          metric("Observed collision frequency", step ? hits / step : 0),
          metric("Union upper bound", Math.min(1, (k * (k - 1)) / (2 * n))),
          metric(
            "Monte Carlo standard error",
            step ? Math.sqrt((theory * (1 - theory)) / step) : 0,
          ),
          metric("Effective tuple size", k),
        ],
        message:
          "Labels stay distinct even if walker coordinates agree. The covariance identity and entropy-budget bound are separately specified reference families; the selected entropy budget is not assigned to the shared-state model.",
        done: step >= 4000,
      }),
      async () => {
        for (let trial = 0; trial < 20; trial++) {
          replacement = [];
          distinct = [];
          let collided = false;
          for (let j = 0; j < k; j++) {
            const label = Math.floor(r() * n);
            replacement.push(label);
            if (distinct.includes(label)) {
              collided = true;
              const remaining = Array.from({ length: n }, (_, i) => i).filter(
                (i) => !distinct.includes(i),
              );
              distinct.push(remaining[Math.floor(r() * remaining.length)]);
            } else distinct.push(label);
          }
          if (collided) hits++;
          step++;
        }
        keep(history, [step, hits / step]);
      },
    );
  },
);

export const demos = [
  keystone,
  drift,
  matching,
  proxy,
  ou,
  boundary,
  comparison,
  survival,
  massLedger,
  reaction,
  chaos,
  limits,
  equilibrium,
  diffusion,
  exchangeability,
  finiteCorrection,
];
