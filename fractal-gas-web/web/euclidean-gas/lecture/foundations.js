import {
  rng,
  mean,
  variance,
  linspace,
  normalPDF,
  histogram,
  line,
  scatter,
  positions,
  velocities,
  pushBounded,
  clamp,
} from "./math.js";

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
const metric = (label, value, unit = "") => ({ label, value, unit });
const chart = (title, xLabel, yLabel, series, extra = {}) => ({
  title,
  xLabel,
  yLabel,
  series,
  ...extra,
});
const norm2 = (p) => p.reduce((s, x) => s + x * x, 0);
const density = (label, xs, lo = -4, hi = 4) =>
  line(label, histogram(xs, lo, hi, 48));
const cloud = (n) =>
  Array.from({ length: n }, (_, i) => [
    (i < n / 2 ? -1 : 1) + 0.24 * Math.cos(i * 2.3),
    0.48 * Math.sin(i * 1.7),
  ]);
const noise = (scale) => ({
  innovation: "gaussian",
  geometry: { kind: "isotropic", scale: { kind: "constant", values: [scale] } },
});
const clone = (x) => structuredClone(x);
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const pick = (weights, random) => {
  let t = random() * sum(weights);
  for (let i = 0; i < weights.length; i++) {
    t -= weights[i];
    if (t <= 0 && weights[i] > 0) return i;
  }
  return weights.findLastIndex((w) => w > 0);
};
const positive = (z, floor = 0.01) => 2 / (1 + Math.exp(-z)) + floor;
const bar = (name, values) => ({
  name,
  points: values.map((v, i) => [i, v]),
  style: "bars",
});
const desc = (
  id,
  title,
  question,
  prediction,
  controls,
  create,
  kind = "WASM experiment",
  explanation = prediction,
) => ({
  id,
  part: "I",
  title,
  question,
  prediction,
  explanation,
  controls,
  create,
  kind,
});
async function config(engine, seed, n = 32) {
  const c = await engine.defaults();
  c.walkers = n;
  c.dimensions = 2;
  c.benchmark = "quadratic";
  c.initial_lower = -1.5;
  c.initial_upper = 1.5;
  c.gas.seed = Number(seed);
  c.gas.precision = "f64";
  c.gas.backend = "cpu";
  c.gas.boundary = { kind: "unbounded" };
  for (const role of ["distance_donors", "cloning_donors"]) {
    c.gas[role].kernel = { kind: "uniform" };
    c.gas[role].law = "independent";
    c.gas[role].history_window = 0;
  }
  return c;
}
function baoab(c, p = {}) {
  c.gas.kinetic = {
    integrator: {
      kind: "baoab",
      positions: "positions",
      velocities: "velocities",
      dt: p.dt ?? 0.04,
      friction: p.gamma ?? 1,
    },
    noise: noise(Math.sqrt(2 * (p.gamma ?? 1) * (p.temperature ?? 0.4))),
  };
  return c;
}
function neutral(c) {
  c.gas.fitness.reward_exponent = 0;
  c.gas.fitness.diversity_exponent = 0;
  return c;
}
async function fixture(run, x, v, alive) {
  return run.set_population(
    JSON.stringify({
      positions: x,
      ...(v ? { velocities: v } : {}),
      ...(alive ? { alive } : {}),
    }),
  );
}
function graph(frame, role, points) {
  const r = frame.report;
  if (!r) return [];
  const b =
    r[role === "distance" ? "distance_companions" : "cloning_companions"];
  const sources =
    role === "distance" ? r.distance_sources : r.clone_plan.sources;
  return b.indices.flatMap((j, k) =>
    b.valid[k] && points[Math.floor(k / b.count)] && sources[j]
      ? [[points[Math.floor(k / b.count)], points[sources[j].slot]]]
      : [],
  );
}
const aliveCount = (f) =>
  f.population.validity.filter(
    (v) => !v.invalid && !v.out_of_bounds && !v.terminated && !v.truncated,
  ).length;
function frozenModel(run, x, v, render, onStep = () => {}) {
  let frame = run.snapshot(),
    tick = 0;
  return {
    async step() {
      await fixture(run, x, v);
      frame = await run.step(1);
      tick++;
      onStep(frame, tick);
    },
    snapshot() {
      return { step: tick, time: tick, ...render(frame, tick) };
    },
    dispose() {
      run.free();
    },
  };
}

// Marginal laws are enumerated before any Monte Carlo outcomes are drawn.
// The greedy recursion averages the uniformly shuffled processing order and
// the Gaussian choice from the remaining pool, including the odd self slot.
export function donorProbabilities(points, width = 1, uniform = false) {
  return points.map((p, i) => {
    const logs = points.map((q, j) =>
      i === j
        ? -Infinity
        : uniform
          ? 0
          : -norm2(p.map((x, k) => x - q[k])) / (2 * width * width),
    );
    const m = Math.max(...logs);
    const w = logs.map((v) => Math.exp(v - m));
    const total = sum(w);
    return w.map((v) => v / total);
  });
}
export function enumerateAssignments(
  points,
  width = 1,
  law = "independent",
  visit,
) {
  const n = points.length,
    prob = donorProbabilities(points, width, law === "uniform"),
    a = Array(n).fill(0);
  if (law === "independent" || law === "uniform") {
    const next = (i, w) => {
      if (i === n) {
        visit(a.slice(), w);
        return;
      }
      for (let j = 0; j < n; j++)
        if (prob[i][j] > 0) {
          a[i] = j;
          next(i + 1, w * prob[i][j]);
        }
    };
    next(0, 1);
  } else {
    const next = (left, w) => {
      if (left.length <= 1) {
        if (left.length) a[left[0]] = left[0];
        visit(a.slice(), w);
        return;
      }
      for (const i of left) {
        const rest = left.filter((j) => j !== i),
          weights = rest.map((j) => (law === "mutual" ? 1 : prob[i][j])),
          total = sum(weights);
        for (let k = 0; k < rest.length; k++) {
          const j = rest[k];
          a[i] = j;
          a[j] = i;
          next(
            rest.filter((v) => v !== j),
            ((w / left.length) * weights[k]) / total,
          );
        }
      }
    };
    next(
      Array.from({ length: n }, (_, i) => i),
      1,
    );
  }
}
function fitnessFor(points, a, alpha = 1, beta = 1) {
  const raw = points.map(norm2).map((x) => -x),
    sep = points.map((p, i) =>
      Math.sqrt(norm2(p.map((x, k) => x - points[a[i]][k])) + 1e-6),
    );
  const channel = (values) => {
    const m = mean(values),
      s = Math.sqrt(variance(values) + 1e-6);
    return values.map((v) => positive((v - m) / s, 1e-6));
  };
  const r = channel(raw),
    d = channel(sep);
  return r.map((v, i) => v ** alpha * d[i] ** beta);
}
export function probabilityField(
  points,
  {
    recipient = 0,
    width = 1,
    cloneWidth = 1,
    saturation = 1,
    law = "independent",
  } = {},
) {
  const n = points.length,
    donor = donorProbabilities(points, cloneWidth)[recipient],
    joint = Array(n).fill(0),
    expectedFitness = Array(n).fill(0),
    fitnessSamples = [],
    assignments = [];
  let mass = 0;
  enumerateAssignments(points, width, law, (a, w) => {
    const f = fitnessFor(points, a);
    mass += w;
    assignments.push({ a, w, f });
    fitnessSamples.push([f[recipient], w]);
    for (let j = 0; j < n; j++) {
      expectedFitness[j] += w * f[j];
      joint[j] +=
        w *
        donor[j] *
        clamp((f[j] - f[recipient]) / (f[recipient] + 1e-6) / saturation, 0, 1);
    }
  });
  return {
    donor,
    joint,
    persistence: 1 - sum(joint),
    fitnessSamples,
    assignments,
    mass,
    clippedMean: donor.map(
      (p, j) =>
        p *
        clamp(
          (expectedFitness[j] - expectedFitness[recipient]) /
            (expectedFitness[recipient] + 1e-6) /
            saturation,
          0,
          1,
        ),
    ),
  };
}

// Piecewise-linear objective contours on a fixed teaching viewport.
function objectiveContours(benchmark) {
  const axis = linspace(-2.2, 2.2, 19),
    segments = [];
  const value = ([x, y]) =>
    benchmark === "sphere"
      ? x * x + y * y
      : x * x +
        y * y -
        10 * Math.cos(2 * Math.PI * x) -
        10 * Math.cos(2 * Math.PI * y) +
        20;
  const levels = benchmark === "sphere" ? [0.25, 1, 2, 4] : [5, 15, 30];
  for (let i = 0; i < axis.length - 1; i++)
    for (let j = 0; j < axis.length - 1; j++) {
      const corners = [
        [axis[i], axis[j]],
        [axis[i + 1], axis[j]],
        [axis[i + 1], axis[j + 1]],
        [axis[i], axis[j + 1]],
      ];
      for (const level of levels)
        for (const ids of [
          [0, 1, 2],
          [0, 2, 3],
        ]) {
          const hits = [];
          for (let e = 0; e < 3; e++) {
            const a = corners[ids[e]],
              b = corners[ids[(e + 1) % 3]],
              va = value(a),
              vb = value(b);
            if (va < level !== vb < level) {
              const t = (level - va) / (vb - va);
              hits.push(a.map((v, k) => v + t * (b[k] - v)));
            }
          }
          if (hits.length === 2) segments.push(hits);
        }
    }
  return segments;
}

const oneStep = desc(
  "I-01",
  "Follow one walker through a complete step",
  "Which operation moved this walker?",
  "Recorded donor choices, copying, and kinetic motion explain every displacement. Replay advances through one committed trace without new random draws.",
  [
    select("walkers", "Walkers", 32, [16, 32, 64]),
    select("benchmark", "Objective", "sphere", ["sphere", "rastrigin"]),
    range("walker", "Selected slot", 0, 0, 15, 1),
  ],
  async ({ params: p, seed, engine }) => {
    const c = baoab(await config(engine, seed, p.walkers), { dt: 0.015 });
    c.benchmark = p.benchmark;
    const contours = objectiveContours(p.benchmark);
    const run = await engine.create(c);
    run.set_trace(true);
    let frame = run.snapshot(),
      trace = [],
      stage = 0,
      tick = 0;
    return {
      async step() {
        if (!trace.length || stage === trace.length - 1) {
          frame = await run.step(1);
          trace = frame.trace;
          stage = 0;
        } else stage++;
        tick++;
      },
      snapshot() {
        const current = trace[stage] ?? frame,
          pts = positions(current),
          i = p.walker % pts.length,
          r = frame.report;
        return {
          step: tick,
          time: Number(frame.step) * 0.015,
          charts: [
            chart(
              `Recorded ${trace[stage]?.stage ?? "initial"} population`,
              "x₁",
              "x₂",
              [
                scatter("Persistent slots", pts),
                scatter(`Selected slot ${i}`, [pts[i]]),
              ],
              {
                segments: graph(
                  frame,
                  "distance",
                  positions(trace[0] ?? frame),
                ),
              },
            ),
            chart(
              "Objective contours and current positions",
              "x₁",
              "x₂",
              [scatter("Walker slots", pts)],
              {
                segments: contours,
                xDomain: [-2.2, 2.2],
                yDomain: [-2.2, 2.2],
              },
            ),
            chart("Selected slot: the committed path", "x₁", "x₂", [
              line(
                "Recorded stages",
                trace.map((t) => positions(t)[i]),
              ),
              scatter("Current stage", [pts[i]]),
            ]),
          ],
          metrics: [
            metric("Gas steps", Number(frame.step)),
            metric("Replay stage", stage + 1),
            metric("Copies", r?.clones ?? 0),
            metric("Pre-clone fitness", r?.pre_clone_fitness.fitness[i] ?? 0),
            metric("Final raw objective", frame.population.rewards.raw[i]),
          ],
          table: {
            columns: ["Stage", "x₁", "x₂", "v₁"],
            rows: trace.map((t) => [
              t.stage,
              ...positions(t)[i],
              velocities(t)[i]?.[0] ?? 0,
            ]),
          },
          message:
            "Eligibility → distance donors → fitness → cloning donors → clone plan → transformed offspring → BAOAB → final reward. The table shows stored states from the last completed step.",
        };
      },
      dispose() {
        run.free();
      },
    };
  },
);

const networks = desc(
  "I-02",
  "Two donor networks, two different questions",
  "Does the walker measuring diversity have to be the one copied?",
  "Distance and cloning draws have separate streams. Mutual rounds give reciprocal edges; an odd leftover uses the configured self-companion rule.",
  [
    select("walkers", "Walkers", 6, [5, 6, 16]),
    select("law", "Distance donor law", "independent", [
      "independent",
      "uniform",
      "mutual",
      "greedy",
    ]),
    range("width", "Distance Gaussian width", 1, 0.1, 3, 0.1),
    range("cloneWidth", "Cloning Gaussian width", 0.7, 0.1, 3, 0.1),
    select("count", "Distance rounds / donors", 1, [1, 2, 4]),
  ],
  async ({ params: p, seed, engine }) => {
    const c = await config(engine, seed, p.walkers),
      x = cloud(p.walkers);
    c.gas.kinetic.integrator.amplitude = 0;
    neutral(c);
    Object.assign(c.gas.distance_donors, {
      count: p.count,
      law:
        p.law === "mutual"
          ? "fisher_yates"
          : p.law === "greedy"
            ? "gaussian_greedy"
            : "independent",
      kernel: ["uniform", "mutual"].includes(p.law)
        ? { kind: "uniform" }
        : { kind: "gaussian", width: p.width },
    });
    c.gas.cloning_donors.kernel = { kind: "gaussian", width: p.cloneWidth };
    const run = await engine.create(c);
    await fixture(run, x);
    const counts = Array(p.walkers).fill(0),
      target = donorProbabilities(
        x,
        p.width,
        ["uniform", "mutual"].includes(p.law),
      )[0];
    if (["mutual", "greedy"].includes(p.law) && p.walkers <= 6) {
      target.fill(0);
      enumerateAssignments(x, p.width, p.law, (a, w) => {
        target[a[0]] += w;
      });
    }
    return frozenModel(
      run,
      x,
      null,
      (f, t) => {
        const draws = sum(counts);
        return {
          charts: [
            chart(
              "Diversity measurement network",
              "x₁",
              "x₂",
              [scatter("Frozen slots", x)],
              { segments: graph(f, "distance", x) },
            ),
            chart(
              "Cloning proposal network",
              "x₁",
              "x₂",
              [scatter("Same frozen slots", x)],
              { segments: graph(f, "cloning", x) },
            ),
            chart("Recipient 0: donor frequency", "Donor slot", "Probability", [
              bar(
                "Measured distance donors",
                counts.map((v) => v / Math.max(1, draws)),
              ),
              line(
                p.walkers > 6 && p.law === "greedy"
                  ? "First-choice kernel (not matching marginal)"
                  : "Exact distance marginal",
                target.map((v, i) => [i, v]),
              ),
              line(
                "Independent cloning law",
                donorProbabilities(x, p.cloneWidth)[0].map((v, i) => [i, v]),
              ),
            ]),
          ],
          metrics: [
            metric("Draws for recipient 0", draws),
            metric("Distance donor rounds", p.count),
            metric(
              "Mutual round",
              f.report?.distance_companions.mutual ? "yes" : "no",
            ),
          ],
          message:
            "Every redraw restores the coordinates before sampling. Edges are actual engine proposals. Reciprocal segments coincide in each mutual round; their union can share vertices.",
        };
      },
      (f) => {
        const b = f.report.distance_companions;
        for (let k = 0; k < b.count; k++)
          if (b.valid[k])
            counts[f.report.distance_sources[b.indices[k]].slot]++;
      },
    );
  },
);

const fitness = desc(
  "I-03",
  "The fitness calculation as an instrument panel",
  "Why can fitness change while a raw reward stays fixed?",
  "Alive-only regularized statistics determine both positive channels, then exponents determine their product.",
  [
    range("alpha", "Reward exponent α", 1, 0, 3),
    range("beta", "Diversity exponent β", 1, 0, 3),
    select("sigma", "Standard deviation floor", 0.01, [0.001, 0.01, 0.1, 1]),
    select("floor", "Positive-map floor", 0.01, [0.0001, 0.01, 0.1]),
    range("outlier", "Outlier radius", 1, 1, 4, 0.1),
    select("exclude", "Outlier eligible", "yes", ["yes", "no"]),
    select("stats", "Statistics", "global", ["global", "local"]),
  ],
  async ({ params: p, seed, engine }) => {
    const c = await config(engine, seed, 8),
      x = Array.from({ length: 8 }, (_, i) => [
        Math.cos((i * Math.PI) / 4) * (i === 7 ? p.outlier : 1),
        Math.sin((i * Math.PI) / 4) * (i === 7 ? p.outlier : 1),
      ]);
    c.gas.kinetic.integrator.amplitude = 0;
    c.gas.fitness.reward_exponent = p.alpha;
    c.gas.fitness.diversity_exponent = p.beta;
    for (const role of ["reward", "diversity"]) {
      c.gas.fitness[`${role}_map`] = {
        kind: "logistic",
        amplitude: 2,
        floor: p.floor,
      };
      c.gas.fitness[`${role}_standardizer`] =
        p.stats === "global"
          ? { kind: "global", sigma_min: p.sigma }
          : {
              kind: "local",
              sigma_min: p.sigma,
              distance: clone(c.gas.distance_donors.distance),
              kernel: { kind: "gaussian", width: 0.5 },
              include_self: false,
            };
    }
    const run = await engine.create(c);
    const alive = x.map((_, i) => i !== 7 || p.exclude === "yes");
    await fixture(run, x, null, alive);
    let frame = await run.step(1),
      tick = 0;
    return {
      async step() {
        await fixture(run, x, null, alive);
        frame = await run.step(1);
        tick++;
      },
      snapshot() {
        const f = frame.report.pre_clone_fitness,
          r = f.reward_z.map((z) => positive(z, p.floor)),
          d = f.diversity_z.map((z) => positive(z, p.floor));
        return {
          step: tick,
          time: tick,
          charts: [
            chart("Positive channels", "Reward channel", "Diversity channel", [
              scatter(
                "Eligible walkers",
                r.flatMap((v, i) => (alive[i] ? [[v, d[i]]] : [])),
              ),
              ...[0.3, 1, 2].flatMap((k) => {
                if (p.beta > 0)
                  return [
                    line(
                      `Fitness ${k}`,
                      linspace(0.05, 2.1, 70)
                        .map((v) => [v, (k / v ** p.alpha) ** (1 / p.beta)])
                        .filter((v) => v[1] <= 2.1),
                    ),
                  ];
                if (p.alpha > 0)
                  return [
                    line(`Fitness ${k}`, [
                      [k ** (1 / p.alpha), 0],
                      [k ** (1 / p.alpha), 2.1],
                    ]),
                  ];
                return [];
              }),
            ]),
            chart("Standardization → positive map", "z", "Positive channel", [
              line(
                "2 sigmoid(z) + floor",
                linspace(-4, 4).map((z) => [z, positive(z, p.floor)]),
              ),
              scatter(
                "Reward measurements",
                f.reward_z.flatMap((z, i) => (alive[i] ? [[z, r[i]]] : [])),
              ),
            ]),
          ],
          metrics: [
            metric("Alive comparison rows", alive.filter(Boolean).length),
            metric("Reward mean (slot 0)", f.reward_stats.mean[0]),
            metric("Regularized scale", f.reward_stats.scale[0]),
            metric(
              "Local fallback rows",
              f.reward_stats.global_fallback.filter(Boolean).length,
            ),
          ],
          table: {
            columns: [
              "Slot",
              "Eligible",
              "Raw cost",
              "Diversity",
              "Reward mean",
              "Reward σ",
              "z reward",
              "z diversity",
              "Reward channel",
              "Diversity channel",
              "Fitness",
            ],
            rows: x.map((_, i) => [
              i,
              alive[i] ? "yes" : "no",
              frame.report.pre_clone_rewards.raw[i],
              f.separation[i],
              f.reward_stats.mean[i],
              f.reward_stats.scale[i],
              f.reward_z[i],
              f.diversity_z[i],
              r[i],
              d[i],
              f.fitness[i],
            ]),
          },
          message:
            "All statistics and fitness values come from the pre-clone engine report. Positive channel values are evaluated from those recorded standardized scores.",
        };
      },
      dispose() {
        run.free();
      },
    };
  },
);

const revival = desc(
  "I-04",
  "Revival, singleton continuation, and extinction",
  "What does fixed population mean when slots lose eligibility?",
  "Four survivors can repopulate sixteen slots; one survivor supplies every revival. With zero survivors the engine emits its extinction event.",
  [
    select("survivors", "Initial eligible slots", 4, [16, 4, 1, 0]),
    select("boundary", "Boundary", "absorbing", ["absorbing", "periodic"]),
    select("jitter", "Copy jitter", 0.05, [0, 0.05, 0.2]),
  ],
  async ({ params: p, seed, engine }) => {
    const c = await config(engine, seed, 16),
      box = { lower: [-2, -2], upper: [2, 2] };
    c.gas.boundary = {
      kind: p.boundary === "periodic" ? "periodic_box" : "absorbing_box",
      field: "positions",
      domain: box,
    };
    if (p.boundary === "periodic")
      for (const key of ["distance_donors", "cloning_donors"])
        c.gas[key].distance.periodic = box;
    c.gas.clone_transform = {
      position_field: "positions",
      jitter: noise(1),
      jitter_amplitude: p.jitter,
      velocity_field: null,
      restitution: null,
    };
    const run = await engine.create(c),
      x = cloud(16);
    await fixture(
      run,
      x,
      null,
      x.map((_, i) => i < p.survivors),
    );
    let frame = run.snapshot(),
      terminal = false,
      tick = 0;
    const history = [[0, aliveCount(frame)]];
    return {
      async step() {
        if (terminal) return;
        try {
          frame = await run.step(1);
          tick++;
          pushBounded(history, [tick, aliveCount(frame)]);
        } catch (error) {
          if (!String(error).toLowerCase().includes("extinct")) throw error;
          terminal = true;
        }
      },
      snapshot() {
        const r = frame.report;
        return {
          step: tick,
          time: tick,
          done: terminal,
          charts: [
            chart(
              "Persistent slot ledger",
              "Slot",
              "Eligible (1) / inactive (0)",
              [
                bar(
                  "Eligibility",
                  frame.population.validity.map((v) =>
                    Number(
                      !v.invalid &&
                        !v.terminated &&
                        !v.out_of_bounds &&
                        !v.truncated,
                    ),
                  ),
                ),
              ],
            ),
            chart("Eligible population", "Step", "k", [
              line("Engine eligible slots", history),
            ]),
          ],
          metrics: [
            metric("Eligible slots", aliveCount(frame)),
            metric("Revived this step", r?.revivals ?? 0),
            metric("State", terminal ? "extinction" : "running"),
          ],
          table: {
            columns: [
              "Slot",
              "Terminated",
              "Outside boundary",
              "Invalid",
              "Truncated",
              "Revival donor",
            ],
            rows: frame.population.validity.map((v, i) => [
              i,
              v.terminated,
              v.out_of_bounds,
              v.invalid,
              v.truncated,
              r?.clone_plan.choices[i].revival
                ? r.clone_plan.sources[
                    r.clone_plan.choices[i].donors[0].pool_index
                  ].slot
                : "—",
            ]),
          },
          message: terminal
            ? "Zero eligible donors: the explicit extinction event stops the experiment and retains its last committed population."
            : "The fixture marks excluded slots as externally terminated. The boundary, invalid-data, and truncation signals remain separate. Engine copy jitter applies to accepted voluntary copies, not revived slots.",
        };
      },
      dispose() {
        run.free();
      },
    };
  },
);

const microscope = desc(
  "I-05",
  "BAOAB under a microscope",
  "At which positions are the two force evaluations made?",
  "B changes velocity, A changes position, and O damps velocity while adding Gaussian innovation. The final B uses the post-A position.",
  [
    range("dt", "Time step h", 0.04, 0.005, 0.1, 0.005),
    range("gamma", "Friction γ", 1, 0.1, 5, 0.1),
    range("temperature", "Thermal parameter T", 0.4, 0, 2, 0.1),
    range("walker", "Selected slot", 0, 0, 15, 1),
  ],
  async ({ params: p, seed, engine }) => {
    const c = neutral(baoab(await config(engine, seed, 16), p)),
      run = await engine.create(c);
    run.set_trace(true);
    await fixture(
      run,
      cloud(16),
      cloud(16).map(([x, y]) => [-y, x]),
    );
    let frame = await run.step(1),
      tick = 0;
    return {
      async step() {
        frame = await run.step(1);
        tick++;
      },
      snapshot() {
        const stages = frame.trace.filter((t) =>
            ["post_transform", "B1", "A1", "O", "A2", "B2"].includes(t.stage),
          ),
          i = p.walker,
          energy = stages.map((t) => [
            0.5 * norm2(velocities(t)[i]),
            0.5 * norm2(positions(t)[i]),
          ]),
          o = stages.find((t) => t.stage === "O"),
          a = stages.find((t) => t.stage === "A1"),
          innovation =
            velocities(o)[i][0] -
            Math.exp(-p.gamma * p.dt) * velocities(a)[i][0];
        return {
          step: tick,
          time: Number(frame.step) * p.dt,
          charts: [
            chart("Recorded B–A–O–A–B phase path", "x₁", "v₁", [
              line(
                "Actual substages",
                stages.map((t) => [positions(t)[i][0], velocities(t)[i][0]]),
              ),
              scatter(
                "Force evaluation positions",
                [stages[0], stages[4]].map((t) => [
                  positions(t)[i][0],
                  velocities(t)[i][0],
                ]),
              ),
            ]),
            chart(
              "Energy at each substage",
              "0: input · 1: B · 2: A · 3: O · 4: A · 5: B",
              "Energy",
              [
                line(
                  "Kinetic",
                  energy.map((v, k) => [k, v[0]]),
                ),
                line(
                  "Potential U = |x|²/2",
                  energy.map((v, k) => [k, v[1]]),
                ),
                line(
                  "Total",
                  energy.map((v, k) => [k, sum(v)]),
                ),
              ],
            ),
          ],
          metrics: [
            metric("O innovation Δv₁", innovation),
            metric(
              "Predicted O variance",
              p.temperature * (1 - Math.exp(-2 * p.gamma * p.dt)),
            ),
            metric(
              "Diffusion factor L",
              Math.sqrt(2 * p.gamma * p.temperature),
            ),
          ],
          table: {
            columns: ["Substage", "x₁", "v₁", "U", "K"],
            rows: stages.map((t, k) => [
              t.stage,
              positions(t)[i][0],
              velocities(t)[i][0],
              energy[k][1],
              energy[k][0],
            ]),
          },
          message:
            "Actual uncapped engine trace, quadratic potential, neutral fitness, Gaussian thermostat. With T=0 the O step is deterministic friction. The energy table follows the selected slot.",
        };
      },
      dispose() {
        run.free();
      },
    };
  },
);

const rewardForce = desc(
  "I-06",
  "Reward attracts copies; potential accelerates motion",
  "What changes when favorable reward and the force point to different places?",
  "Selection follows independently evaluated reward, while the BAOAB gradient follows the confining potential.",
  [
    select("landscape", "Reward / force preset", "aligned", [
      "aligned",
      "shifted",
      "multiwell",
    ]),
    select("walkers", "Walkers", 128, [128, 256]),
    range("alpha", "Reward exponent α", 1, 0, 2, 0.1),
    range("beta", "Diversity exponent β", 0, 0, 1, 0.1),
    range("dt", "Time step", 0.025, 0.005, 0.08, 0.005),
  ],
  async ({ params: p, seed, engine }) => {
    const c = baoab(await config(engine, seed, p.walkers), {
      dt: p.dt,
      temperature: 0.2,
    });
    c.benchmark = p.landscape === "multiwell" ? "rastrigin" : "quadratic";
    c.potential = "quadratic";
    c.reward_shift = p.landscape === "shifted" ? [1, 0] : [];
    c.gas.fitness.reward_exponent = p.alpha;
    c.gas.fitness.diversity_exponent = p.beta;
    const run = await engine.create(c);
    let frame = run.snapshot();
    const reward = [],
      potential = [],
      spread = [],
      fraction = [];
    function record() {
      const x = positions(frame),
        t = Number(frame.step) * p.dt;
      pushBounded(reward, [t, mean(frame.population.rewards.raw)]);
      pushBounded(potential, [t, mean(x.map((v) => norm2(v) / 2))]);
      pushBounded(spread, [
        t,
        variance(x.map((v) => v[0])) + variance(x.map((v) => v[1])),
      ]);
      pushBounded(fraction, [t, (frame.report?.clones ?? 0) / p.walkers]);
    }
    record();
    return {
      async step() {
        frame = await run.step(1);
        record();
      },
      snapshot() {
        const x = positions(frame);
        return {
          step: Number(frame.step),
          time: Number(frame.step) * p.dt,
          charts: [
            chart(
              "Reward cost profile; smaller is better",
              "x₁ (x₂=0)",
              "Cost",
              [
                line(
                  "Reward objective",
                  linspace(-2.5, 2.5).map((v) => [
                    v,
                    p.landscape === "multiwell"
                      ? v * v - 10 * Math.cos(2 * Math.PI * v) + 10
                      : 0.5 * (v - (p.landscape === "shifted" ? 1 : 0)) ** 2,
                  ]),
                ),
                line(
                  "Force potential U",
                  linspace(-2.5, 2.5).map((v) => [v, 0.5 * v * v]),
                ),
              ],
            ),
            chart(
              "Live cloud under −∇U = −x",
              "x₁",
              "x₂",
              [scatter("Walkers", x)],
              { segments: cloud(16).map((q) => [q, q.map((v) => 0.7 * v)]) },
            ),
            chart(
              "Measured reward and confinement",
              "Physical time",
              "Population mean",
              [
                line("Raw reward cost", reward),
                line("Potential energy", potential),
                line("Position variance", spread),
              ],
            ),
            chart(
              "Accepted voluntary copying",
              "Physical time",
              "Fraction",
              [line("Cloned slots / N", fraction)],
              { yDomain: [0, 1] },
            ),
          ],
          metrics: [
            metric("Mean reward cost", mean(frame.population.rewards.raw)),
            metric("Mean x₁", mean(x.map((v) => v[0]))),
            metric("Reward optimum x₁", p.landscape === "shifted" ? 1 : 0),
            metric("Potential optimum x₁", 0),
          ],
          message:
            "Reward minimization and a separate quadratic gradient provider are active in the same real run. Set both exponents to zero to isolate force and thermostat; reset the seed to compare trajectories.",
        };
      },
      dispose() {
        run.free();
      },
    };
  },
);

const field = desc(
  "I-07",
  "A probability field for one selected walker",
  "How much probability belongs to each accepted copy before drawing randomness?",
  "Averaging the entire clipped fitness comparison produces the accepted-copy weights; persistence supplies the remaining probability.",
  [
    select("walkers", "Frozen population N", 5, [4, 5, 6]),
    select("law", "Diversity assignment law", "independent", [
      "independent",
      "greedy",
    ]),
    range("width", "Diversity width", 1, 0.2, 2, 0.2),
    range("cloneWidth", "Cloning width", 1, 0.2, 2, 0.2),
    select("saturation", "Acceptance saturation scale", 1, [0.25, 1, 4]),
    range("recipient", "Recipient slot", 0, 0, 3, 1),
  ],
  async ({ params: p, seed }) => {
    const x = cloud(p.walkers),
      f = probabilityField(x, p),
      random = rng(seed),
      counts = Array(p.walkers).fill(0),
      donors = Array(p.walkers).fill(0),
      weights = f.assignments.map((a) => a.w);
    let samples = 0;
    return {
      async step() {
        for (let n = 0; n < 128; n++) {
          const a = f.assignments[pick(weights, random)],
            j = pick(f.donor, random);
          donors[j]++;
          if (
            random() <
            clamp(
              (a.f[j] - a.f[p.recipient]) /
                (a.f[p.recipient] + 1e-6) /
                p.saturation,
              0,
              1,
            )
          )
            counts[j]++;
          samples++;
        }
      },
      snapshot() {
        return {
          step: samples,
          time: samples,
          charts: [
            chart("Frozen conditioning population", "x₁", "x₂", [
              scatter("Slots", x),
              scatter("Selected recipient", [x[p.recipient]]),
            ]),
            chart(
              "Cloning donor law and conditional acceptance",
              "Candidate slot",
              "Probability",
              [
                bar("Donor probability", f.donor),
                line(
                  "Acceptance given donor",
                  f.joint.map((v, j) => [
                    j,
                    f.donor[j] > 0 ? v / f.donor[j] : 0,
                  ]),
                ),
              ],
            ),
            chart(
              "Accepted-copy weights: predict → sample",
              "Candidate slot",
              "Joint probability",
              [
                bar("Exact averaged clipped acceptance", f.joint),
                line(
                  "Observed accepted-copy frequency",
                  counts.map((v, j) => [j, v / Math.max(1, samples)]),
                ),
                line(
                  "Clip after averaging fitness",
                  f.clippedMean.map((v, j) => [j, v]),
                  { dashed: true },
                ),
              ],
            ),
            chart(
              "Fitness distribution over diversity assignments",
              "Fitness",
              "Probability per bin",
              [
                {
                  name: "Enumerated probability mass",
                  style: "bars",
                  points: Array.from({ length: 24 }, (_, i) => [
                    ((i + 0.5) * 4.01) / 24,
                    f.fitnessSamples.reduce(
                      (s, [v, w]) =>
                        s +
                        (Math.min(23, Math.floor((v / 4.01) * 24)) === i
                          ? w
                          : 0),
                      0,
                    ),
                  ]),
                },
              ],
            ),
          ],
          metrics: [
            metric("Exact accepted-copy probability", sum(f.joint)),
            metric("Persistence probability", f.persistence),
            metric("Mass", f.mass),
            metric("Independent conditional draws", samples),
            metric(
              "Maximum frequency standard error",
              0.5 / Math.sqrt(Math.max(1, samples)),
            ),
          ],
          message:
            "Exact small-law enumerator and seeded conditional Monte Carlo, using the engine’s regularized logistic fitness and acceptance formula. Gaussian greedy integrates random processing order and reciprocal pairing; odd leftover is self-paired.",
        };
      },
      dispose() {},
    };
  },
  "Mathematical model",
);

const mixture = desc(
  "I-08",
  "Resolve a cloning jump into its mixture components",
  "Which part of the post-copy law remains an atom?",
  "Persistence keeps its probability at the old position. Accepted-copy mass broadens around donors; BAOAB then transports each component.",
  [
    range("recipient", "Recipient slot", 0, 0, 3, 1),
    select("jitter", "Copy jitter σ", 0.1, [0, 0.02, 0.1, 0.3]),
    range("temperature", "Thermostat T", 0.4, 0, 2, 0.1),
    range("boundary", "Viable half-width", 1.1, 0.4, 2, 0.1),
  ],
  async ({ params: p, seed }) => {
    const x = cloud(5),
      v = x.map(([a, b]) => [-b, a]),
      f = probabilityField(x, { recipient: p.recipient }),
      weights = [f.persistence, ...f.joint],
      random = rng(seed),
      samples = [],
      final = [],
      phase = [],
      eventCounts = Array(6).fill(0),
      cloneExit = Array(6).fill(0),
      kineticExit = Array(6).fill(0);
    let count = 0;
    return {
      async step() {
        for (let k = 0; k < 128; k++) {
          const event = pick(weights, random),
            j = event === 0 ? p.recipient : event - 1;
          let q = x[j].map(
              (a) => a + (event > 0 ? p.jitter * random.normal() : 0),
            ),
            u = v[j].slice();
          eventCounts[event]++;
          const died = q.some((a) => Math.abs(a) > p.boundary);
          if (died) cloneExit[event]++;
          const q0 = q[0];
          if (!died) {
            u = u.map((a, d) => a - 0.02 * q[d]);
            q = q.map((a, d) => a + 0.02 * u[d]);
            if (q.some((a) => Math.abs(a) > p.boundary)) kineticExit[event]++;
            else {
              const c = Math.exp(-0.04),
                s = Math.sqrt(p.temperature * (1 - c * c));
              u = u.map((a) => c * a + s * random.normal());
              q = q.map((a, d) => a + 0.02 * u[d]);
              if (q.some((a) => Math.abs(a) > p.boundary)) kineticExit[event]++;
              else u = u.map((a, d) => a - 0.02 * q[d]);
            }
          }
          pushBounded(samples, q0, 400);
          pushBounded(final, q[0], 400);
          pushBounded(phase, [q[0], u[0]], 400);
          count++;
        }
      },
      snapshot() {
        const grid = linspace(-2, 2, 161),
          components = f.joint.map((w, j) =>
            line(
              `Donor ${j}: mass ${w.toFixed(3)}`,
              p.jitter
                ? grid.map((q) => [q, w * normalPDF(q, x[j][0], p.jitter)])
                : [
                    [x[j][0], 0],
                    [x[j][0], w],
                  ],
            ),
          );
        return {
          step: count,
          time: count,
          charts: [
            chart(
              p.jitter
                ? "Continuous copy density plus separate persistence mass"
                : "Atomic post-copy law",
              "x₁",
              p.jitter ? "Density / atom mass" : "Atom probability",
              [
                ...components,
                line("Persistence atom (probability stem)", [
                  [x[p.recipient][0], 0],
                  [x[p.recipient][0], f.persistence],
                ]),
                ...(p.jitter
                  ? [
                      line(
                        "Continuous mixture density",
                        grid.map((q) => [
                          q,
                          sum(
                            f.joint.map(
                              (w, j) => w * normalPDF(q, x[j][0], p.jitter),
                            ),
                          ),
                        ]),
                      ),
                    ]
                  : []),
              ],
            ),
            chart(
              "Sampled conditional outcomes (rolling 400)",
              "x₁",
              "Density",
              [
                density("Post-clone samples", samples, -2, 2),
                density("After kinetic schedule", final, -2, 2),
              ],
            ),
            chart("Kinetic pushforward in phase space", "x₁", "v₁", [
              scatter("Conditional outcomes", phase),
            ]),
            chart(
              "Event-conditioned exit estimates",
              "0: persistence; 1…5: donor",
              "Conditional exit probability",
              [
                bar(
                  "Clone-stage exit",
                  cloneExit.map((n, i) => n / Math.max(1, eventCounts[i])),
                ),
                line(
                  "Subsequent kinetic exit",
                  kineticExit.map((n, i) => [
                    i,
                    n / Math.max(1, eventCounts[i]),
                  ]),
                ),
              ],
              { yDomain: [0, 1] },
            ),
          ],
          metrics: [
            metric("Draws", count),
            metric("Persistence mass", f.persistence),
            metric("Copy mass", sum(f.joint)),
            metric("Viable |xₖ| ≤", p.boundary),
          ],
          message:
            "Analytic conditional mixture with literal velocity copying (no restitution), Gaussian jitter, and an explicit quadratic BAOAB reference pushforward. Exit fractions are conditioned on the event label; the histogram uses only the rolling 400 outcomes. The persistence stem is probability, not a density.",
        };
      },
      dispose() {},
    };
  },
  "Mathematical model",
);

const collision = desc(
  "I-09",
  "What an inelastic collision really conserves",
  "How can relative motion shrink while pair momentum stays fixed?",
  "On each accepted mutual pair, coordinate momentum is unchanged and relative kinetic energy is multiplied by a².",
  [
    range("restitution", "Restitution a", 0.5, 0, 1, 0.05),
    range("vx", "First velocity x₁", 1.5, -2, 2, 0.1),
    range("vy", "First velocity x₂", 0.5, -2, 2, 0.1),
  ],
  async ({ params: p, seed, engine }) => {
    const c = baoab(await config(engine, seed, 4), {
      dt: 0.005,
      temperature: 0,
    });
    c.gas.cloning_donors.law = "fisher_yates";
    c.gas.clone_transform = {
      position_field: null,
      jitter: null,
      jitter_amplitude: 0,
      velocity_field: "velocities",
      restitution: p.restitution,
    };
    c.gas.fitness.diversity_exponent = 0;
    c.gas.clone_decision.saturation = 0.01;
    const x = [
        [0.1, 0],
        [0.7, 0],
        [1.3, 0],
        [2, 0],
      ],
      v = [
        [p.vx, p.vy],
        [-1, 1],
        [0.5, -1.5],
        [-1, -0.5],
      ],
      run = await engine.create(c);
    run.set_trace(true);
    await fixture(run, x, v);
    let frame = await run.step(1),
      tick = 0;
    return {
      async step() {
        await fixture(run, x, v);
        frame = await run.step(1);
        tick++;
      },
      snapshot() {
        const before = velocities(
            frame.trace.find((t) => t.stage === "pre_clone"),
          ),
          after = velocities(
            frame.trace.find((t) => t.stage === "post_transform"),
          ),
          r = frame.report,
          pairs = [];
        for (let i = 0; i < 4; i++) {
          const j = r.clone_plan.sources[r.cloning_companions.indices[i]].slot;
          if (i < j) pairs.push([i, j]);
        }
        let momentumError = 0,
          relativeBefore = 0,
          relativeAfter = 0;
        const rows = pairs.map(([i, j]) => {
          const m = before[i].map((a, k) => (a + before[j][k]) / 2),
            e0 =
              0.5 *
              (norm2(before[i].map((a, k) => a - m[k])) +
                norm2(before[j].map((a, k) => a - m[k]))),
            e1 =
              0.5 *
              (norm2(after[i].map((a, k) => a - m[k])) +
                norm2(after[j].map((a, k) => a - m[k]))),
            err = Math.sqrt(
              norm2(m.map((a, k) => after[i][k] + after[j][k] - 2 * a)),
            ),
            active =
              r.clone_plan.choices[i].accepted ||
              r.clone_plan.choices[j].accepted;
          momentumError = Math.max(momentumError, err);
          if (active) {
            relativeBefore += e0;
            relativeAfter += e1;
          }
          return [
            `${i} ↔ ${j}`,
            active ? "collision" : "no accepted copy",
            2 * m[0],
            2 * m[1],
            e0,
            e1,
            active ? p.restitution ** 2 * e0 : e0,
          ];
        });
        return {
          step: tick,
          time: tick,
          charts: [
            chart(
              "Actual disjoint-pair velocity transform",
              "v₁",
              "v₂",
              [
                scatter("Before", before),
                scatter("After transform", after),
                scatter(
                  "Pair centers",
                  pairs.map(([i, j]) =>
                    before[i].map((a, k) => (a + before[j][k]) / 2),
                  ),
                ),
              ],
              {
                segments: pairs.flatMap(([i, j]) => [
                  [before[i], before[j]],
                  [after[i], after[j]],
                ]),
              },
            ),
            chart(
              "Accepted pairs: relative energy identity",
              "Restitution a",
              "Relative kinetic energy",
              [
                line(
                  "Exact a² law",
                  linspace(0, 1).map((a) => [a, a * a * relativeBefore]),
                ),
                scatter("Actual engine transform", [
                  [p.restitution, relativeAfter],
                ]),
              ],
            ),
          ],
          metrics: [
            metric("Largest pair momentum error", momentumError),
            metric("Accepted-pair relative energy", relativeAfter),
            metric(
              "Predicted relative energy",
              p.restitution ** 2 * relativeBefore,
            ),
          ],
          table: {
            columns: [
              "Pair",
              "Event",
              "Momentum x",
              "Momentum y",
              "Relative K before",
              "Relative K after",
              "Prediction",
            ],
            rows,
          },
          message:
            "Actual post-transform states, before kinetics. Rust applies restitution only when at least one member accepts a copy. Pairs with no accepted copy retain their original relative energy. Mutual matching supplies disjoint pairs.",
        };
      },
      dispose() {
        run.free();
      },
    };
  },
);

const geometry = desc(
  "I-10",
  "A flat chart is only one latent geometry",
  "Which object changed: donor distance, noise covariance, or a prescribed metric?",
  "A constant diagonal factor changes live diffusion; a variable metric produces position-dependent ellipses in the mathematical panel.",
  [
    range("anisotropy", "Constant diffusion ratio", 2, 0.5, 3, 0.1),
    range("position", "Selected chart coordinate", 0, -2, 2, 0.1),
    range("velocityWeight", "Donor velocity weight", 1, 0, 3, 0.1),
  ],
  async ({ params: p, seed, engine }) => {
    const c = baoab(await config(engine, seed, 64), { temperature: 0.3 }),
      other = clone(c);
    c.gas.kinetic.noise = {
      innovation: "gaussian",
      geometry: {
        kind: "diagonal",
        factor: {
          kind: "constant",
          values: [Math.sqrt(0.6) * p.anisotropy, Math.sqrt(0.6)],
        },
      },
    };
    for (const conf of [c, other])
      for (const role of ["distance_donors", "cloning_donors"])
        conf.gas[role].distance = {
          kind: "phase_space",
          positions: "positions",
          velocities: "velocities",
          position_scale: 1,
          velocity_scale: 1,
          lambda: p.velocityWeight,
          periodic: null,
        };
    const runs = await Promise.all([engine.create(c), engine.create(other)]);
    let frames = runs.map((r) => r.snapshot());
    const history = [];
    return {
      async step() {
        frames = await Promise.all(runs.map((r) => r.step(1)));
        pushBounded(history, [
          Number(frames[0].step) * 0.04,
          variance(positions(frames[0]).map((x) => x[0])),
          variance(positions(frames[1]).map((x) => x[0])),
        ]);
      },
      snapshot() {
        const angle = linspace(0, 2 * Math.PI, 65),
          ellipse = (center, scales) =>
            angle.map((a) => [
              center[0] + 0.25 * scales[0] * Math.cos(a),
              center[1] + 0.25 * scales[1] * Math.sin(a),
            ]),
          g = [
            1.25 + 0.75 * Math.sin(p.position),
            1.25 + 0.75 * Math.cos(p.position),
          ],
          grid = linspace(-2, 2, 9),
          segments = grid.flatMap((v) => [
            [
              [v, -2],
              [v, 2],
            ],
            [
              [-2, v],
              [2, v],
            ],
          ]);
        return {
          step: Number(frames[0].step),
          time: Number(frames[0].step) * 0.04,
          charts: [
            chart("Live constant anisotropy", "x₁", "x₂", [
              scatter("Diagonal-factor gas", positions(frames[0])),
              scatter("Isotropic gas, same seed", positions(frames[1])),
            ]),
            chart(
              "Prescribed smooth metric: local diffusion ellipses",
              "Chart x₁",
              "Chart x₂",
              [
                line(
                  "Metric G⁻¹/² ellipse",
                  ellipse(
                    [p.position, 0],
                    g.map((v) => 1 / Math.sqrt(v)),
                  ),
                ),
                line(
                  "Constant factor ellipse",
                  ellipse([p.position, 0], [p.anisotropy, 1]),
                ),
              ],
              { segments, xDomain: [-2.5, 2.5], yDomain: [-2.5, 2.5] },
            ),
            chart("Measured position spread", "Physical time", "Variance x₁", [
              line(
                "Diagonal factor",
                history.map((v) => [v[0], v[1]]),
              ),
              line(
                "Isotropic",
                history.map((v) => [v[0], v[2]]),
              ),
            ]),
          ],
          metrics: [
            metric("Prescribed metric eigenvalue 1", g[0]),
            metric("Prescribed metric eigenvalue 2", g[1]),
            metric("Live factor ratio", p.anisotropy),
            metric("Donor velocity weight λ", p.velocityWeight),
            metric("Euclidean ruler: 0 → selected x₁", Math.abs(p.position)),
            metric(
              "Metric straight-path ruler",
              Math.abs(p.position) *
                mean(
                  linspace(0, p.position, 101).map((v) =>
                    Math.sqrt(1.25 + 0.75 * Math.sin(v)),
                  ),
                ),
            ),
          ],
          table: {
            columns: ["Component", "Implementation here"],
            rows: [
              [
                "Reward and force",
                "Live benchmark reward / analytic potential",
              ],
              ["Noise factor", "Live constant diagonal vs isotropic"],
              ["Donor distance", "Live weighted phase-space distance"],
              ["Variable metric ellipses", "Prescribed mathematical field"],
              [
                "Viscosity / rotation / cap",
                "Provider extension; not applied in these runs",
              ],
            ],
          },
          message:
            "Both clouds are real engine runs. The grid panel evaluates G(x)=diag(1.25+0.75 sin x₁,1.25+0.75 cos x₁); its ellipses are a separate variable-metric illustration. Reset with another seed to compare the measured spread.",
        };
      },
      dispose() {
        runs.forEach((r) => r.free());
      },
    };
  },
  "WASM + reference",
);

export const demos = [
  oneStep,
  networks,
  fitness,
  revival,
  microscope,
  rewardForce,
  field,
  mixture,
  collision,
  geometry,
];
