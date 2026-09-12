import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import init, {
  BrowserGas,
  default_config,
} from "../../web/euclidean-gas/engine/cpu/gas.js";
import {
  demos,
  harmonicBudget,
} from "../../web/euclidean-gas/lecture/entropy.js";
import {
  factorial,
  fitnessJet,
  gaussianKL,
  gaussianFisher,
  harmonicState,
  baoab,
  gaussianStep,
  hellinger,
  enumeration,
  matrixExp,
  symmetricEigenvalues,
  kl,
  constraintWidth,
  hypocoerciveCoefficients,
  spectralGapDiagnostic,
  dirichletKernel,
  cosineGaussianVariance,
} from "../../web/euclidean-gas/lecture/entropy-math.js";
await init({
  module_or_path: await readFile(
    new URL("../../web/euclidean-gas/engine/cpu/gas_bg.wasm", import.meta.url),
  ),
});
const engine = {
  defaults: async () => JSON.parse(default_config()),
  create: async (c) => BrowserGas.create(JSON.stringify(c)),
  restore: async (bytes) => BrowserGas.restore(bytes),
};
const create = (id, params = {}) =>
  demos.find((d) => d.id === id).create({ params, seed: 73, engine });
const close = (a, b, tol = 1e-10) =>
  assert.ok(Math.abs(a - b) < tol, `${a} versus ${b}`);
const value = (snapshot, label) =>
  snapshot.metrics.find((m) => m.label === label).value;
function finiteCharts(snapshot) {
  for (const chart of snapshot.charts) {
    for (const series of chart.series ?? [])
      for (const point of series.points)
        for (const n of point)
          assert.ok(n === null || Number.isFinite(n), `${chart.title}: ${n}`);
    for (const row of chart.matrix ?? [])
      for (const n of row) assert.ok(Number.isFinite(n));
  }
}
test("All sixteen Part IV experiments compute finite chart data before and after stepping", async () => {
  assert.equal(demos.length, 16);
  for (const demo of demos) {
    const m = await create(demo.id);
    try {
      finiteCharts(m.snapshot());
      await m.step();
      finiteCharts(m.snapshot());
      assert.ok(m.snapshot().charts.length > 0, demo.id);
    } finally {
      m.dispose();
    }
  }
});
test("Exact Gaussian entropy obeys the kinetic dissipation identity", () => {
  const p = { k: 1.7, gamma: 0.8, theta: 0.6, displacement: 2 };
  for (const t of [0.1, 0.7, 2, 4]) {
    const state = harmonicState(t, p),
      I = gaussianFisher(state.mean, state.covariance, state.target),
      H = (x) => {
        const s = harmonicState(x, p);
        return gaussianKL(s.mean, s.covariance, s.target);
      };
    close(
      (H(t + 1e-5) - H(t - 1e-5)) / 2e-5,
      -p.gamma * p.theta * I[1][1],
      2e-8,
    );
  }
  const initial = harmonicState(0, p);
  close(
    gaussianFisher(initial.mean, initial.covariance, initial.target)[1][1],
    0,
  );
});
test("Gaussian Hellinger exactly splits mass and shape; atomic laws are singular", () => {
  for (const mass of [0.05, 0.4, 1])
    for (const center of [-2, 0, 1]) {
      const h = hellinger({ mass, center });
      close(h.total, h.massTerm + h.shapeTerm);
    }
  close(
    hellinger({ mass: 1, otherMass: 1, center: 0, width: 1, otherWidth: 1 })
      .total,
    0,
  );
  close(hellinger({ atomic: true, mass: 1, otherMass: 1 }).total, 2);
});
test("Taylor fitness derivatives agree with independent finite differences and normalization derivatives cancel", () => {
  const x = 0.13,
    h = 0.0002,
    params = { sigma: 0.15, rho: 0.7 },
    result = fitnessJet(x, params),
    f = (x) => fitnessJet(x, { ...params, n: 0 }).fitness[0];
  const differences = [
    (f(x + h) - f(x - h)) / (2 * h),
    (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2,
    (f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h)) / (2 * h ** 3),
  ];
  differences.forEach((v, i) =>
    close(v, result.fitness[i + 1] * factorial(i + 1), i === 2 ? 0.003 : 2e-5),
  );
  for (let n = 0; n <= 3; n++)
    close(
      result.weights.reduce((s, w) => s + w[n], 0),
      n === 0 ? 1 : 0,
      1e-10,
    );
  const coeff = fitnessJet(x, { n: 12 }).fitness;
  const dx = 0.005,
    taylor = coeff.reduce((s, v, i) => s + v * dx ** i, 0);
  close(taylor, fitnessJet(x + dx, { n: 0 }).fitness[0], 1e-12);
});
test("Enumerated companion laws normalize and probability derivative closes the expectation identity", () => {
  for (const law of ["independent", "matching", "greedy", "shuffled_greedy"]) {
    const x = 0.12,
      h = 1e-5,
      a = enumeration(x - h, 0.7, law),
      b = enumeration(x + h, 0.7, law),
      c = enumeration(x, 0.7, law);
    close(
      c.assignments.reduce((s, a) => s + a.p, 0),
      1,
    );
    const derivative = c.assignments.reduce(
      (s, item, i) =>
        s +
        (item.p * (b.assignments[i].value - a.assignments[i].value)) / (2 * h) +
        ((b.assignments[i].p - a.assignments[i].p) / (2 * h)) * item.value,
      0,
    );
    close(derivative, (b.expected - a.expected) / (2 * h), 1e-8);
    assert.ok(Math.abs(c.expected - c.substitute) > 1e-5);
  }
});
test("BAOAB covariance solves discrete balance and numerical entropy relaxes", async () => {
  for (const h of [0.005, 0.08, 0.3])
    for (const gamma of [0.2, 1, 3]) {
      const k = baoab(h, 1.7, gamma, 0.6),
        next = gaussianStep({ mean: [0, 0], covariance: k.stationary }, k);
      next.covariance.forEach((r, i) =>
        r.forEach((v, j) => close(v, k.stationary[i][j], 1e-12)),
      );
    }
  const m = await create("IV-10");
  const before = m.snapshot().charts[0].series.map((s) => s.points[0][1]);
  for (let i = 0; i < 300; i++) await m.step();
  m.snapshot().charts[0].series.forEach((s, i) =>
    assert.ok(s.points.at(-1)[1] < before[i] * 1e-6),
  );
  m.dispose();
});
test("Matrix exponential and symmetric eigenvalues recover known references", () => {
  const r = matrixExp(
    [
      [0, 1],
      [-1, 0],
    ],
    Math.PI / 2,
  );
  close(r[0][1], 1);
  close(r[0][0], 0);
  const e = symmetricEigenvalues([
    [2, 1],
    [1, 2],
  ]);
  close(e[0], 1);
  close(e[1], 3);
});
test("Frozen graph preserves the weighted mean and closes its energy ledger", async () => {
  const m = await create("IV-12");
  const initial = m.snapshot(),
    E = initial.charts[2].series[0].points[0][1];
  for (let i = 0; i < 150; i++) await m.step();
  const end = m.snapshot();
  close(value(end, "Weighted mean"), value(initial, "Weighted mean"));
  close(value(end, "Weighted dissipation identity residual"), 0);
  assert.ok(end.charts[2].series[0].points.at(-1)[1] < E * 0.1);
  m.dispose();
});
test("Fixed-time refinement uses identical time and resolves the harmonic second-order bias", async () => {
  const m = await create("IV-14"),
    s = m.snapshot();
  assert.ok(s.table.rows.every((r) => r[1] === 5));
  assert.ok(Math.abs(value(s, "Stationary fitted slope") - 2) < 0.01);
  assert.ok(Math.abs(value(s, "Finite-time fitted slope") - 2) < 0.1);
  m.dispose();
});
test("Diameter minorization formula matches its target independently of count", async () => {
  const width = constraintWidth(2, 0.3);
  close(Math.exp(-4 / (2 * width * width)), 0.3);
  for (const N of [2, 64, 512]) {
    const m = await create("IV-15", { N, target: 0.3 });
    const s = m.snapshot();
    assert.ok(
      value(s, "Actual minimum donor probability") >=
        value(s, "Per-candidate probability floor"),
    );
    m.dispose();
  }
});
test("Real WASM full-factor kicks recover the Hessian diffusion covariance", async () => {
  const m = await create("IV-11");
  try {
    for (let i = 0; i < 32; i++) await m.step();
    const s = m.snapshot();
    assert.equal(value(s, "Samples"), 4096);
    assert.ok(value(s, "Covariance max absolute error") < 0.06);
    finiteCharts(s);
  } finally {
    m.dispose();
  }
});
test("Real WASM capstone records both populations, deterministic replay and a restorable checkpoint", async () => {
  const a = await create("IV-16"),
    b = await create("IV-16");
  try {
    for (let i = 0; i < 6; i++) {
      await a.step();
      await b.step();
    }
    const s = a.snapshot();
    assert.equal(s.experiment.configs.length, 8);
    assert.equal(s.step, 6);
    assert.deepEqual(s.charts, b.snapshot().charts);
    assert.ok(value(s, "Independent group A total reward rows") > 64);
    const restored = await engine.restore(a.checkpoint());
    try {
      assert.equal(restored.snapshot().step, 6);
    } finally {
      restored.free();
    }
  } finally {
    a.dispose();
    b.dispose();
  }
});
test("Markov data processing contracts both laws under the same kernel", () => {
  const p = [0.85, 0.1, 0.05],
    q = [0.2, 0.3, 0.5],
    K = [
      [0.7, 0.1, 0.2],
      [0.2, 0.6, 0.2],
      [0.1, 0.2, 0.7],
    ],
    apply = (a) => a.map((_, j) => a.reduce((s, v, i) => s + v * K[i][j], 0));
  assert.ok(kl(apply(p), apply(q)) <= kl(p, q));
});

test("Mathematical models remain finite at every exposed control endpoint", async () => {
  for (const demo of demos.filter((d) => d.kind === "Mathematical model")) {
    for (const control of demo.controls) {
      const values =
        control.type === "range"
          ? [control.min, control.max]
          : control.options.map((option) => option.value);
      for (const parameter of values) {
        const m = await create(demo.id, { [control.key]: parameter });
        try {
          await m.step();
          finiteCharts(m.snapshot());
        } finally {
          m.dispose();
        }
      }
    }
  }
});

test("Every capstone card runs the declared populations and finite measured output", async () => {
  for (const card of ["operators", "survival", "population"]) {
    const m = await create("IV-16", { card });
    try {
      for (let i = 0; i < 8; i++) await m.step();
      const s = m.snapshot();
      finiteCharts(s);
      assert.equal(s.experiment.card, card);
      assert.equal(
        s.experiment.configs[4].walkers,
        card === "population" ? 128 : 64,
      );
      assert.ok(s.experiment.initializationMs > 0);
      assert.ok(s.experiment.warmStepsMs > 0);
      if (card === "survival")
        assert.equal(
          s.experiment.configs[4].gas.boundary.kind,
          "absorbing_box",
        );
    } finally {
      m.dispose();
    }
  }
});

test("Rejected WASM fixture reward evaluation preserves the entire checkpoint", async () => {
  const config = await engine.defaults();
  config.walkers = 4;
  config.benchmark = "sphere";
  config.gas.precision = "f32";
  config.gas.boundary = { kind: "unbounded" };
  const run = await engine.create(config);
  try {
    run.set_trace(true);
    await run.step(1);
    const before = run.snapshot(),
      checkpoint = run.checkpoint();
    await assert.rejects(
      run.set_population(
        JSON.stringify({
          positions: Array.from({ length: 4 }, () => [1e30, 1e30]),
        }),
      ),
      /reward evaluation failed/,
    );
    assert.deepEqual(run.snapshot(), before);
    assert.deepEqual(run.checkpoint(), checkpoint);
  } finally {
    run.free();
  }
  config.initial_lower = 2.9e38;
  config.initial_upper = 3e38;
  config.reward_shift = [-3e38, -3e38];
  await assert.rejects(
    engine.create(config),
    /translated reward coordinate overflowed/,
  );
});

test("Chapter hypocoercive coefficients contract under the audited low-friction stress", async () => {
  const m = await create("IV-01", { k: 2, gamma: 0.2, theta: 0.2 });
  try {
    for (let i = 0; i < 350; i++) await m.step();
    const s = m.snapshot(),
      c = s.charts.find((c) => c.title === "Chapter decay bound"),
      phi = c.series[0].points,
      envelope = c.series[1].points;
    assert.ok(phi.every((p, i) => p[1] <= envelope[i][1] + 1e-12));
    assert.ok(phi.slice(1).every((p, i) => p[1] <= phi[i][1] + 1e-12));
    close(value(s, "η"), 0.00046040515653775324);
    assert.ok(
      s.charts
        .find((c) => c.title.startsWith("Alternative positive"))
        .series[0].points.some((p, i, a) => i && p[1] > a[i - 1][1]),
    );
  } finally {
    m.dispose();
  }
});
test("The killed-kernel eigenmeasure is the selected entropy target and KL stays nonnegative", async () => {
  const m = await create("IV-02", { target: "qsd", killing: 0.5 });
  try {
    for (let i = 0; i < 150; i++) await m.step();
    const s = m.snapshot(),
      points = s.charts.at(-1).series[0].points;
    assert.ok(points.every(([, v]) => v >= 0));
    assert.ok(points.at(-1)[1] < 1e-12);
    assert.ok(value(s, "Cumulative surviving mass") < 1e-10);
  } finally {
    m.dispose();
  }
});
test("Reciprocal Gaussian ratios grow and the conditioned Dirichlet KDE has zero endpoints and unit mass", async () => {
  const a = await create("IV-04", { window: 3 }),
    b = await create("IV-04", { window: 6 }),
    c = await create("IV-04", { reference: "Absorbing" });
  try {
    assert.ok(
      value(b.snapshot(), "Largest q/p") >
        value(a.snapshot(), "Largest q/p") * 100,
    );
    close(
      value(b.snapshot(), "Largest p/q"),
      value(a.snapshot(), "Largest p/q"),
    );
    const s = c.snapshot(),
      points = s.charts[0].series[2].points;
    close(points[0][1], 0);
    close(points.at(-1)[1], 0);
    close(
      points.reduce(
        (sum, p, i) => sum + p[1] * (i === 0 || i === 120 ? 0.5 : 1),
        0,
      ) / 120,
      1,
      1e-12,
    );
    assert.ok(value(s, "Dirichlet smoothing surviving mass") < 1);
  } finally {
    a.dispose();
    b.dispose();
    c.dispose();
  }
});
test("Smoothed atoms and the self-inclusive normalization option preserve their declared mass identities", async () => {
  const m = await create("IV-03", { representation: "Smoothed atoms" }),
    a = await create("IV-06", { N: 256, rho: 0.1, self: "inclusive" });
  try {
    const s = m.snapshot();
    assert.ok(value(s, "Gaussian-smoothed normalized H²") < 2);
    close(value(s, "Identity residual"), 0);
    const t = a.snapshot();
    assert.ok(value(t, "Raw row sum") >= 1);
    close(value(t, "Normalized row sum"), 1);
    close(
      value(t, "Weighted x²"),
      value(t, "Replicated cloud weighted x²"),
      1e-12,
    );
  } finally {
    m.dispose();
    a.dispose();
  }
});
test("Taylor useful-radius diagnostic identifies the observed small-floor divergence", async () => {
  const m = await create("IV-07", { order: 12, radius: 0.5, sigma: 0.02 });
  try {
    const s = m.snapshot();
    assert.ok(value(s, "Maximum displayed Taylor error") > 100);
    assert.ok(value(s, "Largest tested radius with error ≤.001") < 0.3);
    assert.ok(value(s, "Largest tested radius with error ≤.001") >= 0.01);
  } finally {
    m.dispose();
  }
});
test("Shuffled greedy enumeration averages every possible first walker and shares the fixed fitness slice", () => {
  const e = enumeration(0.13, 0.7, "shuffled_greedy");
  for (const a of e.assignments)
    close(a.p, e.rows.reduce((sum, row, i) => sum + row[a.c[i]], 0) / 4);
  close(e.frozen, fitnessJet(0.13, { n: 0 }).fitness[0]);
  assert.ok(
    Math.abs(e.expected - enumeration(0.13, 0.7, "greedy").expected) > 1e-5,
  );
});
test("Weak graph links are reported below resolution with a rigorous cut upper bound", async () => {
  const m = await create("IV-12", { width: 0.2, gap: 3, viscosity: 0.1 });
  try {
    const s = m.snapshot();
    assert.equal(
      value(s, "Normalized spectral gap"),
      "Below numerical resolution",
    );
    assert.ok(value(s, "Two-cluster Rayleigh upper bound") < 4e-43);
    for (let i = 0; i < 100; i++) await m.step();
    close(
      value(m.snapshot(), "Weighted mean"),
      value(s, "Initial weighted mean"),
    );
  } finally {
    m.dispose();
  }
});
test("Stationary variance and a Lipschitz coupling bound match the absolute-error theorem at exact physical duration", () => {
  const b = harmonicBudget(32, 0.5, 1);
  close(b.sampling, (1 - Math.exp(-(1 - 0.5 ** 2 / 4))) / Math.sqrt(64));
  assert.ok(b.currentSampling > b.sampling);
  assert.ok(b.transient >= b.meanTransient);
  const a = harmonicBudget(128, 0.08, 5);
  close(a.steps * a.h, 5);
  assert.equal(a.steps, 63);
});
test("Discrete harmonic horizon scales with friction and covariance versus KL orders stay distinct", async () => {
  const m = await create("IV-10", { h: 0.01, gamma: 0.2, theta: 0.2 });
  try {
    for (let i = 0; i < 350; i++) await m.step();
    const s = m.snapshot();
    assert.ok(s.done);
    assert.ok(s.time >= 120);
    assert.ok(value(s, "Remaining maximum KL") < 1e-8);
    close(value(s, "Velocity variance bias (order h²)"), 0.000005);
  } finally {
    m.dispose();
  }
});
test("Full-factor covariance discrepancy is scaled by its actual covariance sampling uncertainty", async () => {
  const m = await create("IV-11", { lambda: -2, shift: 2.1, angle: 85 });
  try {
    for (let i = 0; i < 32; i++) await m.step();
    const s = m.snapshot();
    assert.ok(value(s, "Largest covariance-entry SE") > 0.1);
    assert.ok(value(s, "Maximum covariance discrepancy / SE") < 4);
  } finally {
    m.dispose();
  }
});
test("Metropolis independent seeds distinguish low-temperature trapping from transition-clock convergence", async () => {
  const m = await create("IV-09", {
    theta: 0.2,
    amplitude: 2,
    k: 0.5,
    wavelength: 2,
  });
  try {
    for (let i = 0; i < 250; i++) await m.step();
    const s = m.snapshot();
    assert.equal(value(s, "Independent seeds"), 4);
    assert.ok(value(s, "Left/right start mean separation") > 3);
    assert.ok(value(s, "Fraction ever crossing x=0") < 0.1);
    close(s.time, 2000);
  } finally {
    m.dispose();
  }
});
test("The absorbing capstone produces actual losses and replica uncertainty with a matched unbounded control", async () => {
  const m = await create("IV-16", { card: "survival", h: 0.04 });
  try {
    for (let i = 0; i < 150; i++) await m.step();
    const s = m.snapshot(),
      ledger = s.charts[1].series;
    assert.ok(
      ledger
        .find((l) => l.name === "Absorbing ±0.35 alive-walker fraction")
        .points.some(([, v]) => v < 1),
    );
    assert.ok(
      ledger
        .find((l) => l.name === "Unbounded control alive-walker fraction")
        .points.every(([, v]) => v === 1),
    );
    assert.ok(value(s, "Absorbing ±0.35 cumulative revivals") > 0);
    assert.equal(s.experiment.replicasPerGroup, 4);
    assert.ok(value(s, "Unbounded control surviving-replica SE") > 0);
  } finally {
    m.dispose();
  }
});

test("Capstone extinction excludes missing observables and exposes survivor counts without a false zero SE", async () => {
  for (const survivors of [0, 1]) {
    let bounded = 0;
    const fixtureEngine = {
      ...engine,
      create: async (c) => {
        const run = await engine.create(c);
        if (c.gas.boundary.kind === "absorbing_box" && bounded++ >= survivors)
          await run.set_population(
            JSON.stringify({
              positions: Array.from({ length: c.walkers }, () => [0, 0]),
              alive: Array(c.walkers).fill(false),
            }),
          );
        return run;
      },
    };
    const m = await demos
      .find((d) => d.id === "IV-16")
      .create({
        params: { card: "survival" },
        seed: 73,
        engine: fixtureEngine,
      });
    try {
      const s = m.snapshot();
      assert.equal(
        value(s, "Absorbing ±0.35 contributing replicas"),
        survivors,
      );
      assert.equal(
        value(s, "Absorbing ±0.35 surviving-replica SE"),
        "Requires 2 survivors",
      );
      assert.equal(s.experiment.contributingReplicas[1], survivors);
      assert.match(
        s.experiment.conditioning,
        /conditional on replica survival/,
      );
      close(
        s.charts[1].series.find(
          (l) => l.name === "Absorbing ±0.35 full-swarm survival",
        ).points[0][1],
        survivors / 4,
      );
      if (!survivors)
        assert.equal(
          value(s, "Absorbing ±0.35 surviving-replica mean radius²"),
          "Extinct",
        );
      finiteCharts(s);
    } finally {
      m.dispose();
    }
  }
});

test("Chapter G eigenvalue metric remains compact at parameter corners", async () => {
  const m = await create("IV-01", { k: 2, gamma: 0.2, theta: 0.2 });
  try {
    assert.ok(value(m.snapshot(), "G eigenvalues").length <= 24);
  } finally {
    m.dispose();
  }
});
