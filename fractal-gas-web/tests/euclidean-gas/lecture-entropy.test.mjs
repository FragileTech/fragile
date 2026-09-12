import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import init, {
  BrowserGas,
  default_config,
} from "../../web/euclidean-gas/engine/cpu/gas.js";
import { demos } from "../../web/euclidean-gas/lecture/entropy.js";
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
  for (const law of ["independent", "matching", "greedy"]) {
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
    assert.equal(s.experiment.configs.length, 2);
    assert.equal(s.step, 6);
    assert.deepEqual(s.charts, b.snapshot().charts);
    assert.ok(value(s, "Run 1 reward rows") > 64);
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
        s.experiment.configs[1].walkers,
        card === "population" ? 128 : 64,
      );
      assert.ok(s.experiment.initializationMs > 0);
      assert.ok(s.experiment.warmStepsMs > 0);
      if (card === "survival")
        assert.equal(
          s.experiment.configs[0].gas.boundary.kind,
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
