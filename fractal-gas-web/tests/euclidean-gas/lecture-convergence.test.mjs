import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import init, {
  BrowserGas,
  default_config,
} from "../../web/euclidean-gas/engine/cpu/gas.js";
import { demos } from "../../web/euclidean-gas/lecture/convergence.js";
import {
  assignment,
  transport,
  perron,
  mv,
  transpose,
  qsd,
  killedKernel,
  reservoir,
  collision,
  geometricPartition,
  linearBAOAB,
  covarianceStep,
} from "../../web/euclidean-gas/lecture/convergence-math.js";
await init({
  module_or_path: await readFile(
    new URL("../../web/euclidean-gas/engine/cpu/gas_bg.wasm", import.meta.url),
  ),
});
const engine = {
  defaults: async () => JSON.parse(default_config()),
  create: async (c) => BrowserGas.create(JSON.stringify(c)),
};
const make = (id, params = {}, seed = 7) =>
  demos.find((d) => d.id === id).create({ params, seed, engine });
const value = (s, label) => s.metrics.find((m) => m.label === label)?.value;
const near = (a, b, tolerance = 1e-9) =>
  assert.ok(Math.abs(a - b) <= tolerance, `${a} ≈ ${b}`);
function finite(s) {
  for (const c of s.charts) {
    for (const row of c.matrix ?? [])
      row.forEach((x) => assert.ok(Number.isFinite(x)));
    for (const series of c.series ?? [])
      for (const row of series.points)
        row.forEach((x) => assert.ok(Number.isFinite(x), `${c.title}: ${x}`));
  }
  for (const m of s.metrics)
    if (typeof m.value === "number")
      assert.ok(Number.isFinite(m.value), m.label);
}

test("All 16 distinct convergence experiments execute real computation and reset deterministically", async () => {
  assert.equal(demos.length, 16);
  assert.equal(new Set(demos.map((d) => d.id)).size, 16);
  for (const d of demos) {
    const a = await make(d.id),
      b = await make(d.id);
    try {
      finite(a.snapshot());
      await a.step();
      await b.step();
      assert.deepEqual(a.snapshot(), b.snapshot(), `${d.id} reset determinism`);
      assert.ok(a.snapshot().step > 0, `${d.id} progresses`);
      finite(a.snapshot());
    } finally {
      a.dispose();
      b.dispose();
    }
  }
});

test("Hungarian matches exhaustive optimum; translation splits off barycenter cost", () => {
  const a = [
      [0, 0],
      [2, 0],
      [0, 2],
      [3, 2],
    ],
    b = [
      [2.4, 1],
      [0.3, 2],
      [2, 3],
      [-1, 0],
    ];
  function permutations(a) {
    return a.length
      ? a.flatMap((v, i) =>
          permutations(a.filter((_, j) => i !== j)).map((p) => [v, ...p]),
        )
      : [[]];
  }
  const brute = Math.min(
    ...permutations([0, 1, 2, 3]).map(
      (p) =>
        a.reduce(
          (s, x, i) => s + x.reduce((t, v, j) => t + (v - b[p[i]][j]) ** 2, 0),
          0,
        ) / 4,
    ),
  );
  near(assignment(a, b).cost, brute);
  const t = transport(a, b);
  near(t.cost, t.centered + t.barycenter);
  assert.ok(t.centered <= t.proxy + 1e-12);
  const shifted = [a[2], a[0], a[3], a[1]].map((x) => [x[0] + 3, x[1] - 2]);
  const translated = transport(a, shifted);
  near(translated.cost, 13);
  near(translated.centered, 0);
  assert.ok(translated.label > translated.cost);
});

test("Complete-linkage partition uses minimum-size and cumulative-contribution rules", () => {
  const points = [
    ...Array.from({ length: 20 }, (_, i) => [i * 0.005, 0]),
    ...Array.from({ length: 5 }, (_, i) => [4 + i * 0.005, 0]),
  ];
  const p = geometricPartition(points, 0.2);
  assert.equal(p.clusters.length, 2);
  assert.equal(p.minimum, 5);
  assert.equal(
    p.high.size,
    25,
    "90% contribution requires both valid clusters in this 20/5 fixture",
  );
  const outlier = geometricPartition([...points.slice(0, 20), [4, 0]], 0.2);
  assert.ok(outlier.high.has(20), "undersized cluster is high-error");
});

test("WASM frozen clone-only fixture and paired transport obey geometry identities", async () => {
  const clone = await make("II-01", { walkers: 64, beta: 0 }),
    pair = await make("II-04", { translation: 2, noise: 0 });
  try {
    for (let i = 0; i < 12; i++) await clone.step();
    assert.ok(value(clone.snapshot(), "Mean high-error Δradius²") < 0);
    near(value(pair.snapshot(), "Centered optimum"), 0);
    assert.ok(value(pair.snapshot(), "Variance proxy") > 0);
    for (let i = 0; i < 4; i++) {
      await pair.step();
      assert.ok(value(pair.snapshot(), "Envelope gap") >= -1e-10);
    }
  } finally {
    clone.dispose();
    pair.dispose();
  }
});

test("OU shared innovations give the exact memory decay path", async () => {
  const m = await make("II-05");
  try {
    for (let i = 0; i < 100; i++) await m.step();
    const c = m
      .snapshot()
      .charts.find((c) => c.title === "Shared-innovation coupling");
    for (let i = 0; i < c.series[0].points.length; i++)
      near(c.series[0].points[i][1], c.series[1].points[i][1], 1e-11);
    assert.ok(value(m.snapshot(), "Velocity variance") > 0.2);
  } finally {
    m.dispose();
  }
});

test("One BAOAB kick has rank one; two steps generate phase-space covariance", () => {
  const { M, g } = linearBAOAB(0.1, 1, 1, 2),
    c1 = covarianceStep(
      [
        [0, 0],
        [0, 0],
      ],
      M,
      g,
    ),
    c2 = covarianceStep(c1, M, g);
  near(c1[0][0] * c1[1][1] - c1[0][1] ** 2, 0, 1e-14);
  assert.ok(c2[0][0] * c2[1][1] - c2[0][1] ** 2 > 0);
});

test("Positive left weights satisfy comparison inequalities and QSD eigenmeasure", () => {
  const A = [
      [0.7, 0.2],
      [0.1, 0.5],
    ],
    { weights, r } = perron(A);
  mv(transpose(A), weights).forEach((x, i) => near(x, r * weights[i], 1e-11));
  assert.ok(r < 1);
  const k = killedKernel(0.1),
    q = qsd(k);
  near(
    q.law.reduce((a, b) => a + b, 0),
    1,
  );
  mv(transpose(k), q.law).forEach((x, i) => near(x, q.alpha * q.law[i], 1e-11));
  assert.ok(q.alpha < 1);
});

test("Starting the killed chain from its QSD preserves conditional shape and geometric survival", async () => {
  const m = await make("II-08", { initial: "qsd" });
  try {
    for (let i = 0; i < 30; i++) await m.step();
    const s = m.snapshot(),
      survival = s.charts.find((c) => c.title === "Unnormalized survival");
    near(
      survival.series[0].points.at(-1)[1],
      value(s, "QSD survival eigenvalue α") ** 30,
      1e-11,
    );
    const shape = s.charts.find(
      (c) => c.title === "Conditional shape among survivors",
    );
    shape.series[0].points.forEach((x, i) =>
      near(x[1], shape.series[2].points[i][1], 1e-10),
    );
  } finally {
    m.dispose();
  }
});

test("Reservoir exact solution conserves ledger and internal replacement cancels", async () => {
  near(reservoir(0, 0.5, 1, 0.3), 0.3);
  near(reservoir(100, 0.5, 1, 0.3), 2 / 3);
  const a = await make("III-01", { attempt: 0 }),
    b = await make("III-01", { attempt: 5 });
  try {
    for (let i = 0; i < 50; i++) {
      await a.step();
      await b.step();
    }
    near(value(a.snapshot(), "Alive mass"), value(b.snapshot(), "Alive mass"));
    near(value(b.snapshot(), "Ledger residual"), 0, 1e-12);
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("Euler stationary generator bias and constant-test residual are exact", async () => {
  const m = await make("III-04", { test: "constant" });
  try {
    await m.step();
    for (const c of m.snapshot().charts.slice(0, 2))
      c.matrix.flat().forEach((v) => near(v, 0));
  } finally {
    m.dispose();
  }
});

test("Local replicator approaches normalized predicted profile with equalized fitness", async () => {
  const m = await make("III-05");
  try {
    const initial = value(m.snapshot(), "L¹ profile error");
    for (let i = 0; i < 200; i++) await m.step();
    near(value(m.snapshot(), "Total quadrature mass"), 1);
    assert.ok(value(m.snapshot(), "L¹ profile error") < initial * 0.01);
    assert.ok(value(m.snapshot(), "Fitness range") < 0.05);
  } finally {
    m.dispose();
  }
});

test("Diffusion separates sine conditional equilibrium from source-balanced parabola", async () => {
  const m = await make("III-06");
  try {
    for (let i = 0; i < 200; i++) await m.step();
    const s = m.snapshot();
    assert.ok(value(s, "Conditional L¹ error") < 1e-8);
    near(
      value(s, "Source-fed mass"),
      value(s, "Parabola mass sL³/(12D)"),
      2e-5,
    );
    assert.ok(value(s, "Source-free mass") < 1e-6);
  } finally {
    m.dispose();
  }
});

test("Finite-label correction vanishes for k=1 and respects union bound", async () => {
  for (const n of [8, 32, 512])
    for (let k = 1; k <= Math.min(n, 32); k++)
      assert.ok(
        collision(n, k) <= Math.min(1, (k * (k - 1)) / (2 * n)) + 1e-12,
      );
  near(collision(64, 1), 0);
  const m = await make("III-08", { tuple: 1 });
  try {
    for (let i = 0; i < 3; i++) await m.step();
    near(value(m.snapshot(), "Observed collision frequency"), 0);
  } finally {
    m.dispose();
  }
});

// Regression cases from the full-horizon teaching audit.
async function finish(m, maxTicks = 1000) {
  let ticks = 0;
  while (!m.snapshot().done && ticks < maxTicks) {
    await m.step();
    ticks++;
  }
  assert.equal(
    m.snapshot().done,
    true,
    "experiment reaches its declared horizon",
  );
  finite(m.snapshot());
  return m.snapshot();
}

test("Keystone fixture has nonempty H/L and satisfies the displayed overlap theorem", async () => {
  for (const fraction of [0.05, 0.15, 0.4]) {
    const m = await make("II-01", { fraction });
    try {
      for (let k = 0; k < 8; k++) {
        await m.step();
        const s = m.snapshot();
        assert.ok(
          value(s, "High-error fraction") > 0 &&
            value(s, "Low-error fraction") > 0,
        );
        assert.equal(value(s, "Theorem hypotheses"), "Satisfied");
        assert.ok(value(s, "Latest low-minus-high fitness gap") > 0);
        assert.ok(value(s, "Latest overlap lower bound") > 0);
        assert.ok(
          value(s, "Latest overlap") + 1e-12 >=
            value(s, "Latest overlap lower bound"),
        );
        near(value(s, "Variance identity residual"), 0, 1e-11);
      }
    } finally {
      m.dispose();
    }
  }
  const neutral = await make("II-01", { alpha: 0, beta: 0 });
  try {
    await neutral.step();
    assert.equal(
      value(neutral.snapshot(), "Theorem hypotheses"),
      "Not satisfied",
    );
    near(value(neutral.snapshot(), "Latest overlap lower bound"), 0);
  } finally {
    neutral.dispose();
  }
});

test("Additive kinetics shows exact positive drift and cannot report a false equilibrium floor", async () => {
  const m = await make("II-02", {
    stage: "kinetic",
    jitter: 0.3,
    replicates: 48,
  });
  try {
    const s = await finish(m);
    near(value(s, "Exact constant variance drift"), 0.174375, 1e-12);
    assert.match(
      value(s, "Supported in-range zero crossing"),
      /No equilibrium floor/,
    );
    assert.ok(
      value(s, "Slope 95% lower") < 0 && value(s, "Slope 95% upper") > 0,
    );
    const c = s.charts.find((c) => c.title === "One-step conditional drift");
    const exact = c.series.find((s) => s.name.startsWith("Exact random-walk"));
    assert.ok(exact.points.every((p) => Math.abs(p[1] - 0.174375) < 1e-12));
    assert.ok(
      c.segments.every(([lo, hi]) => lo[1] <= 0.174375 && hi[1] >= 0.174375),
    );
  } finally {
    m.dispose();
  }
  const noNoise = await make("II-02", { jitter: 0 }),
    jitter = await make("II-02", { jitter: 0.3 });
  try {
    const a = await finish(noNoise),
      b = await finish(jitter);
    assert.equal(typeof value(a, "Supported in-range zero crossing"), "string");
    assert.equal(typeof value(b, "Supported in-range zero crossing"), "number");
    assert.ok(value(b, "Supported in-range zero crossing") > 0.02);
    assert.ok(value(b, "Intercept 95% lower") > 0);
  } finally {
    noNoise.dispose();
    jitter.dispose();
  }
});

test("Sorted one-coordinate transport matches optimal assignment after an individual point edit", async () => {
  const m = await make("II-03", {
    cost: "oneD",
    point: 3,
    pointX: 1.2,
    translation: 0.7,
  });
  try {
    await m.step();
    const s = m.snapshot();
    near(
      value(s, "Sorted one-coordinate cost"),
      value(s, "Optimal W₂²"),
      1e-11,
    );
    near(value(s, "Decomposition error"), 0, 1e-11);
  } finally {
    m.dispose();
  }
});

test("Slow-friction OU reaches eight relaxation times in bounded display updates", async () => {
  const m = await make("II-05", { gamma: 0.1, h: 0.005 });
  try {
    const s = await finish(m, 400);
    near(value(s, "Elapsed friction times γt"), 8, 1e-10);
    assert.equal(s.step, 16000);
    const c = s.charts.find((c) => c.title === "Shared-innovation coupling");
    near(c.series[0].points.at(-1)[1], c.series[1].points.at(-1)[1], 1e-10);
    const mean = s.charts.find((c) => c.title === "Memory of the initial mean");
    assert.ok(mean.series.some((c) => c.name === "Mean 95% upper"));
    const position = s.charts.find(
      (c) => c.title === "Position integration has its own reference",
    );
    assert.ok(position.series[1].points.at(-1)[1] > 0);
    assert.ok(
      s.charts.every((c) =>
        (c.series ?? []).every((series) => series.points.length <= 400),
      ),
    );
  } finally {
    m.dispose();
  }
});

test("Empirical QSD horizon retains survivors while full exact-law mode exposes depletion", async () => {
  const empirical = await make("II-08"),
    full = await make("II-08", { horizon: "exact" });
  try {
    const a = await finish(empirical),
      b = await finish(full);
    assert.ok(a.step < 60);
    assert.ok(value(a, "Surviving replicas") > 0);
    assert.ok(
      Math.min(
        value(a, "Expected surviving replicas"),
        value(a, "Surviving replicas"),
      ) <= value(a, "Empirical stopping threshold"),
    );
    const shape = a.charts.find(
      (c) => c.title === "Conditional shape among survivors",
    );
    assert.equal(shape.segments.length, 3);
    assert.equal(b.step, 160);
    assert.equal(value(b, "Surviving replicas"), 0);
    assert.match(value(b, "Conditional sample stage"), /depleted/);
  } finally {
    empirical.dispose();
    full.dispose();
  }
});

test("Pair histogram permutation null distinguishes sampling roughness from shared labels", async () => {
  const { random, permutationJoint, tanhGaussianVariance } = await import(
    "../../web/euclidean-gas/lecture/convergence-math.js"
  );
  const r = random(7),
    pairs = Array.from({ length: 96 }, () => [
      Math.tanh(r.normal()),
      Math.tanh(r.normal()),
    ]);
  const independent = permutationJoint(pairs, 7),
    shared = permutationJoint(
      pairs.map(([x]) => [x, x]),
      7,
    );
  assert.ok(
    independent.l1 > 0.3,
    "independent finite histogram still looks rough",
  );
  assert.ok(
    independent.lower < independent.l1 && independent.upper > independent.l1,
  );
  assert.ok(independent.pValue > 0.05);
  assert.ok(shared.pValue < 0.01);
  near(tanhGaussianVariance(1.48), 0.4658473000951703, 1e-10);
});

test("Empirical-law engine runs active copying and a measured no-copy comparison", async () => {
  const m = await make("III-03", { replicas: 48 });
  try {
    const s = await finish(m);
    assert.ok(value(s, "Active accepted clone fraction") > 0.1);
    assert.ok(value(s, "Active accepted clone fraction") < 0.5);
    assert.equal(value(s, "Independent completed runs"), 384);
    for (const row of s.table.rows) {
      const [n, , variance, , , , , , marginal, covariance] = row;
      near(variance, marginal + (1 - 1 / n) * covariance, 1e-12);
      assert.ok(row[3] <= row[4]);
      assert.ok(row[6] <= row[7]);
    }
    assert.ok(
      s.charts.find(
        (c) => c.title === "Pair histogram calibrated against independence",
      ).segments.length === 4,
    );
    const stat = value(s, "Selected joint-product L¹"),
      upper = value(s, "Permutation-null 97.5% quantile");
    assert.ok(stat >= 0 && upper > 0);
  } finally {
    m.dispose();
  }
  const noisy = await make("III-03", {
    selection: 0,
    noise: 0.2,
    updates: 12,
    replicas: 48,
  });
  try {
    const s = await finish(noisy);
    near(value(s, "Active accepted clone fraction"), 0);
    near(
      value(s, "Exact no-copy evolved marginal coefficient"),
      0.4658473000951703,
      1e-10,
    );
  } finally {
    noisy.dispose();
  }
});

test("Paired stationary residual exposes O(h) bias and quantifies noisy raw cells", async () => {
  const m = await make("III-04");
  try {
    const s = await finish(m);
    near(value(s, "N=256 paired identity residual"), 0, 1e-12);
    near(
      value(s, "N=256 discrete sampling SE at h=.05"),
      Math.sqrt(8 / 256),
      1e-12,
    );
    for (const row of s.table.rows) {
      const [
        n,
        h,
        discrete,
        expectedDiscrete,
        discreteCI,
        continuous,
        expectedContinuous,
        continuousCI,
        paired,
        expectedPaired,
        pairedCI,
      ] = row;
      near(expectedDiscrete, 0, 1e-12);
      near(expectedContinuous, -h / (1 - h / 2), 1e-12);
      near(discreteCI, 1.96 * Math.sqrt(8 / n), 1e-12);
      near(expectedPaired, h / (1 - h / 2), 1e-12);
      near(paired, discrete - continuous, 1e-12);
      assert.ok(pairedCI < continuousCI);
    }
    const fine64 = s.table.rows.find((r) => r[0] === 64 && r[1] === 0.05);
    assert.ok(fine64[5] > 0, "audit setting has positive noisy raw residual");
    assert.ok(
      Math.abs(fine64[5] - fine64[6]) < fine64[7],
      "sampling interval explains its sign",
    );
  } finally {
    m.dispose();
  }
});

test("Equilibrium relaxation and spatial reconstruction error are measured separately", async () => {
  const low = await make("III-05", { resolution: 48 }),
    high = await make("III-05", { resolution: 192 });
  try {
    const a = await finish(low),
      b = await finish(high);
    assert.ok(
      value(a, "L¹ profile error") < 1e-10 &&
        value(b, "L¹ profile error") < 1e-10,
    );
    assert.ok(
      value(a, "L¹ reconstruction error against 3072-cell reference") > 0.005,
    );
    assert.ok(
      value(b, "L¹ reconstruction error against 3072-cell reference") <
        value(a, "L¹ reconstruction error against 3072-cell reference") / 3,
    );
  } finally {
    low.dispose();
    high.dispose();
  }
});

test("Spectral source error decreases at the predicted cubic tail rate", async () => {
  const errors = [];
  for (const modes of [15, 31, 63]) {
    const m = await make("III-06", { modes });
    try {
      const s = await finish(m);
      const error = value(s, "Stationary source-mass truncation error");
      errors.push(error);
      assert.ok(
        error > 0 && error < value(s, "Source-mass spectral tail bound"),
      );
      assert.ok(
        value(s, "Source-mass truncation error vs 2047 modes") <=
          value(s, "Source-mass spectral tail bound"),
      );
    } finally {
      m.dispose();
    }
  }
  assert.ok(errors[0] / errors[1] > 7 && errors[1] / errors[2] > 7);
  const zero = await make("III-06", { source: 0 });
  try {
    const s = await finish(zero);
    near(value(s, "Source-mass truncation error vs 2047 modes"), 0);
  } finally {
    zero.dispose();
  }
});
