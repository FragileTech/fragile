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
