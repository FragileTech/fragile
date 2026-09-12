import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import init, {
  BrowserGas,
  default_config,
} from "../../web/euclidean-gas/engine/cpu/gas.js";
import {
  demos,
  donorProbabilities,
  enumerateAssignments,
  probabilityField,
} from "../../web/euclidean-gas/lecture/foundations.js";
import { positions, velocities } from "../../web/euclidean-gas/lecture/math.js";
await init({
  module_or_path: await readFile(
    new URL("../../web/euclidean-gas/engine/cpu/gas_bg.wasm", import.meta.url),
  ),
});
const engine = {
  defaults: async () => JSON.parse(default_config()),
  create: (c) => BrowserGas.create(JSON.stringify(c)),
};
const defaultParams = (d) =>
  Object.fromEntries(d.controls.map((c) => [c.key, c.value]));
function finiteCharts(s) {
  assert.ok(s.charts.length > 0);
  for (const c of s.charts)
    for (const series of c.series ?? [])
      for (const pair of series.points) {
        assert.equal(pair.length, 2);
        assert.ok(pair.every(Number.isFinite), `${c.title}: ${pair}`);
      }
  for (const m of s.metrics ?? [])
    if (typeof m.value === "number")
      assert.ok(Number.isFinite(m.value), m.label);
}
for (const d of demos)
  test(`${d.id} has finite real computations and deterministic reset`, async () => {
    let a, b;
    try {
      const args = { params: defaultParams(d), seed: 7, engine };
      a = await d.create(args);
      finiteCharts(a.snapshot());
      await a.step();
      await a.step();
      const result = a.snapshot();
      finiteCharts(result);
      assert.deepEqual(a.snapshot(), result, "snapshot is pure");
      b = await d.create(args);
      await b.step();
      await b.step();
      assert.deepEqual(b.snapshot(), result);
    } finally {
      a?.dispose();
      b?.dispose();
    }
  });
for (const id of ["I-02", "I-04", "I-10"])
  test(`${id} non-default controls run`, async () => {
    const d = demos.find((x) => x.id === id),
      params = {
        ...defaultParams(d),
        ...(id === "I-02"
          ? { walkers: 5, law: "greedy", count: 4 }
          : id === "I-04"
            ? { survivors: 1, boundary: "periodic" }
            : { velocityWeight: 0 }),
      };
    const m = await d.create({ params, seed: 21, engine });
    try {
      await m.step();
      finiteCharts(m.snapshot());
      if (id === "I-04")
        assert.equal(
          m.snapshot().metrics.find((v) => v.label === "Eligible slots").value,
          16,
        );
    } finally {
      m.dispose();
    }
  });
test("zero survivors is a terminal retained frame, not an unhandled failure", async () => {
  const d = demos.find((d) => d.id === "I-04"),
    m = await d.create({
      params: { ...defaultParams(d), survivors: 0 },
      seed: 7,
      engine,
    });
  try {
    await m.step();
    assert.equal(m.snapshot().done, true);
    assert.equal(m.snapshot().metrics[0].value, 0);
  } finally {
    m.dispose();
  }
});
test("exact small laws normalize and mutual assignments are reciprocal", () => {
  const x = [
    [-1, 0],
    [-0.5, 0.1],
    [0.5, 0],
    [1, 0.1],
    [1.5, 0],
  ];
  for (const law of ["independent", "mutual", "greedy"]) {
    let total = 0;
    enumerateAssignments(x, 0.5, law, (a, w) => {
      total += w;
      if (law !== "independent")
        for (let i = 0; i < a.length; i++) assert.equal(a[a[i]], i);
    });
    assert.ok(Math.abs(total - 1) < 1e-10, law);
  }
  const f = probabilityField(x);
  assert.ok(
    Math.abs(f.persistence + f.joint.reduce((a, b) => a + b) - 1) < 1e-12,
  );
  assert.ok(f.joint.every((x) => x >= 0));
  const q = donorProbabilities(x, 0.2);
  assert.ok(
    q.every(
      (row, i) =>
        row[i] === 0 && Math.abs(row.reduce((a, b) => a + b) - 1) < 1e-12,
    ),
  );
});
test("fixture replacement validates before commit and refreshes translated rewards", async () => {
  const c = await engine.defaults();
  c.walkers = 4;
  c.benchmark = "quadratic";
  c.reward_shift = [1, 0];
  c.potential = "quadratic";
  c.gas.precision = "f64";
  c.gas.boundary = { kind: "unbounded" };
  const r = await engine.create(c);
  try {
    const f = await r.set_population(
      JSON.stringify({
        positions: [
          [1, 0],
          [2, 0],
          [0, 0],
          [1, 1],
        ],
        alive: [true, true, false, true],
      }),
    );
    assert.deepEqual(f.population.rewards.raw, [0, 0.5, 0.5, 0.5]);
    assert.equal(f.population.validity[2].terminated, true);
    const before = r.checkpoint();
    await assert.rejects(
      r.set_population(JSON.stringify({ positions: [[0, 0]] })),
    );
    assert.deepEqual(r.checkpoint(), before);
  } finally {
    r.free();
  }
});
test("BAOAB trace does not change the committed run and final force uses post-A position", async () => {
  const c = await engine.defaults();
  c.walkers = 4;
  c.benchmark = "quadratic";
  c.gas.precision = "f64";
  c.gas.boundary = { kind: "unbounded" };
  c.gas.fitness.reward_exponent = 0;
  c.gas.fitness.diversity_exponent = 0;
  c.gas.kinetic = {
    integrator: {
      kind: "baoab",
      positions: "positions",
      velocities: "velocities",
      dt: 0.04,
      friction: 1,
    },
    noise: {
      innovation: "gaussian",
      geometry: { kind: "isotropic", scale: { kind: "constant", values: [0] } },
    },
  };
  const a = await engine.create(c),
    b = await engine.create(c);
  try {
    a.set_trace(true);
    const fa = await a.step(1),
      fb = await b.step(1);
    assert.deepEqual(fa.population, fb.population);
    assert.deepEqual(fa.report, fb.report);
    assert.deepEqual(
      fa.trace.map((t) => t.stage),
      [
        "pre_clone",
        "literal_clone",
        "post_transform",
        "B1",
        "A1",
        "O",
        "A2",
        "B2",
        "post_kinetic",
      ],
    );
    const stage = (name) => fa.trace.find((t) => t.stage === name),
      A = positions(stage("A2")),
      before = velocities(stage("A2")),
      after = velocities(stage("B2"));
    after.forEach((row, i) =>
      row.forEach((v, j) =>
        assert.ok(Math.abs(v - (before[i][j] - 0.02 * A[i][j])) < 1e-12),
      ),
    );
    assert.deepEqual(
      a.checkpoint(),
      b.checkpoint(),
      "traces are not serialized into checkpoint or RNG",
    );
  } finally {
    a.free();
    b.free();
  }
});
test("restitution trace recovers pair momentum and a-squared relative energy", async () => {
  const d = demos.find((d) => d.id === "I-09");
  for (const restitution of [0, 0.5, 1]) {
    const m = await d.create({
      params: { ...defaultParams(d), restitution },
      seed: 7,
      engine,
    });
    try {
      const s = m.snapshot();
      assert.ok(s.metrics[0].value < 1e-12);
      assert.ok(Math.abs(s.metrics[1].value - s.metrics[2].value) < 1e-12);
    } finally {
      m.dispose();
    }
  }
});

test("every declared Part I control endpoint admits a finite experiment", async () => {
  for (const d of demos)
    for (const control of d.controls) {
      const values =
        control.type === "range"
          ? [control.min, control.max]
          : control.options.map((o) => o.value);
      for (const value of values) {
        if (value === control.value) continue;
        let model;
        try {
          model = await d.create({
            params: { ...defaultParams(d), [control.key]: value },
            seed: 13,
            engine,
          });
          await model.step();
          finiteCharts(model.snapshot());
        } catch (e) {
          throw new Error(`${d.id} ${control.key}=${value}: ${e}`, {
            cause: e,
          });
        } finally {
          model?.dispose();
        }
      }
    }
});

test("exact conditional accepted-copy field agrees with independent engine draws", async () => {
  const x = [
      [-1, 0.2],
      [-0.5, 0.1],
      [0.5, 0],
      [1.2, 0.1],
    ],
    n = 1500;
  const c = await engine.defaults();
  c.walkers = 4;
  c.benchmark = "sphere";
  c.gas.precision = "f64";
  c.gas.boundary = { kind: "unbounded" };
  c.gas.kinetic.integrator.amplitude = 0;
  c.gas.distance_donors.kernel = { kind: "gaussian", width: 0.7 };
  c.gas.cloning_donors.kernel = { kind: "gaussian", width: 0.8 };
  const predicted = probabilityField(x, { width: 0.7, cloneWidth: 0.8 }),
    counts = Array(4).fill(0),
    run = await engine.create(c);
  try {
    for (let i = 0; i < n; i++) {
      await run.set_population(JSON.stringify({ positions: x }));
      const { report: r } = await run.step(1);
      if (r.clone_plan.choices[0].accepted)
        counts[r.clone_plan.sources[r.cloning_companions.indices[0]].slot]++;
    }
    predicted.joint.forEach((p, j) =>
      assert.ok(
        Math.abs(counts[j] / n - p) < 0.045,
        `donor ${j}: observed ${counts[j] / n}, predicted ${p}`,
      ),
    );
  } finally {
    run.free();
  }
});
