import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import init, {
  BrowserGas,
  default_config,
  partv_geometry,
  partv_analysis,
} from "../../web/euclidean-gas/engine/cpu/gas.js";
import { demos } from "../../web/euclidean-gas/lecture/fractal.js";
import { parameters } from "../../web/euclidean-gas/lecture/catalog.js";
await init({
  module_or_path: await readFile(
    new URL("../../web/euclidean-gas/engine/cpu/gas_bg.wasm", import.meta.url),
  ),
});
const engine = {
  defaults: async () => JSON.parse(default_config()),
  create: (c) => BrowserGas.create(JSON.stringify(c)),
  geometry: async (r) => partv_geometry(JSON.stringify(r)),
  analysis: async (r) => partv_analysis(JSON.stringify(r)),
};
function finiteCharts(snapshot, id) {
  assert.ok(snapshot.charts.length, id);
  let points = 0;
  for (const chart of snapshot.charts) {
    for (const s of chart.series || [])
      for (const p of s.points) {
        assert.ok(p.every(Number.isFinite), `${id}: ${s.name}: ${p}`);
        points++;
      }
    if (chart.matrix) {
      assert.ok(chart.matrix.flat().every(Number.isFinite), id);
      points += chart.matrix.length;
    }
  }
  assert.ok(points > 0, id);
  for (const m of snapshot.metrics)
    if (typeof m.value === "number")
      assert.ok(Number.isFinite(m.value), `${id}: ${m.label}`);
}
for (const demo of demos)
  test(
    demo.id + " initializes, advances native computations, and replays seed",
    async () => {
      const args = { params: parameters(demo), seed: 7, engine };
      let first;
      const model = await demo.create(args);
      try {
        finiteCharts(model.snapshot(), demo.id);
        await model.step();
        await model.step();
        first = model.snapshot();
        finiteCharts(first, demo.id);
        assert.equal(first.step, 2);
      } finally {
        model.dispose();
      }
      const replay = await demo.create(args);
      try {
        await replay.step();
        await replay.step();
        const result = replay.snapshot();
        assert.deepEqual(result.charts, first.charts);
        assert.deepEqual(result.metrics, first.metrics);
      } finally {
        replay.dispose();
      }
    },
  );
test("WASM metric identities and empirical O covariance use full thermostat prefactor", () => {
  const g = partv_geometry(
    JSON.stringify({
      kind: "metric",
      hessian: [
        [2, 0.3],
        [0.3, 1],
      ],
      epsilon: 0.5,
      policy: "strict",
    }),
  );
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++) {
      const product =
        g.metric[i][0] * g.inverse[0][j] + g.metric[i][1] * g.inverse[1][j];
      assert.ok(Math.abs(product - (i === j ? 1 : 0)) < 1e-12);
    }
  const r = partv_geometry(
    JSON.stringify({
      kind: "ou",
      metric: g.metric,
      gamma: 1.3,
      temperature: 0.7,
      dt: 0.08,
      samples: 16384,
      seed: 19,
    }),
  );
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++) {
      assert.ok(
        Math.abs(
          r.expected_covariance[i][j] -
            0.7 * (1 - Math.exp(-2 * 1.3 * 0.08)) * g.inverse[i][j],
        ) < 1e-12,
      );
      assert.ok(
        Math.abs(r.sample_covariance[i][j] - r.expected_covariance[i][j]) <
          6 * r.covariance_standard_error[i][j],
      );
    }
});
test("WASM dimension reference uses Alexandrov fraction 8/35", () => {
  const r = partv_analysis(
    JSON.stringify({ kind: "dimension", samples: 256, replicas: 32, seed: 9 }),
  );
  assert.ok(
    r.metrics.some(
      (m) =>
        m.reference !== undefined && Math.abs(m.reference - 8 / 35) < 1e-12,
    ),
  );
});

test("Every Part V scientific control endpoint produces finite measured output", async () => {
  for (const demo of demos)
    for (const control of demo.controls) {
      const values =
        control.type === "select"
          ? [control.options[0].value, control.options.at(-1).value]
          : [control.min, control.max];
      for (const value of new Set(values)) {
        const params = parameters(demo, { [control.key]: value });
        const model = await demo.create({ params, seed: 7, engine });
        try {
          await model.step();
          finiteCharts(model.snapshot(), `${demo.id}/${control.key}=${value}`);
        } finally {
          model.dispose();
        }
      }
    }
});
test("Recording batches preserve every microstep through WASM checkpoint continuation", async () => {
  const c = await engine.defaults();
  c.walkers = 8;
  c.dimensions = 2;
  c.gas.precision = "f64";
  c.gas.backend = "cpu";
  const a = await engine.create(c),
    b = await engine.create(c);
  try {
    a.start_recording("{}");
    b.start_recording("{}");
    await a.step(16);
    for (let j = 0; j < 16; j++) await b.step(1);
    assert.deepEqual(a.archive(), b.archive());
    assert.equal(a.archive().steps.length, 16);
    const restored = await BrowserGas.restore(a.checkpoint());
    try {
      assert.deepEqual(restored.archive(), a.archive());
      await restored.step(1);
      await a.step(1);
      assert.deepEqual(restored.fractal_set(), a.fractal_set());
    } finally {
      restored.free();
    }
  } finally {
    a.free();
    b.free();
  }
});

test("Physical time cuts retain partial faces and temporal edges", async () => {
  const { clipPolygonAtTime, clipSegmentAtTime } = await import(
    "../../web/euclidean-gas/lecture/scene.js"
  );
  const face = [
    [0, 0, 0],
    [2, 0, 0],
    [2, 0, 2],
    [0, 0, 2],
  ];
  const cut = clipPolygonAtTime(face, 0.5);
  assert.deepEqual(cut, [
    [0, 0, 0],
    [2, 0, 0],
    [2, 0, 0.5],
    [0, 0, 0.5],
  ]);
  // Shoelace area in the x/time plane: retained area is 2 * 0.5 = 1.
  assert.equal(
    Math.abs(
      cut.reduce((sum, a, i) => {
        const b = cut[(i + 1) % cut.length];
        return sum + a[0] * b[2] - b[0] * a[2];
      }, 0),
    ) / 2,
    1,
  );
  assert.deepEqual(clipSegmentAtTime([0, 0, 0], [2, 4, 2], 0.5), [
    [0, 0, 0],
    [0.5, 1, 0.5],
  ]);
  assert.deepEqual(clipSegmentAtTime([2, 4, 2], [0, 0, 0], 0.5), [
    [0.5, 1, 0.5],
    [0, 0, 0],
  ]);
  assert.deepEqual(clipPolygonAtTime(face, -1), []);
});

test("Harmonic display uses unbiased variance and the correct initial uniform uncertainty", async () => {
  const demo = demos.find((d) => d.id === "V-08");
  const initialVariance = 1.8 ** 2 / 3;
  for (const walkers of [64, 256]) {
    const model = await demo.create({
      params: parameters(demo, { walkers }),
      seed: 7,
      engine,
    });
    try {
      const snapshot = model.snapshot();
      const frame = model.archive().anchors[0].population;
      const xs = frame.observations.fields.positions.values.filter(
        (_, j) => j % 2 === 0,
      );
      const mean = xs.reduce((a, x) => a + x, 0) / walkers;
      const unbiased =
        xs.reduce((a, x) => a + (x - mean) ** 2, 0) / (walkers - 1);
      const chart = snapshot.charts[0];
      assert.ok(
        Math.abs(
          chart.series.find((s) => s.name === "Unbiased measured x₁ variance")
            .points[0][1] - unbiased,
        ) < 1e-14,
      );
      assert.ok(
        Math.abs(
          chart.series.find((s) => s.name === "Exact transient prediction")
            .points[0][1] - initialVariance,
        ) < 1e-14,
      );
      // Uniform fourth central moment a^4/5, independent of native cumulant code.
      const se = Math.sqrt(
        (1.8 ** 4 / 5 -
          ((walkers - 3) / (walkers - 1)) * initialVariance ** 2) /
          walkers,
      );
      assert.ok(
        Math.abs(
          snapshot.result.transient_variance_standard_errors[0][0] - se,
        ) < 1e-14,
      );
    } finally {
      model.dispose();
    }
  }
});

test("Anisotropic harmonic engine matches independent transient moments across replicas", async () => {
  const demo = demos.find((d) => d.id === "V-08");
  const n = 64,
    replicas = 32,
    h = 0.04,
    a = 1.8,
    damping = Math.exp(-h);
  // Independently derive the five BAOAB stages without using Rust's matrices.
  function advance(x, v, xi = 0) {
    v -= (h * x) / 2;
    x += (h * v) / 2;
    v = damping * v + xi;
    x += (h * v) / 2;
    v -= (h * x) / 2;
    return [x, v];
  }
  const colX = advance(1, 0),
    colV = advance(0, 1),
    noise = advance(0, 0, 1);
  const A = [
    [colX[0], colV[0]],
    [colX[1], colV[1]],
  ];
  const checkpoints = [0, 16, 64, 128];
  for (const anisotropy of [1, 8]) {
    let C = [
        [(a * a) / 3, 0],
        [0, 0],
      ],
      initialX = [1, 0];
    const expected = [];
    for (let step = 0; step <= 128; step++) {
      if (checkpoints.includes(step))
        expected.push({
          value: C[0][0],
          se: Math.sqrt(
            (((2 * n) / (n - 1)) * C[0][0] ** 2 -
              ((2 * a ** 4) / 15) * initialX[0] ** 4) /
              (n * replicas),
          ),
        });
      C = A.map((row, i) =>
        A.map(
          (other, j) =>
            row.reduce(
              (sum, v, k) =>
                sum + other.reduce((s, w, l) => s + v * w * C[k][l], 0),
              0,
            ) +
            (0.4 / anisotropy) * (1 - damping ** 2) * noise[i] * noise[j],
        ),
      );
      initialX = A.map((row) => row[0] * initialX[0] + row[1] * initialX[1]);
    }
    const observed = checkpoints.map(() => 0);
    for (let replica = 0; replica < replicas; replica++) {
      const model = await demo.create({
        params: parameters(demo, { walkers: n, anisotropy }),
        seed: 13007 + replica * 7919,
        engine,
      });
      try {
        for (let step = 0; step <= 128; step++) {
          if (checkpoints.includes(step)) {
            const snapshot = model.snapshot();
            const value = snapshot.charts[0].series
              .find((s) => s.name === "Unbiased measured x₁ variance")
              .points.at(-1)[1];
            observed[checkpoints.indexOf(step)] += value / replicas;
          }
          if (step < 128) await model.step();
        }
      } finally {
        model.dispose();
      }
    }
    for (let j = 0; j < checkpoints.length; j++) {
      const z = Math.abs(observed[j] - expected[j].value) / expected[j].se;
      assert.ok(
        z < 4,
        `anisotropy=${anisotropy}, step=${checkpoints[j]}, measured=${observed[j]}, predicted=${expected[j].value}, z=${z}`,
      );
    }
  }
});
