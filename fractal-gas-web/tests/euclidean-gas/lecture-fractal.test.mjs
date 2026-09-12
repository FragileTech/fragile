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
