import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import init, {
  BrowserGas,
  default_config,
  partvi_analysis,
  partvi_archive,
  partvi_run,
  physics_curvature_batch,
} from "../../web/euclidean-gas/engine/cpu/gas.js";
import {
  demos,
  importedResultsModel,
} from "../../web/euclidean-gas/lecture/partvi.js";
import { parameters } from "../../web/euclidean-gas/lecture/catalog.js";
await init({
  module_or_path: await readFile(
    new URL("../../web/euclidean-gas/engine/cpu/gas_bg.wasm", import.meta.url),
  ),
});
const engine = {
  curvatureBatch: (r) => physics_curvature_batch(JSON.stringify(r)),
  defaults: async () => JSON.parse(default_config()),
  create: (c) => BrowserGas.create(JSON.stringify(c)),
  qft: async (r, a) =>
    a
      ? partvi_archive(JSON.stringify(r), JSON.stringify(a))
      : partvi_analysis(JSON.stringify(r)),
  qftRun: (r, c) => partvi_run(JSON.stringify(r), JSON.stringify(c)),
};
function finite(s, id) {
  assert.ok(s.charts.length, id);
  let count = 0;
  for (const c of s.charts)
    for (const a of c.series || [])
      for (const p of a.points) {
        assert.ok(p.every(Number.isFinite), `${id}/${a.name}: ${p}`);
        count++;
      }
  assert.ok(count, id + " contains computed points");
  for (const m of s.metrics)
    if (typeof m.value === "number") assert.ok(Number.isFinite(m.value));
  assert.ok(s.result.model);
}
for (const d of demos)
  test(d.id + " computes and replays compiled results", async () => {
    const p = parameters(d),
      args = { engine, seed: 7, params: p };
    let first;
    const model = await d.create(args);
    try {
      finite(model.snapshot(), d.id);
      await model.step();
      first = model.snapshot();
      finite(first, d.id);
    } finally {
      model.dispose();
    }
    const replay = await d.create(args);
    try {
      await replay.step();
      assert.deepEqual(replay.snapshot().result, first.result);
    } finally {
      replay.dispose();
    }
  });
test("Every Part VI control endpoint is numerically defined", async () => {
  for (const d of demos)
    for (const c of d.controls) {
      if (
        [
          "source",
          "walkers",
          "replicas",
          "horizon",
          "theta",
          "backend",
        ].includes(c.key) ||
        c.key.startsWith("engine_")
      )
        continue;
      const values =
        c.type === "select" ? c.options.map((x) => x.value) : [c.min, c.max];
      for (const v of new Set(values)) {
        let m;
        try {
          m = await d.create({
            engine,
            seed: 7,
            params: parameters(d, { source: "reference", [c.key]: v }),
          });
        } catch (error) {
          throw new Error(
            `${d.id}/${c.key}=${v}: ${error.message || String(error)}`,
          );
        }
        try {
          finite(m.snapshot(), d.id + "/" + c.key + "=" + v);
        } finally {
          m.dispose();
        }
      }
    }
});
test("Recorded color, noise, triplets and field budgets use actual 3D archives", async () => {
  for (const id of [
    2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 21, 32, 34, 35, 36, 39, 48,
    51, 52,
  ]) {
    const d = demos[id - 1];
    const m = await d.create({
      engine,
      seed: 9,
      params: parameters(d, { source: "recorded", walkers: 16 }),
    });
    try {
      finite(m.snapshot(), d.id);
      await m.step();
      finite(m.snapshot(), d.id);
      const a = await m.archive();
      assert.equal(
        a.anchors[0].population.observations.fields.positions.item_shape[0],
        3,
      );
      assert.ok(a.steps.length >= 8);
      assert.ok(m.snapshot().result.details.archive_steps >= 8);
    } finally {
      m.dispose();
    }
  }
});
test("Actual source and Noether continuation experiments have independent groups", async () => {
  for (const id of [19, 22, 45]) {
    const d = demos[id - 1],
      m = await d.create({
        engine,
        seed: 7,
        params: parameters(d, {
          source: "recorded",
          walkers: 16,
          replicas: 8,
          horizon: 1,
        }),
      });
    try {
      finite(m.snapshot(), d.id);
      assert.match(
        m.snapshot().result.model,
        /checkpoint|execut|continuation|engine/i,
      );
    } finally {
      m.dispose();
    }
  }
});
test("Native result imports preserve computed results and reject nonfinite curves", async () => {
  const r = partvi_analysis(
    JSON.stringify({
      experiment: 28,
      parameters: { amplitude: 0.7, rate: 0.3 },
    }),
  );
  const m = importedResultsModel({
    schema: "fragile-partvi-results-v1",
    results: [r],
  });
  assert.deepEqual(m.snapshot().result, r);
  const bad = structuredClone(r);
  bad.plots[0].series[0].points[0][1] = Infinity;
  assert.throws(() =>
    importedResultsModel({
      schema: "fragile-partvi-results-v1",
      results: [bad],
    }),
  );
});
test("Requests reject unsupported identities and malformed parameters", () => {
  for (const r of [
    { experiment: 0 },
    { experiment: 67 },
    { experiment: 3, parameters: [] },
    { experiment: 3, parameters: { n: { nested: 1 } } },
  ])
    assert.throws(() => partvi_analysis(JSON.stringify(r)));
});

test("Curvature CPU batch agrees with the host tensor calculation", async () => {
  const d = demos[38];
  for (const source of ["reference", "recorded"])
    for (const backend of ["cpu_f64", "cpu_f32"]) {
      const model = await d.create({
        engine,
        seed: 7,
        params: parameters(d, { source, backend, walkers: 16 }),
      });
      try {
        const result = model.snapshot().result,
          batch = result.details.backend_calculation;
        assert.equal(batch.backend, "cpu");
        assert.equal(batch.curvatures.length, 1);
        const expected = result.details.curvature.scalar,
          actual = batch.curvatures[0].scalar;
        assert.ok(
          Math.abs(actual - expected) <
            (backend === "cpu_f32" ? 5e-3 : 1e-8) *
              Math.max(1, Math.abs(expected)),
          `${source}/${backend}: ${actual} versus ${expected}`,
        );
      } finally {
        model.dispose();
      }
    }
});

test("Explicit formula labels select their corresponding workbench", async () => {
  const spec = JSON.parse(
    await readFile(
      new URL(
        "../../web/euclidean-gas/lecture/partvi-spec.json",
        import.meta.url,
      ),
    ),
  );
  const index = JSON.parse(
    await readFile(
      new URL(
        "../../web/euclidean-gas/lecture/partvi-formulas.json",
        import.meta.url,
      ),
    ),
  );
  for (const d of spec) {
    const entry = index.find((e) => e.label === d.target);
    if (entry) assert.equal(entry.experiment, d.id);
  }
  assert.equal(
    index.find(
      (e) => e.label === "prop-ym-native-labeled-color-reflection-sign",
    ).experiment,
    "VI-28",
  );
});

test("Archive imports require explicit current trace fields", async () => {
  const d = demos[7],
    model = await d.create({
      engine,
      seed: 7,
      params: parameters(d, { source: "recorded", walkers: 16 }),
    });
  try {
    const archive = await model.archive(),
      request = JSON.stringify({ experiment: 8 });
    const missingProviders = structuredClone(archive);
    delete missingProviders.providers;
    assert.throws(() =>
      partvi_archive(request, JSON.stringify(missingProviders)),
    );
    const missingShifts = structuredClone(archive);
    delete missingShifts.steps[0].noise[0].applied_source_shifts;
    assert.throws(() => partvi_archive(request, JSON.stringify(missingShifts)));
    const missingCoverage = structuredClone(archive);
    missingCoverage.steps[0].field_evaluations[0].available = [];
    assert.throws(() =>
      partvi_archive(request, JSON.stringify(missingCoverage)),
    );
  } finally {
    model.dispose();
  }
});

test("Algorithm controls reach the executed transition and provenance", async () => {
  const d = demos[47];
  for (const override of [
    { engine_memory: 8, engine_dt: 0.1, engine_friction: 0.5 },
    { engine_innovation: "standardized_uniform", engine_viscosity: 0 },
    { engine_boundary: "absorbing_box" },
    { engine_friction: 0 },
  ]) {
    const p = parameters(d, { source: "recorded", walkers: 16, ...override });
    const m = await d.create({ engine, seed: 29, params: p });
    try {
      finite(m.snapshot(), d.id);
      const r = m.snapshot().result;
      assert.equal(r.details.calculation_origin, "executed_algorithm_archive");
      const c = r.details.executed_gas_config;
      assert.equal(c.distance_donors.history_window, Number(p.engine_memory));
      assert.equal(c.cloning_donors.history_window, Number(p.engine_memory));
      assert.equal(c.kinetic.integrator.dt, Number(p.engine_dt));
      assert.equal(c.kinetic.integrator.friction, Number(p.engine_friction));
      assert.equal(c.kinetic.noise.innovation, p.engine_innovation);
      assert.equal(c.boundary.kind, p.engine_boundary);
      assert.equal(c.qft.viscosity.coefficient, Number(p.engine_viscosity));
    } finally {
      m.dispose();
    }
  }
});

test("Periodic engine boundaries and donor distances share the executed domain", async () => {
  const d = demos[17];
  const m = await d.create({
    engine,
    seed: 29,
    params: parameters(d, {
      source: "recorded",
      walkers: 16,
      engine_boundary: "periodic_box",
    }),
  });
  try {
    finite(m.snapshot(), d.id);
    const c = m.snapshot().result.details.executed_gas_config;
    assert.deepEqual(c.distance_donors.distance.periodic, c.boundary.domain);
    assert.deepEqual(c.cloning_donors.distance.periodic, c.boundary.domain);
  } finally {
    m.dispose();
  }
});
