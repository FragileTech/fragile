import test from "node:test";
import assert from "node:assert/strict";
import create from "../../web/optimization/engine/optimization.mjs";
import { NativeOptimization } from "../../web/optimization/native.js";
import {
  Recording,
  importRecording,
} from "../../web/optimization/recording.js";
import {
  validateGeometry,
  projectedCovariance,
} from "../../web/optimization/geometry.js";

for (const algorithm of [
  "wave",
  "graph",
  "fmc",
  "wave_jump",
  "gas",
  "euclidean",
  "cmaes_active",
  "cmaes_bipop",
])
  test(`${algorithm}: diagnostic observers preserve exact seeded trajectory`, async () => {
    const engine = new NativeOptimization(await create());
    try {
      for (const perturbation of algorithm.startsWith("cmaes")
        ? ["gaussian"]
        : [
            "gaussian",
            "adaptive_fractal",
            "cloning_guided",
            ...(algorithm === "wave" || algorithm === "gas"
              ? ["local_covariance"]
              : []),
          ]) {
        const config = {
          algorithm,
          perturbation,
          benchmark: "quadratic",
          dimensions: 3,
          walkers: 16,
          max_walkers: 32,
          seed: 19,
          horizon: 2,
          gas_local_search: false,
          gas_tabu: false,
          potential_force: false,
          periodic: !algorithm.startsWith("cmaes"),
          boundary: algorithm.startsWith("cmaes") ? "cma" : "periodic",
          perturbation_std: 0.1,
        };
        engine.create(config);
        const reference = [engine.snapshot()];
        for (let i = 0; i < 8 && !engine.status().finished; i++)
          reference.push(engine.step());
        const resolved = engine.create({
          ...config,
          geometry_diagnostics: true,
        });
        const recording = new Recording(resolved);
        for (let i = 0; i < reference.length; i++) {
          const frame = i ? engine.step() : engine.snapshot(),
            status = engine.status();
          assert.deepEqual(
            frame,
            reference[i],
            `${algorithm}/${perturbation} step ${i}`,
          );
          validateGeometry(status.geometry, 3);
          recording.append(frame, status);
          if (status.geometry)
            assert.ok(
              JSON.stringify(status.geometry).length <=
                status.geometry_capacity_bytes,
            );
          if (algorithm.startsWith("cmaes")) {
            assert.equal(status.geometry.generation, status.generation);
            assert.equal(status.geometry.restart, status.restarts);
            assert.ok(status.geometry.methods[0].models[0].shape.length === 9);
          } else if (i > 0) {
            assert.ok(
              status.geometry.methods.some((m) => m.id === "cloning_guided"),
            );
            assert.ok(
              status.geometry.methods.some((m) => m.id === "adaptive_fractal"),
            );
            assert.ok(
              status.geometry.methods.some((m) => m.id === "local_covariance"),
            );
          }
        }
        assert.deepEqual(
          importRecording(recording.export()).metadata,
          recording.metadata,
        );
        const before = engine.snapshot();
        engine.setGeometryDiagnostics(false);
        assert.deepEqual(engine.snapshot(), before);
        assert.equal(engine.status().geometry, null);
        engine.setGeometryDiagnostics(true);
        assert.deepEqual(engine.snapshot(), before);
      }
    } finally {
      engine.dispose();
    }
  });

test("same-swarm estimators learn and high-dimensional models retain low-rank projection", async () => {
  const engine = new NativeOptimization(await create());
  try {
    for (const d of [2, 3, 80]) {
      engine.create({
        algorithm: "wave",
        benchmark: "quadratic",
        perturbation: "gaussian",
        dimensions: d,
        walkers: 24,
        max_walkers: 24,
        seed: 3,
        periodic: true,
        geometry_diagnostics: true,
        perturbation_std: 0.2,
      });
      for (let i = 0; i < 8; i++) engine.step();
      const g = engine.status().geometry;
      validateGeometry(g, d);
      for (const id of [
        "local_covariance",
        "adaptive_fractal",
        "cloning_guided",
      ]) {
        const method = g.methods.find((m) => m.id === id);
        assert.ok(method.models.length, `${d}D ${id} has evidence`);
        for (const model of method.models)
          assert.ok(
            projectedCovariance(model, [0, d - 1])
              .flat()
              .every(Number.isFinite),
          );
      }
      if (d > 64)
        assert.equal(
          g.methods.find((m) => m.id === "cloning_guided").models[0]
            .representation,
          "diagonal_low_rank",
        );
      assert.ok(g.events.some((e) => e.kind === "proposal"));
    }
  } finally {
    engine.dispose();
  }
});

test("live budget edits retain diagnostic learning; toggles and boundary changes warm up safely", async () => {
  const engine = new NativeOptimization(await create());
  try {
    engine.create({
      algorithm: "wave",
      benchmark: "quadratic",
      dimensions: 3,
      walkers: 16,
      max_walkers: 16,
      seed: 4,
      perturbation: "gaussian",
      periodic: true,
      geometry_diagnostics: true,
    });
    for (let i = 0; i < 5; i++) engine.step();
    const before = engine.snapshot(),
      methods = engine.status().geometry.methods;
    engine.updateSettings({ max_evaluations: 100000 });
    assert.deepEqual(engine.snapshot(), before);
    assert.deepEqual(engine.status().geometry.methods, methods);
    engine.updateSettings({ boundary: "cma" });
    assert.ok(
      engine
        .status()
        .geometry.methods.filter((m) => !m.active)
        .every((m) => m.models.length === 0),
    );
    engine.step();
    const active = engine.snapshot();
    engine.setGeometryDiagnostics(false);
    assert.deepEqual(engine.snapshot(), active);
    assert.equal(engine.status().geometry, null);
    engine.setGeometryDiagnostics(true);
    assert.ok(
      engine.status().geometry.methods.every((m) => m.models.length === 0),
    );
  } finally {
    engine.dispose();
  }
});

test("planner execution arrows retain replay movements without relearning from replay", async () => {
  const engine = new NativeOptimization(await create());
  try {
    for (const perturbation of ["gaussian", "cloning_guided"]) {
      engine.create({
        algorithm: "fmc",
        benchmark: "quadratic",
        dimensions: 2,
        walkers: 8,
        max_walkers: 8,
        horizon: 2,
        seed: 21,
        perturbation,
        periodic: true,
        geometry_diagnostics: true,
      });
      let execution = false;
      for (let i = 0; i < 8; i++) {
        engine.step();
        const geometry = engine.status().geometry;
        validateGeometry(geometry, 2);
        execution ||= geometry.events.some((e) => e.kind === "execution");
      }
      assert.equal(execution, true, perturbation);
    }
  } finally {
    engine.dispose();
  }
});
