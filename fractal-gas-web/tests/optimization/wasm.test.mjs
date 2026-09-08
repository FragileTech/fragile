import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import {
  NativeOptimization,
  frameInfo,
  row,
} from "../../web/optimization/native.js";
import {
  Recording,
  importRecording,
} from "../../web/optimization/recording.js";
const path = new URL(
  "../../web/optimization/engine/optimization.mjs",
  import.meta.url,
);
const available = existsSync(path);
const create = available ? (await import(path.href)).default : null;
test(
  "WASM catalog, deterministic algorithms, full dimensions, and replay",
  { skip: !available },
  async () => {
    const native = new NativeOptimization(await create());
    assert.equal(native.catalog().benchmarks.length, 37);
    assert.deepEqual(
      native
        .catalog()
        .algorithms.map((a) => a.id)
        .sort(),
      [
        "cmaes_active",
        "cmaes_bipop",
        "euclidean",
        "fmc",
        "gas",
        "graph",
        "wave",
        "wave_jump",
      ],
    );
    assert.equal(native.catalog().perturbations.length, 4);
    for (const objective of ["minimize", "maximize"])
      for (const perturbation of ["gaussian", "uniform", "local_covariance"])
        for (const algorithm of [
          "euclidean",
          "wave",
          "graph",
          "fmc",
          "wave_jump",
        ]) {
          if (
            perturbation === "local_covariance" &&
            !["wave", "fmc", "wave_jump"].includes(algorithm)
          )
            continue;
          const cfg = {
            benchmark: "quadratic",
            dimensions: 5,
            walkers: 16,
            max_walkers: 128,
            algorithm,
            objective,
            perturbation,
            perturbation_std: 0.3,
            horizon: 2,
            periodic: true,
            seed: 7,
          };
          const resolved = native.create(cfg);
          const recording = new Recording(resolved);
          recording.append(native.snapshot());
          for (let i = 0; i < 12; i++) recording.append(native.step());
          const end = recording.frames.at(-1);
          assert.ok(
            objective === "minimize"
              ? end[9] <= recording.frames[0][9]
              : end[9] >= recording.frames[0][9],
          );
          assert.equal(resolved.objective, objective);
          assert.equal(resolved.perturbation, perturbation);
          assert.equal(resolved.perturbation_std, 0.3);
          assert.deepEqual(
            importRecording(recording.export()).frames,
            recording.frames,
          );
          native.create(cfg);
          for (let i = 0; i < 12; i++) native.step();
          assert.deepEqual(native.snapshot(), end);
          assert.equal(row(end, 0).x.length, 5);
        }
    native.dispose();
    assert.throws(() => native.step(), /Invalid/);
  },
);
test(
  "surface evaluation shares formulas and does not consume noise RNG",
  { skip: !available },
  async () => {
    const native = new NativeOptimization(await create());
    native.create({ benchmark: "rosenbrock", dimensions: 2, walkers: 4 });
    const valid = native.snapshot();
    assert.throws(
      () => native.create({ benchmark: "easom", dimensions: 3 }),
      /two dimensions/,
    );
    assert.deepEqual(native.snapshot(), valid);
    assert.throws(
      () => native.sample(new Float32Array([1, 2, 3]), 3),
      /dimensions/,
    );
    assert.deepEqual(
      Array.from(native.sample(new Float32Array([0, 0, 1, 1]), 2)),
      [1, 0],
    );
    native.create({ benchmark: "stochastic_gaussian", walkers: 4 });
    const before = native.snapshot();
    assert.deepEqual(
      Array.from(native.sample(new Float32Array([1, 2, 3]), 3)),
      [0],
    );
    assert.deepEqual(native.snapshot(), before);
    assert.equal(frameInfo(before).d, 3);
    native.dispose();
  },
);

import { spawnSync } from "node:child_process";
const library = new URL(
  "../../build-optimization-native/optimization/libfg_optimization.so",
  import.meta.url,
);
test(
  "native and WASM follow the same optimization random stream",
  { skip: !available || !existsSync(library) },
  async () => {
    const native = new NativeOptimization(await create());
    try {
      for (const objective of ["minimize", "maximize"])
        for (const perturbation of ["gaussian", "uniform", "local_covariance"])
          for (const algorithm of [
            "euclidean",
            "wave",
            "graph",
            "fmc",
            "wave_jump",
          ]) {
            if (
              perturbation === "local_covariance" &&
              !["wave", "fmc", "wave_jump"].includes(algorithm)
            )
              continue;
            const config = {
              algorithm,
              objective,
              perturbation,
              perturbation_std: 0.3,
              horizon: 2,
              benchmark: "quadratic",
              dimensions: 3,
              walkers: 16,
              max_walkers: 128,
              periodic: true,
              seed: 7,
            };
            native.create(config);
            for (let i = 0; i < 8; i++) native.step();
            const result = spawnSync(
              "python3",
              [new URL("./native_reference.py", import.meta.url).pathname],
              {
                input: JSON.stringify({ config, steps: 8 }),
                encoding: "utf8",
                timeout: 10000,
              },
            );
            assert.equal(
              result.status,
              0,
              result.stderr || result.error?.message,
            );
            const expected = JSON.parse(result.stdout),
              actual = native.snapshot();
            assert.equal(actual.length, expected.length);
            expected.forEach((v, i) => {
              if (v === null) assert.ok(!Number.isFinite(actual[i]));
              else
                assert.ok(
                  Math.abs(v - actual[i]) <= 2e-6 * Math.max(1, Math.abs(v)),
                  `${algorithm} word ${i}: ${actual[i]} != ${v}`,
                );
            });
          }
    } finally {
      native.dispose();
    }
  },
);

test(
  "COCO BBOB catalog, instances, fixed budgets and surface isolation",
  { skip: !available },
  async () => {
    const native = new NativeOptimization(await create());
    try {
      const bbob = native
        .catalog()
        .benchmarks.filter((b) => b.suite === "bbob");
      assert.equal(bbob.length, 24);
      for (const entry of bbob) {
        for (const dimensions of entry.dimensions) {
          const cfg = native.create({
            benchmark: entry.id,
            dimensions,
            coco_instance: 2,
            walkers: 8,
          });
          assert.equal(cfg.coco_version, "2.8.2");
          assert.ok(
            cfg.coco_problem_id.endsWith(
              `_i02_d${String(dimensions).padStart(2, "0")}`,
            ),
          );
          assert.ok(Number.isFinite(cfg.reference_minimum));
          assert.equal(cfg.potential_force, false);
          const before = native.snapshot();
          const sampled = native.sample(
            Float32Array.from(
              { length: dimensions },
              (_, k) => [0.37, 0.83, -1.1][k % 3],
            ),
            dimensions,
          );
          assert.ok(Number.isFinite(sampled[0]));
          assert.deepEqual(native.snapshot(), before);
          native.step();
        }
      }
      for (const algorithm of [
        "euclidean",
        "wave",
        "graph",
        "fmc",
        "wave_jump",
      ]) {
        const resolved = native.create({
          benchmark: "bbob_21",
          dimensions: 5,
          coco_instance: 3,
          walkers: 8,
          max_walkers: 64,
          algorithm,
          horizon: 2,
          periodic: true,
          max_evaluations: 65,
          perturbation_std: 0.1,
        });
        const recording = new Recording(resolved);
        recording.append(native.snapshot());
        let stopped = false;
        for (let i = 0; i < 100 && !stopped; i++) {
          const before = native.snapshot();
          try {
            recording.append(native.step());
          } catch (e) {
            assert.match(e.message, /Evaluation budget reached/);
            assert.deepEqual(native.snapshot(), before);
            stopped = true;
          }
          assert.ok(frameInfo(native.snapshot()).evaluations <= 65);
        }
        assert.ok(stopped);
        assert.deepEqual(
          importRecording(recording.export()).frames,
          recording.frames,
        );
        assert.equal(
          importRecording(recording.export()).config.coco_instance,
          3,
        );
      }
      assert.throws(
        () => native.create({ benchmark: "bbob_24", dimensions: 4 }),
        /dimensions/,
      );
      assert.throws(
        () => native.create({ benchmark: "bbob_1", coco_instance: 0 }),
        /instance/,
      );
    } finally {
      native.dispose();
    }
  },
);

test(
  "COCO native/WASM problem parity for all functions and hard instances",
  { skip: !available || !existsSync(library) },
  async () => {
    const native = new NativeOptimization(await create());
    try {
      for (let f = 1; f <= 24; f++) {
        const config = {
          benchmark: `bbob_${f}`,
          dimensions: 5,
          coco_instance: 15,
          algorithm: "wave",
          walkers: 8,
          periodic: true,
          perturbation_std: 0.1,
        };
        native.create(config);
        for (let i = 0; i < 3; i++) native.step();
        const result = spawnSync(
          "python3",
          [new URL("./native_reference.py", import.meta.url).pathname],
          {
            input: JSON.stringify({ config, steps: 3 }),
            encoding: "utf8",
            timeout: 10000,
          },
        );
        assert.equal(result.status, 0, result.stderr || result.error?.message);
        const expected = JSON.parse(result.stdout),
          actual = native.snapshot();
        expected.forEach((v, i) => {
          if (v === null) assert.ok(!Number.isFinite(actual[i]));
          else
            assert.ok(
              Math.abs(v - actual[i]) <= 3e-6 * Math.max(1, Math.abs(v)),
              `${config.benchmark} word ${i}: ${actual[i]} != ${v}`,
            );
        });
      }
    } finally {
      native.dispose();
    }
  },
);

test(
  "GAS switches, adaptive jumps, budgets, and portable recordings",
  { skip: !available },
  async () => {
    const engine = new NativeOptimization(await create());
    try {
      assert.deepEqual(
        engine.catalog().perturbations.find((p) => p.id === "gas_adaptive")
          .algorithms,
        ["gas"],
      );
      for (const gas_tabu of [false, true])
        for (const gas_local_search of [false, true])
          for (const perturbation of [
            "gas_adaptive",
            "gaussian",
            "uniform",
            "local_covariance",
          ])
            for (const objective of ["minimize", "maximize"]) {
              const config = {
                algorithm: "gas",
                benchmark: "sphere",
                dimensions: 3,
                walkers: 8,
                low: -1,
                high: 1,
                gas_tabu,
                gas_local_search,
                perturbation,
                objective,
                gas_local_evaluations: 30,
                max_evaluations: 500,
                seed: 7,
              };
              const resolved = engine.create(config);
              const recording = new Recording(resolved);
              recording.append(engine.snapshot());
              for (let i = 0; i < 3; ++i) recording.append(engine.step());
              assert.equal(resolved.gas_tabu, gas_tabu);
              assert.equal(resolved.gas_local_search, gas_local_search);
              const restored = importRecording(recording.export());
              assert.deepEqual(restored.config, recording.config);
              assert.deepEqual(restored.frames, recording.frames);
              const expected = engine.snapshot();
              engine.create(restored.config);
              for (let i = 0; i < 3; ++i) engine.step();
              assert.deepEqual(engine.snapshot(), expected);
              assert.ok(expected[5] <= config.max_evaluations);
              if (existsSync(library)) {
                const result = spawnSync(
                  "python3",
                  [new URL("./native_reference.py", import.meta.url).pathname],
                  {
                    input: JSON.stringify({ config, steps: 3 }),
                    encoding: "utf8",
                    timeout: 10000,
                  },
                );
                assert.equal(
                  result.status,
                  0,
                  result.stderr || result.error?.message,
                );
                JSON.parse(result.stdout).forEach((v, i) => {
                  assert.ok(
                    Math.abs(v - expected[i]) <=
                      2e-6 * Math.max(1, Math.abs(v)),
                    `GAS ${gas_tabu}/${gas_local_search}/${perturbation}/${objective} word ${i}: ${expected[i]} != ${v}`,
                  );
                });
              }
            }
      engine.create({
        algorithm: "gas",
        benchmark: "constant",
        walkers: 8,
        gas_local_search: false,
        max_evaluations: 16,
      });
      engine.step();
      const before = engine.snapshot();
      assert.throws(() => engine.step(), /Evaluation budget reached/);
      assert.deepEqual(engine.snapshot(), before);
      const noisy = engine.create({
        algorithm: "gas",
        benchmark: "stochastic_gaussian",
        walkers: 8,
      });
      assert.equal(noisy.gas_local_search, false);
      assert.throws(
        () =>
          engine.create({ algorithm: "wave", perturbation: "gas_adaptive" }),
        /not supported/,
      );
    } finally {
      engine.dispose();
    }
  },
);

test(
  "local covariance config, reset, budget and recording in Wave and planners",
  { skip: !available },
  async () => {
    const engine = new NativeOptimization(await create());
    const entry = engine
      .catalog()
      .perturbations.find((p) => p.id === "local_covariance");
    assert.deepEqual(entry.algorithms, ["wave", "fmc", "wave_jump", "gas"]);
    for (const algorithm of entry.algorithms) {
      const cfg = {
        algorithm,
        perturbation: "local_covariance",
        covariance_learning_rate: 0.2,
        benchmark: "rastrigin",
        walkers: 24,
        dimensions: 2,
        periodic: true,
        horizon: 2,
        max_horizon: 4,
        dt_max: 3,
        gas_local_search: false,
        max_evaluations: 2000,
        seed: 17,
      };
      const resolved = engine.create(cfg);
      assert.equal(resolved.covariance_learning_rate, 0.2);
      const recording = new Recording(resolved);
      for (let i = 0; i < 40; ++i) recording.append(engine.step());
      const expected = engine.snapshot();
      assert.ok(frameInfo(expected).evaluations <= 2000);
      assert.equal(
        frameInfo(expected).n,
        ["wave", "gas"].includes(algorithm) ? 24 : 25,
      );
      const imported = importRecording(recording.export());
      assert.deepEqual(imported.frames, recording.frames);
      engine.create(cfg);
      for (let i = 0; i < 40; ++i) engine.step();
      assert.deepEqual(engine.snapshot(), expected);
    }
    for (const algorithm of ["graph", "euclidean"])
      assert.throws(
        () => engine.create({ algorithm, perturbation: "local_covariance" }),
        /not supported/,
      );
    assert.throws(
      () =>
        engine.create({
          algorithm: "wave",
          perturbation: "local_covariance",
          covariance_learning_rate: 2,
        }),
      /covariance/,
    );
    engine.dispose();
  },
);
