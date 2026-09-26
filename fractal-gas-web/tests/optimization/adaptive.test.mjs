import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import create from "../../web/optimization/engine/optimization.mjs";
import { NativeOptimization } from "../../web/optimization/native.js";
import {
  Recording,
  importRecording,
} from "../../web/optimization/recording.js";

test("Wave restores five elites after every out-of-bounds batch", async () => {
  const engine = new NativeOptimization(await create());
  try {
    for (const perturbation of ["gaussian", "cloning_guided"]) {
      const config = engine.create({
        algorithm: "wave",
        benchmark: "quadratic",
        dimensions: 20,
        walkers: 8,
        max_walkers: 8,
        elites: 5,
        seed: 7,
        periodic: false,
        perturbation,
        perturbation_std: 1000000,
        adaptive_min_scale: 1000000,
        adaptive_max_scale: 1000000,
      });
      const initial = engine.snapshot();
      const recording = new Recording(config);
      recording.append(initial, engine.status());
      for (let step = 1; step <= 3; step++) {
        engine.step();
        const frame = engine.snapshot();
        assert.equal(frame[6], 5);
        assert.equal(frame[5], initial[5] + 8 * step);
        assert.equal(frame[9], initial[9]);
        for (let i = 0; i < 5; i++) {
          const offset = 12 + i * 48;
          assert.deepEqual(
            frame.slice(offset, offset + 20),
            initial.slice(offset, offset + 20),
          );
        }
        recording.append(frame, engine.status());
      }
      assert.equal(importRecording(recording.export()).frames.length, 4);
      const legacy = JSON.parse(recording.export());
      legacy.engine = "fgopt-8";
      assert.equal(importRecording(JSON.stringify(legacy)).frames.length, 4);
    }
  } finally {
    engine.dispose();
  }
});

for (const strategy of ["adaptive_fractal", "cloning_guided"])
  test(`${strategy}: native/WASM trajectories and controller recording round trips`, async () => {
    const engine = new NativeOptimization(await create());
    try {
      for (const algorithm of [
        "wave",
        "graph",
        "fmc",
        "wave_jump",
        "gas",
        "euclidean",
      ]) {
        const config = {
          algorithm,
          benchmark: "quadratic",
          dimensions: strategy === "cloning_guided" ? 20 : 2,
          walkers: strategy === "cloning_guided" ? 32 : 8,
          max_walkers: 64,
          periodic: true,
          horizon: 2,
          gas_local_search: false,
          potential_force: false,
          perturbation: strategy,
          perturbation_std: 0.1,
          adaptive_min_scale: 0.002,
          adaptive_max_scale: 0.2,
        };
        engine.create(config);
        for (let i = 0; i < 6; i++) engine.step();
        assert.ok(engine.status().exploration.scale_min >= 0.002);
        assert.ok(engine.status().exploration.scale_max <= 0.2);
        const result = spawnSync(
          "python3",
          [new URL("./native_reference.py", import.meta.url).pathname],
          { input: JSON.stringify({ config, steps: 6 }), encoding: "utf8" },
        );
        assert.equal(result.status, 0, result.stderr);
        const reference = JSON.parse(result.stdout),
          actual = engine.snapshot();
        assert.equal(actual.length, reference.length);
        for (let i = 0; i < actual.length; i++)
          if (reference[i] !== null)
            assert.ok(
              Math.abs(actual[i] - reference[i]) <=
                1e-5 * Math.max(1, Math.abs(reference[i])),
              `${algorithm} index ${i}: ${actual[i]} vs ${reference[i]}`,
            );
      }
      const config = engine.create({
        algorithm: "wave",
        benchmark: "quadratic",
        dimensions: strategy === "cloning_guided" ? 20 : 2,
        walkers: strategy === "cloning_guided" ? 32 : 8,
        max_walkers: 32,
        periodic: true,
        controller_enabled: true,
        max_evaluations: 2000,
        perturbation: strategy,
      });
      const recording = new Recording(config);
      recording.append(engine.snapshot(), engine.status());
      recording.append(engine.step(), engine.status());
      const before = Array.from(engine.snapshot());
      assert.throws(() =>
        engine.updateSettings({ adaptive_min_scale: 2, adaptive_max_scale: 1 }),
      );
      assert.deepEqual(Array.from(engine.snapshot()), before);
      engine.updateSettings({
        adaptive_min_scale: 0.01,
        adaptive_max_scale: 0.02,
      });
      recording.append(engine.snapshot(), {
        ...engine.status(),
        settings_update: true,
      });
      engine.updateSettings({ restart_token: 1 });
      recording.append(engine.step(), engine.status());
      assert.equal(engine.status().controller.round, 1);
      const loaded = importRecording(recording.export());
      assert.equal(loaded.frames.length, 4);
      const archive = engine.exportBasins();
      engine.create({ ...config, seed: 91 });
      engine.importBasins(archive);
      assert.ok(engine.status().controller.basins.every((b) => !b.validated));
      const archiveRecording = new Recording(engine.config());
      archiveRecording.append(engine.snapshot(), engine.status());
      engine.importBasins(archive);
      archiveRecording.append(engine.snapshot(), {
        ...engine.status(),
        settings_update: true,
        archive_update: true,
      });
      assert.equal(importRecording(archiveRecording.export()).frames.length, 2);
    } finally {
      engine.dispose();
    }
  });
