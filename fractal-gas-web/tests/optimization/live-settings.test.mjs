import test from "node:test";
import assert from "node:assert/strict";
import create from "../../web/optimization/engine/optimization.mjs";
import {
  NativeOptimization,
  frameInfo,
} from "../../web/optimization/native.js";
import {
  Recording,
  importRecording,
} from "../../web/optimization/recording.js";

test("live settings preserve counters, frames, and initial configuration", async () => {
  const native = new NativeOptimization(await create());
  try {
    for (const algorithm of ["wave", "euclidean", "gas", "graph"]) {
      const config = native.create({
        algorithm,
        benchmark: "quadratic",
        dimensions: 2,
        walkers: 8,
        max_walkers: 32,
        periodic: true,
        gas_local_search: false,
      });
      const rec = new Recording(config);
      rec.append(native.snapshot(), native.status());
      native.step();
      rec.append(native.snapshot(), native.status());
      const before = native.snapshot(),
        original = rec.frames.map((f) => f.slice());
      const info = frameInfo(before);
      for (const patch of [
        { seed: 99 },
        { algorithm: "gas" },
        { dt_max: 0 },
        { perturbation_std: -1 },
        { walkers: 40 },
      ]) {
        assert.throws(() => native.updateSettings(patch));
        assert.deepEqual(native.snapshot(), before);
      }
      const status = native.updateSettings({
        perturbation: "uniform",
        perturbation_std: 0.1,
        walkers: 12,
        max_walkers: 40,
      });
      const after = native.snapshot();
      assert.equal(frameInfo(after).iteration, info.iteration);
      assert.equal(frameInfo(after).evaluations, info.evaluations);
      assert.equal(frameInfo(after).best, info.best);
      rec.append(after, {
        ...status,
        settings_update: true,
        population_changed: algorithm !== "graph",
      });
      native.step();
      rec.append(native.snapshot(), native.status());
      native.updateSettings({ max_evaluations: 1 });
      assert.equal(native.status().budget_exhausted, true);
      assert.throws(() => native.step(), /Evaluation budget/);
      const resumed = native.updateSettings({ max_evaluations: 0 });
      rec.append(native.snapshot(), { ...resumed, settings_update: true });
      native.step();
      rec.append(native.snapshot(), native.status());
      const loaded = importRecording(rec.export());
      assert.deepEqual(loaded.frames, rec.frames);
      assert.deepEqual(loaded.metadata, rec.metadata);
      assert.deepEqual(rec.frames.slice(0, 2), original);
      assert.equal(loaded.config.walkers, 8);
      assert.equal(loaded.config.perturbation, config.perturbation);
    }
  } finally {
    native.dispose();
  }
});

test("planner patches remain pending through old search and execution", async () => {
  const module = await create(),
    native = new NativeOptimization(module),
    reference = new NativeOptimization(module);
  try {
    for (const algorithm of ["fmc", "wave_jump"]) {
      const config = {
        algorithm,
        walkers: 8,
        max_walkers: 24,
        horizon: 2,
        periodic: true,
      };
      native.create(config);
      reference.create(config);
      native.step();
      reference.step();
      native.updateSettings({ perturbation_std: 0, walkers: 16, horizon: 3 });
      assert.equal(native.status().effective_settings.walkers, 8);
      assert.equal(native.status().pending_settings.walkers, 16);
      native.updateSettings({ max_evaluations: 1 });
      assert.equal(native.status().budget_exhausted, true);
      native.updateSettings({ max_evaluations: 0 });
      for (let i = 0; i < 20 && native.status().pending_settings; ++i) {
        native.step();
        reference.step();
        assert.deepEqual(native.snapshot(), reference.snapshot());
      }
      assert.equal(native.status().pending_settings, null);
      assert.equal(native.config().walkers, 16);
      native.step();
      assert.equal(frameInfo(native.snapshot()).n, 17);
    }
  } finally {
    native.dispose();
    reference.dispose();
  }
});

test("toggling GAS memory and tuning Euclidean forces preserve sessions", async () => {
  const native = new NativeOptimization(await create());
  try {
    native.create({
      algorithm: "gas",
      walkers: 8,
      gas_tabu: false,
      gas_local_search: false,
      periodic: true,
    });
    for (const patch of [
      { gas_tabu: true },
      { walkers: 12 },
      { gas_tabu: false },
      { walkers: 4 },
      { gas_tabu: true },
      { gas_local_search: true, gas_local_evaluations: 4 },
    ]) {
      native.updateSettings(patch);
      native.step();
    }
    native.create({ algorithm: "euclidean", walkers: 8, periodic: true });
    const before = native.snapshot();
    native.updateSettings({
      cloning: false,
      kinetic: false,
      gamma: 2,
      beta: 4,
      delta_t: 0.01,
      substeps: 2,
      companion: "uniform",
      clone_companion: "uniform",
      reward_coef: 2,
      distance_coef: 0,
    });
    assert.deepEqual(native.snapshot(), before);
    native.step();
    const after = native.snapshot();
    const { stride, n } = frameInfo(after);
    for (let i = 0; i < n; i++)
      assert.deepEqual(
        after.slice(12 + i * stride, 15 + i * stride),
        before.slice(12 + i * stride, 15 + i * stride),
      );
    native.updateSettings({ cloning: true, kinetic: true });
    native.step();
  } finally {
    native.dispose();
  }
});
