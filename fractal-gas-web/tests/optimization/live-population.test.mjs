import test from "node:test";
import assert from "node:assert/strict";
import createOptimization from "../../web/optimization/engine/optimization.mjs";
import {
  NativeOptimization,
  frameInfo,
} from "../../web/optimization/native.js";
import {
  Recording,
  importRecording,
} from "../../web/optimization/recording.js";

test("optimization resizing preserves evaluations and variable-size recordings", async () => {
  const native = new NativeOptimization(await createOptimization());
  try {
    for (const objective of ["minimize", "maximize"]) {
      const config = native.create({
        algorithm: "wave",
        benchmark: "quadratic",
        dimensions: 2,
        walkers: 12,
        max_walkers: 24,
        objective,
        periodic: true,
      });
      const rec = new Recording(config);
      rec.append(native.snapshot(), native.status());
      const before = frameInfo(native.snapshot());
      const state = native.setPopulation(4, "cumulative_reward");
      const after = frameInfo(native.snapshot());
      assert.equal(after.iteration, before.iteration);
      assert.equal(after.evaluations, before.evaluations);
      assert.equal(after.n, 4);
      rec.replaceLast(native.snapshot(), {
        ...state,
        population_changed: true,
      });
      native.step();
      rec.append(native.snapshot(), native.status());
      native.setPopulation(24, "virtual_reward");
      rec.replaceLast(native.snapshot(), {
        ...native.status(),
        population_changed: true,
      });
      native.step();
      rec.append(native.snapshot(), native.status());
      assert.deepEqual(importRecording(rec.export()).frames, rec.frames);
      assert.equal(rec.frames.length, 3);
      assert.throws(() => native.setPopulation(25, "virtual_reward"));
    }
    for (const algorithm of ["fmc", "wave_jump"]) {
      native.create({
        algorithm,
        walkers: 8,
        max_walkers: 20,
        horizon: 2,
        consensus_prefix: false,
        periodic: true,
      });
      native.step();
      native.setPopulation(4, "virtual_reward");
      assert.equal(native.status().population.active, 8);
      assert.equal(native.status().pending_settings.walkers, 4);
      native.step();
      native.setPopulation(16, "cumulative_reward");
      assert.equal(native.status().population.pending, true);
      for (
        let i = 0;
        i < 16 &&
        (native.status().pending_settings ||
          native.status().population.pending);
        i++
      )
        native.step();
      assert.equal(native.status().population.active, 16);
    }
    native.create({ algorithm: "graph", walkers: 8, max_walkers: 20 });
    native.setPopulation(12, "virtual_reward");
    assert.equal(native.config().walkers, 12);
    assert.equal(frameInfo(native.snapshot()).n, 8);
  } finally {
    native.dispose();
  }
});
