import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import create from "../../web/optimization/engine/optimization.mjs";
import { NativeOptimization } from "../../web/optimization/native.js";
import {
  Recording,
  importRecording,
} from "../../web/optimization/recording.js";

test("CMA boundary repair: six fractal adapters, native/WASM and live recording", async () => {
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
        dimensions: 20,
        walkers: 8,
        max_walkers: 128,
        elites: 5,
        horizon: 2,
        seed: 7,
        boundary: "cma",
        perturbation: "cloning_guided",
        adaptive_min_scale: 1000,
        adaptive_max_scale: 1000,
        potential_force: false,
        gas_local_search: false,
      };
      engine.create(config);
      for (let step = 0; step < 6; step++) engine.step();
      const actual = engine.snapshot();
      if (algorithm !== "graph")
        assert.equal(actual[6], actual[1], `${algorithm}: all walkers valid`);
      else assert.ok(actual[6] > 0);
      const result = spawnSync(
        "python3",
        [new URL("./native_reference.py", import.meta.url).pathname],
        { input: JSON.stringify({ config, steps: 6 }), encoding: "utf8" },
      );
      assert.equal(result.status, 0, result.stderr);
      const reference = JSON.parse(result.stdout);
      assert.equal(reference.length, actual.length);
      for (let i = 0; i < reference.length; i++)
        if (reference[i] !== null)
          assert.ok(
            Math.abs(actual[i] - reference[i]) <=
              1e-5 * Math.max(1, Math.abs(reference[i])),
            `${algorithm}: ${i}`,
          );
    }
    engine.create({
      algorithm: "wave",
      dimensions: 20,
      walkers: 8,
      elites: 5,
      periodic: true,
    });
    const recording = new Recording(engine.config());
    recording.append(engine.snapshot(), engine.status());
    for (const boundary of ["cma", "none", "periodic"]) {
      const before = engine.snapshot();
      const status = engine.updateSettings({ boundary });
      assert.deepEqual(engine.snapshot(), before);
      assert.equal(status.effective_settings.periodic, boundary === "periodic");
      recording.append(engine.snapshot(), { ...status, settings_update: true });
    }
    const imported = importRecording(recording.export());
    assert.equal(imported.frames.length, 4);
  } finally {
    engine.dispose();
  }
});
