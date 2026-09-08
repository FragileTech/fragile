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
    assert.equal(native.catalog().benchmarks.length, 13);
    for (const algorithm of ["euclidean", "wave", "graph"]) {
      const cfg = {
        benchmark: "quadratic",
        dimensions: 5,
        walkers: 16,
        max_walkers: 128,
        algorithm,
        periodic: true,
        seed: 7,
      };
      const resolved = native.create(cfg);
      const recording = new Recording(resolved);
      recording.append(native.snapshot());
      for (let i = 0; i < 12; i++) recording.append(native.step());
      const end = recording.frames.at(-1);
      assert.ok(end[9] <= recording.frames[0][9]);
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
      for (const algorithm of ["euclidean", "wave", "graph"]) {
        const config = {
          algorithm,
          benchmark: "quadratic",
          dimensions: 3,
          walkers: 16,
          max_walkers: 128,
          periodic: true,
          seed: 7,
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
