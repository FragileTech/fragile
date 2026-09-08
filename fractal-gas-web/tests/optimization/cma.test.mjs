import test from "node:test";
import assert from "node:assert/strict";
import create from "../../web/optimization/engine/optimization.mjs";
import {
  NativeOptimization,
  frameInfo,
  row,
} from "../../web/optimization/native.js";
import {
  Recording,
  importRecording,
} from "../../web/optimization/recording.js";
test("CMA complete generations, precision, reset and metadata playback", async () => {
  const native = new NativeOptimization(await create());
  for (const algorithm of ["cmaes_active", "cmaes_bipop"]) {
    const cfg = {
      algorithm,
      benchmark: "quadratic",
      dimensions: 3,
      seed: 0,
      cma_population: 8,
      max_evaluations: 24,
      periodic: false,
    };
    const resolved = native.create(cfg);
    assert.equal(resolved.precision, "float64");
    assert.equal(resolved.libcmaes_revision.length, 40);
    const recording = new Recording(resolved);
    recording.append(native.snapshot(), native.status());
    for (let i = 0; i < 2; i++)
      recording.append(native.step(), native.status());
    const end = native.snapshot();
    assert.equal(native.status().next_evaluations, 8);
    assert.equal(native.status().budget_exhausted, true);
    assert.throws(() => native.step(), /budget/i);
    for (let i = 0; i < 8; i++) {
      const candidate = row(end, i);
      assert.ok(candidate.x.some((x) => x !== Math.fround(x)));
      assert.ok(
        candidate.x.every((x) => x >= resolved.low && x <= resolved.high),
      );
    }
    const restored = importRecording(recording.export());
    assert.deepEqual(restored.frames, recording.frames);
    assert.deepEqual(restored.metadata, recording.metadata);
    native.create(resolved);
    native.step();
    native.step();
    assert.deepEqual(native.snapshot(), end);
  }
  assert.throws(
    () => native.create({ algorithm: "cmaes_active", periodic: true }),
    /bounded/,
  );
  assert.throws(
    () => native.create({ algorithm: "cmaes_active", max_evaluations: 1 }),
    /first complete/,
  );
  native.dispose();
});
test("BIPOP variable populations finish reproducibly", async () => {
  const native = new NativeOptimization(await create());
  const cfg = {
    algorithm: "cmaes_bipop",
    benchmark: "quadratic",
    dimensions: 2,
    seed: 0,
    cma_runs: 2,
  };
  native.create(cfg);
  const recording = new Recording(cfg);
  for (let i = 0; i < 3000; i++) {
    recording.append(native.snapshot(), native.status());
    if (native.status().finished) break;
    native.step();
  }
  assert.equal(native.status().finished, true);
  assert.match(native.status().stop_reason, /exhausted/);
  assert.ok(new Set(recording.frames.map((f) => frameInfo(f).n)).size > 1);
  const restored = importRecording(recording.export());
  assert.deepEqual(restored.frames, recording.frames);
  native.dispose();
});

test("double benchmark fixtures retain near-optimum precision", async () => {
  const native = new NativeOptimization(await create());
  native.create({
    algorithm: "cmaes_active",
    benchmark: "rosenbrock",
    dimensions: 2,
  });
  const x = new Float64Array([1 + 1e-10, 1]);
  const expected = 100 * (1 - x[0] * x[0]) ** 2 + (1 - x[0]) ** 2;
  assert.equal(native.sample(x, 2)[0], expected);
  assert.equal(native.sample(new Float32Array(x), 2)[0], 0);
  const before = native.snapshot();
  native.sample(x, 2);
  assert.deepEqual(native.snapshot(), before);
  native.dispose();
});
test("CMA interleaved noisy sessions isolate RNG streams", async () => {
  const module = await create();
  const a = new NativeOptimization(module),
    b = new NativeOptimization(module);
  const cfg = {
    algorithm: "cmaes_active",
    benchmark: "stochastic_gaussian",
    dimensions: 3,
    seed: 0,
  };
  a.create(cfg);
  b.create(cfg);
  assert.deepEqual(a.snapshot(), b.snapshot());
  for (let i = 0; i < 15; i++) {
    a.sample(new Float64Array([0, 0, 0]), 3);
    assert.deepEqual(a.step(), b.step());
  }
  a.dispose();
  b.dispose();
});
test("WASM double evaluation agrees with native fixed-coordinate fixtures", async () => {
  const { readFile } = await import("node:fs/promises");
  const fixtures = JSON.parse(
    await readFile(
      new URL("./cma-precision-fixtures.json", import.meta.url),
      "utf8",
    ),
  );
  const native = new NativeOptimization(await create());
  for (const fixture of fixtures) {
    native.create(fixture.config);
    const actual = native.sample(new Float64Array(fixture.x), 2)[0];
    assert.ok(
      Math.abs(actual - fixture.expected) <=
        1e-12 * Math.max(1e-25, Math.abs(fixture.expected)),
    );
  }
  native.dispose();
});
