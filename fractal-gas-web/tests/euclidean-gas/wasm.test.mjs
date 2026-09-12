import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import init, {
  BrowserGas,
  default_config,
  checkpoint_config,
  capabilities,
} from "../../web/euclidean-gas/engine/cpu/gas.js";
import {
  resolveConfig,
  frameMetrics,
  validateLabConfig,
} from "../../web/euclidean-gas/config.js";
await init({
  module_or_path: await readFile(
    new URL("../../web/euclidean-gas/engine/cpu/gas_bg.wasm", import.meta.url),
  ),
});

test("Checkpoint format explicitly versions corrected semantics", () => {
  assert.equal(capabilities().checkpoint_version, 2);
});

test("Squared distance and ordinary distance have the same mean-diversity semantics", async () => {
  const base = JSON.parse(default_config());
  base.walkers = 12;
  base.gas.precision = "f64";
  base.gas.distance_donors.kernel = { kind: "uniform" };
  base.gas.distance_donors.count = 3;
  const squared = structuredClone(base);
  squared.gas.distance_donors.distance.squared = true;
  const runs = [];
  try {
    for (const config of [base, squared])
      runs.push(await BrowserGas.create(JSON.stringify(config)));
    const a = (await runs[0].step(1)).report.pre_clone_fitness;
    const b = (await runs[1].step(1)).report.pre_clone_fitness;
    for (const key of ["separation", "fitness"]) {
      a[key].forEach((x, i) => assert.ok(Math.abs(x - b[key][i]) < 1e-10));
    }
  } finally {
    runs.forEach((run) => run.free());
  }
});

test("WASM uses masked available donors without replacement", async () => {
  const config = JSON.parse(default_config());
  config.walkers = 3;
  config.gas.distance_donors.count = 8;
  config.gas.distance_donors.replacement = false;
  const gas = await BrowserGas.create(JSON.stringify(config));
  try {
    const frame = await gas.step(1);
    const companions = frame.report.distance_companions;
    assert.equal(companions.count, 8);
    for (let i = 0; i < 3; i++)
      assert.equal(
        companions.valid.slice(i * 8, (i + 1) * 8).filter(Boolean).length,
        2,
      );
    const bytes = gas.checkpoint();
    const corrupt = new Uint8Array(bytes.length + 1);
    corrupt.set(bytes);
    await assert.rejects(BrowserGas.restore(corrupt), /trailing/);
  } finally {
    gas.free();
  }
});

test("WASM memory budget rejects oversized reservations explicitly", async () => {
  const config = JSON.parse(default_config());
  config.gas.max_memory_bytes = 1;
  await assert.rejects(
    BrowserGas.create(JSON.stringify(config)),
    /memory budget/,
  );
});

for (const precision of ["f32", "f64"])
  test(`WASM CPU ${precision}: steps, checkpoint replay, scalar rewards, pure landscape`, async () => {
    const config = JSON.parse(default_config());
    config.walkers = 24;
    config.gas.precision = precision;
    const gas = await BrowserGas.create(JSON.stringify(config));
    try {
      assert.equal(gas.snapshot().step, 0);
      const first = await gas.step(2);
      assert.equal(first.step, 2);
      assert.equal(first.population.rewards.raw.length, 24);
      const saved = gas.checkpoint();
      assert.equal(
        JSON.parse(checkpoint_config(saved)).gas.precision,
        precision,
      );
      const before = gas.snapshot();
      gas.landscape(0, 1, 12, [0, 0]);
      assert.deepEqual(
        gas.snapshot(),
        before,
        "landscape must not consume RNG or reward budget",
      );
      const expected = await gas.step(1);
      const restored = await BrowserGas.restore(saved);
      try {
        assert.deepEqual(await restored.step(1), expected);
      } finally {
        restored.free();
      }
      assert.equal(frameMetrics(expected, config).alive, 24);
      await assert.rejects(gas.step(17), /1..=16/);
    } finally {
      gas.free();
    }
  });
test("Browser GPU f64 fails explicitly", async () => {
  const config = JSON.parse(default_config());
  config.gas.backend = "wgpu";
  config.gas.precision = "f64";
  await assert.rejects(BrowserGas.create(JSON.stringify(config)), /f32/);
});
test("Invalid checkpoint fails without modifying a live run", async () => {
  const config = JSON.parse(default_config());
  config.walkers = 8;
  const gas = await BrowserGas.create(JSON.stringify(config));
  try {
    const before = gas.snapshot();
    await assert.rejects(BrowserGas.restore(new Uint8Array([1, 2, 3])));
    assert.deepEqual(gas.snapshot(), before);
  } finally {
    gas.free();
  }
});
const form = {
  benchmark: "rastrigin",
  dimensions: "2",
  walkers: "32",
  seed: "7",
  direction: "minimize",
  backend: "cpu",
  precision: "f32",
  kinetic: "direct_jump",
  amplitude: "0.05",
  dt: "0.01",
  friction: "1",
  "distance-law": "gaussian",
  "clone-law": "fisher_yates",
  "distance-count": "3",
  "kernel-width": "1",
  reducer: "mean",
  distance: "euclidean",
  boundary: "absorbing_box",
  alpha: "1",
  beta: "1",
  "positive-map": "logistic",
  standardizer: "global",
  "sigma-min": "0.001",
  innovation: "gaussian",
  "noise-geometry": "isotropic",
  "noise-scale": "1",
};
test("Lab configuration preserves independent donor roles", () => {
  const config = resolveConfig(JSON.parse(default_config()), form);
  assert.equal(config.gas.distance_donors.count, 3);
  assert.equal(config.gas.cloning_donors.count, 1);
  assert.equal(config.gas.cloning_donors.law, "fisher_yates");
  assert.equal(config.gas.distance_donors.law, "independent");
});
test("UI validation rejects incompatible geometry and precision", () => {
  assert.throws(
    () => validateLabConfig({ ...JSON.parse(default_config()), dimensions: 1 }),
    /2–128/,
  );
  assert.throws(
    () =>
      validateLabConfig({
        ...JSON.parse(default_config()),
        benchmark: "unknown",
      }),
    /supported/,
  );
  assert.throws(
    () =>
      resolveConfig(JSON.parse(default_config()), {
        ...form,
        precision: "f64",
        backend: "wgpu",
      }),
    /f32/,
  );
  assert.throws(
    () =>
      resolveConfig(JSON.parse(default_config()), {
        ...form,
        distance: "phase_space",
      }),
    /velocities/,
  );
  assert.throws(
    () =>
      resolveConfig(JSON.parse(default_config()), {
        ...form,
        distance: "cosine",
        boundary: "periodic_box",
      }),
    /periodic/,
  );
});
test("All four benchmark configurations run in WASM", async () => {
  for (const benchmark of [
    "sphere",
    "rastrigin",
    "rosenbrock",
    "styblinski_tang",
  ]) {
    const config = resolveConfig(JSON.parse(default_config()), {
      ...form,
      benchmark,
    });
    const gas = await BrowserGas.create(JSON.stringify(config));
    try {
      const frame = await gas.step(1);
      assert.equal(frame.step, 1);
      assert(frame.population.rewards.raw.every(Number.isFinite));
    } finally {
      gas.free();
    }
  }
});
test("Langevin, anisotropic noise and mutual companions run in WASM f64", async () => {
  const config = resolveConfig(JSON.parse(default_config()), {
    ...form,
    precision: "f64",
    kinetic: "baoab",
    distance: "phase_space",
    "distance-law": "fisher_yates",
    "noise-geometry": "full",
    "noise-scale": "0.2",
  });
  const gas = await BrowserGas.create(JSON.stringify(config));
  try {
    const frame = await gas.step(2);
    assert.equal(frame.step, 2);
    assert(
      frame.population.observations.fields.velocities.values.every(
        Number.isFinite,
      ),
    );
    assert.equal(frame.report.distance_companions.count, 3);
  } finally {
    gas.free();
  }
});
