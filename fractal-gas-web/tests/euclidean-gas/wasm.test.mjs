import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import init, {
  BrowserGas,
  default_config,
  checkpoint_config,
  capabilities,
  benchmark_catalog,
  objective_info,
} from "../../web/euclidean-gas/engine/cpu/gas.js";
import {
  benchmarkInfo,
  benchmarkParameters,
  dimensionFor,
  resolveConfig as resolveWithCatalog,
  frameMetrics,
  validateLabConfig as validateWithCatalog,
} from "../../web/euclidean-gas/config.js";
await init({
  module_or_path: await readFile(
    new URL("../../web/euclidean-gas/engine/cpu/gas_bg.wasm", import.meta.url),
  ),
});

const catalog = benchmark_catalog();
const resolveConfig = (base, values) =>
  resolveWithCatalog(base, values, catalog);
const validateLabConfig = (config) => validateWithCatalog(config, catalog);

test("Checkpoint format explicitly versions geometry state", () => {
  assert.equal(capabilities().checkpoint_version, 5);
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
test("Every catalog benchmark resolves, runs and draws a landscape in WASM", async () => {
  assert.equal(catalog.benchmarks.length, 38);
  assert.equal(
    catalog.benchmarks.filter((entry) => entry.suite === "bbob").length,
    24,
  );
  for (const entry of catalog.benchmarks) {
    for (const requested of entry.dimensions ? [2, 5] : [2]) {
      const parameters = benchmarkParameters(entry);
      if (entry.dimensionRule) parameters.n_atoms = 3;
      const dimensions = dimensionFor(entry, parameters, requested);
      const config = resolveConfig(JSON.parse(default_config()), {
        ...form,
        benchmark: entry.id,
        parameters,
        dimensions: String(dimensions),
        "initial-box": "domain",
      });
      validateLabConfig(config);
      assert.deepEqual(config.gas.boundary.domain.lower, [
        ...Array(dimensions).fill(entry.bounds[0]),
      ]);
      const gas = await BrowserGas.create(JSON.stringify(config));
      try {
        const frame = await gas.step(1);
        assert.equal(frame.step, 1, entry.id);
        assert(frame.population.rewards.raw.every(Number.isFinite), entry.id);
        const host = entry.objective_execution === "host";
        if (entry.suite === "bbob") {
          assert(host);
          assert.equal(
            frame.execution.host_reward_evaluations,
            frame.reward_evaluations,
          );
        } else if (!host)
          assert.equal(frame.execution.host_reward_evaluations, 0);
        const objective = gas.objective_info();
        assert.deepEqual(objective, objective_info(JSON.stringify(config)));
        const info = benchmarkInfo(config, catalog, objective);
        assert.equal(info.low, entry.bounds[0]);
        assert.equal(info.molecule, entry.id === "lennard_jones");
        const landscape = gas.landscape(
          0,
          1,
          8,
          objective.minimizer || Array(dimensions).fill(0.25),
        );
        assert.equal(landscape.values.length, 64);
        assert.equal(landscape.minimum, objective.minimum);
        if (objective.minimum !== null && !entry.dimensionRule)
          assert(
            landscape.values.every(
              (v) => v === null || v >= objective.minimum - 1e-6,
            ),
            entry.id,
          );
        // Landscape queries leave the run untouched.
        assert.deepEqual(gas.snapshot(), frame);
      } finally {
        gas.free();
      }
    }
  }
});
test("Benchmark parameters and dimension rules are enforced", async () => {
  const base = JSON.parse(default_config());
  const pick = (id) => catalog.benchmarks.find((entry) => entry.id === id);
  assert.equal(dimensionFor(pick("eggholder"), {}, 7), 2);
  assert.equal(dimensionFor(pick("lennard_jones"), { n_atoms: 4 }, 2), 12);
  assert.equal(dimensionFor(pick("bbob_7"), {}, 8), 10);
  assert.equal(dimensionFor(pick("rosenbrock"), {}, 1), 2);
  assert.throws(
    () =>
      resolveConfig(base, { ...form, benchmark: "eggholder", dimensions: "3" }),
    /requires 2 dimensions/,
  );
  assert.throws(
    () =>
      resolveConfig(base, { ...form, benchmark: "bbob_3", dimensions: "4" }),
    /2, 3, 5, 10, 20, 40/,
  );
  assert.throws(
    () =>
      resolveConfig(base, {
        ...form,
        benchmark: "bbob_3",
        parameters: { coco_instance: 0 },
      }),
    /COCO instance/,
  );
  await assert.rejects(
    BrowserGas.create(
      JSON.stringify({ ...base, benchmark: { id: "sphere", alpha: 2 } }),
    ),
  );
  const config = resolveConfig(base, {
    ...form,
    benchmark: "bbob_21",
    parameters: { coco_instance: 3 },
    dimensions: "5",
  });
  assert.deepEqual(config.benchmark, { id: "bbob_21", coco_instance: 3 });
  const gas = await BrowserGas.create(JSON.stringify(config));
  try {
    assert.equal(gas.objective_info().coco_problem_id, "bbob_f021_i03_d05");
    await gas.step(2);
    const bytes = gas.checkpoint();
    assert.deepEqual(
      JSON.parse(checkpoint_config(bytes)).benchmark,
      config.benchmark,
    );
    const restored = await BrowserGas.restore(bytes);
    try {
      assert.deepEqual(await restored.step(1), await gas.step(1));
    } finally {
      restored.free();
    }
  } finally {
    gas.free();
  }
});
test("The benchmark list is generated from the engine catalog", async () => {
  const html = await readFile(
    new URL("../../web/euclidean-gas/index.html", import.meta.url),
    "utf8",
  );
  assert.match(html, /<select id="benchmark"><\/select/);
  assert.equal(capabilities().objective_catalog_version, catalog.version);
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

test("Elite count validation, clone protection and checkpoint continuation", async () => {
  const config = JSON.parse(default_config());
  assert.equal(config.gas.n_elite ?? 0, 0);
  config.walkers = 8;
  config.gas.n_elite = 2;
  validateLabConfig(config);
  for (const count of [-1, 1.5, 9]) {
    const invalid = structuredClone(config);
    invalid.gas.n_elite = count;
    assert.throws(() => validateLabConfig(invalid), /Elite walkers/);
    await assert.rejects(() => BrowserGas.create(JSON.stringify(invalid)));
  }
  const gas = await BrowserGas.create(JSON.stringify(config));
  let resumed;
  try {
    await gas.step(1);
    const bytes = gas.checkpoint();
    assert.equal(JSON.parse(checkpoint_config(bytes)).gas.n_elite, 2);
    resumed = await BrowserGas.restore(bytes);
    const frame = await gas.step(1);
    assert.equal(frame.elite_count, 2);
    assert.ok(
      frame.report.clone_plan.choices
        .slice(0, 2)
        .every((choice) => !choice.accepted),
    );
    assert.deepEqual(await resumed.step(1), frame);
  } finally {
    gas.free();
    resumed?.free();
  }
});
