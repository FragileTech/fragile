import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import "../web/lab/controllers/index.js";
import {
  createController,
  registerController,
} from "../web/lab/controllers/registry.js";
import { runBenchmark, runEpisode } from "../web/lab/experiments.js";
import { bytesOf } from "../web/lab/motion.js";
const scene = {
  name: "Deterministic test",
  size: [100, 100],
  physics: { substeps: 2 },
  bodies: [
    {
      controlled: true,
      position: [50, 50],
      velocity: [2, 1],
      actuator: { kind: "kart" },
    },
  ],
};
const settings = {
  walkers: 12,
  horizon: 4,
  frames: 2,
  elites: 2,
  recording: 2,
  search_iterations: 2,
};
const presetIds = JSON.parse(
  await readFile(
    new URL("../web/lab/scenario-catalog.json", import.meta.url),
    "utf8",
  ),
).map((p) => p.id);
const module = await loadNative(false);
for (const algorithm of ["fmc", "wave-jump", "random", "cem", "icem", "mppi"])
  test(`${algorithm} restores its internal search state exactly with variable channels`, () => {
    const root = new NativeEngine(module, scene),
      a = createController(module, scene, { ...settings, algorithm });
    try {
      a.controller.begin(root.snapshot(), 73);
      a.controller.advance();
      const saved = a.controller.checkpoint();
      while (!a.controller.advance()) {}
      const expected = a.controller.result();
      a.controller.restore(saved);
      while (!a.controller.advance()) {}
      const actual = a.controller.result();
      assert.deepEqual(actual.action, expected.action);
      assert.deepEqual(bytesOf(actual.cloud), bytesOf(expected.cloud));
      assert.deepEqual(actual.tree.meta, expected.tree.meta);
      assert.deepEqual(actual.tree.values, expected.tree.values);
      assert.equal(actual.action.length, 3);
      assert.equal(actual.tree.poseDim, 2);
    } finally {
      root.dispose();
      a.dispose();
    }
  });
test("registered controllers use opaque state and benchmark suites are reproducible", async () => {
  registerController("test_zero", {
    label: "Zero action",
    create: ({ engine }) => ({
      begin(root) {
        engine.restore(root);
      },
      advance() {
        return true;
      },
      result() {
        return {
          action: new Float32Array(engine.descriptor().channels.length),
        };
      },
    }),
  });
  const options = {
    module,
    scenes: [scene, { ...scene, name: "Second scene" }],
    spec: {
      seeds: [3, 11],
      maxFrames: 6,
      goal: { metric: "survival", target: 6 },
      variants: [{ ...settings, algorithm: "test_zero" }],
    },
  };
  const a = await runBenchmark(options),
    b = await runBenchmark(options);
  assert.equal(a.results.length, 4);
  assert.equal(a.perScene.length, 2);
  assert.equal(a.summary[0].successRate, 1);
  for (let i = 0; i < 4; i++)
    for (const key of [
      "reward",
      "frames",
      "collisions",
      "controlEffort",
      "success",
      "completionSeconds",
    ])
      assert.equal(a.results[i][key], b.results[i][key]);
});
test("comparison episodes start from identical complete state", async () => {
  const root = new NativeEngine(module, scene);
  root.step(new Float32Array([0.4, 0.3, 0]), 5);
  try {
    const common = {
      module,
      scene,
      root: { snapshot: root.snapshot(), rows: root.states() },
      seed: 17,
      maxFrames: 8,
      record: true,
      goal: { metric: "survival", target: 8 },
    };
    const a = await runEpisode({
        ...common,
        settings: { ...settings, algorithm: "fmc" },
      }),
      b = await runEpisode({
        ...common,
        settings: { ...settings, algorithm: "cem" },
      });
    assert.deepEqual(
      bytesOf(a.motion.frame(0).state),
      bytesOf(b.motion.frame(0).state),
    );
    assert.notDeepEqual(a.final, b.final);
    assert.equal(a.motion.length, 9);
  } finally {
    root.dispose();
  }
});

test("Wave checkpoints preserve the active population after committing its visible world", () => {
  const e = new NativeEngine(module, scene);
  try {
    e.begin(settings, 37);
    e.waveStep();
    e.restoreRows(e.states(true, settings.walkers).slice(0, e.stride));
    const saved = e.checkpoint();
    e.waveStep();
    const future = e.states(true, settings.walkers),
      tree = e.tree();
    e.restoreCheckpoint(saved);
    e.waveStep();
    assert.deepEqual(
      bytesOf(e.states(true, settings.walkers)),
      bytesOf(future),
    );
    assert.deepEqual(e.tree().meta, tree.meta);
  } finally {
    e.dispose();
  }
});

test("baseline benchmarks report actual physics work and repeat seeded trajectories", async () => {
  const run = () =>
    runBenchmark({
      module,
      scene,
      spec: {
        seeds: [7],
        maxFrames: 4,
        variants: ["fmc", "cem", "icem", "mppi"].map((algorithm) => ({
          ...settings,
          algorithm,
          icem_decay: 2,
        })),
      },
    });
  const a = await run(),
    b = await run();
  for (let i = 0; i < a.results.length; i++) {
    const x = a.results[i],
      y = b.results[i];
    assert.ok(x.simulatorFrames > 0);
    for (const key of ["reward", "controlEffort", "simulatorFrames"])
      assert.equal(x[key], y[key]);
  }
  const [fmc, cem, icem, mppi] = a.results;
  assert.equal(fmc.simulatorFrames, 2 * 12 * 4 * 2);
  assert.equal(cem.simulatorFrames, 2 * 12 * 4 * 2 * 2);
  assert.equal(icem.simulatorFrames, 2 * (12 + 6) * 4 * 2);
  assert.equal(mppi.simulatorFrames, cem.simulatorFrames);
});

for (const algorithm of ["icem", "mppi"])
  test(`${algorithm} plans joint controls for all catalog native scene presets`, async () => {
    for (const name of presetIds) {
      const config = JSON.parse(
        await readFile(
          new URL(`../web/lab/scenarios/${name}.json`, import.meta.url),
          "utf8",
        ),
      );
      const world = new NativeEngine(module, config);
      const strategy = createController(module, config, {
        ...settings,
        algorithm,
        walkers: 8,
        horizon: 2,
      });
      try {
        strategy.controller.begin(world.snapshot(), 3);
        while (!strategy.controller.advance()) {}
        const result = strategy.controller.result();
        assert.equal(result.action.length, world.dim);
        assert.ok(result.action.every(Number.isFinite));
        world.step(result.action, 1);
        assert.equal(world.metrics()[4], 1);
      } finally {
        world.dispose();
        strategy.dispose();
      }
    }
  });
