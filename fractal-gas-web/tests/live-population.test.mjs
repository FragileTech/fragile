import test from "node:test";
import assert from "node:assert/strict";
import { loadNative, NativeEngine } from "../web/lab/native.js";
const controlModule = await loadNative();
const scene = {
  size: [1000, 1000],
  bodies: [{ position: [500, 500], controlled: true }],
};
const cfg = {
  walkers: 8,
  max_walkers: 32,
  elites: 2,
  horizon: 10,
  frames: 1,
  recording: 2,
};

test("packed Wave resizing and checkpoint continuation are deterministic", () => {
  const a = new NativeEngine(controlModule, scene),
    b = new NativeEngine(controlModule, scene);
  try {
    a.begin(cfg, 19);
    a.advance();
    const world = a.snapshot(),
      iteration = a.metrics()[8];
    for (const count of [4, 24, 6, 18]) {
      a.setPopulation(count, "virtual_reward");
      assert.equal(a.populationStatus().active, count);
      assert.equal(a.states(true).length, count * a.stride);
      assert.deepEqual(a.snapshot(), world);
      assert.equal(a.metrics()[8], iteration);
    }
    const state = a.states(true),
      status = a.populationStatus();
    for (const count of [0, 1, 33]) assert.throws(() => a.setPopulation(count));
    assert.throws(() => a.setPopulation(8, "unknown"));
    assert.deepEqual(a.populationStatus(), status);
    assert.deepEqual(a.states(true), state);
    b.restoreCheckpoint(a.checkpoint());
    assert.deepEqual(a.populationStatus(), b.populationStatus());
    for (let i = 0; i < 3; i++) {
      a.advance();
      b.advance();
    }
    assert.deepEqual(a.states(true), b.states(true));
    assert.deepEqual(a.tree(), b.tree());
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("pending counts survive checkpoints and begin applies the latest request", () => {
  const a = new NativeEngine(controlModule, scene),
    b = new NativeEngine(controlModule, scene);
  try {
    a.begin({ ...cfg, horizon: 1 }, 19);
    a.advance();
    const plan = a.planResult();
    a.setPopulation(20, "cumulative_reward");
    a.setPopulation(16, "virtual_reward");
    assert.equal(a.populationStatus().active, 8);
    assert.equal(a.populationStatus().requested, 16);
    assert.deepEqual(a.planResult(), plan);
    b.restoreCheckpoint(a.checkpoint());
    assert.deepEqual(b.populationStatus(), a.populationStatus());
    // begin from the same structural settings preserves the pending native count.
    a.begin({ ...cfg, horizon: 1 }, 20);
    assert.equal(a.populationStatus().active, 16);
    assert.equal(a.populationStatus().pending, false);
    b.begin({ ...cfg, horizon: 1 }, 20);
    assert.equal(b.populationStatus().active, 16);
    a.advance();
    b.advance();
    assert.deepEqual(a.states(true), b.states(true));
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("no-op requests preserve the next search iteration and random stream", () => {
  const a = new NativeEngine(controlModule, scene),
    b = new NativeEngine(controlModule, scene);
  try {
    a.begin(cfg, 1);
    a.advance();
    b.restoreCheckpoint(a.checkpoint());
    a.setPopulation(8, "cumulative_reward");
    a.advance();
    b.advance();
    assert.deepEqual(a.states(true), b.states(true));
    assert.deepEqual(a.tree(), b.tree());
  } finally {
    a.dispose();
    b.dispose();
  }
});

test("control recordings retain population settings and pending requests", async () => {
  const { MotionRecording, WorldCapture } = await import(
    "../web/lab/motion.js"
  );
  const { exportRecording, importRecording } = await import(
    "../web/lab/archive.js"
  );
  const engine = new NativeEngine(controlModule, scene);
  try {
    const record = new MotionRecording(engine.info, engine.snapshot());
    record.scene = scene;
    record.settings = cfg;
    const capture = new WorldCapture(engine, ({ packet, label }) =>
      record.append(packet, label),
    );
    capture.capture(engine.neutralAction(), 0, "Initial");
    const population = {
      active: 8,
      requested: 20,
      maximum: 32,
      removal_policy: "cumulative_reward",
      pending: true,
    };
    record.addPopulationChange({ population, decision: 0 });
    const loaded = importRecording(
      exportRecording(scene, cfg, [], record),
    ).motion;
    assert.equal(loaded.length, 1);
    assert.deepEqual(loaded.populationChanges, record.populationChanges);
    assert.equal(loaded.rewardConfiguration().settings.walkers, 20);
    assert.equal(loaded.rewardConfiguration().settings.max_walkers, 32);
    assert.throws(() =>
      loaded.restorePopulationChanges([
        { frame: 0, population: { ...population, requested: 33 } },
      ]),
    );
  } finally {
    engine.dispose();
  }
});
