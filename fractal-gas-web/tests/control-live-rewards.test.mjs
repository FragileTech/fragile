import test from "node:test";
import assert from "node:assert/strict";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import { prepareRewardEngines } from "../web/lab/live-rewards.js";
import { MotionRecording, WorldCapture, bytesOf } from "../web/lab/motion.js";
import { exportRecording, importRecording } from "../web/lab/archive.js";

const module = await loadNative();
const scene = {
  version: 1,
  size: [100, 100],
  physics: { dt: 0.1, substeps: 2 },
  bodies: [
    { position: [50, 50], velocity: [2, -3], drag: 0, controlled: true },
  ],
};
test("search-branch lookup distinguishes coefficient-only changes by decision", () => {
  const engine = new NativeEngine(module, scene);
  try {
    const root = engine.snapshot();
    const motion = new MotionRecording(engine.info, root, 0.1);
    motion.scene = scene;
    motion.settings = { reward_coef: 1 };
    motion.rewardChanges = [
      { root, scene, decision: 2, settings: { reward_coef: 2 } },
      { root, scene, decision: 5, settings: { reward_coef: 3 } },
    ];
    assert.equal(
      motion.rewardConfigurationForRoot(root, 2).settings.reward_coef,
      1,
    );
    assert.equal(
      motion.rewardConfigurationForRoot(root, 5).settings.reward_coef,
      2,
    );
    assert.equal(
      motion.rewardConfigurationForRoot(root, 6).settings.reward_coef,
      3,
    );
  } finally {
    engine.dispose();
  }
});
test("live replacements preserve bytes, change future rewards, and roll back invalid settings", () => {
  const engine = new NativeEngine(module, scene);
  let next;
  try {
    engine.reset(73);
    engine.step(engine.neutralAction(), 5);
    const before = engine.states();
    next = prepareRewardEngines(
      engine,
      scene,
      { ...scene, rewards: { distance_squared: 2 } },
      {},
    );
    assert.deepEqual(bytesOf(next.engine.states()), bytesOf(before));
    next.engine.step(next.engine.neutralAction(), 3);
    assert.ok(Math.abs(next.engine.results()[0] - 0.78) < 1e-5);
    assert.throws(
      () =>
        prepareRewardEngines(
          engine,
          scene,
          { ...scene, rewards: { distance_squared: -1 } },
          {},
        ),
      RangeError,
    );
    assert.deepEqual(bytesOf(engine.states()), bytesOf(before));
  } finally {
    engine.dispose();
    next?.engine.dispose();
    next?.predict.dispose();
  }
});
test("continuous archive keeps reward boundaries and restores omitted-default scene snapshots", () => {
  const engine = new NativeEngine(module, scene);
  let next, restored;
  try {
    const motion = new MotionRecording(engine.info, engine.snapshot(), 0.1);
    motion.scene = scene;
    motion.settings = { distance_coef: 1, reward_coef: 1 };
    const capture = new WorldCapture(engine, (data) =>
      motion.append(data.packet, data.label),
    );
    capture.capture(engine.neutralAction());
    capture.step(engine.neutralAction(), 4, 1);
    const original = motion.pack();
    next = prepareRewardEngines(
      engine,
      scene,
      { ...scene, rewards: { distance_squared: 2 } },
      { reward_coef: 2 },
    );
    motion.addRewardChange({
      scene: next.scene,
      settings: next.coefficients,
      coefficients: next.coefficients,
      root: next.engine.snapshot(),
      tick: 4,
    });
    capture.engine = next.engine;
    capture.capture(next.engine.neutralAction(), 1, "Reward settings changed");
    capture.step(next.engine.neutralAction(), 3, 2);
    const imported = importRecording(
      exportRecording(next.scene, next.coefficients, [], motion),
    ).motion;
    assert.deepEqual(imported.pack().subarray(0, original.length), original);
    assert.equal(imported.rewardChanges.length, 1);
    assert.equal(imported.rewardConfiguration(0).scene.rewards, undefined);
    assert.equal(
      imported.rewardConfiguration().scene.rewards.distance_squared,
      2,
    );
    const base = imported.rewardConfiguration(0);
    restored = prepareRewardEngines(
      next.engine,
      next.scene,
      base.scene,
      base.settings,
      imported.rows(0),
      base.root,
    );
    assert.deepEqual(
      bytesOf(restored.engine.states()),
      bytesOf(imported.rows(0)),
    );
    assert.throws(() =>
      restored.engine.restore(imported.rewardConfiguration().root),
    );
    const malformed = structuredClone(imported.rewardChanges);
    malformed[0].frame = imported.length;
    assert.throws(() => imported.restoreRewardChanges(malformed));
  } finally {
    engine.dispose();
    next?.engine.dispose();
    next?.predict.dispose();
    restored?.engine.dispose();
    restored?.predict.dispose();
  }
});
