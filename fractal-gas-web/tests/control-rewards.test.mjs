import test from "node:test";
import assert from "node:assert/strict";
import {
  REWARD_TERMS,
  coefficientValues,
  rewardValues,
  withRewards,
} from "../web/lab/reward-settings.js";
import { loadNative, NativeEngine } from "../web/lab/native.js";
const scene = {
  version: 1,
  size: [100, 100],
  physics: { dt: 0.1, substeps: 2 },
  bodies: [
    { position: [50, 50], velocity: [2, -3], drag: 0, controlled: true },
  ],
};
test("reward edits preserve scene geometry and cargo configuration", () => {
  const source = {
    ...scene,
    rewards: { pickup: 17, custom: 9 },
    cargo: { unload_seconds: 3 },
  };
  const values = rewardValues(source);
  assert.equal(values.full_reward, 17);
  assert.equal(values.distance_squared, 1);
  const updated = withRewards(source, {
    ...values,
    distance_squared: 2,
    full_reward: 35,
  });
  assert.equal(updated.rewards.custom, 9);
  assert.deepEqual(updated.bodies, source.bodies);
  assert.deepEqual(updated.cargo, { unload_seconds: 3, full_reward: 35 });
  assert.equal(source.rewards.distance_squared, undefined);
  const roundtrip = JSON.parse(JSON.stringify(updated));
  assert.deepEqual(rewardValues(roundtrip), {
    ...values,
    distance_squared: 2,
    full_reward: 35,
  });
  assert.equal(withRewards(scene, rewardValues(scene)).cargo, undefined);
});
test("all reward controls validate numeric limits", () => {
  for (const term of REWARD_TERMS) {
    for (const bad of [-1, term.max + 1, Infinity, NaN, "2", undefined])
      assert.throws(
        () => withRewards(scene, { ...rewardValues(scene), [term.key]: bad }),
        RangeError,
      );
  }
});
const module = await loadNative();
test("reweighting preserves every state byte and changes future rewards", () => {
  const engine = new NativeEngine(module, scene);
  let updated;
  try {
    engine.reset(73);
    engine.step(engine.neutralAction(), 5);
    const oldState = engine.states();
    assert.ok(Math.abs(engine.results()[0] - 0.65) < 0.0001);
    const next = withRewards(scene, {
      ...rewardValues(scene),
      distance_squared: 2,
    });
    updated = new NativeEngine(module, next);
    assert.deepEqual(updated.info, engine.info);
    updated.restoreRows(oldState);
    assert.deepEqual(
      new Uint8Array(updated.states().buffer),
      new Uint8Array(oldState.buffer),
    );
    const snapshot = updated.snapshot();
    updated.step(updated.neutralAction(), 3);
    const total = updated.results()[0];
    assert.ok(
      Math.abs(total - 0.78) < 0.0001,
      `2 × (0.2² + 0.3²) × 3 = 0.78, got ${total}`,
    );
    const future = updated.states();
    updated.restore(snapshot);
    let separate = 0;
    for (let i = 0; i < 3; i++) {
      updated.step(updated.neutralAction(), 1);
      separate += updated.results()[0];
    }
    assert.ok(Math.abs(total - separate) < 1e-6);
    assert.deepEqual(updated.states(), future);
    // A positive weight gives no reward for a stationary vehicle.
    const stationary = new NativeEngine(module, {
      ...next,
      bodies: [{ ...scene.bodies[0], velocity: [0, 0] }],
    });
    try {
      stationary.step(stationary.neutralAction(), 5);
      assert.equal(stationary.results()[0], 0);
    } finally {
      stationary.dispose();
    }
  } finally {
    engine.dispose();
    updated?.dispose();
  }
});

test("coefficient defaults and limits preserve explicit zero", () => {
  assert.deepEqual(coefficientValues(), { distance_coef: 1, reward_coef: 1 });
  assert.deepEqual(coefficientValues({ distance_coef: 0, reward_coef: 10 }), {
    distance_coef: 0,
    reward_coef: 10,
  });
  for (const key of ["distance_coef", "reward_coef"])
    for (const value of [-1, 11, Infinity, NaN, "2"])
      assert.throws(() => coefficientValues({ [key]: value }), RangeError);
});
test("explicit zero disables default movement reward", () => {
  const engine = new NativeEngine(module, {
    ...scene,
    rewards: { distance_squared: 0 },
  });
  try {
    engine.step(engine.neutralAction(), 5);
    assert.equal(engine.results()[0], 0);
  } finally {
    engine.dispose();
  }
});
