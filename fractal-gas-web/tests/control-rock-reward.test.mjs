import test from "node:test";
import assert from "node:assert/strict";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import {
  REWARD_TERMS,
  rewardValues,
  withRewards,
} from "../web/lab/reward-settings.js";
const module = await loadNative();
const base = {
  task: "harvest",
  size: [100, 100],
  physics: { dt: 0.1, substeps: 2 },
  bodies: [
    { controlled: true, position: [10, 20], velocity: [2, 0], drag: 0 },
    { cargo: true, position: [20, 20], velocity: [2, 0], drag: 0 },
    { controlled: true, position: [30, 20], velocity: [2, 0], drag: 0 },
  ],
  tethers: [
    { a: 0, b: 1, stiffness: 0, damping: 0 },
    { a: 2, b: 1, stiffness: 0, damping: 0 },
  ],
};
const weights = Object.fromEntries(REWARD_TERMS.map((t) => [t.key, 0]));
const fixture = (scene = base, coefficient = 3) =>
  new NativeEngine(
    module,
    withRewards(scene, { ...weights, hooked_rock_distance: coefficient }),
  );
const close = (actual, expected) =>
  assert.ok(Math.abs(actual - expected) < 1e-4, `${actual} != ${expected}`);

test("hooked rock travel is linear, counted once, and invariant to action batching", () => {
  const e = fixture();
  try {
    const root = e.snapshot();
    e.step(e.neutralAction(), 5);
    close(e.results()[0], 3); // 3 reward/m × 2 m/s × 0.1 s × 5, despite two hooks.
    const expected = e.snapshot();
    e.restore(root);
    let total = 0;
    for (let i = 0; i < 5; i++) {
      e.step(e.neutralAction(), 1);
      total += e.results()[0];
    }
    close(total, 3);
    assert.deepEqual(e.snapshot(), expected);
    assert.equal(rewardValues(base).hooked_rock_distance, 1);
  } finally {
    e.dispose();
  }
});

test("stationary/spinning rocks, unhooked rocks, noncargo and zero weights earn nothing", () => {
  for (const scenario of [
    { ...base, tethers: [] },
    {
      ...base,
      bodies: base.bodies.map((b, i) =>
        i === 1 ? { ...b, velocity: [0, 0] } : b,
      ),
    },
    {
      ...base,
      bodies: base.bodies.map((b, i) => (i === 1 ? { ...b, cargo: false } : b)),
    },
  ]) {
    const e = fixture(scenario);
    try {
      const rows = e.states();
      rows[8 + 5 * e.bodies + 1] = 4;
      e.restoreRows(rows);
      e.step(e.neutralAction(), 1);
      assert.equal(e.results()[0], 0);
    } finally {
      e.dispose();
    }
  }
  const e = fixture(base, 0);
  try {
    e.step(e.neutralAction(), 1);
    assert.equal(e.results()[0], 0);
  } finally {
    e.dispose();
  }
});

test("different hooked rocks sum their distances", () => {
  const e = fixture({
    ...base,
    bodies: [
      ...base.bodies,
      { cargo: true, position: [40, 20], velocity: [4, 0], drag: 0 },
    ],
    tethers: [base.tethers[0], { ...base.tethers[1], b: 3 }],
  });
  try {
    e.step(e.neutralAction(), 1);
    close(e.results()[0], 1.8);
  } finally {
    e.dispose();
  }
});

test("hook eligibility starts on the next frame and excludes inactive towing vehicles", () => {
  const source = {
    ...base,
    bodies: base.bodies.map((b, i) =>
      i === 1 ? { ...b, position: [12, 20] } : b,
    ),
    tethers: [{ ...base.tethers[0], b: -1, automatic: true, hook_range: 3 }],
  };
  const e = fixture(source);
  try {
    e.step(e.neutralAction(), 1);
    assert.equal(e.results()[0], 0);
    e.step(e.neutralAction(), 1);
    close(e.results()[0], 0.6);
    const rows = e.states();
    new Uint32Array(rows.buffer)[e.info[5]] = 0;
    e.restoreRows(rows);
    e.step(e.neutralAction(), 1);
    assert.equal(e.results()[0], 0);
  } finally {
    e.dispose();
  }
});

test("delivery respawns do not count as travel and snapshots replay exactly", () => {
  const e = fixture({
    ...base,
    bodies: base.bodies.map((b, i) => (i === 1 ? { ...b, respawn: true } : b)),
    bases: [{ position: [20.2, 20], radius: 1 }],
  });
  try {
    const root = e.snapshot();
    e.step(e.neutralAction(), 1);
    close(e.results()[0], 0.6);
    assert.equal(e.metrics()[5], 1);
    const after = e.snapshot();
    e.restore(root);
    e.step(e.neutralAction(), 1);
    close(e.results()[0], 0.6);
    assert.deepEqual(e.snapshot(), after);
    e.step(e.neutralAction(), 1);
    assert.equal(e.results()[0], 0);
  } finally {
    e.dispose();
  }
});
