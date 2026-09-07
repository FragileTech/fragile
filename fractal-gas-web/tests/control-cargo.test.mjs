import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import { MotionRecording, WorldCapture } from "../web/lab/motion.js";
import { BodyLayer } from "../web/lab/visuals/body-layer.js";
import { CargoVisuals } from "../web/lab/visuals/cargo.js";
import * as T from "../web/lab/vendor/three.module.js";
const module = await loadNative(false);
const catalog = JSON.parse(
  await readFile(new URL("../web/lab/agent-catalog.json", import.meta.url)),
);
function config(type = "harvester") {
  return {
    // Cargo tests exercise collection/unloading semantics, not flight dynamics.
    // Keep this shared fixture planar now that drones auto-enable flight mode.
    environment: { flight: false },
    size: [64, 44],
    task: "forage",
    agent_types: catalog,
    bodies: [{ agent_type: type, position: [10, 10] }],
    cargo: { capacity: 5, unload_seconds: 2 },
    refineries: [{ position: [30, 20], radius: 5 }],
    pickups: Array.from({ length: 8 }, () => ({
      position: [10, 10],
      radius: 0.4,
    })),
    respawn_seconds: 3,
    rewards: { progress: 0, collision: 0, pickup: 10, delivery: 100 },
  };
}
const near = (a, b, t = 1e-3) => assert(Math.abs(a - b) < t, `${a} != ${b}`);
function move(e, x, y) {
  const r = e.states();
  r[8] = x;
  r[8 + e.bodies] = y;
  e.restoreRows(r);
}
for (const type of ["harvester", "drone"])
  test(`${type}: capacity, gradual reward, interrupted unloading and repeated cycles`, () => {
    const s = config(type),
      e = new NativeEngine(module, s);
    try {
      const a = e.neutralAction(),
        off = e.info[15];
      e.step(a);
      near(e.metrics()[0], 60);
      assert.deepEqual([...e.states().slice(off, off + 4)], [5, 1, 0, 1]);
      assert.equal(e.metrics()[6], 5);
      e.step(a, 20);
      assert.equal(e.metrics()[6], 5);
      near(e.metrics()[0], 0);
      move(e, 30, 20);
      e.step(a, 60);
      near(e.states()[off], 2.5);
      near(e.states()[off + 2], 2.5);
      near(e.metrics()[0], 50);
      const saved = e.snapshot();
      e.step(a, 60);
      near(e.metrics()[0], 50);
      assert.equal(e.states()[off], 0);
      assert.equal(e.metrics()[5], 1);
      const future = e.snapshot();
      e.restore(saved);
      move(e, 10, 10);
      e.step(a, 10);
      near(e.states()[off], 2.5);
      assert.equal(e.metrics()[6], 5);
      near(e.metrics()[0], 0);
      move(e, 30, 20);
      e.step(a, 60);
      assert.equal(e.states()[off], 0);
      near(e.states()[off + 2], 5);
      assert.equal(e.states()[off + 1], 0);
      e.restore(saved);
      e.step(a, 60);
      assert.deepEqual(e.snapshot(), future);
      move(e, 10, 10);
      e.step(a);
      assert.equal(e.states()[off], 3);
      assert.equal(e.states()[off + 3], 1);
      near(e.metrics()[0], 30);
      move(e, 30, 20);
      e.step(a, 60);
      assert.equal(e.states()[off], 3);
      near(e.metrics()[0], 0); // Partial loads do not unload.
      const r = e.states();
      for (let i = 0; i < 2; i++) {
        r[e.info[8] + i * 3] = 30;
        r[e.info[8] + i * 3 + 1] = 20;
        r[e.info[8] + i * 3 + 2] = 0;
      }
      e.restoreRows(r);
      e.step(a);
      assert.equal(e.states()[off], 5);
      assert.equal(e.states()[off + 3], 2);
      near(e.metrics()[0], 30);
      e.step(a, 120);
      assert.equal(e.metrics()[5], 2);
      near(e.states()[off + 2], 10);
      e.reset(7);
      assert.deepEqual([...e.states().slice(off, off + 4)], [0, 0, 0, 0]);
    } finally {
      e.dispose();
    }
  });
test("Contested drops have one owner; full vehicles yield to collectors; concurrent discharge", () => {
  const s = config();
  s.bodies = [
    { agent_type: "drone", position: [10, 10] },
    { agent_type: "drone", position: [12, 10] },
  ];
  s.pickups = Array.from({ length: 10 }, () => ({
    position: [11, 10],
    radius: 2,
  }));
  const e = new NativeEngine(module, s);
  try {
    const a = e.neutralAction(),
      off = e.info[15];
    e.step(a);
    assert.equal(e.metrics()[6], 10);
    assert.equal(e.states()[off], 5);
    assert.equal(e.states()[off + 4], 5);
    near(e.metrics()[0], 120);
    const r = e.states();
    r[8] = 29;
    r[9] = 31;
    r[10] = r[11] = 20;
    e.restoreRows(r);
    e.step(a, 120);
    assert.equal(e.metrics()[5], 2);
    near(e.metrics()[0], 200);
    near(e.states()[off + 2] + e.states()[off + 6], 10);
  } finally {
    e.dispose();
  }
});
test("Cargo snapshots, gather, observations, checkpoints and replay preserve mid-unload state", () => {
  const s = config(),
    e = new NativeEngine(module, s);
  try {
    const off = e.info[15],
      a = e.neutralAction();
    e.step(a);
    move(e, 30, 20);
    e.step(a, 30);
    const root = e.snapshot(),
      motion = new MotionRecording(e.info, root, s.physics?.dt ?? 1 / 60),
      capture = new WorldCapture(e, ({ packet }) => motion.append(packet));
    capture.step(a, 100, 1);
    motion.validate();
    assert(motion.events.some((x) => x.label.includes("tank empty")));
    const future = e.snapshot();
    e.restore(root);
    e.step(a, 100);
    assert.deepEqual(e.snapshot(), future);
    e.restore(root);
    e.begin({ walkers: 8, horizon: 2, frames: 1 }, 7);
    const checkpoint = e.checkpoint();
    e.restoreCheckpoint(checkpoint);
    assert.deepEqual(e.snapshot(), root);
    const parent = new T.Group(),
      layer = new BodyLayer(s, e.info, parent, e.channels, {
        style: "steampunk",
      }),
      visual = new CargoVisuals(s, e.info, layer, parent, "steampunk");
    visual.update(e.states());
    const pose = visual.fill.instanceMatrix.array.slice();
    e.step(a, 20);
    visual.update(e.states());
    assert.notDeepEqual(visual.fill.instanceMatrix.array, pose);
    e.restore(root);
    visual.update(e.states());
    assert.deepEqual(visual.fill.instanceMatrix.array, pose);
    assert(e.info[11] >= 12);
    const bad = e.states();
    bad[off] = 6;
    assert.throws(() => e.restoreRows(bad), /cargo/);
    assert.deepEqual(e.snapshot(), root);
  } finally {
    e.dispose();
  }
  const batch = new NativeEngine(module, s, 2);
  try {
    batch.step(batch.neutralAction());
    const rows = batch.states();
    rows[batch.info[15] + 2] = 2;
    batch.restoreRows(rows);
    batch.gather(Int32Array.from([1, 0]));
    near(batch.states()[batch.info[15] + 2], 0);
    near(batch.states()[batch.stride + batch.info[15] + 2], 2);
  } finally {
    batch.dispose();
  }
});
test("Cargo configuration validation and legacy scene behavior", () => {
  for (const cargo of [
    { capacity: 0 },
    { capacity: 1.5 },
    { unload_seconds: 0 },
    { full_reward: -1 },
  ])
    assert.throws(() => new NativeEngine(module, { ...config(), cargo }));
  assert.throws(
    () => new NativeEngine(module, { ...config(), refineries: [] }),
  );
  const s = config();
  delete s.cargo;
  const e = new NativeEngine(module, s);
  try {
    e.step(e.neutralAction());
    assert.equal(e.metrics()[6], 8);
    assert.equal(e.info[15], 0);
  } finally {
    e.dispose();
  }
});
