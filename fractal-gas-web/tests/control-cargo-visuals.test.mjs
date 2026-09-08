import test from "node:test";
import assert from "node:assert/strict";
import * as T from "../web/lab/vendor/three.module.js";
import { CargoVisuals, CARGO_LOAD_SECONDS } from "../web/lab/visuals/cargo.js";

function fixture(count = 1, capacity = 5) {
  const info = new Uint32Array(16);
  info[1] = count;
  info[15] = 8 + count * 6;
  const state = new Float32Array(info[15] + count * 4);
  const bits = new Uint32Array(state.buffer);
  const scene = {
    physics: { dt: 1 / 60 },
    cargo: { capacity },
    refineries: [{ position: [20, 20], radius: 6 }],
  };
  const layer = {
    controlled: Array.from({ length: count }, (_, i) => i),
    models: Array.from({ length: count }, () => new T.Group()),
    presentations: Array.from({ length: count }, () => new T.Group()),
    bodies: Array.from({ length: count }, (_, i) => ({
      visual: { model: ["rocket", "kart", "drone", "harvester"][i % 4] },
    })),
    active: Array(count).fill(true),
    inView: Array(count).fill(true),
    presentationVersion: 0,
  };
  const visual = new CargoVisuals(
    scene,
    info,
    layer,
    new T.Group(),
    "futuristic",
  );
  const update = (tick, held, delivered = 0, phase = 0) => {
    bits[0] = tick;
    for (let c = 0; c < count; c++) {
      state[info[15] + c * 4] = held;
      state[info[15] + c * 4 + 1] = phase;
      state[info[15] + c * 4 + 2] = delivered;
    }
    visual.update(state);
  };
  return { info, state, bits, scene, layer, visual, update };
}
function snapshot(mesh) {
  return [...mesh.instanceMatrix.array];
}
function position(mesh, index = 0) {
  const m = new T.Matrix4();
  mesh.getMatrixAt(index, m);
  return new T.Vector3().setFromMatrixPosition(m);
}

test("observed gains load through intake; counts are immediate and stationary frames freeze", () => {
  const { visual, update, state } = fixture();
  update(0, 0);
  assert.equal(visual.fill.count, 0);
  update(1, 3);
  assert.equal(visual.entries[0].amount, 3);
  assert.equal(visual.entries[0].status, "Loading");
  assert.equal(visual.entries[0].loadingAmount, 3);
  update(10, 3);
  assert(visual.transfer.count > 0);
  assert(visual.fill.count > 0);
  const original = state.slice(),
    pose = snapshot(visual.transfer);
  visual.animate();
  visual.update(state);
  assert.deepEqual(snapshot(visual.transfer), pose);
  assert.deepEqual(state, original);
  update(Math.ceil(CARGO_LOAD_SECONDS * 60) + 2, 3);
  assert.equal(visual.entries[0].status, "Carrying");
  assert.equal(visual.transfer.count, 0);
});

test("initial frames, discontinuities and explicit seeks never invent pickups", () => {
  const { visual, update } = fixture();
  update(100, 4);
  assert.equal(visual.entries[0].status, "Carrying");
  update(101, 5);
  assert.equal(visual.entries[0].status, "Loading");
  visual.resetTransitions();
  assert.equal(visual.entries[0].status, "Full");
  assert.equal(visual.transfer.count, 0);
  update(50, 2);
  assert.equal(visual.entries[0].status, "Carrying");
  update(500, 5);
  assert.equal(visual.entries[0].status, "Full");
  update(500, 1);
  assert.equal(visual.entries[0].loadingAmount, 0);
});

test("unloading requires native return phase and refinery contact; collected total stays exact", () => {
  const { visual, update, state, info, layer } = fixture();
  update(0, 5, 0, 1);
  assert.equal(visual.entries[0].status, "Full");
  assert.equal(visual.transfer.count, 0);
  state[8] = state[8 + info[1]] = 20;
  update(1, 4.5, 0.5, 1);
  assert.equal(visual.entries[0].status, "Unloading");
  assert.equal(visual.entries[0].collected, 5);
  assert(visual.transfer.count > 0);
  update(2, 4.5, 0.5, 0);
  assert.equal(visual.entries[0].status, "Carrying");
  assert.equal(visual.transfer.count, 0);
  layer.models[0].scale.setScalar(2);
  update(3, 0, 5, 0);
  assert.equal(visual.entries[0].status, "Empty");
  assert.equal(visual.entries[0].collected, 5);
  assert.equal(visual.fill.count, 0);
});

test("off shows the full authoritative load without transfer uploads, and attachment follows yaw and bob", () => {
  const { visual, update, layer, state, info } = fixture();
  update(0, 0);
  update(1, 5);
  update(8, 5);
  visual.setAnimationsEnabled(false);
  const rest = snapshot(visual.fill),
    version = visual.transfer.instanceMatrix.version;
  assert.equal(visual.fill.count, 5);
  assert.equal(visual.transfer.visible, false);
  update(9, 3);
  assert.equal(visual.fill.count, 3);
  assert.equal(visual.transfer.instanceMatrix.version, version);
  update(10, 5);
  assert.deepEqual(snapshot(visual.fill), rest);
  visual.resetTransitions();
  visual.setAnimationsEnabled(true);
  const base = position(visual.fill);
  layer.presentations[0].position.z = 0.2;
  layer.presentationVersion++;
  visual.animate();
  assert(Math.abs(position(visual.fill).z - base.z - 0.2) < 1e-6);
  visual.setAnimationsEnabled(false);
  state[8 + 4 * info[1]] = Math.PI / 2;
  update(11, 5);
  const rotated = position(visual.fill);
  assert(Math.abs(rotated.x + base.y) < 1e-6);
  assert(Math.abs(rotated.y - base.x) < 1e-6);
});

for (const count of [1, 16, 64, 128])
  test(`${count} vehicles retain three shared draws and bounded geometry with culling`, () => {
    const { visual, update, layer, state, info } = fixture(count, 10000);
    for (let i = 0; i < count; i++) state[8 + i] = state[8 + info[1] + i] = 20;
    update(0, 10000, 0, 1);
    const meshes = [visual.fill, visual.meter, visual.transfer];
    const triangles = meshes.reduce(
      (sum, m) =>
        sum +
        m.count *
          ((m.geometry.index?.count ?? m.geometry.attributes.position.count) /
            3),
      0,
    );
    assert.equal(meshes.filter((m) => m.visible).length, 3);
    assert(triangles <= count * 106);
    const version = visual.fill.instanceMatrix.version;
    visual.animate();
    visual.animate();
    assert.equal(visual.fill.instanceMatrix.version, version);
    layer.inView.fill(false);
    layer.presentationVersion++;
    visual.animate();
    assert(meshes.every((m) => m.count === 0 && !m.visible));
    layer.inView.fill(true);
    layer.active.fill(false);
    layer.presentationVersion++;
    visual.animate();
    assert(meshes.every((m) => !m.visible));
  });

test("invalid display fields are clamped and noncargo scenes create no GPU/DOM entries", () => {
  const { visual, update } = fixture();
  update(0, NaN, Infinity);
  assert.equal(visual.entries[0].amount, 0);
  assert.equal(visual.entries[0].collected, 0);
  update(1, 1e9, -4);
  assert.equal(visual.entries[0].amount, 5);
  assert(snapshot(visual.fill).every(Number.isFinite));
  const none = new CargoVisuals(
    {},
    new Uint32Array(16),
    {},
    new T.Group(),
    "futuristic",
  );
  none.update(new Float32Array());
  none.animate();
  none.resetTransitions();
  assert.deepEqual(none.entries, []);
  assert.equal(none.group, undefined);
});
