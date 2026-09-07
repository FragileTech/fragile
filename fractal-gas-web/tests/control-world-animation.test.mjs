import test from "node:test";
import assert from "node:assert/strict";
import * as T from "../web/lab/vendor/three.module.js";
import { animateWorld, WorldDynamics } from "../web/lab/visuals/world.js";
import { CargoVisuals } from "../web/lab/visuals/cargo.js";

test("World animation restores authored rest and caches motion bindings", () => {
  const root = new T.Group(),
    spin = new T.Group(),
    gimbal = new T.Group();
  spin.userData.motion = "world-spin";
  spin.rotation.z = 0.3;
  gimbal.userData = { motion: "world-gimbal", axisIndex: 1, speed: 0.4 };
  gimbal.rotation.y = -0.2;
  root.add(spin, gimbal);
  animateWorld(root, 3);
  const pose = [spin.rotation.z, gimbal.rotation.y];
  root.traverse = () => {
    throw new Error("per-frame traversal");
  };
  animateWorld(root, 20);
  animateWorld(root, 3);
  assert.deepEqual([spin.rotation.z, gimbal.rotation.y], pose);
  animateWorld(root, 10, { enabled: false });
  assert.equal(spin.rotation.z, 0.3);
  assert.equal(gimbal.rotation.y, -0.2);
});

test("Disabled cargo keeps amounts and meters accurate without transfer updates", () => {
  const info = new Uint32Array(16);
  info[1] = 1;
  info[15] = 20;
  const presentation = new T.Group();
  const body = new T.Group();
  const layer = {
    controlled: [0],
    models: [body],
    presentations: [presentation],
    bodies: [{ visual: { model: "drone" } }],
  };
  const scene = {
    cargo: { capacity: 5 },
    refineries: [{ position: [0, 0], radius: 2 }],
  };
  const visual = new CargoVisuals(
    scene,
    info,
    layer,
    new T.Group(),
    "futuristic",
  );
  const state = new Float32Array(24);
  state[20] = 3;
  state[21] = 1;
  const original = state.slice();
  visual.update(state);
  const rest = visual.fill.instanceMatrix.array.slice();
  presentation.position.z = 0.2;
  visual.animate();
  assert.notDeepEqual(visual.fill.instanceMatrix.array, rest);
  visual.setAnimationsEnabled(false);
  assert.deepEqual(visual.fill.instanceMatrix.array, rest);
  assert.equal(visual.transfer.visible, false);
  const version = visual.transfer.instanceMatrix.version;
  state[20] = 1;
  visual.update(state);
  assert.equal(visual.transfer.instanceMatrix.version, version);
  assert.notDeepEqual(visual.fill.instanceMatrix.array, rest);
  state[20] = 3;
  assert.deepEqual(state, original);
});

test("World off hides cosmetics immediately and still updates pickups and tethers", () => {
  const info = new Uint32Array(16);
  info[5] = 20;
  info[7] = 24;
  info[8] = 28;
  const scene = {
    pickups: [{ position: [0, 0], radius: 0.4 }],
    tethers: [{ a: 0 }],
  };
  const layer = {
    models: [new T.Group(), new T.Group()],
    controlled: [],
    bodies: [],
  };
  const dynamics = new WorldDynamics(
    scene,
    info,
    "futuristic",
    layer,
    new T.Group(),
    new T.Group(),
  );
  // Fake loaded cosmetic props to exercise the off path without GLB I/O.
  dynamics.bursts[0] = new T.Group();
  const effect = new T.Group(),
    thrust = new T.Group();
  dynamics.effects.push({ effect, thrust });
  dynamics.setAnimationsEnabled(false);
  assert.equal(effect.visible, false);
  assert.equal(thrust.visible, false);
  assert.equal(dynamics.bursts[0].visible, false);
  const state = new Float32Array(32),
    bits = new Uint32Array(state.buffer);
  bits[24] = 2;
  state[8] = 1;
  state[9] = 4;
  state[28] = 3;
  state[29] = 5;
  const original = state.slice();
  dynamics.update(state, []);
  assert.equal(dynamics.pickups[0].visible, true);
  assert.equal(dynamics.pickups[0].position.x, 3);
  assert.equal(dynamics.tethers[0].group.visible, true);
  assert.equal(dynamics.tethers[0].cable.scale.z, 3);
  dynamics.cargo.animate = () => {
    throw new Error("disabled cosmetic work");
  };
  dynamics.animate(100);
  assert.deepEqual(state, original);
  state[30] = 2;
  bits[24] = 0;
  dynamics.update(state, []);
  assert.equal(dynamics.pickups[0].visible, false);
  assert.equal(dynamics.tethers[0].group.visible, false);
});

test("crowd effects use instance visibility and skip culled bodies", () => {
  const source = new T.Group();
  source.visible = false; // Instanced originals are hidden, not inactive.
  const effect = new T.Group();
  const dynamics = {
    animationsEnabled: true,
    cargo: { animate() {} },
    bodyLayer: { models: [source], instances: [{}], inView: [true] },
    effects: [{ i: 0, model: "drone", effect, active: true, speed: 0 }],
  };
  WorldDynamics.prototype.animate.call(dynamics, 1);
  assert.equal(effect.visible, true);
  dynamics.bodyLayer.inView[0] = false;
  WorldDynamics.prototype.animate.call(dynamics, 2);
  assert.equal(effect.visible, false);
});
