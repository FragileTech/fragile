import test from "node:test";
import assert from "node:assert/strict";
import * as T from "../web/lab/vendor/three.module.js";
import { ActionEffects } from "../web/lab/visuals/action-effects.js";
function fixture(kind = "vector", count = 1) {
  const layer = {
    controlled: Array.from({ length: count }, (_, i) => i),
    models: Array.from({ length: count }, () => new T.Group()),
    presentations: Array.from({ length: count }, () => new T.Object3D()),
    commands: Array.from({ length: count }, () => ({
      kind,
      thrust: 0,
      throttle: 0,
      steering: 0,
      brake: 0,
      forceX: 0,
      forceY: 0,
      torque: 0,
      thrusters: [],
    })),
    active: Array(count).fill(true),
    inView: Array(count).fill(true),
    presentationVersion: 0,
  };
  return { layer, effects: new ActionEffects(layer, new T.Group()) };
}
test("signed torque jets form the native counterclockwise or clockwise couple", () => {
  const { layer, effects } = fixture();
  const matrix = new T.Matrix4(),
    p = new T.Vector3(),
    q = new T.Quaternion(),
    s = new T.Vector3();
  for (const sign of [-1, 1]) {
    layer.commands[0].torque = sign;
    layer.presentationVersion++;
    effects.update();
    assert.equal(effects.jets.count, 2);
    let moment = 0;
    for (let i = 0; i < 2; i++) {
      effects.jets.getMatrixAt(i, matrix);
      matrix.decompose(p, q, s);
      const force = new T.Vector3(-1, 0, 0).applyQuaternion(q);
      moment += p.x * force.y - p.y * force.x;
    }
    assert.ok(moment * sign > 0);
    assert.equal(effects.lamps.count, 0);
  }
  layer.commands[0].torque = 0;
  layer.presentationVersion++;
  effects.update();
  assert.equal(effects.jets.count, 0);
  effects.dispose();
});
test("sideways and reverse thrusters keep independent command signs", () => {
  const { layer, effects } = fixture("thrusters");
  layer.commands[0].thrusters = [
    { value: 1, position: [0, 0.5], direction: [1, 0] },
    { value: -1, position: [0, -0.5], direction: [1, 0] },
    { value: 0.5, position: [0, 0], direction: [0, 1] },
  ];
  effects.update();
  assert.equal(effects.jets.count, 3);
  const m = new T.Matrix4(),
    axis = new T.Vector3();
  for (const [i, x, y] of [
    [0, -1, 0],
    [1, 1, 0],
    [2, 0, -1],
  ]) {
    effects.jets.getMatrixAt(i, m);
    axis.set(1, 0, 0).transformDirection(m);
    assert.ok(Math.abs(axis.x - x) < 1e-6 && Math.abs(axis.y - y) < 1e-6);
  }
  effects.dispose();
});
test("static cues update without animation and unchanged frames do not upload", () => {
  const { layer, effects } = fixture("kart");
  layer.commands[0].brake = 1;
  layer.commands[0].throttle = -1;
  effects.update();
  assert.equal(effects.lamps.count, 4);
  const v = effects.lamps.instanceMatrix.version;
  assert.equal(effects.update(), false);
  assert.equal(effects.lamps.instanceMatrix.version, v);
  layer.commands[0].brake = 0;
  layer.commands[0].throttle = 0;
  layer.presentationVersion++;
  effects.update();
  assert.equal(effects.lamps.count, 0);
  effects.dispose();
});
test("crowd culling, attachment and per-agent geometry budgets", () => {
  const { layer, effects } = fixture("thrusters", 128);
  assert.equal(effects.jets.material.forceSinglePass, true);
  assert.equal(effects.guides.material.forceSinglePass, true);
  for (const c of layer.commands)
    c.thrusters = Array.from({ length: 32 }, () => ({
      value: 1,
      position: [0, 0],
      direction: [1, 0],
    }));
  effects.update();
  assert.equal(effects.jets.count, 4096);
  assert.equal(effects.jets.geometry.attributes.position.count / 3, 4);
  assert.equal((effects.jets.count * 4) / 128, 128);
  assert.equal(
    effects.group.children.filter((m) => m.isInstancedMesh).length,
    2,
  );
  layer.inView.fill(false);
  layer.inView[0] = true;
  layer.presentationVersion++;
  effects.update();
  assert.equal(effects.jets.count, 32);
  layer.presentations[0].position.z = 0.2;
  layer.presentationVersion++;
  effects.update();
  const m = new T.Matrix4();
  effects.jets.getMatrixAt(0, m);
  assert.ok(Math.abs(m.elements[14] - (effects.jetHeights[0] + 0.2)) < 1e-6);
  layer.active[0] = false;
  layer.presentationVersion++;
  effects.update();
  assert.equal(effects.jets.visible, false);
  effects.dispose();
});
test("guides select one agent, label commands and stay within two draws /256 triangles", () => {
  const { layer, effects } = fixture("holonomic", 2);
  layer.commands[0].forceX = 1;
  layer.commands[0].forceY = -0.5;
  layer.commands[0].torque = -1;
  const text = effects.updateGuides(99, true);
  assert.match(text, /Body 1 · Commands/);
  assert.match(text, /Force Y -50%/);
  assert.match(text, /Torque -100%/);
  assert.ok(effects.guideGeometry.drawRange.count / 3 <= 256);
  assert.equal(effects.guides.visible, true);
  effects.updateGuides(1, true);
  assert.equal(effects.guideBody, 1);
  effects.updateGuides(1, false);
  assert.equal(effects.guides.visible, false);
  assert.equal(effects.label, "");
  effects.updateGuides(0, true);
  layer.inView[0] = false;
  layer.presentationVersion++;
  effects.update();
  assert.equal(effects.guides.visible, false);
  layer.inView[0] = true;
  layer.presentationVersion++;
  effects.update();
  assert.equal(effects.guides.visible, true);
  effects.dispose();
});

test("authored lamp sockets are cached through model normalization and cosmetic poses", () => {
  const layer = {
    controlled: [0],
    models: [new T.Group()],
    presentations: [new T.Object3D()],
    commands: [{ kind: "kart", brake: 1, throttle: 0 }],
    active: [true],
    inView: [true],
    presentationVersion: 0,
  };
  const model = layer.models[0],
    normalized = new T.Group();
  normalized.scale.setScalar(0.5);
  model.add(normalized);
  model.scale.setScalar(2);
  for (const y of [-0.5, 0.5]) {
    const socket = new T.Object3D();
    socket.userData.effectSocket = "brake";
    socket.position.set(-2, y, 1);
    normalized.add(socket);
  }
  const effects = new ActionEffects(layer, new T.Group());
  model.traverse = () => {
    throw new Error("per-frame traversal");
  };
  layer.presentations[0].position.z = 0.1;
  effects.update();
  const m = new T.Matrix4();
  effects.lamps.getMatrixAt(0, m);
  assert.ok(Math.abs(m.elements[12] + 2) < 1e-6);
  assert.ok(Math.abs(m.elements[14] - 1.2) < 1e-6);
  effects.dispose();
});
