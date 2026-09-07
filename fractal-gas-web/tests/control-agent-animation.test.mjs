import test from "node:test";
import assert from "node:assert/strict";
import * as T from "../web/lab/vendor/three.module.js";
import { BodyLayer } from "../web/lab/visuals/body-layer.js";
import {
  createAgentModel,
  animatedParts,
  animateAgent,
} from "../web/lab/visuals/registry.js";

function fixture(kind, count = 1) {
  const scene = {
    bodies: Array.from({ length: count }, () => ({
      controlled: true,
      visual: { model: kind },
      position: [0, 0],
    })),
  };
  const info = new Uint32Array(16);
  info[1] = count;
  info[5] = 8 + 5 * count;
  const state = new Float32Array(8 + 6 * count);
  const bits = new Uint32Array(state.buffer);
  for (let i = 0; i < count; i++) {
    bits[info[5] + i] = 1;
    state[8 + 2 * count + i] = -1;
  }
  const layer = new BodyLayer(scene, info, new T.Group());
  layer.update(state, new Float32Array(count * 2).fill(0.5));
  layer.testState = state;
  return layer;
}

test("cosmetic hover leaves root and shadow anchored; off restores all poses and steering", () => {
  const layer = fixture("drone");
  const root = layer.models[0];
  const position = root.position.clone(),
    rotation = root.rotation.clone();
  const shadow = root.getObjectByName("contact-shadow");
  const shadowPosition = shadow.position.clone();
  layer.animate(0.1, 20);
  assert.notEqual(layer.presentations[0].position.z, 0);
  assert.deepEqual(root.position, position);
  assert(root.rotation.equals(rotation));
  assert.deepEqual(shadow.position, shadowPosition);
  layer.setAnimationsEnabled(false);
  assert.equal(layer.presentations[0].position.z, 0);
  for (const { part, rotation: rest } of layer.animations[0]) {
    if (part.userData.motion === "thrust") assert.equal(part.visible, false);
    else assert(part.rotation.equals(rest));
  }
  layer.animate(0.1, 50, { playing: true });
  assert.equal(layer.animationTime, 0);
});

test("wheels roll backwards only during playback, stay grounded, and reset on seek", () => {
  const layer = fixture("kart");
  const wheel = layer.animations[0].find(
    (x) => x.part.userData.motion === "wheel",
  );
  const wheelPosition = wheel.part.position.clone();
  layer.animate(0.1, 0, { playing: true });
  assert(wheel.part.rotation.y < wheel.rotation.y);
  const angle = wheel.part.rotation.y;
  layer.animate(0.1, 0, { playing: false });
  assert.equal(wheel.part.rotation.y, angle);
  assert.deepEqual(wheel.part.position, wheelPosition);
  layer.setAnimationsEnabled(false);
  assert(wheel.part.rotation.equals(wheel.rotation));
  const steering = layer.animations[0].find(
    (x) => x.part.userData.motion === "steer",
  );
  assert.equal(steering.part.rotation.z, steering.rotation.z + 0.5 * 0.35);
  layer.setAnimationsEnabled(true);
  layer.animate(0.1, 0, { playing: true });
  layer.resetAnimation();
  assert.equal(layer.animationInputs[0].wheelTravel, 0);
});

test("crowds throttle to 30 Hz, keep instancing and skip culled agents", () => {
  const layer = fixture("drone", 17);
  const count = layer.instances.length;
  layer.animate(1 / 60, 0);
  assert.equal(layer.animationTime, 0);
  layer.inView[0] = false;
  layer.animate(1 / 60, 0);
  assert.equal(layer.animationTime, 1 / 30);
  assert.equal(layer.presentations[0].position.z, 0);
  assert.notEqual(layer.presentations[1].position.z, 0);
  assert.equal(layer.instances.length, count);
});

test("workshop profile resets deterministically without shared material changes", () => {
  for (const kind of ["rocket", "drone", "kart", "harvester"]) {
    const model = createAgentModel({ model: kind });
    const parts = animatedParts(model, { kind, style: "steampunk" });
    const materials = [];
    model.traverse((p) => {
      if (p.isMesh) materials.push([p.material, p.material.version]);
    });
    animateAgent(parts, {
      idleTime: 2,
      time: 2,
      speed: 1,
      thrust: 0.6,
      steer: 0.5,
    });
    animateAgent(parts, { enabled: false });
    assert.equal(parts.pose.position.z, 0);
    for (const [material, version] of materials)
      assert.equal(material.version, version);
    for (const { part, position, rotation } of parts.presentation) {
      assert.deepEqual(part.position, position);
      assert(part.rotation.equals(rotation));
    }
  }
});

test("ground chassis banks and pitches while intake freezes on pause", () => {
  for (const kind of ["kart", "harvester"]) {
    const layer = fixture(kind);
    const wheel = layer.animations[0].find(
      (x) => x.part.userData.motion === "wheel",
    );
    const restPosition = wheel.part.position.clone();
    layer.animate(0.1, 0, { playing: true });
    assert.notEqual(layer.presentations[0].rotation.x, 0);
    assert.notEqual(layer.presentations[0].rotation.y, 0);
    assert.deepEqual(wheel.part.position, restPosition);
    const intake = layer.animations[0].find((x) => /intake/i.test(x.part.name));
    if (intake) {
      const angle = intake.part.rotation.y;
      layer.animate(0.1, 0, { playing: false });
      assert.equal(intake.part.rotation.y, angle);
    }
    layer.setAnimationsEnabled(false);
    assert.equal(layer.presentations[0].rotation.x, 0);
    assert.equal(layer.presentations[0].rotation.y, 0);
  }
});

test("crowd state deliveries retain the last cosmetic pose on throttled frames", () => {
  const layer = fixture("kart", 17);
  layer.animate(1 / 30, 0, { playing: true });
  const wheel = layer.animations[0].find(
    (x) => x.part.userData.motion === "wheel",
  );
  const angle = wheel.part.rotation.y;
  const pitch = layer.presentations[0].rotation.y;
  new Uint32Array(layer.testState.buffer)[0] = 600;
  layer.update(layer.testState, new Float32Array(34).fill(0.5));
  layer.animate(1 / 60, 0, { playing: true });
  assert.equal(wheel.part.rotation.y, angle);
  assert.equal(layer.presentations[0].rotation.y, pitch);
});
