import test from "node:test";
import assert from "node:assert/strict";
import { AnimationClock } from "../web/lab/animation-clock.js";
import { LabRenderer } from "../web/lab/renderer.js";

test("idle clock is bounded, reusable, and independent of simulation playback", () => {
  const clock = new AnimationClock();
  const frame = clock.tick(100);
  assert.equal(clock.tick(120), frame);
  assert.equal(frame.idleTime, 0.02);
  assert.equal(frame.playing, false);
  clock.setPlayback({ playing: true, speed: 4 });
  clock.tick(10000);
  assert.equal(frame.dt, 0.05);
  assert.equal(frame.speed, 4);
  assert.equal(frame.idleTime, 0.07);
  clock.tick(10010, false);
  clock.tick(50000);
  assert.equal(frame.dt, 0, "No hidden/off time is replayed on resume");
  clock.reset(3);
  clock.tick(50020);
  assert.equal(frame.idleTime, 3, "Seek resets to a reproducible phase");
});

test("renderer off path skips all cosmetic callbacks; hidden frames skip rendering", () => {
  const oldDocument = globalThis.document;
  const oldRaf = globalThis.requestAnimationFrame;
  let cosmetic = 0,
    cargoUpdates = 0,
    readoutUpdates = 0,
    renders = 0;
  globalThis.document = { hidden: false };
  globalThis.requestAnimationFrame = () => 1;
  try {
    const noop = () => {};
    const renderer = {
      animationsEnabled: false,
      animationClock: new AnimationClock(),
      animationStep: {},
      canvas: { clientHeight: 400 },
      camera: {},
      world: {},
      reactors: [],
      bodyLayer: { updateLod: noop, animate: () => cosmetic++ },
      worldDynamics: {
        pickupBatch: { updateLod: noop },
        cargo: { animate: () => cargoUpdates++ },
        animate: () => cosmetic++,
      },
      refreshCargoReadout: () => readoutUpdates++,
      renderer: {
        getContext: () => ({ isContextLost: () => false }),
        render: () => renders++,
        info: { render: { calls: 2, triangles: 10 } },
      },
    };
    LabRenderer.prototype.animate.call(renderer, 100);
    assert.equal(cosmetic, 0);
    assert.equal(renders, 1);
    assert.equal(cargoUpdates, 1);
    assert.equal(readoutUpdates, 1);
    assert.equal(renderer.performance.animationCpuMs, 0);
    renderer.animationsEnabled = true;
    LabRenderer.prototype.animate.call(renderer, 120);
    assert.equal(cosmetic, 2);
    document.hidden = true;
    LabRenderer.prototype.animate.call(renderer, 50000);
    assert.equal(cosmetic, 2);
    assert.equal(renders, 2);
    assert.equal(cargoUpdates, 2, "Hidden frames skip cargo updates");
    assert.equal(readoutUpdates, 2, "Hidden frames skip readout updates");
  } finally {
    if (oldDocument === undefined) delete globalThis.document;
    else globalThis.document = oldDocument;
    if (oldRaf === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = oldRaf;
  }
});

test("toggle reapplies current gameplay state after resetting cosmetic poses", () => {
  const calls = [];
  const state = new Float32Array([1, 2]),
    action = [0.4];
  const renderer = {
    animationClock: new AnimationClock(),
    simulationTime: 3,
    bodyLayer: { setAnimationsEnabled: (value) => calls.push(["body", value]) },
    worldDynamics: {
      setAnimationsEnabled: (value) => calls.push(["world", value]),
    },
    state,
    action,
    reactors: [],
    update: (s, a) => {
      assert.equal(s, state);
      assert.equal(a, action);
      calls.push(["state"]);
    },
  };
  LabRenderer.prototype.setAnimationsEnabled.call(renderer, true);
  assert.deepEqual(calls, [["body", true], ["world", true], ["state"]]);
});
