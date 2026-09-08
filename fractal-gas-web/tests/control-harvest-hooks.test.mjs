import test from "node:test";
import assert from "node:assert/strict";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import { prepareRewardEngines } from "../web/lab/live-rewards.js";
import { harvestPresentation, withHookMass } from "../web/lab/harvest-hooks.js";
import { rewardValues } from "../web/lab/reward-settings.js";
const module = await loadNative();
const base = {
  task: "harvest",
  size: [100, 100],
  physics: { dt: 0.1, substeps: 4 },
  environment: { flight: false },
  bodies: [
    { controlled: true, position: [40, 40], drag: 0, angular_drag: 0 },
    { cargo: true, position: [60, 40], drag: 0 },
  ],
  bases: [{ position: [80, 40], radius: 2 }],
  tethers: [
    {
      a: 0,
      b: -1,
      automatic: true,
      rest_length: 2.5,
      hook_range: 2.8,
      stiffness: 0,
      damping: 0,
    },
  ],
  rewards: { progress: 1, distance_squared: 0, catch: 0 },
};
const near = (a, b, tol = 1e-4) =>
  assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);
function pose(e, body, p, v = [0, 0]) {
  const r = e.states(),
    n = e.bodies;
  r[8 + body] = p[0];
  r[8 + n + body] = p[1];
  r[8 + 2 * n + body] = v[0];
  r[8 + 3 * n + body] = v[1];
  e.restoreRows(r);
}
function latch(e, target) {
  const r = e.states();
  new Uint32Array(r.buffer)[e.info[7]] = target;
  e.restoreRows(r);
}
test("harvesting has a persistent observed body and compatible presentation", () => {
  const e = new NativeEngine(module, base);
  try {
    assert.equal(e.bodies, 3);
    const s = harvestPresentation(base, e.info);
    assert.equal(s.bodies.length, e.bodies);
    assert.equal(s.tethers[0].a, 2);
    assert.equal(s.tethers[1].b, 2);
    assert.equal(s.tethers[1].permanent, true);
    assert.equal(harvestPresentation(base, [1, 2]), base);
    const before = e.snapshot();
    e.step(e.neutralAction(), 3);
    const after = e.snapshot();
    e.restore(before);
    e.step(e.neutralAction(), 3);
    assert.deepEqual(e.snapshot(), after);
  } finally {
    e.dispose();
  }
});
test("empty progress uses hook, not rocket, and moving away is negative", () => {
  for (const speed of [2, -2]) {
    const e = new NativeEngine(module, base);
    try {
      pose(e, 2, [50, 40], [speed, 0]);
      const before = 20 - 10;
      e.step(e.neutralAction(), 1);
      const r = e.states();
      near(e.results()[0], before - Math.abs(60 - r[10]));
      assert.equal(Math.sign(e.results()[0]), Math.sign(speed));
      near(r[8], 40);
    } finally {
      e.dispose();
    }
  }
});
test("catch is measured at hook and every re-catch earns the bonus", () => {
  const e = new NativeEngine(module, {
    ...base,
    rewards: { progress: 0, distance_squared: 0, catch: 10 },
  });
  try {
    for (let i = 0; i < 3; i++) {
      pose(e, 2, [58, 40]);
      latch(e, 0);
      e.step(e.neutralAction(), 1);
      near(e.results()[0], 10);
      assert.equal(new Uint32Array(e.states().buffer)[e.info[7]], 2);
      e.step(e.neutralAction(), 1);
      near(e.results()[0], 0);
    }
  } finally {
    e.dispose();
  }
});
test("attached progress points toward discharge and baseline is per frame", () => {
  const e = new NativeEngine(module, {
    ...base,
    rewards: { progress: 1, distance_squared: 1, catch: 0 },
  });
  try {
    pose(e, 0, [40, 40], [2, 0]);
    pose(e, 1, [60, 40], [3, 0]);
    latch(e, 2);
    const root = e.snapshot();
    e.step(e.neutralAction(), 5);
    near(e.results()[0], 1.7);
    const end = e.snapshot();
    e.restore(root);
    let reward = 0;
    for (let i = 0; i < 5; i++) {
      e.step(e.neutralAction(), 1);
      reward += e.results()[0];
    }
    near(reward, 1.7);
    assert.deepEqual(e.snapshot(), end);
  } finally {
    e.dispose();
  }
});
test("delivery/respawn and excluded imported reward weights add nothing", () => {
  const e = new NativeEngine(module, {
    ...base,
    bodies: base.bodies.map((b) => ({ ...b, respawn: !!b.cargo })),
    rewards: {
      progress: 0,
      distance_squared: 0,
      catch: 0,
      delivery: 1000,
      collision: 1000,
      hooked_rock_distance: 1000,
    },
  });
  try {
    pose(e, 1, [80, 40]);
    latch(e, 2);
    e.step(e.neutralAction(), 1);
    near(e.results()[0], 0);
    assert.equal(new Uint32Array(e.states().buffer)[4], 1);
    assert.equal(new Uint32Array(e.states().buffer)[e.info[7] + 2], 3);
    assert.equal(
      rewardValues({ ...base, rewards: { delivery: 1000 } }).delivery,
      0,
    );
  } finally {
    e.dispose();
  }
});
test("empty hook loads and turns rocket; mass edits preserve state bytes", () => {
  const scene = {
    ...base,
    tethers: [
      { ...base.tethers[0], stiffness: 35, damping: 6, break_force: 0 },
    ],
    environment: { flight: true, downward_gravity: 9.81 },
  };
  const e = new NativeEngine(module, scene);
  let next;
  try {
    pose(e, 2, [43, 36]);
    next = prepareRewardEngines(e, scene, withHookMass(scene, 1), {});
    assert.deepEqual(e.states(), next.engine.states());
    e.step(e.neutralAction(), 1);
    next.engine.step(next.engine.neutralAction(), 1);
    const a = e.states(),
      b = next.engine.states(),
      n = e.bodies;
    assert.ok(
      Math.abs(a[8 + 5 * n]) > 0.001,
      "off-center tension must rotate rocket",
    );
    assert.ok(Math.abs(a[8 + 2 * n]) > 0.001, "empty hook must pull rocket");
    assert.ok(
      Math.abs(b[8 + 2 * n]) > Math.abs(a[8 + 2 * n]),
      "heavier hook must pull more",
    );
    assert.equal(
      new Uint32Array(a.buffer)[e.info[7] + 2],
      3,
      "permanent cable cannot break",
    );
    for (const mass of [0, NaN, Infinity, 101])
      assert.throws(() => withHookMass(scene, mass));
  } finally {
    e.dispose();
    next?.engine.dispose();
    next?.predict.dispose();
  }
});

test("legacy recorded layouts restore without adding hook state", () => {
  const scene = { ...base, legacy_hook_layout: true };
  const e = new NativeEngine(module, scene);
  try {
    assert.equal(e.bodies, base.bodies.length);
    const snapshot = e.snapshot();
    e.step(e.neutralAction(), 1);
    e.restore(snapshot);
    assert.deepEqual(e.snapshot(), snapshot);
    assert.equal(harvestPresentation(scene, e.info), scene);
  } finally {
    e.dispose();
  }
});

test("fleet changes and rock resizing preserve one hook per rocket", async () => {
  const { readFile } = await import("node:fs/promises");
  const { configureVehicleCount } = await import("../web/lab/vehicle-scene.js");
  const { configureRocks } = await import("../web/lab/rock-scene.js");
  const preset = JSON.parse(
    await readFile(
      new URL("../web/lab/scenarios/harvest.json", import.meta.url),
    ),
  );
  for (const count of [1, 3, 2]) {
    const scene = configureRocks(configureVehicleCount(preset, count), {
      scale: 0.5,
      count: 3,
    });
    const e = new NativeEngine(module, scene);
    try {
      const shown = harvestPresentation(scene, e.info);
      assert.equal(shown.bodies.filter((b) => b.hook).length, count);
      assert.equal(e.bodies, scene.bodies.length + count);
      assert.equal(
        new Set(
          shown.tethers.filter((t) => t.owner != null).map((t) => t.owner),
        ).size,
        count,
      );
      e.step(e.neutralAction(), 2);
      assert.ok(e.states().every(Number.isFinite));
    } finally {
      e.dispose();
    }
  }
});

test("mass changes survive motion archive boundaries and resumed physics", async () => {
  const { MotionRecording, WorldCapture } = await import(
    "../web/lab/motion.js"
  );
  const { exportRecording, importRecording } = await import(
    "../web/lab/archive.js"
  );
  const e = new NativeEngine(module, base);
  let next, resumed;
  try {
    const motion = new MotionRecording(e.info, e.snapshot(), base.physics.dt);
    motion.scene = base;
    motion.settings = {};
    const capture = new WorldCapture(e, (data) =>
      motion.append(data.packet, data.label),
    );
    capture.capture(e.neutralAction());
    next = prepareRewardEngines(e, base, withHookMass(base, 2), {});
    motion.addRewardChange({
      scene: next.scene,
      coefficients: next.coefficients,
      settings: {},
      root: next.engine.snapshot(),
      tick: 0,
      decision: 0,
    });
    const captureNext = new WorldCapture(next.engine, (data) =>
      motion.append(data.packet, data.label),
    );
    captureNext.capture(next.engine.neutralAction());
    const restored = importRecording(exportRecording(base, {}, [], motion));
    const configuration = restored.motion.rewardConfiguration();
    assert.equal(configuration.scene.hook_mass, 2);
    assert.equal(
      restored.motion.rewardConfiguration(0).scene.hook_mass,
      undefined,
    );
    resumed = new NativeEngine(module, configuration.scene);
    resumed.restore(configuration.root);
    next.engine.step(next.engine.neutralAction(), 3);
    resumed.step(resumed.neutralAction(), 3);
    assert.deepEqual(resumed.snapshot(), next.engine.snapshot());
  } finally {
    e.dispose();
    next?.engine.dispose();
    next?.predict.dispose();
    resumed?.dispose();
  }
});
