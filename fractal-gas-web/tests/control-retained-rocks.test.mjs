import test from "node:test";
import assert from "node:assert/strict";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import { configureRocks } from "../web/lab/rock-scene.js";
import { readFile } from "node:fs/promises";
const module = await loadNative();
const scene = {
  task: "harvest",
  size: [100, 100],
  keep_delivered_rocks: true,
  physics: { dt: 0.1, substeps: 2 },
  environment: { flight: false },
  bodies: [
    { controlled: true, position: [10, 10], drag: 0 },
    {
      cargo: true,
      respawn: true,
      position: [50, 50],
      radius: 0.3,
      mass: 0.1,
      drag: 0,
      angular_drag: 0,
    },
  ],
  bases: [{ position: [50, 50], radius: 4 }],
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
  rewards: { progress: 0, distance_squared: 0, catch: 10 },
};
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-4, `${a} != ${b}`);
function pose(e, i, x, y, vx = 0, vy = 0) {
  const r = e.states(),
    n = e.bodies;
  r[8 + i] = x;
  r[8 + n + i] = y;
  r[8 + 2 * n + i] = vx;
  r[8 + 3 * n + i] = vy;
  e.restoreRows(r);
}
function attach(e, t = 0, b = 1) {
  const r = e.states();
  new Uint32Array(r.buffer)[e.info[7] + 2 * t] = b + 1;
  e.restoreRows(r);
}
const step = (e) => e.step(e.neutralAction(), 1);
const flags = (e) => new Uint32Array(e.states().buffer)[e.info[5] + 1];
const deliveries = (e) => new Uint32Array(e.states().buffer)[4];
const target = (e, t = 0) =>
  new Uint32Array(e.states().buffer)[e.info[7] + 2 * t];

test("inner delivery retains motion, counts once and detaches every hook", () => {
  const s = structuredClone(scene);
  s.bodies.push({ controlled: true, position: [20, 10], drag: 0 });
  s.tethers.push({ ...s.tethers[0], a: 2 });
  const e = new NativeEngine(module, s);
  try {
    pose(e, 1, 50, 50, 1, 2);
    const r = e.states();
    r[8 + 4 * e.bodies + 1] = 0.4;
    r[8 + 5 * e.bodies + 1] = 0.5;
    e.restoreRows(r);
    attach(e, 0);
    attach(e, 1);
    step(e);
    const after = e.states();
    close(after[9], 50.1);
    close(after[8 + e.bodies + 1], 50.2);
    close(after[8 + 2 * e.bodies + 1], 1);
    close(after[8 + 3 * e.bodies + 1], 2);
    close(after[8 + 4 * e.bodies + 1], 0.45);
    close(after[8 + 5 * e.bodies + 1], 0.5);
    assert.equal(flags(e), 3);
    assert.equal(deliveries(e), 1);
    assert.equal(target(e, 0), 0);
    assert.equal(target(e, 1), 0);
    assert.equal(e.results()[0], 0);
    assert.ok(target(e, 2));
    assert.ok(target(e, 3));
    e.step(e.neutralAction(), 3);
    assert.equal(deliveries(e), 1);
    assert.equal(flags(e), 3);
  } finally {
    e.dispose();
  }
});

test("inner and outer thresholds prevent repeated delivery and early re-catching", () => {
  const e = new NativeEngine(module, scene);
  try {
    pose(e, 1, 52, 50);
    step(e);
    assert.equal(deliveries(e), 0);
    pose(e, 1, 51.9, 50);
    step(e);
    assert.equal(deliveries(e), 1);
    for (const x of [52.5, 50, 53.5, 51, 54]) {
      pose(e, 1, x, 50);
      pose(e, 2, x - 1, 50);
      step(e);
      assert.equal(flags(e), 3);
      assert.equal(target(e), 0);
      assert.equal(deliveries(e), 1);
      assert.equal(e.results()[0], 0);
    }
    pose(e, 1, 54.01, 50);
    pose(e, 2, 53, 50);
    step(e);
    assert.equal(flags(e), 1);
    assert.equal(target(e), 2);
    assert.equal(e.results()[0], 10);
    pose(e, 1, 50, 50);
    step(e);
    assert.equal(flags(e), 3);
    assert.equal(deliveries(e), 2);
    assert.equal(target(e), 0);
    assert.equal(e.results()[0], 0);
  } finally {
    e.dispose();
  }
});

test("release requires exiting all overlapping drop zones", () => {
  const e = new NativeEngine(module, {
    ...scene,
    bases: [...scene.bases, { position: [55, 50], radius: 4 }],
  });
  try {
    step(e);
    pose(e, 1, 54.5, 50);
    pose(e, 2, 53.5, 50);
    step(e);
    assert.equal(flags(e), 3);
    assert.equal(target(e), 0);
    assert.equal(deliveries(e), 1);
    pose(e, 1, 59, 50);
    step(e);
    assert.equal(flags(e), 3);
    pose(e, 1, 59.01, 50);
    pose(e, 2, 58, 50);
    step(e);
    assert.equal(flags(e), 1);
    assert.equal(target(e), 2);
  } finally {
    e.dispose();
  }
});

test("locked rocks are excluded from approach rewards, including release transitions", () => {
  const e = new NativeEngine(module, {
    ...scene,
    rewards: { progress: 1, distance_squared: 0, catch: 0 },
  });
  try {
    step(e);
    pose(e, 2, 45, 50, 1, 0);
    step(e);
    assert.equal(e.results()[0], 0);
    pose(e, 1, 53.95, 50, 1, 0);
    step(e);
    assert.equal(flags(e), 1);
    assert.equal(e.results()[0], 0);
    pose(e, 1, 55, 50);
    pose(e, 2, 45, 50, 1, 0);
    step(e);
    close(
      e.results()[0],
      (0.1 * Math.exp(-0.15 * 0.05) * (1 + Math.exp(-0.15 * 0.05))) / 2,
    );
  } finally {
    e.dispose();
  }
});

test("retained cargo remains collidable", () => {
  const e = new NativeEngine(module, {
    ...scene,
    bodies: [
      ...scene.bodies,
      { position: [53, 50], radius: 0.3, mass: 0.1, drag: 0 },
    ],
  });
  try {
    step(e);
    pose(e, 2, 51, 50, -5, 0);
    e.step(e.neutralAction(), 3);
    assert.equal(flags(e), 3);
    assert.ok(Math.abs(e.states()[8 + 2 * e.bodies + 1]) > 0.1);
    assert.equal(deliveries(e), 1);
  } finally {
    e.dispose();
  }
});

test("lock state snapshots and action batching replay exactly", () => {
  const e = new NativeEngine(module, scene);
  try {
    step(e);
    pose(e, 1, 53.8, 50, 1, 0);
    pose(e, 2, 54, 52);
    const root = e.snapshot();
    e.step(e.neutralAction(), 5);
    const end = e.snapshot(),
      reward = e.results()[0];
    e.restore(root);
    assert.equal(flags(e), 3);
    let total = 0;
    for (let i = 0; i < 5; i++) {
      step(e);
      total += e.results()[0];
    }
    assert.deepEqual(e.snapshot(), end);
    close(total, reward);
    const other = new NativeEngine(module, scene);
    try {
      other.restore(root);
      other.step(other.neutralAction(), 5);
      assert.deepEqual(other.snapshot(), end);
    } finally {
      other.dispose();
    }
  } finally {
    e.dispose();
  }
});

test("disabled and omitted options retain full-radius delivery and respawn rules", () => {
  for (const keep of [false, undefined])
    for (const respawn of [false, true]) {
      const s = {
        ...scene,
        keep_delivered_rocks: keep,
        bodies: scene.bodies.map((b) => ({
          ...b,
          respawn: b.cargo && respawn,
        })),
      };
      const e = new NativeEngine(module, s);
      try {
        pose(e, 1, 53, 50);
        step(e);
        assert.equal(deliveries(e), 1);
        assert.equal(flags(e), respawn ? 1 : 2);
        if (respawn)
          assert.ok(
            Math.hypot(e.states()[9] - 50, e.states()[8 + e.bodies + 1] - 50) >
              4,
          );
      } finally {
        e.dispose();
      }
    }
});

test("Collaborative mining retains a delivered rock without trapping it on either hook", async () => {
  const stock = JSON.parse(
    await readFile(
      new URL("../web/lab/scenarios/mining.json", import.meta.url),
    ),
  );
  const configured = configureRocks(stock, {
    scale: 1,
    count: 1,
    keepDeliveredRocks: true,
  });
  const e = new NativeEngine(module, configured);
  try {
    const cargo = configured.bodies.findIndex((body) => body.cargo);
    const base = configured.bases[0];
    pose(e, cargo, ...base.position);
    step(e);
    assert.equal(deliveries(e), 1);
    assert.equal(target(e, 0), 0);
    assert.equal(target(e, 1), 0);
    assert.ok(target(e, 2), "first rocket keeps its physical hook");
    assert.ok(target(e, 3), "second rocket keeps its physical hook");
    assert.equal(new Uint32Array(e.states().buffer)[e.info[5] + cargo], 3);
    const deliveredY = e.states()[8 + e.bodies + cargo];
    // Use the shipped gravity, drag, hulls and two spring/hook assemblies.
    // A retained rock must fall out of the lock rather than remain frozen.
    for (let frame = 0; frame < 120; frame++) step(e);
    const after = e.states();
    assert.ok(after[8 + e.bodies + cargo] < deliveredY - base.radius);
    assert.equal(new Uint32Array(after.buffer)[e.info[5] + cargo], 1);
    assert.equal(deliveries(e), 1);
    pose(e, cargo, 12, 20);
    pose(e, 0, 10, 23);
    pose(e, 1, 14, 23);
    pose(e, 3, 10.3, 20);
    pose(e, 4, 13.7, 20);
    step(e);
    assert.equal(target(e, 0), cargo + 1);
    assert.equal(target(e, 1), cargo + 1);
    assert.equal(deliveries(e), 1);
  } finally {
    e.dispose();
  }
});

test("Collaborative mining delivers and respawns when retention is disabled", async () => {
  const stock = JSON.parse(
    await readFile(
      new URL("../web/lab/scenarios/mining.json", import.meta.url),
    ),
  );
  const configured = configureRocks(stock, {
    scale: 1,
    count: 1,
    keepDeliveredRocks: false,
  });
  const e = new NativeEngine(module, configured);
  try {
    const cargo = configured.bodies.findIndex((body) => body.cargo);
    const base = configured.bases[0];
    // Between the retained inner disk and full delivery boundary: opt-out must
    // deliver here, detach both towing hooks, and move the existing body slot.
    pose(e, cargo, base.position[0] + base.radius * 0.75, base.position[1]);
    step(e);
    assert.equal(deliveries(e), 1);
    assert.equal(target(e, 0), 0);
    assert.equal(target(e, 1), 0);
    const after = e.states();
    assert.equal(new Uint32Array(after.buffer)[e.info[5] + cargo], 1);
    assert.ok(
      Math.hypot(
        after[8 + cargo] - base.position[0],
        after[8 + e.bodies + cargo] - base.position[1],
      ) > base.radius,
      "delivered rock must leave the base and respawn elsewhere",
    );
    e.step(e.neutralAction(), 3);
    assert.equal(deliveries(e), 1);
  } finally {
    e.dispose();
  }
});

test("toggle defaults, edits and JSON exports preserve delivery configuration", async () => {
  for (const name of ["harvest", "mining"]) {
    const stock = JSON.parse(
      await readFile(
        new URL(`../web/lab/scenarios/${name}.json`, import.meta.url),
      ),
    );
    const count = name === "harvest" ? 5 : 1;
    assert.equal(stock.keep_delivered_rocks, true);
    for (const keep of [false, true]) {
      const edited = configureRocks(stock, {
        scale: 1,
        count,
        keepDeliveredRocks: keep,
      });
      assert.equal(
        JSON.parse(JSON.stringify(edited)).keep_delivered_rocks,
        keep,
      );
      assert.equal(
        configureRocks(edited, { scale: 0.5, count }).keep_delivered_rocks,
        keep,
      );
    }
    assert.throws(() =>
      configureRocks(stock, {
        scale: 1,
        count,
        keepDeliveredRocks: "true",
      }),
    );
  }
  assert.throws(
    () => new NativeEngine(module, { ...scene, keep_delivered_rocks: "true" }),
  );
  const old = new NativeEngine(module, {
    ...scene,
    keep_delivered_rocks: false,
  });
  const retained = new NativeEngine(module, scene);
  try {
    assert.deepEqual(old.info, retained.info);
  } finally {
    old.dispose();
    retained.dispose();
  }
});
