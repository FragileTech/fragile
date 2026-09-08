import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { configureRocks, rockOptions } from "../web/lab/rock-scene.js";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import { inside, clearance } from "../web/lab/ants-scene.js";
const module = await loadNative();
const radius = (b) =>
  b.vertices?.length
    ? Math.max(...b.vertices.map((p) => Math.hypot(...p)))
    : b.radius;
for (const name of ["harvest", "mining"]) {
  const template = JSON.parse(
    await readFile(
      new URL(`../web/lab/scenarios/${name}.json`, import.meta.url),
    ),
  );
  test(`${name}: sizes and counts have clear positions and preserve settings`, () => {
    const before = structuredClone(template);
    for (const scale of [0.1, 0.5, 1, 2])
      for (const count of name === "mining" ? [1] : [1, 5, 20]) {
        const scene = configureRocks(template, { scale, count });
        assert.deepEqual(rockOptions(JSON.parse(JSON.stringify(scene))), {
          scale,
          count,
          weight: 1,
        });
        const rocks = scene.bodies.filter((b) => b.cargo);
        assert.equal(rocks.length, count);
        for (const b of rocks) {
          assert.equal(b.respawn, true);
          assert(inside(b.position, scene.boundary));
          assert(clearance(b.position, scene.boundary) >= radius(b));
          for (const hole of scene.holes)
            assert(
              !inside(b.position, hole) &&
                clearance(b.position, hole) >= radius(b),
            );
          for (const other of scene.bodies)
            if (b !== other)
              assert(
                Math.hypot(
                  b.position[0] - other.position[0],
                  b.position[1] - other.position[1],
                ) >=
                  radius(b) + radius(other),
              );
        }
        assert.equal(rocks[0].mass, template.bodies.find((b) => b.cargo).mass);
        const reset = configureRocks(scene, { scale: 1, count });
        assert(
          Math.abs(
            radius(reset.bodies.find((b) => b.cargo)) -
              radius(template.bodies.find((b) => b.cargo)),
          ) < 1e-10,
        );
        const engine = new NativeEngine(module, {
          ...scene,
          keep_delivered_rocks: false,
        });
        try {
          engine.step(engine.neutralAction(), 1);
          assert([...engine.results()].every(Number.isFinite));
        } finally {
          engine.dispose();
        }
      }
    assert.deepEqual(template, before);
    for (const options of [
      { scale: 0, count: 1 },
      { scale: 3, count: 1 },
      { scale: 1, count: 0 },
      { scale: 1, count: 21 },
      { scale: 1, count: 1.5 },
    ])
      assert.throws(() => configureRocks(template, options), RangeError);
  });
  test(`${name}: weight scales rock mass and survives later edits`, () => {
    const before = structuredClone(template);
    const count = name === "mining" ? 1 : 5;
    const originalRocks = template.bodies
      .filter((body) => body.cargo)
      .slice(0, count);
    const originalHulls = originalRocks.map((body) => body.vertices);
    for (const weight of [0.01, 0.1, 1, 10]) {
      const changed = configureRocks(template, { scale: 1, count, weight });
      assert.deepEqual(rockOptions(changed), { scale: 1, count, weight });
      const rocks = changed.bodies.filter((body) => body.cargo);
      for (const [i, rock] of rocks.entries()) {
        assert.equal(rock.mass, originalRocks[i].mass * weight);
        assert.deepEqual(rock.vertices, originalHulls[i]);
      }
      const reset = configureRocks(changed, { scale: 0.5, count, weight: 1 });
      assert.deepEqual(rockOptions(reset), { scale: 0.5, count, weight: 1 });
      for (const [i, rock] of reset.bodies
        .filter((body) => body.cargo)
        .entries())
        assert.ok(Math.abs(rock.mass - originalRocks[i].mass) < 1e-10);
    }
    assert.deepEqual(template, before);
    for (const weight of [0, 0.009, 10.01, Infinity, NaN])
      assert.throws(
        () => configureRocks(template, { scale: 1, count, weight }),
        RangeError,
      );
  });
  test(`${name}: repeated deliveries replenish every slot and replay exactly`, () => {
    const scene = configureRocks(template, {
      scale: 2,
      count: name === "mining" ? 1 : 20,
    });
    const engine = new NativeEngine(module, {
      ...scene,
      keep_delivered_rocks: false,
    });
    try {
      engine.reset(73);
      const cargo = scene.bodies.findIndex((b) => b.cargo);
      for (let delivery = 1; delivery <= 50; delivery++) {
        const rows = engine.states();
        rows[8 + cargo] = scene.bases[0].position[0];
        rows[8 + engine.bodies + cargo] = scene.bases[0].position[1];
        rows[8 + 2 * engine.bodies + cargo] = rows[
          8 + 3 * engine.bodies + cargo
        ] = 0;
        engine.restoreRows(rows);
        const snapshot = engine.snapshot();
        engine.step(engine.neutralAction(), 1);
        const result = engine.states();
        const words = new Uint32Array(result.buffer);
        assert.equal(words[4], delivery);
        for (let i = 0; i < scene.bodies.length; i++)
          if (scene.bodies[i].cargo)
            assert.equal(words[8 + 6 * engine.bodies + i], 1);
        engine.restore(snapshot);
        engine.step(engine.neutralAction(), 1);
        assert.deepEqual(engine.states(), result);
      }
    } finally {
      engine.dispose();
    }
  });
}

test("stock mining docks sit in the upper-left quadrant", async () => {
  for (const name of ["harvest", "mining"]) {
    const scene = JSON.parse(
      await readFile(
        new URL(`../web/lab/scenarios/${name}.json`, import.meta.url),
      ),
    );
    const [x, y] = scene.bases[0].position;
    assert(x < scene.size[0] / 2);
    assert(y > scene.size[1] / 2);
    assert(inside(scene.bases[0].position, scene.boundary));
    assert(
      clearance(scene.bases[0].position, scene.boundary) >=
        scene.bases[0].radius,
    );
  }
});

test("hook stiffness survives edits and changes spring extension", () => {
  const source = {
    task: "harvest",
    size: [100, 100],
    bodies: [
      { controlled: true, position: [40, 50], velocity: [-2, 0], drag: 0 },
      { cargo: true, position: [44, 50], velocity: [2, 0], drag: 0 },
    ],
    tethers: [
      {
        a: 0,
        b: 1,
        stiffness: 35,
        damping: 0,
        break_force: 1e9,
        rest_length: 4,
      },
    ],
  };
  const extensions = [];
  for (const stiffness of [1, 1000000]) {
    const changed = configureRocks(source, { scale: 0.1, count: 1, stiffness });
    assert.equal(changed.tethers[0].stiffness, stiffness);
    const roundtrip = configureRocks(JSON.parse(JSON.stringify(changed)), {
      scale: 0.2,
      count: 1,
    });
    assert.equal(roundtrip.tethers[0].stiffness, stiffness);
    const e = new NativeEngine(module, changed);
    try {
      e.step(e.neutralAction(), 30);
      const rows = e.states();
      assert.ok(rows.every(Number.isFinite));
      const hook = e.bodies - 1;
      extensions.push(
        Math.abs(
          Math.hypot(
            rows[9] - rows[8 + hook],
            rows[8 + e.bodies + 1] - rows[8 + e.bodies + hook],
          ) - rows[e.info[7] + 1],
        ),
      );
    } finally {
      e.dispose();
    }
  }
  assert.ok(extensions[0] > 1);
  assert.ok(extensions[1] < 0.01);
  assert.equal(source.tethers[0].stiffness, 35);
  for (const stiffness of [-1, NaN, Infinity, 1000001])
    assert.throws(
      () => configureRocks(source, { scale: 1, count: 1, stiffness }),
      RangeError,
    );
  for (const scale of [0, 0.09])
    assert.throws(
      () => configureRocks(source, { scale, count: 1 }),
      RangeError,
    );
});
