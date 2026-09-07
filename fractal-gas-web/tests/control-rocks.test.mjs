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
    for (const scale of [0.5, 1, 2])
      for (const count of name === "mining" ? [1] : [1, 5, 20]) {
        const scene = configureRocks(template, { scale, count });
        assert.deepEqual(rockOptions(JSON.parse(JSON.stringify(scene))), {
          scale,
          count,
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
        const engine = new NativeEngine(module, scene);
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
  test(`${name}: repeated deliveries replenish every slot and replay exactly`, () => {
    const scene = configureRocks(template, {
      scale: 2,
      count: name === "mining" ? 1 : 20,
    });
    const engine = new NativeEngine(module, scene);
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
