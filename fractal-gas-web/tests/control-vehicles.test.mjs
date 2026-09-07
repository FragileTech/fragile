import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import {
  configureVehicleCount,
  vehicleCount,
} from "../web/lab/vehicle-scene.js";
import { resolveBodies } from "../web/lab/agent-types.js";
import { inside, clearance } from "../web/lab/ants-scene.js";
import { loadNative, NativeEngine } from "../web/lab/native.js";
const directory = new URL("../web/lab/scenarios/", import.meta.url);
const module = await loadNative(false);
for (const file of await readdir(directory)) {
  const template = JSON.parse(await readFile(new URL(file, directory), "utf8"));
  test(`${file}: configurable fleets preserve the world and run in the engine`, () => {
    const before = structuredClone(template);
    assert.deepEqual(
      configureVehicleCount(template, vehicleCount(template)),
      template,
    );
    for (const count of [1, 7, 128]) {
      const scene = configureVehicleCount(template, count);
      assert.equal(vehicleCount(scene), count);
      const bodies = resolveBodies(scene);
      assert.deepEqual(
        scene.bodies.filter((_, i) => !bodies[i].controlled),
        template.bodies.filter(
          (_, i) => !resolveBodies(template)[i].controlled,
        ),
      );
      for (const body of bodies.slice(template.bodies.length)) {
        const r = body.radius;
        assert(inside(body.position, scene.boundary));
        assert(clearance(body.position, scene.boundary) >= r);
        for (const hole of scene.holes || []) {
          assert(!inside(body.position, hole));
          assert(clearance(body.position, hole) >= r);
        }
        for (const other of bodies) {
          if (other === body) continue;
          const otherRadius = other.vertices?.length
            ? Math.max(...other.vertices.map(([x, y]) => Math.hypot(x, y)))
            : other.radius;
          assert(
            Math.hypot(
              body.position[0] - other.position[0],
              body.position[1] - other.position[1],
            ) >=
              r + otherRadius,
          );
        }
      }
      const engine = new NativeEngine(module, scene);
      try {
        assert.equal(engine.controlled, count);
        engine.step(engine.neutralAction(), 1);
        assert([...engine.states()].every(Number.isFinite));
      } finally {
        engine.dispose();
      }
    }
    assert.deepEqual(template, before);
  });
}
test("Count validation and tether remapping", () => {
  const scene = {
    size: [20, 20],
    boundary: [
      [0, 0],
      [20, 0],
      [20, 20],
      [0, 20],
    ],
    bodies: [
      { controlled: true, position: [2, 2], radius: 0.5 },
      { controlled: true, position: [4, 2], radius: 0.5 },
      { position: [6, 2], radius: 0.5 },
    ],
    tethers: [
      { a: 0, b: 2 },
      { a: 1, b: 2 },
    ],
  };
  for (const count of [0, -1, 1.5, 129, NaN, "5"])
    assert.throws(() => configureVehicleCount(scene, count), /integer/);
  assert.deepEqual(configureVehicleCount(scene, 1).tethers, [{ a: 0, b: 1 }]);
});
