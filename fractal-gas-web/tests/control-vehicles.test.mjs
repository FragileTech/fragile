import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import {
  configureVehicleCount,
  configureVehicleType,
  vehicleCount,
  vehicleType,
} from "../web/lab/vehicle-scene.js";
import {
  resolveBodies,
  resolveAgentTypes,
  flightMode,
  VEHICLE_TYPES,
} from "../web/lab/agent-types.js";
import { actionLayout } from "../web/lab/actions.js";
import { inside, clearance } from "../web/lab/ants-scene.js";
import { loadNative, NativeEngine } from "../web/lab/native.js";
const directory = new URL("../web/lab/scenarios/", import.meta.url);
const module = await loadNative(false);
const catalog = JSON.parse(
  await readFile(
    new URL("../web/lab/agent-catalog.json", import.meta.url),
    "utf8",
  ),
);
for (const file of await readdir(directory)) {
  const template = JSON.parse(await readFile(new URL(file, directory), "utf8"));
  for (const type of VEHICLE_TYPES) {
    test(`${file}: switch all vehicles to ${type}, preserving the environment`, () => {
      const before = structuredClone(template);
      const scene = configureVehicleType(template, type);
      const oldBodies = resolveBodies(template),
        bodies = resolveBodies(scene);
      assert.equal(vehicleType(scene), type);
      assert.equal(vehicleCount(scene), vehicleCount(template));
      const {
        bodies: old,
        description: oldDescription,
        ...oldWorld
      } = template;
      const {
        bodies: changed,
        description: newDescription,
        ...newWorld
      } = scene;
      assert.deepEqual(newWorld, oldWorld);
      const expected = resolveAgentTypes(scene.agent_types).get(type);
      for (const [i, body] of bodies.entries()) {
        if (!oldBodies[i].controlled) {
          assert.deepEqual(body, oldBodies[i]);
          continue;
        }
        assert.deepEqual(body.position, oldBodies[i].position);
        assert.equal(body.angle, oldBodies[i].angle);
        for (const [key, value] of Object.entries(expected.physics))
          assert.deepEqual(body[key], value, key);
        assert.deepEqual(body.visual, expected.visual);
      }
      assert.equal(flightMode(scene), ["rocket", "drone"].includes(type));
      const roundTrip = JSON.parse(JSON.stringify(scene));
      assert.deepEqual(roundTrip, scene);
      const engine = new NativeEngine(module, roundTrip);
      try {
        assert.equal(
          engine.dim,
          vehicleCount(scene) * (type === "rocket" ? 2 : 3),
        );
        assert.equal(actionLayout(scene).length, engine.dim);
        engine.step(engine.neutralAction(), 3);
        assert([...engine.states()].every(Number.isFinite));
        const checkpoint = engine.snapshot();
        engine.step(engine.neutralAction(), 2);
        const future = engine.snapshot();
        engine.restore(checkpoint);
        engine.step(engine.neutralAction(), 2);
        assert.deepEqual(engine.snapshot(), future);
      } finally {
        engine.dispose();
      }
      const larger = configureVehicleCount(scene, 7);
      assert.equal(vehicleCount(larger), 7);
      assert.equal(vehicleType(larger), type);
      assert.deepEqual(template, before);
    });
  }
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
test("Switching clears old physics, preserves instance state and explicit flight, and supports imports", () => {
  const scene = {
    name: "Edited scene",
    environment: { flight: false },
    agent_types: catalog,
    bodies: [
      {
        agent_type: "rocket",
        position: [5, 5],
        velocity: [1, 2],
        angle: 0.5,
        omega: 0.2,
        mass: 55,
        inertia: 90,
        vertices: [
          [0, 0],
          [1, 0],
          [0, 1],
        ],
        restitution: 0.9,
        friction: 1.4,
        actuator: { kind: "vector" },
        flight_capable: true,
        visual: { model: "rocket", color: "red" },
        cargo: true,
        respawn: true,
      },
      { position: [9, 9], cargo: true, radius: 1 },
    ],
    tethers: [{ a: 0, b: 1, stiffness: 35 }],
    rewards: { movement: 2 },
  };
  let converted = scene;
  for (const type of [...VEHICLE_TYPES, "rocket"]) {
    converted = configureVehicleType(converted, type);
    assert.equal(vehicleType(converted), type);
    assert.equal(flightMode(converted), false);
    assert.deepEqual(converted.bodies[0].velocity, [1, 2]);
    assert.equal(converted.bodies[0].omega, 0.2);
    assert.equal(converted.bodies[0].cargo, true);
    assert.equal(converted.bodies[0].respawn, true);
    for (const key of [
      "inertia",
      "vertices",
      "restitution",
      "friction",
      "visual",
      "mass",
      "actuator",
    ])
      assert.equal(converted.bodies[0][key], undefined);
    assert.deepEqual(converted.tethers, scene.tethers);
    assert.deepEqual(converted.rewards, scene.rewards);
  }
  const imported = {
    bodies: [{ controlled: true, position: [3, 4], mass: 88 }],
  };
  const saved = structuredClone(imported);
  assert.equal(vehicleType(imported), undefined);
  const drone = configureVehicleType(imported, "drone", catalog);
  assert.equal(resolveBodies(drone)[0].mass, 0.7);
  assert.equal(vehicleType(drone), "drone");
  assert.deepEqual(imported, saved);
  assert.throws(
    () => configureVehicleType(imported, "unknown", catalog),
    /Choose/,
  );
  assert.throws(() => configureVehicleType(imported, "drone"), /Missing/);
  assert.throws(
    () => configureVehicleType({ bodies: [] }, "drone", catalog),
    /no vehicles/,
  );
  assert.equal(
    vehicleType({
      agent_types: catalog,
      bodies: [{ agent_type: "rocket" }, { agent_type: "drone" }],
    }),
    undefined,
  );
  assert.equal(
    vehicleType({
      agent_types: catalog,
      bodies: [{ agent_type: "thruster_tug" }],
    }),
    undefined,
  );
  assert.equal(vehicleType({ bodies: [] }), undefined);
});
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

test("Ants fleet changes preserve custom instructions and world settings", async () => {
  const scene = JSON.parse(
    await readFile(new URL("ants.json", directory), "utf8"),
  );
  scene.description = "5 harvesters. Custom cargo and replenishment rules.";
  scene.cargo.capacity = 9;
  scene.respawn_seconds = 7;
  const changed = configureVehicleCount(
    configureVehicleType(scene, "rocket"),
    3,
  );
  assert.equal(
    changed.description,
    "3 rockets. Custom cargo and replenishment rules.",
  );
  assert.deepEqual(changed.cargo, scene.cargo);
  assert.equal(changed.respawn_seconds, 7);
  assert.deepEqual(
    changed.bodies.map((body) => body.position),
    scene.bodies.slice(0, 3).map((body) => body.position),
  );
  scene.description = "My edited foraging experiment";
  assert.equal(
    configureVehicleType(scene, "drone").description,
    scene.description,
  );
});
