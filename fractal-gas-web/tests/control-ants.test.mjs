import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  configureAntsScene,
  antsOptionsFromScene,
  DEFAULT_ANTS_OPTIONS,
} from "../web/lab/ants-scene.js";
import { resolveBodies } from "../web/lab/agent-types.js";
import { loadNative, NativeEngine } from "../web/lab/native.js";

const template = JSON.parse(
  await readFile(
    new URL("../web/lab/scenarios/ants.json", import.meta.url),
    "utf8",
  ),
);

test("Ants configuration validates input and leaves the template intact", () => {
  assert.equal(DEFAULT_ANTS_OPTIONS.count, 5);
  assert.equal(template.bodies.length, 5);
  const before = structuredClone(template);
  for (const count of [0, -1, 1.5, 129, NaN, Infinity, "48", undefined])
    assert.throws(
      () => configureAntsScene(template, { agentType: "harvester", count }),
      /integer from 1 to 128/,
    );
  assert.throws(
    () => configureAntsScene(template, { agentType: "kart", count: 48 }),
    /Harvesters or Drones/,
  );
  assert.deepEqual(
    configureAntsScene(template, DEFAULT_ANTS_OPTIONS),
    template,
  );
  const changed = configureAntsScene(template, {
    agentType: "drone",
    count: 7,
  });
  assert.deepEqual(antsOptionsFromScene(changed), {
    agentType: "drone",
    count: 7,
  });
  assert.equal(
    antsOptionsFromScene({ ...changed, name: "Custom experiment" }),
    undefined,
  );
  changed.bodies[0].position[0] = 999;
  assert.deepEqual(template, before);
});

test("Both vehicle types have correct physics and clear starts at supported counts", async () => {
  const module = await loadNative(false);
  for (const agentType of ["harvester", "drone"])
    for (const count of [1, 48, 128]) {
      const scene = configureAntsScene(template, { agentType, count });
      const bodies = resolveBodies(scene);
      assert.equal(bodies.length, count);
      for (const [i, body] of bodies.entries()) {
        assert.equal(body.visual.model, agentType);
        assert.equal(body.mass, agentType === "harvester" ? 8 : 0.7);
        assert.equal(
          body.actuator.kind,
          agentType === "harvester" ? "kart" : "holonomic",
        );
        const [x, y] = body.position;
        assert(Math.min(x - 2, 62 - x, y - 2, 42 - y) >= body.radius);
        assert(
          Math.hypot(
            Math.max(29 - x, 0, x - 33),
            Math.max(17 - y, 0, y - 28),
          ) >= body.radius,
        );
        for (const other of bodies.slice(i + 1))
          assert(
            Math.hypot(x - other.position[0], y - other.position[1]) >=
              body.radius + other.radius,
          );
      }
      assert.deepEqual(JSON.parse(JSON.stringify(scene)), scene);
      const engine = new NativeEngine(module, scene);
      try {
        assert.equal(engine.controlled, count);
        assert.equal(engine.dim, count * 3);
        engine.step(engine.neutralAction(), 3);
        assert.equal(engine.metrics()[4], 3);
        assert.equal(engine.metrics()[3], 0);
      } finally {
        engine.dispose();
      }
    }
});

test("A fully collected pool repeatedly respawns after three seconds and replays exactly", async () => {
  const module = await loadNative(false);
  const seededPositions = [];
  for (const agentType of ["harvester", "drone"])
    for (const seed of [17, 29]) {
      const scene = configureAntsScene(template, { agentType, count: 1 });
      delete scene.cargo; // Explicit legacy unlimited-forage coverage.
      scene.bodies[0].position = [10, 10];
      scene.pickups.forEach((drop) => {
        drop.position = [10, 10];
      });
      const engine = new NativeEngine(module, scene);
      try {
        engine.reset(seed);
        const offset = engine.info[8],
          action = engine.neutralAction();
        for (let cycle = 0; cycle < 3; cycle++) {
          // Bring the entire pool to the collector to exercise full depletion.
          const row = engine.states();
          for (let i = 0; i < 24; i++) {
            row[offset + i * 3] = row[8];
            row[offset + i * 3 + 1] = row[9];
          }
          engine.restoreRows(row);
          engine.step(action, 1);
          assert.equal(engine.metrics()[6], (cycle + 1) * 24);
          for (let i = 0; i < 24; i++)
            assert.equal(engine.states()[offset + i * 3 + 2], 3);
          const saved = engine.snapshot();
          engine.step(action, 179);
          for (let i = 0; i < 24; i++)
            assert(engine.states()[offset + i * 3 + 2] > 0);
          engine.step(action, 2);
          const fresh = engine.states(),
            future = engine.snapshot();
          for (let i = 0; i < 24; i++) {
            const [x, y, timer] = fresh.slice(
              offset + i * 3,
              offset + i * 3 + 3,
            );
            assert.equal(timer, 0);
            assert(x > 2 && x < 62 && y > 2 && y < 42);
            assert(!(x > 29 && x < 33 && y > 17 && y < 28));
          }
          assert.equal(engine.metrics()[3], 0);
          engine.restore(saved);
          engine.step(action, 181);
          assert.deepEqual(engine.snapshot(), future);
          if (cycle === 0)
            seededPositions.push(fresh.slice(offset, offset + 72));
        }
        engine.reset(seed);
        assert.equal(engine.metrics()[6], 0);
      } finally {
        engine.dispose();
      }
    }
  assert.notDeepEqual(seededPositions[0], seededPositions[1]);
  assert.deepEqual(seededPositions[0], seededPositions[2]);
});
