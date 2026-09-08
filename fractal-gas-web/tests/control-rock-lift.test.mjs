import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { rockLiftBudget } from "../web/lab/rock-lift.js";
import { configureRocks } from "../web/lab/rock-scene.js";
import { withActionMultipliers } from "../web/lab/action-settings.js";
import { loadNative, NativeEngine } from "../web/lab/native.js";

const mining = JSON.parse(
  await readFile(new URL("../web/lab/scenarios/mining.json", import.meta.url)),
);
const module = await loadNative();

test("stock flight rockets start upright with enough thrust to bank while towing", async () => {
  for (const name of ["harvest", "mining"]) {
    const scene = JSON.parse(
      await readFile(
        new URL(`../web/lab/scenarios/${name}.json`, import.meta.url),
      ),
    );
    const rock = Math.max(
      ...scene.bodies.filter((b) => b.cargo).map((b) => b.mass),
    );
    for (const body of scene.bodies.filter((b) => b.controlled)) {
      assert.ok(Math.abs(body.angle - Math.PI / 2) < 1e-6);
      assert.ok(
        body.thrust / Math.sqrt(2) >
          (body.mass + scene.hook_mass + rock) * 9.81,
        "one rocket must support the heaviest stock rock even banked 45 degrees",
      );
    }
    assert.equal(scene.physics.lethal_walls, false);
    assert.equal(scene.rewards.distance_squared, 0);
    assert.ok(scene.rewards.catch > 0 && scene.rewards.progress > 0);
  }
});

test("mining lift budget accounts for rocket weight and respects thrust settings", () => {
  const budget = rockLiftBudget(mining);
  assert.equal(budget.thrust, 48);
  assert.ok(Math.abs(budget.load - 26.8794) < 1e-8);
  assert.equal(budget.canLift, true);
  assert.ok(budget.suggestedWeight >= 1);
  const light = configureRocks(mining, {
    scale: 1,
    count: 1,
    weight: budget.suggestedWeight,
  });
  assert.ok(rockLiftBudget(light).canLift);
  assert.ok((1.25 + light.bodies[2].mass) * 9.81 <= 0.8 * 24);
  assert.equal(rockLiftBudget(light).suggestedWeight, budget.suggestedWeight);
  const boosted = withActionMultipliers(mining, { rocket: { thrust: 10 } });
  assert.equal(rockLiftBudget(boosted).thrust, 480);
  assert.equal(rockLiftBudget(boosted).canLift, true);
  assert.ok(rockLiftBudget(boosted).suggestedWeight > budget.suggestedWeight);
  const disabled = withActionMultipliers(mining, { rocket: { thrust: 0 } });
  assert.equal(rockLiftBudget(disabled).suggestedWeight, null);
  assert.equal(rockLiftBudget(mining, 0.01).canLift, true);
});

test("lift estimates handle gravity overrides, hook settings, and unsupported actuators", () => {
  assert.equal(
    rockLiftBudget({ ...mining, environment: { flight: false } }),
    null,
  );
  assert.equal(
    rockLiftBudget({ ...mining, environment: { downward_gravity: 0 } }),
    null,
  );
  assert.match(
    rockLiftBudget({ ...mining, tethers: [] }).unavailable,
    /tow hook/,
  );
  assert.match(
    rockLiftBudget({
      ...mining,
      tethers: mining.tethers.map((t) => ({ ...t, stiffness: 0 })),
    }).unavailable,
    /positive stiffness/,
  );
  const scene = structuredClone(mining);
  scene.agent_types.rocket.physics.actuator = { kind: "kart" };
  assert.match(rockLiftBudget(scene).unavailable, /rocket and drone/);
  scene.agent_types.rocket.physics.actuator = {
    kind: "holonomic",
    action_multipliers: { force_x: 0, force_y: 2 },
  };
  assert.equal(rockLiftBudget(scene).thrust, 96);
});

// Use the shipped polygon and actuator definitions in a clear vertical takeoff
// area, so navigation and lethal walls do not obscure the force balance.
function takeoffScene(weight) {
  const scene = configureRocks(mining, { scale: 1, count: 1, weight });
  scene.size = [200, 200];
  delete scene.boundary;
  scene.holes = [];
  scene.gravity = [];
  scene.bases = [];
  scene.physics.lethal_walls = false;
  scene.bodies = [scene.bodies[0], scene.bodies[2]];
  scene.bodies[0].position = [100, 5];
  scene.bodies[0].angle = Math.PI / 2;
  scene.bodies[1].position = [100, 1.3649];
  scene.tethers = [
    { ...scene.tethers[0], a: 0, b: 1, rest_length: 5 - 1.3649 },
  ];
  return scene;
}

test("WASM physics lifts the default mining rock off the floor, but not an overloaded rock", () => {
  for (const weight of [10, 1, rockLiftBudget(mining).suggestedWeight]) {
    const engine = new NativeEngine(module, takeoffScene(weight));
    try {
      engine.reset(0);
      const initial = engine.states();
      const action = Float32Array.of(1, 0);
      const checkpoint = engine.snapshot();
      engine.step(action, 120);
      const result = engine.states();
      assert.ok([...result].every(Number.isFinite));
      const rise =
        result[8 + engine.bodies + 1] - initial[8 + engine.bodies + 1];
      if (weight === 10) assert.ok(rise < 0.1, `heavy rock rose ${rise}`);
      else {
        assert.ok(rise > 2, `lighter rock rose only ${rise}`);
        assert.ok(
          result[8 + 3 * engine.bodies + 1] > 2,
          "rock must still have upward velocity",
        );
      }
      // Confirm deterministic continuation through the same floor contact.
      engine.restore(checkpoint);
      engine.step(action, 120);
      assert.deepEqual(engine.states(), result);
    } finally {
      engine.dispose();
    }
  }
});

test("every stock asteroid lifts with one rocket and the default hook at 1×", async () => {
  const harvest = JSON.parse(
    await readFile(
      new URL("../web/lab/scenarios/harvest.json", import.meta.url),
    ),
  );
  const baseline = configureRocks(harvest, { scale: 1, count: 5, weight: 1 });
  const rocks = baseline.bodies.filter((b) => b.cargo);
  assert.deepEqual(
    rocks.map((b) => b.mass),
    [0.03, 0.05, 0.02, 0.02, 0.04],
  );
  assert.equal(baseline.rock_options.weight, 1);
  const budget = rockLiftBudget(baseline);
  assert.equal(budget.thrust, 24);
  assert.ok(Math.abs(budget.load - 12.753) < 1e-8);
  assert.ok(budget.load <= 0.8 * budget.thrust);
  assert.ok(budget.suggestedWeight >= 1);
  for (const rock of rocks) {
    const scene = structuredClone(baseline);
    scene.size = [200, 200];
    delete scene.boundary;
    scene.holes = [];
    scene.bases = [];
    scene.physics.lethal_walls = false;
    scene.environment = { flight: true };
    const radius = Math.max(...rock.vertices.map(([x, y]) => Math.hypot(x, y)));
    const cargoY = radius + 0.1;
    const vehicle = {
      ...scene.bodies[0],
      position: [100, cargoY + radius + 0.2 + 0.65 + 2.5],
      angle: Math.PI / 2,
    };
    scene.bodies = [vehicle, { ...rock, position: [100, cargoY] }];
    scene.tethers = [{ ...scene.tethers[0], a: 0, b: 1 }];
    const engine = new NativeEngine(module, scene);
    try {
      const initial = engine.states();
      engine.step(Float32Array.of(1, 0), 120);
      const rows = engine.states();
      const rise = rows[8 + engine.bodies + 1] - initial[8 + engine.bodies + 1];
      assert.ok(rise > 2, `${rock.mass} kg rock rose only ${rise} m at 1×`);
      assert.ok(
        rows[8 + 3 * engine.bodies + 1] > 1,
        "cargo must still be climbing",
      );
      assert.equal(engine.results()[2], 0, "lift must not terminate the world");
    } finally {
      engine.dispose();
    }
  }
});
