import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { actionLayout, manualAction } from "../web/lab/actions.js";
import { loadNative, NativeEngine } from "../web/lab/native.js";
import { configureRocks } from "../web/lab/rock-scene.js";
import {
  actionMultiplierValues,
  actionSettingGroups,
  withActionMultipliers,
} from "../web/lab/action-settings.js";

const module = await loadNative(false);
const mining = JSON.parse(
  await readFile(new URL("../web/lab/scenarios/mining.json", import.meta.url)),
);

const scene = {
  agent_types: {
    rocket: {
      label: "Rocket",
      physics: {
        controlled: true,
        actuator: {
          kind: "vector",
          action_multipliers: { thrust: 2, torque: 0.5 },
        },
      },
    },
    drone: {
      label: "Drone",
      physics: {
        controlled: true,
        actuator: { kind: "holonomic" },
      },
    },
  },
  bodies: [
    { agent_type: "rocket", position: [10, 10] },
    { agent_type: "drone", position: [20, 20] },
    { position: [30, 30] },
  ],
};

test("action settings expose one multiplier per controlled agent type channel", () => {
  const groups = actionSettingGroups(scene);
  assert.deepEqual(
    groups.map(({ name, label, channels }) => ({ name, label, channels })),
    [
      { name: "rocket", label: "Rocket", channels: ["thrust", "torque"] },
      {
        name: "drone",
        label: "Drone",
        channels: ["force_x", "force_y", "torque"],
      },
    ],
  );
  assert.deepEqual(actionMultiplierValues(scene, groups), {
    rocket: { thrust: 2, torque: 0.5 },
    drone: { force_x: 1, force_y: 1, torque: 1 },
  });
});

test("action multipliers are stored in the problem's agent type properties", () => {
  const next = withActionMultipliers(scene, {
    rocket: { thrust: 10, torque: 0 },
    drone: { force_x: 3 },
  });
  assert.equal(
    next.agent_types.rocket.physics.actuator.action_multipliers.thrust,
    10,
  );
  assert.equal(
    next.agent_types.rocket.physics.actuator.action_multipliers.torque,
    0,
  );
  assert.equal(
    next.agent_types.drone.physics.actuator.action_multipliers.force_x,
    3,
  );
  assert.equal(
    scene.agent_types.rocket.physics.actuator.action_multipliers.thrust,
    2,
  );
  assert.throws(
    () => withActionMultipliers(scene, { rocket: { torque: 10.1 } }),
    /between 0 and 10/,
  );
});

test("keyboard actions use the expanded native ranges", () => {
  const channels = actionLayout(scene);
  assert.deepEqual(
    channels.slice(0, 2).map(({ low, high }) => [low, high]),
    [
      [0, 2],
      [-0.5, 0.5],
    ],
  );
  assert.deepEqual(
    [...manualAction(channels, new Set(["w", "a"]))].slice(0, 2),
    [2, 0.5],
  );
});

test("native channels accept the expanded and disabled ranges", () => {
  const engine = new NativeEngine(module, scene);
  try {
    assert.deepEqual(
      engine.channels.map(({ name, low, high }) => [name, low, high]),
      [
        ["thrust", 0, 2],
        ["torque", -0.5, 0.5],
        ["force_x", -1, 1],
        ["force_y", -1, 1],
        ["torque", -1, 1],
      ],
    );
    const disabled = new NativeEngine(
      module,
      withActionMultipliers(scene, {
        rocket: { thrust: 0, torque: 0 },
      }),
    );
    try {
      assert.equal(disabled.channels[0].low, disabled.channels[0].high);
      assert.equal(disabled.channels[1].low, disabled.channels[1].high);
    } finally {
      disabled.dispose();
    }
  } finally {
    engine.dispose();
  }
});

test("rock edits preserve later per-agent actuator multipliers", () => {
  const light = configureRocks(mining, { scale: 1, count: 1, weight: 0.01 });
  const boosted = withActionMultipliers(light, {
    rocket: { thrust: 10, torque: 10 },
  });
  assert.equal(boosted.bodies[0].actuator, undefined);
  assert.equal(boosted.bodies[2].mass, 0.0024);
  const engine = new NativeEngine(module, boosted);
  try {
    assert.deepEqual(
      engine.channels.slice(0, 2).map(({ low, high }) => [low, high]),
      [
        [0, 10],
        [-10, 10],
      ],
    );
    const action = engine.neutralAction();
    action[0] = engine.channels[0].high;
    const inspected = engine.inspect(action);
    const force = Array.from({ length: inspected.length / 8 }, (_, i) =>
      inspected.slice(i * 8, i * 8 + 8),
    ).find((row) => row[0] === 0 && row[1] === 0);
    assert.ok(force);
    // Column 7 is the force magnitude, so the check survives spawn-angle tweaks.
    assert.ok(force[7] > 159);
  } finally {
    engine.dispose();
  }
});
