import test from "node:test";
import assert from "node:assert/strict";
import {
  actionLayout,
  createActionBinding,
  normalizedAction,
  visualInput,
} from "../web/lab/actions.js";
import {
  animatedParts,
  animateAgent,
  createAgentModel,
} from "../web/lab/visuals/registry.js";
import { BodyLayer } from "../web/lab/visuals/body-layer.js";
import * as T from "../web/lab/vendor/three.module.js";

test("signed normalization uses asymmetric endpoints and rejects invalid or disabled input", () => {
  const c = { low: -2, high: 4 };
  for (const [input, expected] of [
    [-1, -0.5],
    [2, 0.5],
    [-9, -1],
    [9, 1],
    [0, 0],
    [NaN, 0],
    [Infinity, 0],
    [undefined, 0],
  ])
    assert.equal(normalizedAction(c, input), expected);
  for (const channel of [
    { low: 0, high: 0 },
    { low: 1, high: 2 },
    { low: -1, high: -0.2 },
    { low: -1, high: NaN },
    { ...c, enabled: false },
    { ...c, disabled: true },
  ])
    assert.equal(normalizedAction(channel, 1), 0);
  assert.equal(normalizedAction({ low: 0, high: 4 }, -1), 0);
});

test("cached commands preserve every actuator channel and return reusable storage", () => {
  for (const kind of ["vector", "kart", "holonomic", "thrusters"]) {
    const body = {
      controlled: true,
      actuator: {
        kind,
        thrusters: [
          { position: [0, 1], direction: [1, 0], reversible: true },
          { position: [0, -1], direction: [0, 1] },
        ],
      },
    };
    const channels = actionLayout({ bodies: [body] });
    const binding = createActionBinding(channels, body, 0),
      commands = binding.commands;
    const thrusters = commands.thrusters;
    assert.equal(
      binding.sample(channels.map((_, i) => (i % 2 ? 1 : -0.5))),
      commands,
    );
    if (kind === "vector") {
      assert.equal(commands.thrust, 0);
      assert.equal(commands.torque, 1);
    }
    if (kind === "kart") {
      assert.equal(commands.throttle, -0.5);
      assert.equal(commands.steering, 1);
      assert.equal(commands.brake, 0);
    }
    if (kind === "holonomic") {
      assert.equal(commands.forceX, -0.5);
      assert.equal(commands.forceY, 1);
      assert.equal(commands.torque, -0.5);
    }
    if (kind === "thrusters") {
      assert.equal(thrusters[0].value, -0.5);
      assert.equal(thrusters[1].value, 1);
      assert.deepEqual(thrusters[0].position, [0, 1]);
      assert.deepEqual(thrusters[1].direction, [0, 1]);
    }
    binding.sample();
    assert.equal(commands.thrusters, thrusters);
    for (const field of [
      "thrust",
      "throttle",
      "steering",
      "brake",
      "forceX",
      "forceY",
      "torque",
    ])
      assert.equal(commands[field], 0);
    assert(commands.thrusters.every((t) => t.value === 0));
  }
});

test("body bindings isolate agents, normalize configured multipliers and clear inactive controls", () => {
  const body = {
    controlled: true,
    actuator: { kind: "vector", action_multipliers: { thrust: 4, torque: 0 } },
  };
  const channels = actionLayout({ bodies: [body, { controlled: true }] });
  const b = createActionBinding(channels, body, 0);
  assert.equal(b.sample([2, 1, 1, 1]).thrust, 0.5);
  assert.equal(b.commands.torque, 0);
  body.disabled = true;
  assert.equal(b.sample([4, 0, 1, 1]).thrust, 0);
  assert.deepEqual(
    visualInput(
      [
        { body: 0, name: "force_y", low: -2, high: 4 },
        { body: 0, name: "brake", low: 0, high: 1 },
      ],
      [4, 1],
      0,
    ),
    { thrust: 0, steer: 0 },
  );
});

test("main exhaust is static when motion is off, proportional, immediate and zero at no thrust", () => {
  const parts = animatedParts(createAgentModel({ model: "rocket" }), {
    kind: "rocket",
  });
  const flame = parts.find((p) => p.part.userData.motion === "thrust");
  const commands = createActionBinding(
    [{ body: 0, name: "thrust", low: 0, high: 2 }],
    { controlled: true },
    0,
  ).commands;
  let previous = 0;
  for (const thrust of [0.1, 0.5, 1]) {
    commands.thrust = thrust;
    animateAgent(parts, { commands, enabled: false, time: 123 });
    assert(flame.part.visible);
    assert(flame.part.scale.x > previous);
    previous = flame.part.scale.x;
  }
  const length = flame.part.scale.x;
  animateAgent(parts, { commands, enabled: false, time: 456 });
  assert.equal(flame.part.scale.x, length);
  commands.thrust = 0;
  animateAgent(parts, { commands, enabled: false });
  assert.equal(flame.part.visible, false);
  commands.kind = "thrusters";
  commands.thrust = 1;
  animateAgent(parts, { commands, enabled: true });
  assert.equal(flame.part.visible, false);
});

test("drone lean follows signed XY commands and off removes only cosmetic bank", () => {
  const parts = animatedParts(createAgentModel({ model: "drone" }), {
    kind: "drone",
  });
  const binding = createActionBinding(
    actionLayout({
      bodies: [{ controlled: true, actuator: { kind: "holonomic" } }],
    }),
    { actuator: { kind: "holonomic" } },
    0,
  );
  animateAgent(parts, { commands: binding.sample([1, -1, 0.5]), idleTime: 1 });
  assert(parts.pose.rotation.x > 0);
  assert(parts.pose.rotation.y > 0);
  animateAgent(parts, { commands: binding.sample([-1, 1, -0.5]), idleTime: 1 });
  assert(parts.pose.rotation.x < 0);
  assert(parts.pose.rotation.y < 0);
  animateAgent(parts, { commands: binding.commands, enabled: false });
  assert.equal(parts.pose.rotation.x, 0);
  assert.equal(parts.pose.rotation.y, 0);
});

test("state updates apply static commands immediately even between crowd animation ticks", () => {
  const count = 17,
    scene = {
      bodies: Array.from({ length: count }, () => ({
        controlled: true,
        visual: { model: "rocket" },
      })),
    };
  const info = new Uint32Array(16);
  info[5] = 8 + 5 * count;
  const state = new Float32Array(8 + 6 * count),
    bits = new Uint32Array(state.buffer);
  for (let b = 0; b < count; b++) bits[info[5] + b] = 1;
  const layer = new BodyLayer(scene, info, new T.Group()),
    action = new Float32Array(count * 2);
  layer.update(state, action);
  layer.animate(1 / 30, 0);
  const flame = layer.animations[0].find(
    (p) => p.part.userData.motion === "thrust",
  );
  action[0] = 0.75;
  layer.update(state, action);
  assert.equal(layer.commands[0].thrust, 0.75);
  assert(flame.part.visible);
  const version = layer.presentationVersion;
  layer.animate(1 / 60, 0);
  assert.equal(layer.presentationVersion, version);
  layer.setAnimationsEnabled(false);
  assert(flame.part.visible);
  action[0] = 0;
  layer.update(state, action);
  assert.equal(flame.part.visible, false);
  assert.equal(layer.active[0], true);
});
