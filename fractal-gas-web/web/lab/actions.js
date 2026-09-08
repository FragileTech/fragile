import { resolveBodies } from "./agent-types.js";
// UI metadata only. Native compilation defines and validates the actual layout.
export function actionLayout(scene) {
  const channels = [];
  resolveBodies(scene).forEach((body, index) => {
    if (!body.controlled) return;
    const a = body.actuator || { kind: "vector" };
    const add = (name, low = -1, extra = {}) => {
      const configured = Number(a.action_multipliers?.[name]);
      const multiplier = Number.isFinite(configured)
        ? Math.max(0, Math.min(10, configured))
        : 1;
      channels.push({
        body: index,
        name,
        low: low * multiplier || 0,
        high: multiplier,
        ...extra,
      });
    };
    switch (a.kind || "vector") {
      case "vector":
        add("thrust", 0);
        add("torque");
        break;
      case "kart":
        add("throttle");
        add("steering");
        add("brake", 0);
        break;
      case "holonomic":
        add("force_x");
        add("force_y");
        add("torque");
        break;
      case "thrusters":
        a.thrusters.forEach((t, i) =>
          add(`thruster_${i}`, t.reversible ? -1 : 0, { thruster: t }),
        );
        break;
      default:
        throw new Error(`Unknown actuator kind: ${a.kind}`);
    }
  });
  return channels;
}
export function manualAction(channels, keys, body = channels[0]?.body) {
  const u = Float32Array.from(channels, (c) =>
      Math.max(c.low, Math.min(c.high, 0)),
    ),
    forward = Number(keys.has("w")) - Number(keys.has("s")),
    turn = Number(keys.has("a")) - Number(keys.has("d")),
    side = Number(keys.has("q")) - Number(keys.has("e"));
  channels.forEach((c, i) => {
    if (c.body !== body) return;
    let value = 0;
    if (["thrust", "throttle", "force_x"].includes(c.name)) value = forward;
    else if (c.name === "force_y") value = side;
    else if (["torque", "steering"].includes(c.name)) value = turn;
    else if (c.name === "brake") value = Number(keys.has(" "));
    else if (c.thruster) {
      const [x, y] = c.thruster.direction || [1, 0],
        [px, py] = c.thruster.position || [0, 0];
      value = forward * x + side * y + turn * (px * y - py * x);
    }
    const magnitude = Math.max(Math.abs(c.low), Math.abs(c.high));
    u[i] = Math.max(c.low, Math.min(c.high, value * magnitude));
  });
  return u;
}
// Normalize each half of an asymmetric channel independently. Missing, disabled,
// non-finite and zero-width controls cannot produce a visible command.
export function normalizedAction(channel, value) {
  if (
    !channel ||
    channel.enabled === false ||
    channel.disabled ||
    !Number.isFinite(value)
  )
    return 0;
  const { low, high } = channel;
  if (
    !Number.isFinite(low) ||
    !Number.isFinite(high) ||
    low > 0 ||
    high < 0 ||
    low > high
  )
    return 0;
  if (value > 0) return high > 0 ? Math.min(1, value / high) : 0;
  if (value < 0) return low < 0 ? Math.max(-1, value / -low) : 0;
  return 0;
}

const commandFields = {
  thrust: "thrust",
  throttle: "throttle",
  steering: "steering",
  brake: "brake",
  force_x: "forceX",
  force_y: "forceY",
  torque: "torque",
};
function actionKind(channels, bodyDef, bodyIndex) {
  if (bodyDef.actuator?.kind) return bodyDef.actuator.kind;
  for (const channel of channels)
    if (channel?.body === bodyIndex && typeof channel.name === "string") {
      if (channel.name.startsWith("thruster_")) return "thrusters";
      if (channel.name === "throttle" || channel.name === "steering")
        return "kart";
      if (channel.name === "force_x" || channel.name === "force_y")
        return "holonomic";
    }
  return "vector";
}
export function createActionBinding(channels, bodyDef = {}, bodyIndex = 0) {
  const commands = {
    kind: actionKind(channels, bodyDef, bodyIndex),
    thrust: 0,
    throttle: 0,
    steering: 0,
    brake: 0,
    forceX: 0,
    forceY: 0,
    torque: 0,
    thrusters: [],
  };
  const entries = [];
  const vector = (value, fallback) =>
    Array.isArray(value) &&
    value.length >= 2 &&
    value.slice(0, 2).every(Number.isFinite)
      ? value.slice(0, 2)
      : [...fallback];
  const thrusterEntry = (spec = {}) => ({
    value: 0,
    position: vector(spec.position, [0, 0]),
    direction: vector(spec.direction, [1, 0]),
  });
  for (const spec of bodyDef.actuator?.thrusters || [])
    commands.thrusters.push(thrusterEntry(spec));
  channels.forEach((channel, index) => {
    if (channel?.body !== bodyIndex || typeof channel.name !== "string") return;
    const match = /^thruster_(\d+)$/.exec(channel.name);
    if (match) {
      const at = Number(match[1]);
      if (!Number.isSafeInteger(at)) return;
      while (commands.thrusters.length <= at)
        commands.thrusters.push(thrusterEntry());
      if (channel.thruster)
        commands.thrusters[at] = thrusterEntry(channel.thruster);
      entries.push({
        channel,
        index,
        target: commands.thrusters[at],
        field: "value",
      });
    } else if (commandFields[channel.name])
      entries.push({
        channel,
        index,
        target: commands,
        field: commandFields[channel.name],
      });
  });
  return {
    commands,
    sample(action) {
      commands.thrust =
        commands.throttle =
        commands.steering =
        commands.brake =
        commands.forceX =
        commands.forceY =
        commands.torque =
          0;
      for (const thruster of commands.thrusters) thruster.value = 0;
      if (
        bodyDef.controlled === false ||
        bodyDef.enabled === false ||
        bodyDef.disabled
      )
        return commands;
      for (const entry of entries)
        entry.target[entry.field] = normalizedAction(
          entry.channel,
          action?.[entry.index],
        );
      return commands;
    },
  };
}

// Legacy scenery callers receive normalized forward propulsion and turn input.
// Lateral force, braking and independent jets must not masquerade as main thrust.
export function visualInput(channels, action, body) {
  let thrust = 0,
    steer = 0;
  for (let i = 0; i < channels.length; i++) {
    const c = channels[i];
    if (c.body !== body) continue;
    const v = normalizedAction(c, action?.[i]);
    if (c.name === "torque" || c.name === "steering") steer = v;
    else if (
      c.name === "thrust" ||
      c.name === "throttle" ||
      c.name === "force_x"
    )
      thrust = Math.max(0, v);
  }
  return { thrust, steer };
}
export const treePoseDim = (tree) => tree.poseDim ?? tree.dim;
export const treeWidth = (tree) => 3 + tree.dim + treePoseDim(tree);
