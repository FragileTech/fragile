import { resolveBodies } from "./agent-types.js";
// UI metadata only. Native compilation defines and validates the actual layout.
export function actionLayout(scene) {
  const channels = [];
  resolveBodies(scene).forEach((body, index) => {
    if (!body.controlled) return;
    const a = body.actuator || { kind: "vector" };
    const add = (name, low = -1, extra = {}) =>
      channels.push({ body: index, name, low, high: 1, ...extra });
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
    u[i] = Math.max(c.low, Math.min(c.high, value));
  });
  return u;
}
export function visualInput(channels, action, body) {
  let thrust = 0,
    steer = 0;
  channels.forEach((c, i) => {
    if (c.body !== body) return;
    const v = action?.[i] || 0;
    if (["torque", "steering"].includes(c.name)) steer = v;
    else if (c.name !== "brake") thrust = Math.max(thrust, Math.abs(v));
  });
  return { thrust, steer };
}
export const treePoseDim = (tree) => tree.poseDim ?? tree.dim;
export const treeWidth = (tree) => 3 + tree.dim + treePoseDim(tree);
