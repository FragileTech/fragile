const radians = Math.PI / 180;

export function presetOrbit(top, side) {
  return {
    yaw: 0,
    pitch: top ? Math.atan2(90, 0.001) : side ? 0 : Math.atan2(60, 45),
    distance: top ? Math.hypot(90, 0.001) : side ? 90 : 75,
  };
}

// The picking plane must stay at least 5 degrees off the view direction, so
// every camera control shares these limits.
export function pitchLimits(side) {
  return [(side ? -80 : 5) * radians, (side ? 80 : 89.9) * radians];
}

const clampPitch = (pitch, side) => {
  const [low, high] = pitchLimits(side);
  return Math.max(low, Math.min(high, pitch));
};

export function dragOrbit(orbit, dx, dy, side) {
  const yaw = orbit.yaw - dx * 0.008;
  return {
    ...orbit,
    yaw: Math.atan2(Math.sin(yaw), Math.cos(yaw)),
    pitch: clampPitch(orbit.pitch + dy * 0.006, side),
  };
}

// Retreat only when necessary to keep the arena and raised scenery ahead of
// the near plane. Orthographic magnification does not depend on this distance.
export function orbitPosition(orbit, arena, center, side) {
  const direction = [
    Math.cos(orbit.pitch) * Math.sin(orbit.yaw),
    -Math.cos(orbit.pitch) * Math.cos(orbit.yaw),
    Math.sin(orbit.pitch),
  ];
  let minDepth = Infinity,
    maxDepth = -Infinity;
  for (const x of [arena.minX, arena.maxX]) {
    for (const y of [arena.minY, arena.maxY]) {
      const depth =
        (x - center[0]) * direction[0] +
        (y - center[1]) * direction[side ? 2 : 1];
      minDepth = Math.min(minDepth, depth);
      maxDepth = Math.max(maxDepth, depth);
    }
  }
  const padding = 16;
  const distance = Math.max(orbit.distance, maxDepth + padding + 1);
  const target = [center[0], side ? 0 : center[1], side ? center[1] : 0];
  return {
    target,
    position: target.map((value, i) => value + distance * direction[i]),
    far: Math.max(300, distance - minDepth + padding + 1),
  };
}

// Screen basis of the orbit camera in world coordinates.
export function orbitBasis(orbit) {
  const cp = Math.cos(orbit.pitch),
    sp = Math.sin(orbit.pitch),
    cy = Math.cos(orbit.yaw),
    sy = Math.sin(orbit.yaw);
  return {
    toCamera: [cp * sy, -cp * cy, sp],
    right: [cy, sy, 0],
    up: [-sp * sy, sp * cy, cp],
  };
}

// Flight views stand the simulation plane upright: sim (x, y, z) is drawn at
// world (x, -z, y).
export function simAxisWorld(axis, sign, side) {
  const v = [0, 0, 0];
  v["xyz".indexOf(axis)] = sign;
  return side ? [v[0], -v[2], v[1]] : v;
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// The closest permitted view looking down a simulation axis.
export function snapOrbit(orbit, axis, sign, side) {
  const w = simAxisWorld(axis, sign, side);
  return {
    ...orbit,
    yaw: Math.hypot(w[0], w[1]) > 1e-9 ? Math.atan2(w[0], -w[1]) : 0,
    pitch: clampPitch(Math.asin(w[2]), side),
  };
}

// Axis handles of the navigation gizmo: screen x/y in [-1, 1] (y up), depth
// toward the viewer, and whether the pitch limits allow snapping to the axis.
export function gizmoAxes(orbit, side) {
  const basis = orbitBasis(orbit);
  const axes = [];
  for (const axis of "xyz")
    for (const sign of [1, -1]) {
      const w = simAxisWorld(axis, sign, side);
      const snapped = orbitBasis(snapOrbit(orbit, axis, sign, side)).toCamera;
      axes.push({
        axis,
        sign,
        x: dot(w, basis.right),
        y: dot(w, basis.up),
        depth: dot(w, basis.toCamera),
        enabled: dot(w, snapped) > Math.SQRT1_2,
      });
    }
  return axes;
}

export function lerpOrbit(a, b, t) {
  const turn = Math.atan2(Math.sin(b.yaw - a.yaw), Math.cos(b.yaw - a.yaw));
  const yaw = a.yaw + turn * t;
  return {
    ...b,
    yaw: Math.atan2(Math.sin(yaw), Math.cos(yaw)),
    pitch: a.pitch + (b.pitch - a.pitch) * t,
  };
}
