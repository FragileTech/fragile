const radians = Math.PI / 180;

export function presetOrbit(top, side) {
  return {
    yaw: 0,
    pitch: top ? Math.atan2(90, 0.001) : side ? 0 : Math.atan2(60, 45),
    distance: top ? Math.hypot(90, 0.001) : side ? 90 : 75,
  };
}

export function dragOrbit(orbit, dx, dy, side) {
  const yaw = orbit.yaw - dx * 0.008;
  return {
    ...orbit,
    yaw: Math.atan2(Math.sin(yaw), Math.cos(yaw)),
    pitch: Math.max(
      (side ? -80 : 5) * radians,
      Math.min((side ? 80 : 89.9) * radians, orbit.pitch + dy * 0.006),
    ),
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
