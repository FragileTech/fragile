// Deterministic geometry-following driver for physical circuit acceptance tests.
// Uses real actions and integration; never edits or teleports the engine state.
export function kartDriver(scene) {
  const points = scene.environment.centerline;
  let distance = 0;
  const segments = points.map((a, i) => {
    const b = points[(i + 1) % points.length];
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const segment = { a, b, length, start: distance };
    distance += length;
    return segment;
  });
  const sample = (s) => {
    s = ((s % distance) + distance) % distance;
    const segment =
      segments.find((p) => p.start + p.length >= s) || segments.at(-1);
    const t = (s - segment.start) / segment.length;
    return segment.a.map((v, i) => v + (segment.b[i] - v) * t);
  };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  return (engine) => {
    const state = engine.states();
    const x = state[8],
      y = state[9],
      angle = state[12];
    const speed = Math.hypot(state[10], state[11]);
    let closest = Infinity,
      along = 0;
    for (const { a, b, length, start } of segments) {
      const t = clamp(
        ((x - a[0]) * (b[0] - a[0]) + (y - a[1]) * (b[1] - a[1])) / length ** 2,
        0,
        1,
      );
      const d =
        (x - a[0] - t * (b[0] - a[0])) ** 2 +
        (y - a[1] - t * (b[1] - a[1])) ** 2;
      if (d < closest) {
        closest = d;
        along = start + length * t;
      }
    }
    const lookahead = 1.3 + speed * 0.18;
    const target = sample(along + lookahead);
    const dx = target[0] - x,
      dy = target[1] - y;
    const heading = Math.atan2(dy, dx) - angle;
    const error = Math.atan2(Math.sin(heading), Math.cos(heading));
    const curvature = (2 * Math.sin(error)) / Math.max(0.5, Math.hypot(dx, dy));
    const steering = clamp(
      Math.atan(1.1 * curvature * (1 + 5 / 12)) / 0.55,
      -1,
      1,
    );
    const desiredSpeed = clamp(5 / (1 + Math.abs(curvature) * 10), 1.2, 5);
    const throttle = clamp(
      (desiredSpeed - speed) * 0.5 + (desiredSpeed * 0.65) / 12,
      0,
      1,
    );
    const brake = clamp((speed - desiredSpeed) * 0.15, 0, 1);
    return new Float32Array([throttle, steering, brake]);
  };
}
