export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export const mean = (xs) =>
  xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0;
export const variance = (xs) => {
  const m = mean(xs);
  return mean(xs.map((x) => (x - m) ** 2));
};
export const linspace = (a, b, n = 101) =>
  Array.from({ length: n }, (_, i) => a + ((b - a) * i) / Math.max(1, n - 1));
export function rng(seed = 7) {
  let state = Number(seed) >>> 0;
  const random = () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  random.normal = () =>
    Math.sqrt(-2 * Math.log(Math.max(Number.MIN_VALUE, random()))) *
    Math.cos(2 * Math.PI * random());
  return random;
}
export function histogram(xs, lo, hi, bins = 32) {
  if (!(hi > lo) || bins < 1)
    throw new Error("Histogram needs an ordered domain and bins.");
  const counts = Array(bins).fill(0),
    width = (hi - lo) / bins;
  for (const x of xs) {
    if (Number.isFinite(x) && x >= lo && x <= hi)
      counts[Math.min(bins - 1, Math.floor((x - lo) / width))]++;
  }
  return counts.map((n, i) => [
    lo + (i + 0.5) * width,
    n / (Math.max(1, xs.length) * width),
  ]);
}
export const normalPDF = (x, mu = 0, sigma = 1) =>
  sigma > 0
    ? Math.exp(-0.5 * ((x - mu) / sigma) ** 2) /
      (sigma * Math.sqrt(2 * Math.PI))
    : 0;
export function covariance2(points) {
  if (!points.length)
    return [
      [0, 0],
      [0, 0],
    ];
  const x = mean(points.map((p) => p[0])),
    y = mean(points.map((p) => p[1]));
  const xx = mean(points.map((p) => (p[0] - x) ** 2));
  const yy = mean(points.map((p) => (p[1] - y) ** 2));
  const xy = mean(points.map((p) => (p[0] - x) * (p[1] - y)));
  return [
    [xx, xy],
    [xy, yy],
  ];
}
export const line = (name, points, options = {}) => ({
  name,
  points,
  style: "line",
  ...options,
});
export const scatter = (name, points, options = {}) => ({
  name,
  points,
  style: "points",
  ...options,
});
function rows(frame, key) {
  const field = frame?.population?.observations?.fields?.[key];
  if (!field) return [];
  const width = field.item_shape.reduce((a, b) => a * b, 1);
  return Array.from({ length: field.values.length / width }, (_, i) =>
    Array.from(field.values.slice(i * width, (i + 1) * width)),
  );
}
export const positions = (frame) => rows(frame, "positions");
export const velocities = (frame) => rows(frame, "velocities");
export function pushBounded(array, value, limit = 400) {
  array.push(value);
  if (array.length > limit) array.splice(0, array.length - limit);
  return array;
}
