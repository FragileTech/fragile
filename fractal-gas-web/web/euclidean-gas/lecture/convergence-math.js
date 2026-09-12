// Numerical kernels used by the convergence lecture experiments.
export const avg = (a) => a.reduce((s, x) => s + x, 0) / (a.length || 1);
export const varOf = (a) => {
  const m = avg(a);
  return avg(a.map((x) => (x - m) ** 2));
};
export const norm2 = (a, b) => a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0);
export const center = (a) => a[0].map((_, j) => avg(a.map((x) => x[j])));
export const centered = (a) => {
  const m = center(a);
  return a.map((x) => x.map((v, j) => v - m[j]));
};
export const spread = (a) =>
  avg(
    centered(a).map((x) =>
      norm2(
        x,
        x.map(() => 0),
      ),
    ),
  );
export function random(seed) {
  let s = seed >>> 0;
  const r = () => {
    s += 0x6d2b79f5;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  r.normal = () =>
    Math.sqrt(-2 * Math.log(Math.max(r(), 1e-15))) *
    Math.cos(2 * Math.PI * r());
  return r;
}
export function assignment(a, b) {
  // Hungarian algorithm, O(N³), exact equal-mass discrete transport objective.
  const n = a.length,
    costs = a.map((x) => b.map((y) => norm2(x, y)));
  const u = Array(n + 1).fill(0),
    v = [...u],
    p = [...u],
    way = [...u];
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const min = Array(n + 1).fill(Infinity),
      used = Array(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity,
        j1 = 0;
      for (let j = 1; j <= n; j++)
        if (!used[j]) {
          const c = costs[i0 - 1][j - 1] - u[i0] - v[j];
          if (c < min[j]) {
            min[j] = c;
            way[j] = j0;
          }
          if (min[j] < delta) {
            delta = min[j];
            j1 = j;
          }
        }
      for (let j = 0; j <= n; j++)
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else min[j] -= delta;
      j0 = j1;
    } while (p[j0]);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  const pairs = Array(n);
  for (let j = 1; j <= n; j++) pairs[p[j] - 1] = j - 1;
  return { cost: avg(pairs.map((j, i) => costs[i][j])), pairs, costs };
}
export function transport(a, b) {
  const ac = centered(a),
    bc = centered(b),
    opt = assignment(a, b),
    co = assignment(ac, bc);
  return {
    ...opt,
    centered: co.cost,
    barycenter: norm2(center(a), center(b)),
    proxy: spread(a) + spread(b),
    label: avg(a.map((x, i) => norm2(x, b[i]))),
  };
}
export const mv = (a, x) =>
  a.map((row) => row.reduce((s, v, j) => s + v * x[j], 0));
export const mm = (a, b) =>
  a.map((row) =>
    b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)),
  );
export const transpose = (a) => a[0].map((_, i) => a.map((r) => r[i]));
export function perron(a) {
  let w = Array(a.length).fill(1);
  for (let k = 0; k < 1000; k++) {
    let z = mv(transpose(a), w),
      m = Math.max(...z);
    if (!m) return { weights: w, r: 0 };
    z = z.map((x) => x / m);
    if (Math.max(...z.map((x, i) => Math.abs(x - w[i]))) < 1e-13) {
      w = z;
      break;
    }
    w = z;
  }
  const ratios = mv(transpose(a), w).map((x, i) => x / w[i]);
  return { weights: w, r: Math.max(...ratios) };
}
export function killedKernel(kill = 0.1, conservative = false) {
  return [
    [0.65, 0.25, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.3, 0.6],
  ].map((r, i) =>
    r.map((x) => x * (conservative ? 1 : 1 - kill * [1.5, 0.4, 1][i])),
  );
}
export function qsd(kernel) {
  const p = perron(kernel),
    sum = p.weights.reduce((a, b) => a + b, 0);
  return { law: p.weights.map((x) => x / sum), alpha: p.r };
}
export function reservoir(t, kill, revive, initial) {
  const total = kill + revive,
    eq = revive / total;
  return total ? eq + (initial - eq) * Math.exp(-total * t) : initial;
}
export function collision(n, k) {
  let p = 1;
  for (let j = 0; j < k; j++) p *= 1 - j / n;
  return 1 - p;
}
export function geometricPartition(points, epsilon = 0.3) {
  let clusters = points.map((_, i) => [i]);
  while (true) {
    let best = Infinity,
      pair = null;
    for (let i = 0; i < clusters.length; i++)
      for (let j = i + 1; j < clusters.length; j++) {
        const d = Math.max(
          ...clusters[i].flatMap((a) =>
            clusters[j].map((b) => Math.sqrt(norm2(points[a], points[b]))),
          ),
        );
        if (d < best) {
          best = d;
          pair = [i, j];
        }
      }
    if (!pair || best > 2 * epsilon) break;
    const [i, j] = pair;
    clusters[i].push(...clusters[j]);
    clusters.splice(j, 1);
  }
  const mu = center(points),
    minimum = Math.max(5, Math.ceil(0.05 * points.length));
  const valid = clusters
    .filter((c) => c.length >= minimum)
    .map((c) => ({
      c,
      contribution: c.length * norm2(center(c.map((i) => points[i])), mu),
    }))
    .sort((a, b) => b.contribution - a.contribution);
  const high = new Set(clusters.filter((c) => c.length < minimum).flat()),
    total = valid.reduce((s, c) => s + c.contribution, 0);
  let sum = 0;
  for (const g of valid) {
    if (sum >= 0.9 * total) break;
    g.c.forEach((i) => high.add(i));
    sum += g.contribution;
  }
  return { high, clusters, minimum };
}
export function histogram(a, lo, hi, bins = 24) {
  const h = Array(bins).fill(0),
    dx = (hi - lo) / bins;
  for (const x of a) {
    const k = Math.floor((x - lo) / dx);
    if (k >= 0 && k < bins) h[k]++;
  }
  return h.map((v, i) => [lo + (i + 0.5) * dx, v / Math.max(1, a.length) / dx]);
}
export const grid = (a, b, n = 81) =>
  Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
export const gaussian = (x, mu = 0, sigma = 1) =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
export function linearBAOAB(h, gamma, temp, stiffness) {
  const a = Math.exp(-gamma * h),
    s = Math.sqrt(temp * (1 - a * a));
  const B = [
      [1, 0],
      [(-stiffness * h) / 2, 1],
    ],
    A = [
      [1, h / 2],
      [0, 1],
    ],
    O = [
      [1, 0],
      [0, a],
    ],
    M = mm(B, mm(A, mm(O, mm(A, B))));
  const g = mv(mm(B, A), [0, s]);
  return { M, g };
}
export function covarianceStep(c, M, g) {
  const z = mm(M, mm(c, transpose(M)));
  return z.map((r, i) => r.map((v, j) => v + g[i] * g[j]));
}
