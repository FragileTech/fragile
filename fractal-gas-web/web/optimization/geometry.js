// Pure geometry routines shared by rendering, recording validation, and tests.
export const geometryColors = {
  cmaes_active: 0xffc56e,
  cmaes_bipop: 0xffc56e,
  local_covariance: 0x6ee7ff,
  adaptive_fractal: 0xbd98ff,
  cloning_guided: 0xff8dab,
  gaussian: 0x9ee7ba,
  uniform: 0x9ee7ba,
  gas_adaptive: 0x9ee7ba,
};
export const geometryNames = {
  cmaes_active: "CMA-ES",
  cmaes_bipop: "BIPOP CMA-ES",
  local_covariance: "Local Gaussian",
  adaptive_fractal: "Adaptive Fractal",
  cloning_guided: "Clone-guided",
  gaussian: "Gaussian reference",
  uniform: "Uniform reference",
  gas_adaptive: "GAS adaptive reference",
};
const finiteArray = (v, n) =>
  Array.isArray(v) && v.length === n && v.every(Number.isFinite);
export function validateGeometry(g, dimensions) {
  if (g == null) return;
  if (
    g.version !== 1 ||
    g.dimensions !== dimensions ||
    !Array.isArray(g.methods) ||
    g.methods.length > 8 ||
    !Array.isArray(g.events) ||
    g.events.length > 512
  )
    throw new Error("Invalid geometry diagnostics");
  let coefficients = 0;
  for (const method of g.methods) {
    if (
      !Object.hasOwn(geometryNames, method.id) ||
      !Array.isArray(method.models) ||
      method.models.length > 32
    )
      throw new Error("Invalid covariance method");
    for (const model of method.models) {
      const c = model.columns;
      if (
        !finiteArray(model.anchor, dimensions) ||
        !Number.isFinite(model.scale) ||
        model.scale < 0 ||
        !Number.isInteger(c) ||
        c < 1 ||
        c > dimensions + 1 ||
        !finiteArray(model.shape, dimensions * c) ||
        !["dense", "diagonal", "diagonal_low_rank"].includes(
          model.representation,
        ) ||
        (model.representation === "dense" && c !== dimensions) ||
        (model.representation === "diagonal" && c !== 1) ||
        (model.transform != null && model.transform !== "cma")
      )
        throw new Error("Invalid covariance model");
      for (const key of ["drift", "field"])
        if (model[key] != null && !finiteArray(model[key], dimensions))
          throw new Error("Invalid learned vector");
      coefficients += model.shape.length;
    }
  }
  if (coefficients > 2000000)
    throw new Error("Geometry diagnostics exceed the coefficient limit");
  for (const event of g.events)
    if (
      !finiteArray(event.origin, dimensions) ||
      !finiteArray(event.destination, dimensions) ||
      !["proposal", "cloning", "kinetic", "execution"].includes(event.kind)
    )
      throw new Error("Invalid jump vector");
}
function entry(model, i, j) {
  const c = model.columns,
    s = model.shape;
  if (model.representation === "dense")
    return (s[i * c + j] + s[j * c + i]) / 2;
  let value = i === j ? s[i * c] : 0;
  if (model.representation === "diagonal_low_rank")
    for (let k = 1; k < c; k++) value += s[i * c + k] * s[j * c + k];
  return value;
}
export function projectedCovariance(model, axes, normalized = false) {
  const d = model.anchor.length;
  let trace = 0;
  for (let i = 0; i < d; i++) trace += entry(model, i, i);
  const scale = normalized ? (trace > 0 ? d / trace : 0) : model.scale ** 2;
  return axes.map((i) =>
    axes.map((j) => (i < d && j < d ? entry(model, i, j) * scale : 0)),
  );
}
// Symmetric Jacobi decomposition; repeated eigenvalues and rank-deficient
// projections require no orientation conventions or divisions by eigenvalue gaps.
export function covarianceFactor(matrix) {
  const n = matrix.length,
    a = matrix.map((r) => r.slice()),
    v = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => +(i === j)),
    );
  const norm = Math.max(...a.flat().map(Math.abs), Number.MIN_VALUE);
  if (!a.flat().every(Number.isFinite))
    throw new Error("Nonfinite covariance projection");
  for (let iteration = 0; iteration < 40; iteration++) {
    let p = 0,
      q = 1;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        if (Math.abs(a[i][j]) > Math.abs(a[p][q])) {
          p = i;
          q = j;
        }
    if (Math.abs(a[p][q]) <= norm * 1e-14) break;
    const theta = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]),
      c = Math.cos(theta),
      s = Math.sin(theta);
    const pp = a[p][p],
      qq = a[q][q],
      pq = a[p][q];
    a[p][p] = c * c * pp - 2 * s * c * pq + s * s * qq;
    a[q][q] = s * s * pp + 2 * s * c * pq + c * c * qq;
    a[p][q] = a[q][p] = 0;
    for (let k = 0; k < n; k++) {
      if (k !== p && k !== q) {
        const kp = a[k][p],
          kq = a[k][q];
        a[k][p] = a[p][k] = c * kp - s * kq;
        a[k][q] = a[q][k] = s * kp + c * kq;
      }
      const kp = v[k][p],
        kq = v[k][q];
      v[k][p] = c * kp - s * kq;
      v[k][q] = s * kp + c * kq;
    }
  }
  if (a.some((r, i) => r[i] < -norm * 1e-10))
    throw new Error("Covariance is not positive semidefinite");
  return v.map((r) => r.map((x, j) => x * Math.sqrt(Math.max(0, a[j][j]))));
}
export function cmaBoundary(x, low, high) {
  const half = (high - low) / 2,
    al = Math.min(half, (1 + Math.abs(low)) / 20),
    au = Math.min(half, (1 + Math.abs(high)) / 20);
  const bottom = low - 2 * al - half,
    top = high + 2 * au + half,
    period = 2 * (high - low + al + au);
  if (x < bottom) x += period * (1 + Math.trunc((bottom - x) / period));
  if (x > top) x -= period * (1 + Math.trunc((x - top) / period));
  if (x < low - al) x += 2 * (low - al - x);
  if (x > high + au) x -= 2 * (x - high - au);
  if (x < low + al) x = low + (x - (low - al)) ** 2 / (4 * al);
  else if (x > high - au) x = high - (x - (high + au)) ** 2 / (4 * au);
  return x;
}
export function covarianceSegments(model, axes, normalized, low, high) {
  const dimension = axes.length,
    factor = covarianceFactor(projectedCovariance(model, axes, normalized));
  const point = (u) =>
    axes.map((axis, i) => {
      let x =
        model.anchor[axis] + factor[i].reduce((sum, f, j) => sum + f * u[j], 0);
      if (model.transform === "cma") x = cmaBoundary(x, low, high);
      return x;
    });
  const rings = [];
  if (dimension === 2) rings.push((t) => [Math.cos(t), Math.sin(t)]);
  else {
    // Three great circles plus latitude/longitude rings make depth legible.
    for (let k = 0; k < 6; k++) {
      const angle = (k * Math.PI) / 6;
      rings.push((t) => [
        Math.cos(t) * Math.cos(angle),
        Math.cos(t) * Math.sin(angle),
        Math.sin(t),
      ]);
    }
    for (const z of [-0.75, -0.4, 0, 0.4, 0.75])
      rings.push((t) => [
        Math.sqrt(1 - z * z) * Math.cos(t),
        Math.sqrt(1 - z * z) * Math.sin(t),
        z,
      ]);
  }
  const segments = [];
  for (const ring of rings)
    for (let k = 0; k < 64; k++)
      segments.push([
        point(ring((k * 2 * Math.PI) / 64)),
        point(ring(((k + 1) * 2 * Math.PI) / 64)),
      ]);
  return segments;
}
export function periodicSegments(origin, destination, low, high, periodic) {
  if (!periodic) return [[origin, destination]];
  const width = high - low,
    wrap = (x) => low + ((((x - low) % width) + width) % width);
  const start = origin.map(wrap),
    delta = destination.map((x, i) => {
      let v = (x - origin[i]) % width;
      if (v > width / 2) v -= width;
      if (v < -width / 2) v += width;
      return v;
    });
  const times = [0, 1];
  for (let i = 0; i < start.length; i++) {
    const t =
      delta[i] > 0
        ? (high - start[i]) / delta[i]
        : delta[i] < 0
          ? (low - start[i]) / delta[i]
          : Infinity;
    if (t > 0 && t < 1) times.push(t);
  }
  times.sort((a, b) => a - b);
  const segments = [];
  for (let k = 1; k < times.length; k++) {
    const a = times[k - 1],
      b = times[k],
      mid = (a + b) / 2;
    if (b - a < 1e-14) continue;
    const shift = start.map(
      (x, i) => Math.floor((x + delta[i] * mid - low) / width) * width,
    );
    segments.push([
      start.map((x, i) => x + delta[i] * a - shift[i]),
      start.map((x, i) => x + delta[i] * b - shift[i]),
    ]);
  }
  return segments;
}
