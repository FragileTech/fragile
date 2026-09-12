// Taylor coefficients store f^(n)/n!: all operations propagate exact truncated series.
export function jet(value, n = 3, variable = false) {
  const a = Array(n + 1).fill(0);
  a[0] = value;
  if (variable && n) a[1] = 1;
  return a;
}
export const add = (a, b) => a.map((v, i) => v + b[i]);
export const scale = (a, b) => a.map((v) => v * b);
export const mul = (a, b) =>
  a.map((_, n) => a.slice(0, n + 1).reduce((s, v, k) => s + v * b[n - k], 0));
export function inv(a) {
  const b = jet(1 / a[0], a.length - 1);
  for (let n = 1; n < a.length; n++)
    b[n] =
      -a.slice(1, n + 1).reduce((s, v, k) => s + v * b[n - k - 1], 0) / a[0];
  return b;
}
export function exp(a) {
  const b = jet(Math.exp(a[0]), a.length - 1);
  for (let n = 1; n < a.length; n++)
    for (let k = 1; k <= n; k++) b[n] += (k * a[k] * b[n - k]) / n;
  return b;
}
export function log(a) {
  const b = jet(Math.log(a[0]), a.length - 1),
    r = inv(a);
  for (let n = 1; n < a.length; n++)
    for (let k = 1; k <= n; k++) b[n] += (k * a[k] * r[n - k]) / n;
  return b;
}
export const pow = (a, p) => exp(scale(log(a), p));
export const div = (a, b) => mul(a, inv(b));
export const factorial = (n) => (n < 2 ? 1 : n * factorial(n - 1));
export function fitnessJet(
  x,
  {
    n = 3,
    N = 4,
    rho = 0.7,
    sigma = 0.15,
    delta = 0.1,
    assignment = null,
    frozenDenominator = false,
  } = {},
) {
  const c = (v) => jet(v, n),
    xs = Array.from({ length: N }, (_, i) =>
      i ? c(-1.2 + (2.4 * i) / (N - 1)) : jet(x, n, true),
    );
  const square = (a) => mul(a, a),
    difference = (a, b) => add(a, scale(b, -1));
  const raw = xs.map((y) =>
    exp(scale(square(difference(xs[0], y)), -0.5 / rho ** 2)),
  );
  let total = raw.reduce(add, c(0));
  if (frozenDenominator) total = c(total[0]);
  const w = raw.map((a) => div(a, total));
  const channel = (measurements) => {
    const mu = w.reduce((s, a, i) => add(s, mul(a, measurements[i])), c(0));
    const variance = w.reduce(
      (s, a, i) => add(s, mul(a, square(difference(measurements[i], mu)))),
      c(0),
    );
    const z = div(
      difference(measurements[0], mu),
      pow(add(variance, c(sigma * sigma)), 0.5),
    );
    return {
      mu,
      variance,
      z,
      value: add(scale(inv(add(c(1), exp(scale(z, -1)))), 2), c(1e-3)),
    };
  };
  const reward = channel(xs.map((y) => scale(square(y), -1)));
  const diversity = channel(
    xs.map((y, i) =>
      pow(
        add(
          square(difference(y, xs[assignment?.[i] ?? (i + 1) % N])),
          c(delta * delta),
        ),
        0.5,
      ),
    ),
  );
  return {
    fitness: mul(reward.value, diversity.value),
    mean: diversity.mu,
    variance: diversity.variance,
    score: diversity.z,
    weights: w,
  };
}
export const transpose = (a) => a[0].map((_, i) => a.map((r) => r[i]));
export const mm = (a, b) =>
  a.map((r) => b[0].map((_, j) => r.reduce((s, v, k) => s + v * b[k][j], 0)));
export const madd = (a, b) => a.map((r, i) => r.map((v, j) => v + b[i][j]));
export const mscale = (a, b) => a.map((r) => r.map((v) => v * b));
export const mv = (a, b) =>
  a.map((r) => r.reduce((s, v, i) => s + v * b[i], 0));
export const identity = (n) =>
  Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => +(i === j)),
  );
export function matrixExp(a, t = 1) {
  const norm = Math.max(
      ...a.map((r) => r.reduce((s, v) => s + Math.abs(v * t), 0)),
    ),
    squarings = Math.max(0, Math.ceil(Math.log2(norm || 1)));
  const b = mscale(a, t / 2 ** squarings);
  let out = identity(a.length),
    term = identity(a.length);
  for (let k = 1; k <= 28; k++) {
    term = mscale(mm(term, b), 1 / k);
    out = madd(out, term);
  }
  for (let k = 0; k < squarings; k++) out = mm(out, out);
  return out;
}
export const determinant = (a) => a[0][0] * a[1][1] - a[0][1] * a[1][0];
export const inverse = (a) =>
  mscale(
    [
      [a[1][1], -a[0][1]],
      [-a[1][0], a[0][0]],
    ],
    1 / determinant(a),
  );
export function gaussianKL(m, c, target) {
  const q = inverse(target),
    trace = mm(q, c)[0][0] + mm(q, c)[1][1];
  return Math.max(
    0,
    0.5 *
      (trace +
        m.reduce((s, v, i) => s + v * mv(q, m)[i], 0) -
        2 +
        Math.log(determinant(target) / determinant(c))),
  );
}
export function gaussianFisher(m, c, target) {
  const q = inverse(target),
    b = madd(q, mscale(inverse(c), -1)),
    qm = mv(q, m);
  return madd(
    mm(mm(b, c), transpose(b)),
    qm.map((x) => qm.map((y) => x * y)),
  );
}
export function harmonicState(
  t,
  { k = 1, gamma = 1, theta = 1, displacement = 2 } = {},
) {
  const target = [
      [theta / k, 0],
      [0, theta],
    ],
    a = matrixExp(
      [
        [0, 1],
        [-k, -gamma],
      ],
      t,
    );
  return { mean: mv(a, [displacement, 0]), covariance: target, target };
}
export function baoab(h, k = 1, gamma = 1, theta = 1) {
  const b = [
      [1, 0],
      [(-h * k) / 2, 1],
    ],
    a = [
      [1, h / 2],
      [0, 1],
    ],
    o = [
      [1, 0],
      [0, Math.exp(-gamma * h)],
    ];
  const M = mm(b, mm(a, mm(o, mm(a, b)))),
    z = mv(mm(b, a), [0, Math.sqrt(theta * -Math.expm1(-2 * gamma * h))]);
  return {
    M,
    Q: z.map((x) => z.map((y) => x * y)),
    stationary: [
      [theta / k, 0],
      [0, theta * (1 - (h * h * k) / 4)],
    ],
  };
}
export function gaussianStep(state, kernel) {
  return {
    mean: mv(kernel.M, state.mean),
    covariance: madd(
      mm(mm(kernel.M, state.covariance), transpose(kernel.M)),
      kernel.Q,
    ),
  };
}
export function hellinger({
  mass = 1,
  otherMass = 0.6,
  center = 1,
  width = 1,
  otherWidth = 0.8,
  atomic = false,
} = {}) {
  const affinity = atomic
    ? 0
    : Math.sqrt(
        (2 * width * otherWidth) / (width * width + otherWidth * otherWidth),
      ) *
      Math.exp(
        (-center * center) / (4 * (width * width + otherWidth * otherWidth)),
      );
  const shape = 2 - 2 * affinity,
    massTerm = (Math.sqrt(mass) - Math.sqrt(otherMass)) ** 2;
  return {
    shape,
    massTerm,
    shapeTerm: Math.sqrt(mass * otherMass) * shape,
    total: mass + otherMass - 2 * Math.sqrt(mass * otherMass) * affinity,
    w2: Math.hypot(center, width - otherWidth),
  };
}
export const kl = (p, q) =>
  Math.max(
    0,
    p.reduce((s, v, i) => s + (v > 0 ? v * Math.log(v / q[i]) : 0), 0),
  );
export const normalize = (a) => {
  const z = a.reduce((s, v) => s + v, 0);
  return a.map((v) => v / z);
};
export function enumeration(x, width = 0.7, law = "independent", rho = 0.7) {
  const xs = [x, -0.4, 0.4, 1.2],
    N = 4;
  const rows = xs.map((a, i) =>
    normalize(
      xs.map((b, j) =>
        i === j ? 0 : Math.exp(-((a - b) ** 2) / (2 * width ** 2)),
      ),
    ),
  );
  let assignments = [];
  if (law === "independent") {
    const visit = (c, p) => {
      if (c.length === N) {
        assignments.push({ c, p });
        return;
      }
      const i = c.length;
      for (let j = 0; j < N; j++) if (j !== i) visit([...c, j], p * rows[i][j]);
    };
    visit([], 1);
  } else {
    assignments = [
      [1, 0, 3, 2],
      [2, 3, 0, 1],
      [3, 2, 1, 0],
    ].map((c) => ({
      c,
      p:
        law === "matching"
          ? Math.exp(
              -xs.reduce((s, a, i) => s + (a - xs[c[i]]) ** 2, 0) /
                (4 * width ** 2),
            )
          : law === "shuffled_greedy"
            ? rows.reduce((s, row, i) => s + row[c[i]], 0) / N
            : rows[0][c[0]],
    }));
    const probs = normalize(assignments.map((a) => a.p));
    assignments.forEach((a, i) => (a.p = probs[i]));
  }
  // Use exactly the same fixed cloud and fitness formula in every assignment.
  const evaluate = (measurements) => {
    const weights = normalize(
      xs.map((y) => Math.exp(-((x - y) ** 2) / (2 * rho * rho))),
    );
    const channel = (m) => {
      const mu = weights.reduce((s, w, i) => s + w * m[i], 0),
        v = weights.reduce((s, w, i) => s + w * (m[i] - mu) ** 2, 0);
      return (
        0.001 + 2 / (1 + Math.exp(-(m[0] - mu) / Math.sqrt(v + 0.15 ** 2)))
      );
    };
    return channel(measurements) * channel(xs.map((y) => -y * y));
  };
  const expectedMeasurements = Array(N).fill(0);
  let expected = 0;
  for (const item of assignments) {
    item.measurements = xs.map((a, i) => Math.hypot(a - xs[item.c[i]], 0.1));
    item.value = evaluate(item.measurements);
    expected += item.p * item.value;
    item.measurements.forEach(
      (d, i) => (expectedMeasurements[i] += item.p * d),
    );
  }
  return {
    expected,
    substitute: evaluate(expectedMeasurements),
    frozen: evaluate(xs.map((a, i) => Math.hypot(a - xs[(i + 1) % N], 0.1))),
    assignments,
    rows,
  };
}
export function symmetricEigenvalues(matrix) {
  const a = matrix.map((r) => [...r]),
    n = a.length;
  for (let iter = 0; iter < 100 * n * n; iter++) {
    let p = 0,
      q = 1;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        if (Math.abs(a[i][j]) > Math.abs(a[p][q])) {
          p = i;
          q = j;
        }
    if (Math.abs(a[p][q]) < 1e-12) break;
    const angle = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]),
      c = Math.cos(angle),
      s = Math.sin(angle);
    const rot = identity(n);
    rot[p][p] = c;
    rot[q][q] = c;
    rot[p][q] = s;
    rot[q][p] = -s;
    const next = mm(transpose(rot), mm(a, rot));
    for (let i = 0; i < n; i++) a[i] = next[i];
  }
  return a.map((r, i) => r[i]).sort((a, b) => a - b);
}
export function constraintWidth(diameter, target) {
  return diameter / Math.sqrt(2 * Math.log(1 / target));
}

export function hypocoerciveCoefficients({ k = 1, gamma = 1, theta = 1 } = {}) {
  const D = gamma * theta,
    C = Math.max(theta, theta / k),
    eta = D / (2 * (1 + 2 * k + (2 * k + gamma + 2) ** 2));
  return {
    eta,
    rate: eta / (C / 2 + 3 * eta),
    C,
    D,
    G: [
      [2 * eta, eta],
      [eta, 2 * eta],
    ],
  };
}
export function spectralGapDiagnostic(K) {
  const degrees = K.map((row) => row.reduce((s, v) => s + v, 0));
  const S = K.map((row, i) =>
    row.map((v, j) => +(i === j) - v / Math.sqrt(degrees[i] * degrees[j])),
  );
  const values = symmetricEigenvalues(S),
    half = Math.floor(K.length / 2);
  const left = degrees.slice(0, half).reduce((a, b) => a + b),
    right = degrees.slice(half).reduce((a, b) => a + b);
  const cut = K.slice(0, half).reduce(
    (s, row) => s + row.slice(half).reduce((a, b) => a + b),
    0,
  );
  const upperBound = cut * (1 / left + 1 / right),
    resolution = 1e-11;
  const resolved =
    values[1] > resolution && values[1] <= upperBound + resolution;
  return { value: resolved ? values[1] : null, upperBound, resolution, values };
}
export function dirichletKernel(x, y, h) {
  if (x <= 0 || x >= 1) return 0;
  const phi = (z) =>
    Math.exp((-z * z) / (2 * h * h)) / (Math.sqrt(2 * Math.PI) * h);
  let sum = 0;
  for (let k = -3; k <= 3; k++) sum += phi(x - y + 2 * k) - phi(x + y + 2 * k);
  return Math.max(0, sum);
}
export function cosineGaussianVariance(mean, variance) {
  const expectation = Math.exp(-variance / 2) * Math.cos(mean);
  return Math.max(
    0,
    (1 + Math.exp(-2 * variance) * Math.cos(2 * mean)) / 2 - expectation ** 2,
  );
}
