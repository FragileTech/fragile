// Davidson likelihood and deterministic Laplace inference, shared by browser and CLI.
export const RANKING_CRITERIA = [
  "correctness",
  "relevance",
  "completeness",
  "clarity",
  "overall",
];
export const MODEL_VERSION = "davidson-1";
export const PRIORS = Object.freeze({ strength: 2, log_tie: 1.5, position: 1 });
export const ELO_FACTOR = 400 / Math.LN10;
export function random(seed = 7) {
  let x = 2166136261;
  for (const c of String(seed)) x = Math.imul(x ^ c.charCodeAt(0), 16777619);
  return () => {
    x += 0x6d2b79f5;
    let t = Math.imul(x ^ (x >>> 15), 1 | x);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function shuffle(items, rng) {
  const a = [...items];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
export function quantile(values, q) {
  if (!values.length) return null;
  const v = [...values].sort((a, b) => a - b),
    p = (v.length - 1) * q,
    lo = Math.floor(p);
  return v[lo] + (v[Math.ceil(p)] - v[lo]) * (p - lo);
}
const zeros = (n) => Array.from({ length: n }, () => new Float64Array(n));
export function probabilities(a, b, logTie = 0, position = 0) {
  const z = [a + position / 2, b - position / 2, logTie + (a + b) / 2],
    m = Math.max(...z);
  const p = z.map((v) => Math.exp(v - m)),
    total = p.reduce((s, v) => s + v, 0);
  return p.map((v) => v / total);
}
const FEATURES = [
  [1, 0, 0, 0.5],
  [0, 1, 0, -0.5],
  [0.5, 0.5, 1, 0],
];
// Outcomes are indexed A, B, tie. Each paired presentation has weight one half.
export function objective(
  theta,
  observations,
  n,
  priors = PRIORS,
  hessian = true,
) {
  const d = n + 2,
    gradient = new Float64Array(d),
    H = hessian ? zeros(d) : null;
  let value = 0;
  for (let i = 0; i < d; i++) {
    const sd =
        i < n ? priors.strength : i === n ? priors.log_tie : priors.position,
      precision = 1 / (sd * sd);
    value += 0.5 * theta[i] * theta[i] * precision;
    gradient[i] = theta[i] * precision;
    if (H) H[i][i] = precision;
  }
  for (const o of observations) {
    const ids = [o.i, o.j, n, n + 1],
      p = probabilities(theta[o.i], theta[o.j], theta[n], theta[n + 1]);
    const w = o.weight ?? 0.5;
    value -= w * Math.log(Math.max(1e-300, p[o.y]));
    const mu = [0, 0, 0, 0];
    for (let k = 0; k < 3; k++)
      for (let a = 0; a < 4; a++) mu[a] += p[k] * FEATURES[k][a];
    for (let a = 0; a < 4; a++) {
      gradient[ids[a]] += w * (mu[a] - FEATURES[o.y][a]);
      if (H)
        for (let b = 0; b < 4; b++) {
          let covariance = -mu[a] * mu[b];
          for (let k = 0; k < 3; k++)
            covariance += p[k] * FEATURES[k][a] * FEATURES[k][b];
          H[ids[a]][ids[b]] += w * covariance;
        }
    }
  }
  return { value, gradient, hessian: H };
}
export function cholesky(matrix) {
  const n = matrix.length,
    L = zeros(n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = matrix[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (!(s > 0) || !Number.isFinite(s))
          throw Error("Ranking curvature is not positive definite");
        L[i][j] = Math.sqrt(s);
      } else L[i][j] = s / L[j][j];
    }
  return L;
}
function solve(L, b) {
  const n = L.length,
    x = new Float64Array(n),
    y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let v = b[i];
    for (let j = 0; j < i; j++) v -= L[i][j] * y[j];
    y[i] = v / L[i][i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let v = y[i];
    for (let j = i + 1; j < n; j++) v -= L[j][i] * x[j];
    x[i] = v / L[i][i];
  }
  return x;
}
function inverse(L) {
  const n = L.length,
    A = zeros(n);
  for (let j = 0; j < n; j++) {
    const e = new Float64Array(n);
    e[j] = 1;
    const col = solve(L, e);
    for (let i = 0; i < n; i++) A[i][j] = col[i];
  }
  return A;
}
function normal(rng) {
  return (
    Math.sqrt(-2 * Math.log(Math.max(1e-15, rng()))) *
    Math.cos(2 * Math.PI * rng())
  );
}
export function fitDavidson(
  ids,
  observations,
  { seed = 7, draws = 2000, priors = PRIORS, sensitivity = true } = {},
) {
  const n = ids.length,
    d = n + 2,
    theta = new Float64Array(d);
  let converged = false,
    iterations = 0,
    fit;
  for (; iterations < 60; iterations++) {
    fit = objective(theta, observations, n, priors);
    if (Math.max(...fit.gradient.map(Math.abs)) < 1e-7) {
      converged = true;
      break;
    }
    const direction = solve(cholesky(fit.hessian), fit.gradient);
    const descent = direction.reduce((s, v, i) => s + v * fit.gradient[i], 0);
    let scale = 1,
      accepted = false;
    for (let attempt = 0; attempt < 25; attempt++, scale *= 0.5) {
      const candidate = theta.map((v, i) => v - scale * direction[i]);
      if (
        objective(candidate, observations, n, priors, false).value <=
        fit.value - 1e-4 * scale * descent
      ) {
        theta.set(candidate);
        accepted = true;
        break;
      }
    }
    if (!accepted) break;
  }
  fit = objective(theta, observations, n, priors);
  const L = cholesky(fit.hessian),
    covariance = inverse(L),
    neighbors = Array.from({ length: n }, () => new Set());
  for (const o of observations) {
    neighbors[o.i].add(o.j);
    neighbors[o.j].add(o.i);
  }
  const components = new Array(n).fill(null),
    groups = [];
  for (let i = 0; i < n; i++)
    if (components[i] === null && neighbors[i].size) {
      const group = [],
        queue = [i];
      components[i] = groups.length;
      while (queue.length) {
        const j = queue.pop();
        group.push(j);
        for (const k of neighbors[j])
          if (components[k] === null) {
            components[k] = groups.length;
            queue.push(k);
          }
      }
      groups.push(group);
    }
  const samples = ids.map(() => []),
    ranks = ids.map(() => []),
    rng = random(seed);
  for (let iteration = 0; iteration < draws; iteration++) {
    const z = Float64Array.from({ length: d }, () => normal(rng)),
      delta = new Float64Array(d);
    for (let i = d - 1; i >= 0; i--) {
      let v = z[i];
      for (let j = i + 1; j < d; j++) v -= L[j][i] * delta[j];
      delta[i] = v / L[i][i];
    }
    const strength = theta.slice(0, n).map((v, i) => v + delta[i]),
      mean = strength.reduce((a, b) => a + b, 0) / n;
    for (let i = 0; i < n; i++)
      samples[i].push(1500 + ELO_FACTOR * (strength[i] - mean));
    for (const group of groups)
      [...group]
        .sort((a, b) => strength[b] - strength[a] || a - b)
        .forEach((i, rank) => ranks[i].push(rank + 1));
  }
  const ratings = ids.map((id, i) => ({
    id,
    strength: theta[i],
    elo: neighbors[i].size ? 1500 + ELO_FACTOR * theta[i] : null,
    interval:
      neighbors[i].size && draws
        ? [quantile(samples[i], 0.025), quantile(samples[i], 0.975)]
        : null,
    rank_interval: ranks[i].length
      ? [quantile(ranks[i], 0.025), quantile(ranks[i], 0.975)]
      : null,
    top_five: ranks[i].length
      ? ranks[i].filter((r) => r <= 5).length / ranks[i].length
      : null,
    opponents: neighbors[i].size,
    component: components[i],
    scope: groups.length > 1 ? "Within component only" : "Connected cohort",
    prior_shift: null,
    unreliable: !converged,
  }));
  if (sensitivity && observations.length) {
    const alternatives = [0.5, 2].map((scale) =>
      fitDavidson(ids, observations, {
        draws: 0,
        sensitivity: false,
        priors: { ...priors, strength: priors.strength * scale },
      }),
    );
    for (let i = 0; i < n; i++) {
      ratings[i].prior_shift = Math.max(
        ...alternatives.map(
          (a) => Math.abs(a.theta[i] - theta[i]) * ELO_FACTOR,
        ),
      );
      ratings[i].prior_sensitive = ratings[i].prior_shift > 100;
    }
  }
  return {
    version: MODEL_VERSION,
    ids,
    theta: Array.from(theta),
    covariance: covariance.map((r) => Array.from(r)),
    ratings,
    converged,
    iterations,
    gradient_max: Math.max(...fit.gradient.map(Math.abs)),
    objective: fit.value,
    priors: { ...priors },
    draws,
    components: groups.length,
    unobserved: neighbors.filter((s) => !s.size).length,
    observations: observations.length,
    evidence_weight: observations.reduce((s, o) => s + (o.weight ?? 0.5), 0),
    position_effect: theta[n + 1],
    log_tie: theta[n],
  };
}
export function predictPair(
  fit,
  a,
  b,
  { presented = false, draws = 0, seed = 7 } = {},
) {
  const i = fit.ids.indexOf(a),
    j = fit.ids.indexOf(b),
    n = fit.ids.length;
  if (
    i < 0 ||
    j < 0 ||
    !fit.ratings[i].opponents ||
    !fit.ratings[j].opponents ||
    fit.ratings[i].component !== fit.ratings[j].component ||
    !fit.converged
  )
    return null;
  const p = probabilities(
    fit.theta[i],
    fit.theta[j],
    fit.theta[n],
    presented ? fit.theta[n + 1] : 0,
  );
  const variance = Math.max(
    0,
    fit.covariance[i][i] + fit.covariance[j][j] - 2 * fit.covariance[i][j],
  );
  const rng = random(seed);
  let greater = 0;
  for (let k = 0; k < draws; k++)
    if (fit.theta[i] - fit.theta[j] + Math.sqrt(variance) * normal(rng) > 0)
      greater++;
  return {
    win: p[0],
    loss: p[1],
    tie: p[2],
    preference_share: p[0] + 0.5 * p[2],
    strength_difference: fit.theta[i] - fit.theta[j],
    strength_interval: [
      fit.theta[i] - fit.theta[j] - 1.96 * Math.sqrt(variance),
      fit.theta[i] - fit.theta[j] + 1.96 * Math.sqrt(variance),
    ],
    stronger_probability: draws ? greater / draws : null,
  };
}
// Expected A-optimal reduction for the local categorical Fisher information.
// Sum of rank-one approximations is used only as a scheduling heuristic, not inference.
export function informationGain(fit, a, b) {
  const i = fit.ids.indexOf(a),
    j = fit.ids.indexOf(b);
  if (i < 0 || j < 0) return 0;
  const p = probabilities(fit.theta[i], fit.theta[j], fit.log_tie, 0),
    variance =
      fit.covariance[i][i] + fit.covariance[j][j] - 2 * fit.covariance[i][j];
  const fisher = (p[0] + p[1] - (p[0] - p[1]) ** 2) / 4;
  let norm = 0;
  for (let k = 0; k < fit.ids.length; k++)
    norm += (fit.covariance[k][i] - fit.covariance[k][j]) ** 2;
  return (fisher * norm) / (1 + fisher * variance);
}
