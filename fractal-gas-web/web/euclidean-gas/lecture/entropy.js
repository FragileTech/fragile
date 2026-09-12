import {
  rng,
  linspace,
  mean,
  variance,
  normalPDF,
  histogram,
  covariance2,
  line,
  scatter,
  positions,
  velocities,
  pushBounded,
} from "./math.js";
import {
  jet,
  add,
  scale,
  mul,
  div,
  exp,
  factorial,
  fitnessJet,
  gaussianKL,
  gaussianFisher,
  harmonicState,
  baoab,
  gaussianStep,
  mm,
  mv,
  madd,
  mscale,
  transpose,
  matrixExp,
  identity,
  hellinger,
  kl,
  normalize,
  enumeration,
  symmetricEigenvalues,
  constraintWidth,
} from "./entropy-math.js";
import { alive, frameMetrics } from "../config.js";

const range = (key, label, value, min, max, step = 0.1) => ({
  key,
  label,
  type: "range",
  value,
  min,
  max,
  step,
});
const select = (key, label, value, options) => ({
  key,
  label,
  type: "select",
  value,
  options: options.map((v) =>
    typeof v === "object" ? v : { value: v, label: String(v) },
  ),
});
const metric = (label, value, unit = "") => ({ label, value, unit });
const chart = (title, xLabel, yLabel, series, extra = {}) => ({
  title,
  xLabel,
  yLabel,
  series,
  ...extra,
});
const staticModel = (data) => ({
  async step() {},
  snapshot() {
    return { step: 0, time: 0, done: true, ...data };
  },
  dispose() {},
});
const model = (advance, snapshot, dispose = () => {}) => ({
  step: advance,
  snapshot,
  dispose,
});
const defaults = (controls, params) =>
  Object.fromEntries(controls.map((c) => [c.key, params[c.key] ?? c.value]));
const descriptor = (
  number,
  title,
  question,
  prediction,
  explanation,
  controls,
  create,
  kind = "Mathematical model",
) => ({
  id: `IV-${String(number).padStart(2, "0")}`,
  part: "IV",
  title,
  question,
  prediction,
  explanation,
  controls,
  create: (context) =>
    create({ ...context, params: defaults(controls, context.params ?? {}) }),
  kind,
});
const ellipse = (m, c) =>
  linspace(0, 2 * Math.PI, 81).map((t) => {
    const a = Math.sqrt(c[0][0]),
      b = c[1][0] / a,
      d = Math.sqrt(Math.max(0, c[1][1] - b * b));
    return [
      m[0] + 2 * a * Math.cos(t),
      m[1] + 2 * (b * Math.cos(t) + d * Math.sin(t)),
    ];
  });
const gammaControl = range("gamma", "Friction γ", 1, 0.2, 3, 0.1),
  thetaControl = range("theta", "Temperature θ", 1, 0.2, 2, 0.1);
const d1 = descriptor(
  1,
  "How velocity noise reaches position information",
  "How does velocity noise remove a position displacement?",
  "Velocity information starts at zero; transport creates it and friction dissipates entropy.",
  "Exact harmonic kinetic Gaussian: dH/dt = −γθ Iᵥ. The equilibrium covariance is diag(θ/κ, θ); all displayed Fisher terms use this same target.",
  [
    range("k", "Stiffness κ", 1, 0.5, 2),
    gammaControl,
    range("displacement", "Initial displacement", 2, 0, 3),
    thetaControl,
  ],
  async ({ params: p }) => {
    let step = 0;
    const histories = Array.from({ length: 6 }, () => []),
      G = [
        [1, 0.2],
        [0.2, 1],
      ];
    const update = () => {
      const state = harmonicState(step * 0.04, p),
        H = gaussianKL(state.mean, state.covariance, state.target),
        I = gaussianFisher(state.mean, state.covariance, state.target);
      const phi = H + I[0][0] + 0.4 * I[0][1] + I[1][1];
      [H, I[0][0], I[1][1], I[0][1], phi, -p.gamma * p.theta * I[1][1]].forEach(
        (v, i) => pushBounded(histories[i], [step * 0.04, v]),
      );
    };
    update();
    return model(
      async () => {
        step++;
        update();
      },
      () => {
        const s = harmonicState(step * 0.04, p);
        return {
          step,
          time: step * 0.04,
          done: step >= 350,
          charts: [
            chart(
              "Phase-space density: two-standard-deviation contour",
              "x",
              "v",
              [
                line("Equilibrium", ellipse([0, 0], s.target), {
                  dashed: true,
                }),
                line("Evolving Gaussian", ellipse(s.mean, s.covariance)),
              ],
            ),
            chart(
              "Entropy and information transfer",
              "Physical time",
              "Information",
              [
                ...["H", "Iₓ", "Iᵥ", "Iₓᵥ", "Φ with G = [[1,.2],[.2,1]]"].map(
                  (n, i) => line(n, histories[i]),
                ),
              ],
            ),
            chart(
              "Exact ordinary entropy dissipation",
              "Physical time",
              "dH/dt",
              [line("−γθ Iᵥ", histories[5])],
            ),
          ],
          metrics: [
            metric("G eigenvalues", "0.8, 1.2"),
            metric("ac − b²", 0.96),
            metric("Entropy H", histories[0].at(-1)[1]),
          ],
          message:
            "The cross term transfers information between x and v. Positive G gives the displayed information functional, evaluated along the exact harmonic flow.",
        };
      },
    );
  },
);

const d2 = descriptor(
  2,
  "Entropy contracts toward the right target",
  "Which reference distribution makes contraction visible?",
  "Applying the same Markov kernel to both laws contracts KL; an invariant target stays fixed.",
  "A three-state reset kernel and state-dependent killing are evaluated as finite probability vectors. The killed distribution is explicitly normalized by its surviving mass.",
  [
    range("reset", "Reset strength", 0.3, 0.05, 0.8, 0.05),
    select("target", "Reference target", "invariant", [
      "invariant",
      "different",
    ]),
    range("killing", "State-dependent killing", 0, 0, 0.5, 0.05),
  ],
  async ({ params: p }) => {
    const invariant = [0.2, 0.3, 0.5],
      q = p.target === "invariant" ? invariant : [0.7, 0.2, 0.1],
      kernel = invariant.map((_, i) =>
        invariant.map((r, j) => (i === j ? 1 - p.reset : 0) + p.reset * r),
      );
    let state = [0.85, 0.1, 0.05],
      time = 0,
      mass = 1;
    const history = [];
    let ledger = [];
    const apply = (a) =>
      a.map((_, j) => a.reduce((s, v, i) => s + v * kernel[i][j], 0));
    const advance = () => {
      const before = kl(state, q),
        next = apply(state),
        qnext = apply(q),
        survivors = next.map((v, i) => v * (1 - (p.killing * i) / 2)),
        survival = survivors.reduce((a, b) => a + b, 0),
        conditioned = normalize(survivors);
      ledger = [
        metric("KL before", before),
        metric("KL(Kp ∥ Kq)", kl(next, qnext)),
        metric("KL(Kp ∥ q)", kl(next, q)),
        metric("One-step survival", survival),
        metric("Conditioning change in KL", kl(conditioned, q) - kl(next, q)),
      ];
      state = conditioned;
      mass *= survival;
      time++;
      pushBounded(history, [time, kl(state, q)]);
    };
    history.push([0, kl(state, q)]);
    return model(
      async () => advance(),
      () => ({
        step: time,
        time,
        done: time >= 150,
        charts: [
          chart("Probability laws", "State", "Probability", [
            {
              name: "Evolving p",
              points: state.map((v, i) => [i, v]),
              style: "bars",
            },
            line(
              "Reference q",
              q.map((v, i) => [i, v]),
            ),
          ]),
          {
            title: "Markov kernel (rows sum to one)",
            matrix: kernel,
            rowLabels: ["0", "1", "2"],
            columnLabels: ["0", "1", "2"],
          },
          chart("Conditioned entropy to selected target", "Updates", "KL", [
            line("KL(p ∥ q)", history),
          ]),
        ],
        metrics: [metric("Cumulative surviving mass", mass), ...ledger],
        message: `Kernel: Kᵢⱼ = (1−r)δᵢⱼ + rπⱼ. Killing survival weights: 1, ${1 - p.killing / 2}, ${1 - p.killing}.`,
        experiment: { kernel, target: q },
      }),
    );
  },
);

const d3 = descriptor(
  3,
  "Separate how much survives from where it survives",
  "Can mass and shape errors move independently?",
  "Changing only mass changes the reaction term; shifting the center changes shape and transport.",
  "The Gaussian affinity gives the Hellinger mass–shape identity exactly. W₂ is evaluated between the normalized Gaussian laws.",
  [
    range("mass", "Second mass", 0.6, 0.05, 1, 0.05),
    range("center", "Second center", 1, -2, 2),
    range("width", "Second width", 0.8, 0.2, 1.5),
    select("representation", "Comparison measure", "Gaussian", [
      "Gaussian",
      "Atoms",
    ]),
  ],
  async ({ params: p, seed }) => {
    const h = hellinger({
        otherMass: p.mass,
        center: p.center,
        otherWidth: p.width,
        atomic: p.representation === "Atoms",
      }),
      grid = linspace(-5, 5, 151),
      random = rng(seed);
    const atoms = Array.from(
      { length: 64 },
      () => p.center + p.width * random.normal(),
    );
    return staticModel({
      charts: [
        chart(
          "Finite measures: total heights include mass",
          "x",
          "Density / atom mass",
          [
            line(
              "Unit-mass Gaussian",
              grid.map((x) => [x, normalPDF(x)]),
            ),
            p.representation === "Atoms"
              ? scatter(
                  "64 atoms, each mass m/64",
                  atoms.map((x) => [x, p.mass / 64]),
                )
              : line(
                  "Second measure",
                  grid.map((x) => [
                    x,
                    p.mass * normalPDF(x, p.center, p.width),
                  ]),
                ),
          ],
        ),
        chart(
          "Exact squared Hellinger ledger",
          "Contribution",
          "Squared distance",
          [
            {
              name: "Mass / shape / total",
              style: "bars",
              points: [
                [0, h.massTerm],
                [1, h.shapeTerm],
                [2, h.total],
              ],
            },
          ],
        ),
      ],
      metrics: [
        metric("Mass term", h.massTerm),
        metric("Weighted shape term", h.shapeTerm),
        metric("Squared Hellinger", h.total),
        ...(p.representation === "Gaussian"
          ? [
              metric("Normalized W₂", h.w2),
              metric(
                "Additive D = √(H² + W₂²)",
                Math.sqrt(h.total + h.w2 * h.w2),
              ),
              metric("Pure reaction path action", h.total),
              ...(p.mass === 1
                ? [metric("Equal-mass transport path action", h.w2 * h.w2)]
                : []),
            ]
          : [metric("Atomic/continuous normalized H²", 2)]),
        metric("Identity residual", h.total - h.massTerm - h.shapeTerm),
      ],
      message:
        "H²(mp,nq) = (√m−√n)² + √mn H²(p,q). The atomic mode compares the finite point measure directly to a continuous Gaussian.",
    });
  },
);

const d4 = descriptor(
  4,
  "Density estimates belong to a stated region",
  "How does the comparison change when we widen the window?",
  "Gaussian density ratios grow in the tails; absorbing densities vanish at the endpoints.",
  "Exact reference densities and seeded sample coverage are evaluated on the declared plotting region.",
  [
    select("reference", "Reference model", "Gaussian", [
      "Gaussian",
      "Absorbing",
    ]),
    range("window", "Tail window / interior reach", 3, 1, 6, 0.25),
    range("bandwidth", "Gaussian KDE bandwidth", 0.1, 0.02, 0.2, 0.02),
  ],
  async ({ params: p, seed }) => {
    const absorbed = p.reference === "Absorbing",
      lo = absorbed ? 0 : -p.window,
      hi = absorbed ? 1 : p.window,
      grid = linspace(lo, hi, 121),
      f = (x) =>
        absorbed ? (Math.PI / 2) * Math.sin(Math.PI * x) : normalPDF(x),
      g = (x) => (absorbed ? 6 * x * (1 - x) : normalPDF(x, 0.7, 1.3)),
      random = rng(seed),
      samples = [];
    while (samples.length < 512) {
      if (!absorbed) samples.push(random.normal());
      else {
        const x = random();
        if (random() < Math.sin(Math.PI * x)) samples.push(x);
      }
    }
    const kde = (x) => mean(samples.map((y) => normalPDF(x, y, p.bandwidth))),
      inside = grid.filter((x) =>
        absorbed
          ? x > 1 / (p.window * 10) && x < 1 - 1 / (p.window * 10)
          : true,
      );
    const ratios = inside.map((x) => [x, f(x) / g(x)]),
      counts = histogram(samples, lo, hi, 32).map(([x, d]) => [
        x,
        (d * 512 * (hi - lo)) / 32,
      ]);
    return staticModel({
      charts: [
        chart("Named reference laws and KDE", "x", "Density", [
          line(
            "p exact",
            grid.map((x) => [x, f(x)]),
          ),
          line(
            "q exact",
            grid.map((x) => [x, g(x)]),
          ),
          line(
            `KDE h=${p.bandwidth}`,
            grid.map((x) => [x, kde(x)]),
            { dashed: true },
          ),
        ]),
        chart("Ratio on the declared interior", "x", "p / q", [
          line("Exact ratio", ratios),
          line(
            "log(p/q)",
            ratios.map(([x, r]) => [x, Math.log(r)]),
          ),
        ]),
        chart("Coverage of the plotted region", "Bin center", "Sample count", [
          { name: "Counts", style: "bars", points: counts },
        ]),
        {
          title: "Absorbing heat reference u(t,x)=e^(−π²t) sin(πx)",
          matrix: linspace(0.02, 0.3, 16).map((t) =>
            linspace(0, 1, 32).map(
              (x) => Math.exp(-(Math.PI ** 2) * t) * Math.sin(Math.PI * x),
            ),
          ),
          rowLabels: linspace(0.02, 0.3, 16).map((t) => t.toFixed(2)),
        },
      ],
      metrics: [
        metric("Samples", 512),
        metric(
          "Samples within plotting window",
          samples.filter((x) => x >= lo && x <= hi).length,
        ),
        metric("Interior min p", Math.min(...inside.map(f))),
        metric("Largest interior ratio", Math.max(...ratios.map((r) => r[1]))),
      ],
      message: absorbed
        ? "Absorbing reference on (0,1); both analytic curves vanish at the boundary."
        : "Unbounded Gaussian laws. Expanding the window includes tails, while histogram mass outside the window stays unplotted.",
    });
  },
);

const d5 = descriptor(
  5,
  "Move one coordinate and watch normalization differentiate",
  "Which derivative terms come from the denominator?",
  "Differentiating every factor preserves normalization; small positive floors resolve the near-collision slice.",
  "Exact Taylor arithmetic differentiates the chapter’s self-inclusive Gaussian-localized reward × diversity fitness through order three. All companions remain fixed.",
  [
    range("rho", "Localization width ρ", 0.7, 0.15, 2, 0.05),
    range("sigma", "Standard-deviation floor", 0.15, 0.02, 0.5, 0.02),
    range("delta", "Distance floor δ", 0.1, 0.02, 0.3, 0.02),
    select("N", "Walkers", 4, [4, 8, 16]),
  ],
  async ({ params: p }) => {
    const xs = linspace(-1.5, 1.5, 101),
      values = xs.map((x) => fitnessJet(x, p));
    const fd = (x, h, order) => {
      const f = (y) => fitnessJet(y, { ...p, n: 0 }).fitness[0];
      return order === 1
        ? (f(x + h) - f(x - h)) / (2 * h)
        : order === 2
          ? (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
          : (f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h)) /
            (2 * h ** 3);
    };
    const exact = fitnessJet(0.13, p).fitness,
      fdErrors = [1, 2, 3].map((n) =>
        Math.abs(fd(0.13, 0.0001, n) - exact[n] * factorial(n)),
      );
    return staticModel({
      charts: [
        chart(
          "Self-inclusive local statistics",
          "Selected coordinate x₀",
          "Value",
          ["mean", "variance", "score", "fitness"].map((key) =>
            line(
              key,
              xs.map((x, i) => [x, values[i][key][0]]),
            ),
          ),
        ),
        chart(
          "Exact derivatives of fixed-companion fitness",
          "x₀",
          "Directional derivative",
          [1, 2, 3].map((n) =>
            line(
              `D${n}F`,
              xs.map((x, i) => [x, values[i].fitness[n] * factorial(n)]),
            ),
          ),
        ),
        chart("The denominator contribution", "x₀", "First derivative", [
          line(
            "Full quotient derivative",
            xs.map((x, i) => [x, values[i].fitness[1]]),
          ),
          line(
            "Frozen denominator counterexample",
            xs.map((x) => [
              x,
              fitnessJet(x, { ...p, frozenDenominator: true }).fitness[1],
            ]),
            { dashed: true },
          ),
        ]),
      ],
      metrics: fdErrors.map((v, i) =>
        metric(`D${i + 1} finite-difference absolute error at x=.13`, v),
      ),
      message:
        "Fixed alive/candidate stratum, cⱼ=(j+1) mod N. rⱼ=−xⱼ², dⱼ=√((xⱼ−x꜀)²+δ²), A=2, η=.001, α=β=1. Independent centered finite differences use h=10⁻⁴.",
    });
  },
);

const d6 = descriptor(
  6,
  "Locality weights in dense and sparse swarms",
  "What grows when many walkers occupy one small ball?",
  "Raw kernel sums and neighbor counts grow with N; normalized probability mass remains one.",
  "Query-field localization separates geometric counts, raw Gaussian row sums, and normalized statistics; identical replicated clouds have the same normalized average.",
  [
    select("N", "Walkers", 64, [16, 64, 256]),
    range("rho", "Localization width ρ", 0.7, 0.1, 2),
    select("geometry", "Cloud geometry", "clustered", ["uniform", "clustered"]),
  ],
  async ({ params: p }) => {
    const xs = Array.from({ length: p.N }, (_, i) =>
        p.geometry === "uniform"
          ? -2 + (4 * i) / (p.N - 1)
          : (i % 2 ? -1 : 1) + 0.15 * Math.sin(i * 2.4),
      ),
      radii = linspace(0.01, 4, 101),
      raw = xs.map((x) => Math.exp((-x * x) / (2 * p.rho ** 2))),
      w = normalize(raw),
      average = xs.reduce((s, x, i) => s + w[i] * x * x, 0);
    const third = (rho) => {
      const c = (v) => jet(v, 3),
        q = jet(0.13, 3, true),
        raw = xs.map((x) => {
          const d = add(c(x), scale(q, -1));
          return exp(scale(mul(d, d), -0.5 / rho ** 2));
        }),
        z = raw.reduce(add, c(0));
      return Math.abs(
        raw.reduce((s, w, i) => add(s, scale(div(w, z), xs[i] ** 2)), c(0))[3] *
          6,
      );
    };
    const derivativeBound = (rho) => {
      const D = Math.max(...xs.map((x) => Math.abs(x - 0.13))),
        L1 = D / rho ** 2,
        L2 = 1 / rho ** 2;
      return Math.max(...xs.map((x) => x * x)) * (18 * L1 * L2 + 26 * L1 ** 3);
    };
    return staticModel({
      charts: [
        chart(
          "The cloud and its locality weights",
          "Position",
          "Normalized weight",
          [
            scatter(
              "Weight per walker",
              xs.map((x, i) => [x, w[i]]),
            ),
          ],
        ),
        chart("Grow a query ball centered at zero", "Radius", "Count / mass", [
          line(
            "Geometric neighbor count",
            radii.map((r) => [r, xs.filter((x) => Math.abs(x) <= r).length]),
          ),
          line(
            "Raw Gaussian mass in ball",
            radii.map((r) => [
              r,
              raw.reduce((s, a, i) => s + (Math.abs(xs[i]) <= r ? a : 0), 0),
            ]),
          ),
          line(
            "Normalized mass in ball",
            radii.map((r) => [
              r,
              w.reduce((s, a, i) => s + (Math.abs(xs[i]) <= r ? a : 0), 0),
            ]),
          ),
        ]),
        chart(
          "Bandwidth controls derivative scale",
          "ρ",
          "|D³ weighted x²|",
          [
            line(
              "Exact Taylor derivative",
              linspace(0.2, 2, 41).map((r) => [r, Math.max(1e-16, third(r))]),
            ),
            line(
              "Chapter W₃ bound × max x²",
              linspace(0.2, 2, 41).map((r) => [r, derivativeBound(r)]),
              { dashed: true },
            ),
          ],
          { yScale: "log" },
        ),
      ],
      metrics: [
        metric(
          "Raw row sum",
          raw.reduce((a, b) => a + b, 0),
        ),
        metric(
          "Normalized row sum",
          w.reduce((a, b) => a + b, 0),
        ),
        metric("Weighted x²", average),
        metric(
          "Replicated cloud weighted x²",
          normalize([...raw, ...raw]).reduce(
            (s, v, i) => s + v * xs[i % p.N] ** 2,
            0,
          ),
        ),
        metric(
          "Effective sample size (additional diagnostic)",
          1 / w.reduce((s, v) => s + v * v, 0),
        ),
      ],
      message:
        "The fixed query is x=0. Cumulative curves use actual geometric radius, not a relabeled effective sample size. Derivative slice uses exact Taylor arithmetic at query x=.13. The bounded-distance estimate is max(x²)[18L₁L₂+26L₁³], with L₁=D/ρ², L₂=ρ⁻², D=max|xⱼ−.13|.",
    });
  },
);
const d7 = descriptor(
  7,
  "Follow the growth of higher derivatives",
  "How do derivative coefficients control a Taylor approximation?",
  "Taylor recovery improves as order grows within a sufficiently small radius; regularizers change that useful radius.",
  "Truncated-series arithmetic computes the first twelve Taylor coefficients without finite-difference subtraction. Set-partition counts organize the Faà di Bruno chain rule.",
  [
    range("order", "Taylor order", 6, 1, 12, 1),
    range("radius", "Expansion radius", 0.2, 0.01, 0.5, 0.01),
    range("sigma", "Standard-deviation floor", 0.15, 0.02, 0.5, 0.02),
  ],
  async ({ params: p }) => {
    const coeff = fitnessJet(0.13, { n: 12, sigma: p.sigma }).fitness,
      grid = linspace(0.13 - p.radius, 0.13 + p.radius, 101),
      bell = Array(7).fill(0);
    bell[0] = 1;
    const choose = (n, k) => factorial(n) / (factorial(k) * factorial(n - k));
    for (let n = 0; n < 6; n++)
      for (let k = 0; k <= n; k++) bell[n + 1] += choose(n, k) * bell[k];
    const polynomial = (x, n) =>
        coeff.slice(0, n + 1).reduce((s, v, i) => s + v * (x - 0.13) ** i, 0),
      errors = grid.map((x) =>
        Math.abs(
          fitnessJet(x, { n: 0, sigma: p.sigma }).fitness[0] -
            polynomial(x, p.order),
        ),
      );
    return staticModel({
      charts: [
        chart(
          "Taylor reconstruction of the actual regularized fitness",
          "x₀",
          "Fitness",
          [
            line(
              "Exact fitness",
              grid.map((x) => [
                x,
                fitnessJet(x, { n: 0, sigma: p.sigma }).fitness[0],
              ]),
            ),
            line(
              `Taylor order ${p.order}`,
              grid.map((x) => [x, polynomial(x, p.order)]),
              { dashed: true },
            ),
            line(
              "Order 2",
              grid.map((x) => [x, polynomial(x, 2)]),
              { dashed: true },
            ),
          ],
        ),
        chart(
          "Computed Taylor coefficients through order twelve",
          "n",
          "|DⁿF| / n!",
          [
            line(
              "Absolute coefficients",
              coeff
                .slice(1)
                .map((v, i) => [i + 1, Math.max(1e-20, Math.abs(v))]),
            ),
          ],
          { yScale: "log" },
        ),
        chart(
          "Set partitions in the chain rule",
          "Derivative order",
          "Bell number",
          [
            {
              name: "Partitions",
              style: "bars",
              points: bell.slice(1).map((v, i) => [i + 1, v]),
            },
          ],
        ),
        chart(
          "Auxiliary smooth nonanalytic function at the origin",
          "x",
          "exp(−1/x²)",
          [
            line(
              "Exact smooth function",
              linspace(-0.5, 0.5, 101).map((x) => [
                x,
                x ? Math.exp(-1 / x ** 2) : 0,
              ]),
            ),
            line(
              "Every Taylor polynomial at 0",
              [
                [-0.5, 0],
                [0.5, 0],
              ],
              { dashed: true },
            ),
          ],
        ),
      ],
      table: {
        columns: ["Order", "Set partitions", "Taylor coefficient"],
        rows: bell.slice(1).map((v, i) => [i + 1, v, coeff[i + 1]]),
      },
      metrics: [
        metric("Maximum displayed Taylor error", Math.max(...errors)),
        metric("Expansion center", 0.13),
      ],
      message:
        "The coefficient graph displays the coefficients of this frozen fitness fixture. For n=3 the partitions are {123}, {12|3}, {13|2}, {23|1}, {1|2|3}: multiplicities 1,3,1.",
    });
  },
);

const d8 = descriptor(
  8,
  "Differentiate an averaged stochastic update",
  "Where does the probability derivative appear?",
  "Expected sampled fitness differs from fitness of expected measurements; the probability term closes the derivative identity.",
  "All 81 independent assignments or all three perfect matchings are enumerated for a fixed four-walker candidate set. Greedy pairing takes the first available row in order.",
  [
    range("width", "Companion width", 0.7, 0.2, 2),
    range("rho", "Localization width", 0.7, 0.2, 2),
    select("law", "Joint assignment law", "independent", [
      "independent",
      "matching",
      "greedy",
    ]),
  ],
  async ({ params: p }) => {
    const xs = linspace(-1.4, 1, 61),
      h = 1e-4,
      calc = (x) => enumeration(x, p.width, p.law, p.rho),
      all = xs.map(calc),
      derivatives = xs.map((x) => {
        const a = calc(x - h),
          b = calc(x + h),
          c = calc(x);
        let value = 0,
          probability = 0;
        for (let i = 0; i < c.assignments.length; i++) {
          value +=
            (c.assignments[i].p *
              (b.assignments[i].value - a.assignments[i].value)) /
            (2 * h);
          probability +=
            ((b.assignments[i].p - a.assignments[i].p) / (2 * h)) *
            c.assignments[i].value;
        }
        return {
          total: (b.expected - a.expected) / (2 * h),
          value,
          probability,
        };
      });
    return staticModel({
      charts: [
        chart(
          "Three different fitness fields",
          "x₀",
          "Fitness",
          ["frozen", "substitute", "expected"].map((key) =>
            line(
              key,
              xs.map((x, i) => [x, all[i][key]]),
            ),
          ),
        ),
        chart("Derivative of the finite expectation", "x₀", "Derivative", [
          line(
            "D E[F]",
            xs.map((x, i) => [x, derivatives[i].total]),
          ),
          line(
            "Σ p DF",
            xs.map((x, i) => [x, derivatives[i].value]),
          ),
          line(
            "Σ (Dp) F",
            xs.map((x, i) => [x, derivatives[i].probability]),
          ),
          line(
            "Sum of the two terms",
            xs.map((x, i) => [
              x,
              derivatives[i].value + derivatives[i].probability,
            ]),
            { dashed: true },
          ),
        ]),
      ],
      metrics: [
        metric("Enumerated assignments", all[0].assignments.length),
        metric(
          "Max derivative closure residual",
          Math.max(
            ...derivatives.map((d) =>
              Math.abs(d.total - d.value - d.probability),
            ),
          ),
        ),
        metric(
          "Probability mass",
          all[30].assignments.reduce((s, a) => s + a.p, 0),
        ),
      ],
      message:
        "Cloud = (x₀, −.4, .5, 1.2), δ=.1, σ=.15, α=β=1; self-inclusive local statistics. Matching weights are proportional to exp(−Σpaired d²/(2ε²)). Greedy ordering is 0,1,2,3.",
    });
  },
);

const d9 = descriptor(
  9,
  "A nonconvex potential can still have a controlled Gibbs law",
  "What changes when a bounded ripple creates wells?",
  "Negative local curvature can coexist with a confining Gaussian envelope; barriers slow movement between wells.",
  "The named potential U(x)=κx²/2 + a cos(2πx/ℓ) has bounded perturbation oscillation 2a. Two seeded random-walk Metropolis ensembles target its quadrature-normalized Gibbs law exactly.",
  [
    range("amplitude", "Ripple amplitude a", 0.5, 0, 2),
    thetaControl,
    range("wavelength", "Ripple wavelength ℓ", 2, 0.5, 2),
    range("k", "Confinement κ", 1, 0.5, 2),
  ],
  async ({ params: p, seed }) => {
    const random = rng(seed),
      L = Math.max(
        7,
        Math.sqrt(((2 * p.theta) / p.k) * (26 + (2 * p.amplitude) / p.theta)),
      ),
      grid = linspace(-L, L, 801),
      dx = grid[1] - grid[0],
      U = (x) =>
        (p.k * x * x) / 2 +
        p.amplitude * Math.cos((2 * Math.PI * x) / p.wavelength),
      logw = (x) => -U(x) / p.theta,
      raw = grid.map((x) => Math.exp(logw(x))),
      Z =
        dx *
        raw.reduce((s, v, i) => s + v * (i === 0 || i === 800 ? 0.5 : 1), 0),
      density = (x) => Math.exp(logw(x)) / Z;
    const binWidth = (2 * L) / 40,
      binTarget = normalize(
        Array.from({ length: 40 }, (_, i) => {
          const left = -L + i * binWidth;
          return Array.from(
            { length: 16 },
            (_, j) =>
              (density(left + ((j + 0.5) * binWidth) / 16) * binWidth) / 16,
          ).reduce((a, b) => a + b, 0);
        }),
      );
    let groups = [Array(128).fill(-2), Array(128).fill(2)],
      step = 0,
      accept = 0,
      proposals = 0;
    const histories = [[], []],
      entropy = [[], []];
    const update = () =>
      groups.forEach((g, j) => {
        pushBounded(histories[j], [step, mean(g)]);
        const bins = histogram(g, -L, L, 40),
          diagnostic = kl(
            normalize(bins.map(([, d]) => d * binWidth)),
            binTarget,
          );
        pushBounded(entropy[j], [step, diagnostic]);
      });
    update();
    return model(
      async () => {
        for (let batch = 0; batch < 8; batch++)
          groups = groups.map((g) =>
            g.map((x) => {
              const y = x + 0.6 * Math.sqrt(p.theta) * random.normal();
              proposals++;
              if (Math.log(random()) < logw(y) - logw(x)) {
                accept++;
                return y;
              }
              return x;
            }),
          );
        step += 8;
        update();
      },
      () => ({
        step,
        time: step,
        done: step >= 2000,
        charts: [
          chart("Potential and curvature on the confining line", "x", "Value", [
            line(
              "U",
              grid.filter((_, i) => i % 4 === 0).map((x) => [x, U(x)]),
            ),
            line(
              "U″",
              grid
                .filter((_, i) => i % 4 === 0)
                .map((x) => [
                  x,
                  p.k -
                    p.amplitude *
                      ((2 * Math.PI) / p.wavelength) ** 2 *
                      Math.cos((2 * Math.PI * x) / p.wavelength),
                ]),
            ),
          ]),
          chart(
            "Two initial ensembles and the same Gibbs target",
            "x",
            "Density",
            [
              line(
                "Quadrature Gibbs law",
                grid.map((x) => [x, density(x)]),
              ),
              ...groups.map((g, i) =>
                line(`Ensemble ${i + 1} histogram`, histogram(g, -L, L, 40)),
              ),
            ],
          ),
          chart(
            "Observable relaxation from opposite starts",
            "Metropolis transitions / particle",
            "Mean x",
            histories.map((v, i) => line(`Start ${i ? "right" : "left"}`, v)),
          ),
          chart(
            "Explicit 40-bin marginal entropy diagnostic",
            "Transitions / particle",
            "Discrete histogram KL",
            entropy.map((v, i) => line(`Ensemble ${i + 1}`, v)),
          ),
        ],
        metrics: [
          metric(
            "Gaussian reference C_LS (Ent≤2C_LS Dirichlet)",
            p.theta / p.k,
          ),
          metric(
            "Bounded-perturbation multiplier",
            Math.exp((2 * p.amplitude) / p.theta),
          ),
          metric(
            "Minimum curvature",
            p.k - p.amplitude * ((2 * Math.PI) / p.wavelength) ** 2,
          ),
          metric("Metropolis acceptance", proposals ? accept / proposals : 0),
          metric("Quadrature half-window L", L),
        ],
        message:
          "Metropolis Gibbs reference experiment; time axis counts Metropolis transitions. The density quadrature uses 801 nodes; Gaussian-tail envelope chooses L from exp(−κL²/(2θ)+2a/θ) ≤ e⁻²⁶. Histogram KL uses 40 bins with integrated reference masses (16 midpoint samples per bin); both distributions are normalized in [−L,L]. Finite-sample bias remains visible.",
      }),
    );
  },
);

const d10 = descriptor(
  10,
  "Follow entropy toward the numerical equilibrium",
  "Which Gaussian law is invariant under a numerical BAOAB step?",
  "Both initial laws relax to the discrete covariance; its velocity variance approaches the continuous value as h shrinks.",
  "For κ=1, numerical BAOAB has Cₕ=diag(θ, θ(1−h²/4)). Every covariance update is M C Mᵀ + Q, using the actual linear splitting matrix.",
  [range("h", "Timestep h", 0.08, 0.01, 0.3, 0.01), gammaControl, thetaControl],
  async ({ params: p }) => {
    const kernel = baoab(p.h, 1, p.gamma, p.theta),
      target = kernel.stationary,
      continuous = [
        [p.theta, 0],
        [0, p.theta],
      ],
      histories = [[], []];
    let states = [
        {
          mean: [2, 0],
          covariance: [
            [0.4, 0],
            [0, 0.5],
          ],
        },
        {
          mean: [-2, 1],
          covariance: [
            [2, 0],
            [0, 1.5],
          ],
        },
      ],
      step = 0;
    const update = () =>
      states.forEach((s, i) =>
        pushBounded(histories[i], [
          step * p.h,
          gaussianKL(s.mean, s.covariance, target),
        ]),
      );
    update();
    return model(
      async () => {
        states = states.map((s) => gaussianStep(s, kernel));
        step++;
        update();
      },
      () => ({
        step,
        time: step * p.h,
        done: step >= 350,
        charts: [
          chart(
            "Entropy relative to the numerical invariant law",
            "Physical time",
            "Exact Gaussian KL",
            histories.map((h, i) => line(`Initial law ${i + 1}`, h)),
          ),
          chart("Continuous / discrete covariance ellipses", "x", "v", [
            line("Continuous Gibbs", ellipse([0, 0], continuous)),
            line("Numerical invariant", ellipse([0, 0], target), {
              dashed: true,
            }),
            ...states.map((s, i) =>
              line(`Current law ${i + 1}`, ellipse(s.mean, s.covariance)),
            ),
          ]),
          chart(
            "Numerical-target discrepancy under refinement",
            "h",
            "KL(Cₕ ∥ C)",
            [
              line(
                "Exact Gaussian KL",
                linspace(0.005, 0.3, 61).map((h) => [
                  h,
                  gaussianKL(
                    [0, 0],
                    baoab(h, 1, p.gamma, p.theta).stationary,
                    continuous,
                  ),
                ]),
              ),
            ],
            { xScale: "log", yScale: "log" },
          ),
        ],
        metrics: [
          metric("Stationary x variance", target[0][0]),
          metric("Stationary v variance", target[1][1]),
          metric(
            "Discrete balance residual",
            Math.max(
              ...madd(
                gaussianStep({ mean: [0, 0], covariance: target }, kernel)
                  .covariance,
                mscale(target, -1),
              )
                .flat()
                .map(Math.abs),
            ),
          ),
        ],
        message:
          "Kinetic harmonic BAOAB, Gaussian innovations, no selection or killing. The displayed KL is evaluated analytically from the evolving mean/covariance.",
      }),
    );
  },
);
async function kineticConfig(
  engine,
  seed,
  N = 64,
  h = 0.04,
  gamma = 1,
  theta = 1,
) {
  const config = await engine.defaults();
  config.benchmark = "sphere";
  config.potential = "quadratic";
  config.walkers = N;
  config.dimensions = 2;
  config.initial_lower = -1;
  config.initial_upper = 1;
  config.gas.seed = Number(seed) >>> 0;
  config.gas.backend = "cpu";
  config.gas.precision = "f64";
  config.gas.boundary = { kind: "unbounded" };
  config.gas.fitness.reward_exponent = 0;
  config.gas.fitness.diversity_exponent = 0;
  config.gas.kinetic.integrator = {
    kind: "baoab",
    positions: "positions",
    velocities: "velocities",
    dt: h,
    friction: gamma,
  };
  config.gas.kinetic.noise = {
    innovation: "gaussian",
    geometry: {
      kind: "isotropic",
      scale: { kind: "constant", values: [Math.sqrt(2 * gamma * theta)] },
    },
  };
  return config;
}
const d11 = descriptor(
  11,
  "Hessian eigenvalues set the adaptive noise ellipse",
  "Which Hessian direction gets the largest velocity kick?",
  "The smallest positive shifted eigenvalue produces the broadest diffusion direction.",
  "The actual Rust full-factor noise module samples Σ=(H+εI)⁻¹ᐟ². Identical zero initial states isolate one BAOAB O-stage kick; the known drift and OU factors are divided out.",
  [
    range("lambda", "First Hessian eigenvalue", 0.5, -2, 3),
    range("shift", "Spectral shift ε", 1, 0.1, 5),
    range("angle", "Eigenvector rotation (degrees)", 30, 0, 180, 5),
  ],
  async ({ params: p, seed, engine }) => {
    const eigen = [p.lambda + p.shift, 2 + p.shift];
    if (eigen[0] <= 0)
      return staticModel({
        charts: [],
        metrics: [metric("Smallest shifted eigenvalue", eigen[0])],
        message:
          "Choose ε > −λmin(H) so that the shifted metric is positive definite.",
      });
    const a = (p.angle * Math.PI) / 180,
      c = Math.cos(a),
      s = Math.sin(a),
      R = [
        [c, -s],
        [s, c],
      ],
      factor = mm(
        mm(R, [
          [1 / Math.sqrt(eigen[0]), 0],
          [0, 1 / Math.sqrt(eigen[1])],
        ]),
        transpose(R),
      ),
      D = mm(factor, transpose(factor));
    const h = 0.1,
      config = await kineticConfig(engine, seed, 128, h);
    config.gas.kinetic.noise.geometry = {
      kind: "full",
      factor: { kind: "constant", values: factor.flat() },
    };
    const gas = await engine.create(config),
      zero = Array.from({ length: 128 }, () => [0, 0]);
    let samples = [],
      step = 0;
    return model(
      async () => {
        await gas.set_population(
          JSON.stringify({ positions: zero, velocities: zero }),
        );
        const f = await gas.step(1),
          gain = Math.sqrt(-Math.expm1(-2 * h) / 2) * (1 - (h * h) / 4);
        samples.push(...velocities(f).map((v) => [v[0] / gain, v[1] / gain]));
        if (samples.length > 4096) samples = samples.slice(-4096);
        step++;
      },
      () => {
        const observed =
          samples.length > 1
            ? covariance2(samples)
            : [
                [0, 0],
                [0, 0],
              ];
        return {
          step,
          time: step,
          done: step >= 32,
          charts: [
            chart(
              "Normalized O-stage kicks from the Rust full factor",
              "Velocity kick 1",
              "Velocity kick 2",
              [
                scatter(
                  "WASM samples",
                  samples.filter(
                    (_, i) =>
                      i % Math.max(1, Math.floor(samples.length / 700)) === 0,
                  ),
                ),
                line("Predicted 2σ ellipse", ellipse([0, 0], D)),
              ],
            ),
            { title: "Predicted velocity diffusion D", matrix: D },
            { title: "Sample covariance", matrix: observed },
            chart(
              "Spectral margin and strongest diffusion",
              "Shift ε",
              "Largest diffusion eigenvalue",
              [
                line(
                  "1 / (ε + λmin)",
                  linspace(
                    Math.max(0.01, -Math.min(p.lambda, 2) + 0.05),
                    5,
                    101,
                  ).map((e) => [e, 1 / (e + Math.min(p.lambda, 2))]),
                ),
              ],
            ),
          ],
          metrics: [
            metric("Smallest metric eigenvalue", Math.min(...eigen)),
            metric("Samples", samples.length),
            metric(
              "Covariance max absolute error",
              Math.max(...madd(observed, mscale(D, -1)).flat().map(Math.abs)),
            ),
          ],
          message:
            "Each sample is v_after / [(1−h²/4) √((1−e^(−2h))/2)] with h=.1, γ=1, zero initial state, U=|x|²/2. Noise factor is Σ, so these normalized kicks have covariance D.",
        };
      },
      () => gas.free(),
    );
  },
  "WASM + reference",
);

const d12 = descriptor(
  12,
  "The right weighted average reveals alignment",
  "Which mean survives row-normalized interactions?",
  "The degree-weighted mean stays fixed and weighted relative energy decays; weak links slow alignment.",
  "A frozen symmetric Gaussian graph generates dv/dt=ν(P−I)v. Matrix exponentiation computes its evolution and the normalized Laplacian spectrum.",
  [
    range("width", "Neighbor width", 0.8, 0.2, 2),
    range("viscosity", "Viscosity ν", 1, 0.1, 3),
    range("gap", "Gap between clusters", 1.5, 0.3, 3),
  ],
  async ({ params: p }) => {
    const xs = [-p.gap - 0.2, -p.gap, -p.gap + 0.3, 0.1, 0.5, 0.7],
      K = xs.map((x, i) =>
        xs.map((y, j) =>
          i === j ? 0 : Math.exp(-((x - y) ** 2) / (2 * p.width * p.width)),
        ),
      ),
      degrees = K.map((r) => r.reduce((a, b) => a + b, 0)),
      P = K.map((r, i) => r.map((v) => v / degrees[i])),
      L = P.map((r, i) => r.map((v, j) => p.viscosity * (v - +(i === j)))),
      symmetric = K.map((r, i) =>
        r.map((v, j) => +(i === j) - v / Math.sqrt(degrees[i] * degrees[j])),
      ),
      spectrum = symmetricEigenvalues(symmetric),
      v0 = [2, -1, 1, -2, 0.5, -0.5],
      mass = degrees.reduce((a, b) => a + b, 0),
      weighted = (v) => v.reduce((s, x, i) => s + x * degrees[i], 0) / mass,
      center = weighted(v0);
    let step = 0,
      v = v0;
    const history = [[], [], [], []];
    const update = () => {
      const energy =
          0.5 * v.reduce((s, x, i) => s + degrees[i] * (x - center) ** 2, 0),
        dissipation =
          -0.5 *
          p.viscosity *
          K.reduce(
            (s, row, i) =>
              s + row.reduce((ss, k, j) => ss + k * (v[i] - v[j]) ** 2, 0),
            0,
          ),
        dv = mv(L, v),
        direct = v.reduce(
          (s, x, i) => s + degrees[i] * (x - center) * dv[i],
          0,
        );
      [mean(v), weighted(v), energy, direct - dissipation].forEach((x, i) =>
        pushBounded(history[i], [step * 0.05, x]),
      );
    };
    update();
    return model(
      async () => {
        step++;
        v = mv(matrixExp(L, step * 0.05), v0);
        update();
      },
      () => ({
        step,
        time: step * 0.05,
        done: step >= 350,
        charts: [
          chart(
            "Frozen graph: node heights are current velocities",
            "Position",
            "Velocity",
            [
              scatter(
                "Walkers",
                xs.map((x, i) => [x, v[i]]),
              ),
            ],
            {
              segments: xs
                .flatMap((x, i) =>
                  xs.slice(i + 1).map((y, j) =>
                    K[i][i + j + 1] > 0.02
                      ? [
                          [x, v[i]],
                          [y, v[i + j + 1]],
                        ]
                      : null,
                  ),
                )
                .filter(Boolean),
            },
          ),
          chart("Two averages", "Physical time", "Mean velocity", [
            line("Ordinary mean", history[0]),
            line("Degree-weighted mean", history[1]),
          ]),
          chart(
            "Dissipation",
            "Physical time",
            "Degree-weighted relative energy",
            [line("Energy", history[2])],
          ),
          { title: "Row-normalized alignment P", matrix: P },
        ],
        metrics: [
          metric("Normalized spectral gap", spectrum[1]),
          metric("Weighted mean", weighted(v)),
          metric("Initial weighted mean", center),
          metric(
            "Weighted dissipation identity residual",
            history[3].at(-1)[1],
          ),
        ],
        message:
          "Frozen positions, zero self-edges. The ledger checks dE/dt = −ν/2 Σᵢⱼ Kᵢⱼ(vᵢ−vⱼ)² directly against the computed ODE derivative.",
      }),
    );
  },
);

function harmonicBudget(N, h, T, gamma = 1) {
  const kernel = baoab(h, 1, gamma, 1),
    steps = Math.round(T / h);
  let numerical = {
    mean: [2, 0],
    covariance: [
      [1, 0],
      [0, 1],
    ],
  };
  // One matrix power plus stationary-covariance identity avoids thousands of steps.
  let power = identity(2),
    base = kernel.M,
    n = steps;
  while (n) {
    if (n % 2) power = mm(power, base);
    base = mm(base, base);
    n = Math.floor(n / 2);
  }
  numerical = {
    mean: mv(power, numerical.mean),
    covariance: madd(
      kernel.stationary,
      mm(
        mm(power, madd(numerical.covariance, mscale(kernel.stationary, -1))),
        transpose(power),
      ),
    ),
  };
  const cosine = (s) => Math.exp(-s.covariance[1][1] / 2) * Math.cos(s.mean[1]),
    target = Math.exp(-0.5),
    numericalStationary = Math.exp(-kernel.stationary[1][1] / 2),
    mu = cosine(numerical),
    second =
      (1 +
        Math.exp(-2 * numerical.covariance[1][1]) *
          Math.cos(2 * numerical.mean[1])) /
      2;
  return {
    kernel,
    numerical,
    steps,
    target,
    mu,
    populationBias: 0,
    sampling: Math.sqrt(Math.max(0, second - mu * mu) / N),
    discretization: Math.abs(numericalStationary - target),
    transient: Math.abs(mu - numericalStationary),
  };
}
const d13 = descriptor(
  13,
  "Spend effort on the largest error term",
  "Which change most improves the bounded observable cos(v)?",
  "Increasing N reduces sampling noise; timestep and transient errors respond to h and duration.",
  "Independent harmonic BAOAB walkers have zero population-interaction bias. The numerical law and stationary target yield exact bias, transient, and sampling variance for cos(v).",
  [
    select("N", "Walkers", 128, [32, 128, 512, 2048]),
    range("h", "Timestep h", 0.08, 0.01, 0.5, 0.01),
    range("T", "Physical duration", 5, 1, 30, 1),
  ],
  async ({ params: p, seed }) => {
    const budget = harmonicBudget(p.N, p.h, p.T),
      random = rng(seed);
    let step = 0,
      observed = [];
    const errorHistory = [];
    return model(
      async () => {
        const sample = mean(
          Array.from({ length: p.N }, () =>
            Math.cos(
              budget.numerical.mean[1] +
                Math.sqrt(budget.numerical.covariance[1][1]) * random.normal(),
            ),
          ),
        );
        observed.push(sample);
        step++;
        pushBounded(errorHistory, [step, Math.abs(sample - budget.target)]);
      },
      () => ({
        step,
        time: budget.steps * p.h,
        done: step >= 128,
        charts: [
          chart(
            "Exact harmonic error budget",
            "Population / sampling / timestep / transient",
            "Magnitude",
            [
              {
                name: "Component",
                style: "bars",
                points: [
                  [0, 0],
                  [1, budget.sampling],
                  [2, budget.discretization],
                  [3, budget.transient],
                ],
              },
            ],
          ),
          chart(
            "Independent complete-run estimates",
            "Replica",
            "Absolute observable error",
            [
              scatter("Measured replica error", errorHistory),
              line(
                "Bias + 2 standard errors",
                [
                  [
                    0,
                    budget.discretization +
                      budget.transient +
                      2 * budget.sampling,
                  ],
                  [
                    Math.max(1, step),
                    budget.discretization +
                      budget.transient +
                      2 * budget.sampling,
                  ],
                ],
                { dashed: true },
              ),
            ],
          ),
          chart(
            "Population tradeoff at fixed h and T",
            "N",
            "Error budget",
            [
              line(
                "One-standard-error sampling + deterministic terms",
                [32, 64, 128, 256, 512, 1024, 2048].map((N) => [
                  N,
                  harmonicBudget(N, p.h, p.T).sampling +
                    budget.discretization +
                    budget.transient,
                ]),
              ),
            ],
            { xScale: "log", yScale: "log" },
          ),
        ],
        metrics: [
          metric("Target E cos(v)", budget.target),
          metric("Numerical E cos(v)", budget.mu),
          metric("Population interaction bias", 0),
          metric("Sampling standard error", budget.sampling),
          metric("Replica mean", observed.length ? mean(observed) : 0),
          metric("Whole replicas", step),
        ],
        message:
          "Samples are drawn from the exact finite-time Gaussian law of the BAOAB kernel; independent particles make b_N=0. The 2-SE line is a diagnostic band, not a deterministic bound.",
        experiment: {
          observable: "cos(v)",
          target: "continuous Gibbs, θ=1",
          N: p.N,
          h: p.h,
          steps: budget.steps,
        },
      }),
    );
  },
);

const d14 = descriptor(
  14,
  "Observe the timestep convergence rate",
  "What remains when sampling error is removed from refinement?",
  "Fixed-time harmonic weak error and stationary velocity bias shrink quadratically with h in the resolved range.",
  "Exact first/second moments remove sampling noise. The finite-time reference uses exp(TA); the numerical result uses the actual BAOAB transition matrix.",
  [
    select("T", "Fixed physical duration", 5, [5, 10]),
    gammaControl,
    select("observable", "Observable", "cos(v)", ["cos(v)", "v²"]),
  ],
  async ({ params: p }) => {
    const hs = [0.08, 0.04, 0.02, 0.01, 0.005],
      values = hs.map((requestedH) => {
        const T = p.T,
          h = T / Math.round(T / requestedH),
          b = harmonicBudget(128, h, T, p.gamma),
          A = matrixExp(
            [
              [0, 1],
              [-1, -p.gamma],
            ],
            T,
          ),
          m = mv(A, [2, 0]),
          C = [
            [1, 0],
            [0, 1],
          ],
          observable = (s) =>
            p.observable === "v²"
              ? s.covariance[1][1] + s.mean[1] ** 2
              : Math.exp(-s.covariance[1][1] / 2) * Math.cos(s.mean[1]),
          truth = observable({ mean: m, covariance: C }),
          numerical = observable(b.numerical),
          target = p.observable === "v²" ? 1 : Math.exp(-0.5),
          stationary = observable({
            mean: [0, 0],
            covariance: b.kernel.stationary,
          });
        return {
          h,
          T,
          steps: b.steps,
          error: Math.abs(numerical - truth),
          stationary: Math.abs(stationary - target),
        };
      }),
      slope = (key) => {
        const x = values.map((v) => Math.log(v.h)),
          y = values.map((v) => Math.log(Math.max(1e-30, v[key]))),
          mx = mean(x),
          my = mean(y);
        return (
          x.reduce((s, v, i) => s + (v - mx) * (y[i] - my), 0) /
          x.reduce((s, v) => s + (v - mx) ** 2, 0)
        );
      };
    return staticModel({
      charts: [
        chart(
          "Finite-time weak observable error",
          "h",
          "Absolute error",
          [
            line(
              p.observable,
              values.map((v) => [v.h, v.error]),
            ),
          ],
          { xScale: "log", yScale: "log" },
        ),
        chart(
          "Stationary observable bias",
          "h",
          "Absolute bias",
          [
            line(
              p.observable,
              values.map((v) => [v.h, v.stationary]),
            ),
          ],
          { xScale: "log", yScale: "log" },
        ),
        chart(
          "Computational cost at fixed physical time",
          "h",
          "BAOAB steps",
          [
            line(
              "Steps",
              values.map((v) => [v.h, v.steps]),
            ),
          ],
          { xScale: "log", yScale: "log" },
        ),
      ],
      table: {
        columns: [
          "h",
          "Matched physical T",
          "Steps",
          "Finite-time error",
          "Stationary bias",
        ],
        rows: values.map((v) => [v.h, v.T, v.steps, v.error, v.stationary]),
      },
      metrics: [
        metric("Finite-time fitted slope", slope("error")),
        metric("Stationary fitted slope", slope("stationary")),
        metric("Sampling uncertainty", 0),
      ],
      message:
        "Conservative all-alive harmonic reference: α=β=0. Every h is evaluated at its displayed matched physical duration, and exact continuous moments supply the reference. The requested h is adjusted to T/round(T/h), so all rows end at exactly the same T.",
    });
  },
);

const d15 = descriptor(
  15,
  "Explore parameter constraints before tuning",
  "How does companion reach translate into a width?",
  "A normalized-measure minorization target gives a width independent of population size; per-candidate reach falls with N.",
  "Gaussian diameter inequalities and the OU variance formula are evaluated directly. The displayed donor fixture spans the declared core diameter.",
  [
    range("diameter", "Declared core diameter D", 2, 0.5, 5),
    select("N", "Alive count", 64, [2, 16, 64, 256, 512]),
    range("target", "Measure target m*", 0.2, 0.01, 0.9, 0.01),
    range("gap", "Fitness gap Fdonor − Fi", 0.5, 0, 3),
    range("cap", "Clone probability cap", 1, 0.1, 4, 0.1),
  ],
  async ({ params: p }) => {
    const width = constraintWidth(p.diameter, p.target),
      xs = linspace(0, p.diameter, p.N),
      weights = normalize(
        xs.slice(1).map((x) => Math.exp((-x * x) / (2 * width * width))),
      ),
      pointFloor = p.target / (p.N - 1),
      h = 0.04,
      gamma = 1,
      diffusion = Math.sqrt(2),
      clone = (g) => Math.min(1, Math.max(0, g / (1 + 1e-6)) / p.cap);
    return staticModel({
      charts: [
        chart(
          "Width needed for a normalized measure target",
          "Declared diameter D",
          "Required width ε",
          [
            line(
              "Sufficient ε",
              linspace(0.5, 5, 81).map((D) => [
                D,
                constraintWidth(D, p.target),
              ]),
            ),
          ],
        ),
        chart(
          "Actual donor probabilities and the analytic floor",
          "Candidate position",
          "Probability",
          [
            scatter(
              "Exact Gaussian probability",
              xs.slice(1).map((x, i) => [x, weights[i]]),
            ),
            line(
              "m* / (N−1)",
              [
                [0, pointFloor],
                [p.diameter, pointFloor],
              ],
              { dashed: true },
            ),
          ],
        ),
        chart(
          "Clone probability after score clipping",
          "Fitness gap",
          "Accepted probability",
          [
            line(
              "min(1, max(0, gap/(Fi+ε))/pmax)",
              linspace(-1, 3, 81).map((x) => [x, clone(x)]),
            ),
          ],
        ),
      ],
      metrics: [
        metric("Required companion width", width),
        metric("Normalized measure floor", p.target),
        metric("Per-candidate probability floor", pointFloor),
        metric("Actual minimum donor probability", Math.min(...weights)),
        metric("Clone probability at selected gap", clone(p.gap)),
        metric(
          "O-stage variance (γ=1,h=.04,Σ=√2)",
          (diffusion ** 2 * -Math.expm1(-2 * gamma * h)) / (2 * gamma),
        ),
      ],
      message:
        "Reference walker is at 0; candidates span (0,D]. Fi=1 and clone ε=10⁻⁶. For a fixed pointwise target p*, feasibility requires (N−1)p*<1. The thermal formula uses diffusion factor Σ, not its covariance.",
    });
  },
);

const d16 = descriptor(
  16,
  "Build a reproducible experiment from the lecture’s question",
  "Do independent seeded runs show the predicted relaxation?",
  "The harmonic reference relaxes toward E|x|²=2; selection and absorption cards reveal their effects on reward, variance, and survival.",
  "Each card runs two real WASM populations with separate seeds. Initialization and step timings are measured independently; JSON export records the full configuration and observable definition.",
  [
    select("card", "Experiment card", "kinetic", [
      { value: "kinetic", label: "Conservative kinetic reference" },
      { value: "operators", label: "Operator intuition" },
      { value: "survival", label: "Absorbing full swarm" },
      { value: "population", label: "Fixed-time population comparison" },
    ]),
    select("N", "Base walkers", 64, [64, 128, 256]),
    range("h", "Timestep h", 0.04, 0.01, 0.1, 0.01),
    gammaControl,
  ],
  async ({ params: p, seed, engine }) => {
    const start = performance.now(),
      configs = [];
    for (let i = 0; i < 2; i++) {
      const c = await kineticConfig(
        engine,
        (Number(seed) + i) >>> 0,
        p.card === "population" ? p.N * (i + 1) : p.N,
        p.h,
        p.gamma,
        1,
      );
      if (p.card === "operators" || p.card === "survival") {
        c.gas.fitness.reward_exponent = 1;
        c.gas.fitness.diversity_exponent = 1;
      }
      if (p.card === "survival")
        c.gas.boundary = {
          kind: "absorbing_box",
          field: "positions",
          domain: { lower: [-1.5, -1.5], upper: [1.5, 1.5] },
        };
      configs.push(c);
    }
    const runs = [];
    try {
      for (const c of configs) runs.push(await engine.create(c));
    } catch (error) {
      runs.forEach((r) => r.free());
      throw error;
    }
    const initializationMs = performance.now() - start;
    let frames = runs.map((r) => r.snapshot()),
      step = 0,
      stepMs = 0,
      firstStepMs = null;
    const histories = Array.from({ length: 2 }, () =>
      Array.from({ length: 6 }, () => []),
    );
    const collect = () =>
      frames.forEach((f, i) => {
        const m = frameMetrics(f, configs[i]),
          ps = positions(f).filter((_, j) =>
            alive(f.population.validity[j], configs[i].gas.include_truncated),
          ),
          observable = ps.length
            ? mean(ps.map((x) => x.reduce((s, v) => s + v * v, 0)))
            : 0,
          cloudVariance = ps.length
            ? variance(ps.map((x) => x[0])) + variance(ps.map((x) => x[1]))
            : 0;
        [
          m.best,
          m.mean,
          m.alive / configs[i].walkers,
          cloudVariance,
          observable,
          m.evaluations,
        ].forEach((v, j) => pushBounded(histories[i][j], [f.step * p.h, v]));
      });
    collect();
    return {
      ...model(
        async () => {
          const start = performance.now();
          for (let i = 0; i < runs.length; i++) {
            const m = frameMetrics(frames[i], configs[i]);
            if (m.alive > 0) frames[i] = await runs[i].step(1);
          }
          const elapsed = performance.now() - start;
          if (firstStepMs === null) firstStepMs = elapsed;
          else stepMs += elapsed;
          step++;
          collect();
        },
        () => ({
          step,
          time: step * p.h,
          done:
            step >= 300 ||
            frames.every((f, i) => frameMetrics(f, configs[i]).alive === 0),
          charts: [
            chart(
              "Raw reward under two reproducible seeds",
              "Physical time",
              "Raw reward",
              histories.flatMap((h, i) => [
                line(`Run ${i + 1} mean`, h[1]),
                line(`Run ${i + 1} best`, h[0], { dashed: true }),
              ]),
            ),
            chart(
              "Survival and spread",
              "Physical time",
              "Alive fraction / cloud variance",
              histories.flatMap((h, i) => [
                line(`Run ${i + 1} alive fraction`, h[2]),
                line(`Run ${i + 1} cloud variance`, h[3], { dashed: true }),
              ]),
            ),
            chart("Observable ⟨|x|²⟩", "Physical time", "Mean squared radius", [
              ...histories.map((h, i) => line(`Run ${i + 1}`, h[4])),
              ...(p.card === "kinetic" || p.card === "population"
                ? [
                    line(
                      "Continuous Gibbs target",
                      [
                        [0, 2],
                        [Math.max(0.1, step * p.h), 2],
                      ],
                      { dashed: true },
                    ),
                  ]
                : []),
            ]),
            chart(
              "Reward evaluation budget",
              "Submitted reward rows",
              "Mean raw reward",
              histories.map((h, i) =>
                line(
                  `Run ${i + 1}`,
                  h[1].map((point, j) => [h[5][j][1], point[1]]),
                ),
              ),
            ),
          ],
          metrics: [
            metric("Initialization", initializationMs, "ms"),
            metric("First step pair", firstStepMs ?? 0, "ms"),
            metric(
              "Mean subsequent step pair",
              step > 1 ? stepMs / (step - 1) : 0,
              "ms",
            ),
            metric("Run 1 reward rows", frames[0].reward_evaluations),
            metric("Run 2 reward rows", frames[1].reward_evaluations),
          ],
          message: `${p.card} card: CPU WASM f64, U=|x|²/2, Σ=√(2γ)I, θ=1. ${p.card === "survival" ? "Alive fractions include absorbing outcomes; extinction stays in the report." : "Physical time and actual submitted reward rows are both shown."}`,
          experiment: {
            card: p.card,
            configs,
            seeds: configs.map((c) => c.gas.seed),
            observable: "mean squared position radius among alive walkers",
            conditioning:
              p.card === "survival"
                ? "finite absorbing/revival swarm"
                : "all alive",
            backend: "cpu WASM f64",
            initializationMs,
            firstStepMs,
            warmStepsMs: stepMs,
          },
        }),
        () => runs.forEach((r) => r.free()),
      ),
      checkpoint() {
        return runs[0].checkpoint();
      },
    };
  },
  "WASM experiment",
);

export const demos = [
  d1,
  d2,
  d3,
  d4,
  d5,
  d6,
  d7,
  d8,
  d9,
  d10,
  d11,
  d12,
  d13,
  d14,
  d15,
  d16,
];
