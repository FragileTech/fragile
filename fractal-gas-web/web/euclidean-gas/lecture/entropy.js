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
  hypocoerciveCoefficients,
  spectralGapDiagnostic,
  dirichletKernel,
  cosineGaussianVariance,
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
    const histories = Array.from({ length: 8 }, () => []),
      coefficients = hypocoerciveCoefficients(p),
      { eta, rate, G } = coefficients;
    const update = () => {
      const state = harmonicState(step * 0.04, p),
        H = gaussianKL(state.mean, state.covariance, state.target),
        I = gaussianFisher(state.mean, state.covariance, state.target);
      const phi = H + 2 * eta * (I[0][0] + I[0][1] + I[1][1]),
        arbitrary = H + I[0][0] + 0.4 * I[0][1] + I[1][1];
      [
        H,
        I[0][0],
        I[1][1],
        I[0][1],
        phi,
        -p.gamma * p.theta * I[1][1],
        (histories[4][0]?.[1] ?? phi) * Math.exp(-rate * step * 0.04),
        arbitrary,
      ].forEach((v, i) => pushBounded(histories[i], [step * 0.04, v]));
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
                ...["H", "Iₓ", "Iᵥ", "Iₓᵥ", "Chapter Φ_G"].map((n, i) =>
                  line(n, histories[i]),
                ),
              ],
            ),
            chart("Chapter decay bound", "Physical time", "Modified entropy", [
              line("Chapter Φ_G", histories[4]),
              line("Φ_G(0) exp(−rt)", histories[6], { dashed: true }),
            ]),
            chart(
              "Alternative positive coefficients: measured trajectory",
              "Physical time",
              "Alternative Φ",
              [line("G=[[1,.2],[.2,1]]", histories[7])],
            ),
            chart(
              "Exact ordinary entropy dissipation",
              "Physical time",
              "dH/dt",
              [line("−γθ Iᵥ", histories[5])],
            ),
          ],
          metrics: [
            metric(
              "G eigenvalues",
              `${eta.toPrecision(4)}, ${(3 * eta).toPrecision(4)}`,
            ),
            metric("ac − b²", 3 * eta ** 2),
            metric("Chapter rate r", rate),
            metric("η", eta),
            metric("Gaussian LSI C", coefficients.C),
            metric("Entropy H", histories[0].at(-1)[1]),
          ],
          message:
            "Chapter coefficients: G=η[[2,1],[1,2]], η=γθ/[2(1+2κ+(2κ+γ+2)²)], C=max(θ/κ,θ), r=η/(C/2+3η). The chapter functional contracts below its envelope; the alternative positive matrix illustrates why the coefficient choice matters.",
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
      { value: "invariant", label: "Reset-kernel invariant" },
      { value: "qsd", label: "Killed-kernel QSD" },
      "different",
    ]),
    range("killing", "State-dependent killing", 0, 0, 0.5, 0.05),
  ],
  async ({ params: p }) => {
    const invariant = [0.2, 0.3, 0.5],
      kernel = invariant.map((_, i) =>
        invariant.map((r, j) => (i === j ? 1 - p.reset : 0) + p.reset * r),
      );
    const killed = kernel.map((row) =>
      row.map((v, j) => v * (1 - (p.killing * j) / 2)),
    );
    let qsd = [...invariant];
    for (let i = 0; i < 600; i++)
      qsd = normalize(
        qsd.map((_, j) => qsd.reduce((s, v, k) => s + v * killed[k][j], 0)),
      );
    const q =
      p.target === "invariant"
        ? invariant
        : p.target === "qsd"
          ? qsd
          : [0.7, 0.2, 0.1];
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
        metrics: [
          metric("Cumulative surviving mass", mass),
          metric(
            "Selected reference",
            p.target === "qsd"
              ? "Conditioned killed-kernel eigenmeasure"
              : p.target === "invariant"
                ? "Reset-kernel invariant"
                : "Alternative fixed law",
          ),
          ...ledger,
        ],
        message: `Kernel: Kᵢⱼ = (1−r)δᵢⱼ + rπⱼ. Killing survival weights: 1, ${1 - p.killing / 2}, ${1 - p.killing}.`,
        experiment: { kernel, killedKernel: killed, qsd, target: q },
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
      "Smoothed atoms",
    ]),
    range("bandwidth", "Atomic smoothing bandwidth", 0.2, 0.05, 0.5, 0.05),
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
    const smoothed = (x) =>
      mean(atoms.map((y) => normalPDF(x, y, p.bandwidth)));
    if (p.representation === "Smoothed atoms") {
      const qgrid = linspace(-16, 16, 1601),
        affinity =
          0.02 *
          qgrid.reduce(
            (sum, x) => sum + Math.sqrt(normalPDF(x) * smoothed(x)),
            0,
          );
      h.shape = 2 - 2 * affinity;
      h.shapeTerm = Math.sqrt(p.mass) * h.shape;
      h.total = h.massTerm + h.shapeTerm;
    }
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
                    p.mass *
                      (p.representation === "Smoothed atoms"
                        ? smoothed(x)
                        : normalPDF(x, p.center, p.width)),
                  ]),
                ),
          ],
        ),
        chart(
          p.representation === "Smoothed atoms"
            ? "Squared Hellinger ledger: Gaussian-mixture quadrature"
            : "Exact squared Hellinger ledger",
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
          : [
              metric(
                p.representation === "Atoms"
                  ? "Atomic/continuous normalized H²"
                  : "Gaussian-smoothed normalized H²",
                h.shape,
              ),
            ]),
        metric("Identity residual", h.total - h.massTerm - h.shapeTerm),
      ],
      message:
        "H²(mp,nq) = (√m−√n)² + √mn H²(p,q). The atomic mode compares a point measure directly. Smoothed atoms use the displayed Gaussian bandwidth and affinity quadrature on[−16,16], spacing.02; changing bandwidth changes the compared measure.",
    });
  },
);

const d4 = descriptor(
  4,
  "Density estimates belong to a stated region",
  "How does the comparison change when we widen the window?",
  "The broader Gaussian q makes q/p grow in the tails while p/q stays bounded; absorbing densities vanish at both endpoints.",
  "Exact density ratios are separated by orientation. A Dirichlet heat-kernel density estimate respects absorbing endpoints and is normalized conditional on remaining in (0,1).",
  [
    select("reference", "Reference model", "Gaussian", [
      "Gaussian",
      "Absorbing",
    ]),
    range("window", "Tail window / interior reach", 3, 1, 6, 0.25),
    range("bandwidth", "KDE bandwidth", 0.1, 0.02, 0.2, 0.02),
  ],
  async ({ params: p, seed }) => {
    const absorbed = p.reference === "Absorbing",
      lo = absorbed ? 0 : -p.window,
      hi = absorbed ? 1 : p.window,
      grid = linspace(lo, hi, 121),
      f = (x) =>
        absorbed
          ? x === 0 || x === 1
            ? 0
            : (Math.PI / 2) * Math.sin(Math.PI * x)
          : normalPDF(x),
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
    const ordinary = grid.map((x) =>
        mean(samples.map((y) => normalPDF(x, y, p.bandwidth))),
      ),
      killed = absorbed
        ? grid.map((x) =>
            mean(samples.map((y) => dirichletKernel(x, y, p.bandwidth))),
          )
        : ordinary;
    const normalizer = absorbed
      ? killed.reduce(
          (s, v, i) => s + v * (i === 0 || i === 120 ? 0.5 : 1),
          0,
        ) / 120
      : 1;
    const estimates = killed.map((v) => v / normalizer),
      inside = grid.filter(
        (x) =>
          !absorbed || (x > 1 / (p.window * 10) && x < 1 - 1 / (p.window * 10)),
      ),
      pq = inside.map((x) => [x, f(x) / g(x)]),
      qp = inside.map((x) => [x, g(x) / f(x)]);
    return staticModel({
      charts: [
        chart("Named reference laws and region-aware KDE", "x", "Density", [
          line(
            "p exact",
            grid.map((x) => [x, f(x)]),
          ),
          line(
            "q exact",
            grid.map((x) => [x, g(x)]),
          ),
          line(
            absorbed
              ? "Dirichlet KDE, conditionally normalized"
              : `Gaussian KDE h=${p.bandwidth}`,
            grid.map((x, i) => [x, estimates[i]]),
          ),
          ...(absorbed
            ? [
                line(
                  "Ordinary KDE: boundary-bias comparison",
                  grid.map((x, i) => [x, ordinary[i]]),
                  { dashed: true },
                ),
              ]
            : []),
        ]),
        chart(
          "Both ratio orientations",
          "x",
          "Density ratio",
          [
            line(
              absorbed
                ? "p/q (bounded interior ratio)"
                : "p/q (globally bounded)",
              pq,
            ),
            line(
              absorbed
                ? "q/p (bounded interior ratio)"
                : "q/p (grows in the tails)",
              qp,
            ),
          ],
          { yScale: "log" },
        ),
        chart("Log ratios on the declared region", "x", "Log density ratio", [
          line(
            "log(p/q)",
            pq.map(([x, r]) => [x, Math.log(r)]),
          ),
          line(
            "log(q/p)",
            qp.map(([x, r]) => [x, Math.log(r)]),
          ),
        ]),
        chart("Coverage of the plotted region", "Bin center", "Sample count", [
          {
            name: "Counts",
            style: "bars",
            points: histogram(samples, lo, hi, 32).map(([x, d]) => [
              x,
              (d * 512 * (hi - lo)) / 32,
            ]),
          },
        ]),
        ...(absorbed
          ? [
              {
                title: "Absorbing heat reference u(t,x)=e^(−π²t)sin(πx)",
                matrix: linspace(0.02, 0.3, 16).map((t) =>
                  linspace(0, 1, 32).map(
                    (x) =>
                      Math.exp(-Math.PI * Math.PI * t) * Math.sin(Math.PI * x),
                  ),
                ),
                rowLabels: linspace(0.02, 0.3, 16).map((t) => t.toFixed(2)),
              },
            ]
          : []),
      ],
      metrics: [
        metric("Samples", 512),
        metric(
          "Samples within plotting window",
          samples.filter((x) => x >= lo && x <= hi).length,
        ),
        metric("Interior min p", Math.min(...inside.map(f))),
        metric("Largest p/q", Math.max(...pq.map((p) => p[1]))),
        metric("Largest q/p", Math.max(...qp.map((p) => p[1]))),
        ...(absorbed
          ? [
              metric("Dirichlet KDE left endpoint", estimates[0]),
              metric("Dirichlet KDE right endpoint", estimates.at(-1)),
              metric("Dirichlet smoothing surviving mass", normalizer),
            ]
          : [
              metric(
                "Global analytic upper bound on p/q",
                1.3 * Math.exp(0.49 / (2 * (1.3 ** 2 - 1))),
              ),
            ]),
      ],
      message: absorbed
        ? "Dirichlet image kernels vanish at 0 and 1. Their mass is measured before conditioning; the displayed corrected KDE integrates to 1 on the plot grid. The ordinary KDE exposes its boundary bias. Both exact density ratios remain bounded near these endpoints."
        : "p=N(0,1), q=N(.7,1.3²). p/q has global maximum 1.8541692; q/p grows without bound. Extending the window reveals the same orientation in the ratio and log-ratio panels.",
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
    const peakIndex = values.reduce(
        (best, value, i) =>
          Math.abs(value.fitness[3]) > Math.abs(values[best].fitness[3])
            ? i
            : best,
        0,
      ),
      peak = xs[peakIndex],
      peakExact = values[peakIndex].fitness[3] * 6;
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
      metrics: [
        ...fdErrors.map((v, i) =>
          metric(`D${i + 1} finite-difference absolute error at x=.13`, v),
        ),
        metric("Peak |D³F| coordinate", peak),
        metric("Peak analytic D³F", peakExact),
      ],
      table: {
        columns: ["FD spacing", "D³F at peak", "Absolute error"],
        rows: [0.001, 0.0003, 0.0001, 0.00003].map((h) => [
          h,
          fd(peak, h, 3),
          Math.abs(fd(peak, h, 3) - peakExact),
        ]),
      },
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
    select("self", "Query convention", "external", [
      { value: "external", label: "External query field" },
      { value: "inclusive", label: "Add fixed self atom at query" },
    ]),
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
    if (p.self === "inclusive") {
      xs.push(0);
      raw.push(1);
      const z = raw.reduce((a, b) => a + b);
      w.splice(0, w.length, ...raw.map((v) => v / z));
    }
    const third = (rho) => {
      const c = (v) => jet(v, 3),
        q = jet(0, 3, true),
        raw = xs.map((x) => {
          const d = add(c(x), scale(q, -1));
          return p.self === "inclusive" && x === 0
            ? c(1)
            : exp(scale(mul(d, d), -0.5 / rho ** 2));
        }),
        z = raw.reduce(add, c(0));
      return Math.abs(
        raw.reduce((s, w, i) => add(s, scale(div(w, z), xs[i] ** 2)), c(0))[3] *
          6,
      );
    };
    const derivativeBound = (rho) => {
      const D = Math.max(...xs.map((x) => Math.abs(x))),
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
              linspace(0.1, 2, 49).map((r) => [r, Math.max(1e-16, third(r))]),
            ),
            line(
              "Chapter W₃ bound × max x²",
              linspace(0.1, 2, 49).map((r) => [r, derivativeBound(r)]),
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
        metric(
          "Weighted x²",
          xs.reduce((s, x, i) => s + w[i] * x * x, 0),
        ),
        metric(
          "Replicated cloud weighted x²",
          normalize([...raw, ...raw]).reduce(
            (s, v, i) => s + v * xs[i % xs.length] ** 2,
            0,
          ),
        ),
        metric(
          "Effective sample size (additional diagnostic)",
          1 / w.reduce((s, v) => s + v * v, 0),
        ),
      ],
      message:
        "The query and derivative coordinate are both x=0. External queries have no self atom; the inclusive option adds a unit Gaussian self-weight that stays constant under query differentiation. Cumulative curves use actual geometric radius, not a relabeled effective sample size. Derivative slice uses exact Taylor arithmetic at query x=0. The bounded-distance estimate is max(x²)[18L₁L₂+26L₁³], with L₁=D/ρ², L₂=ρ⁻², D=max|xⱼ|.",
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
    const radiusErrors = linspace(0.01, 0.5, 50).map((r) => [
        r,
        Math.max(
          ...linspace(0.13 - r, 0.13 + r, 61).map((x) =>
            Math.abs(
              fitnessJet(x, { n: 0, sigma: p.sigma }).fitness[0] -
                polynomial(x, p.order),
            ),
          ),
        ),
      ]),
      useful = radiusErrors.filter(([, e]) => e <= 0.001).at(-1)?.[0] ?? 0,
      C = 1 + Math.abs(coeff[0]),
      B = Math.max(
        ...coeff.slice(1).map((v, i) => (Math.abs(v) / C) ** (1 / (i + 1))),
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
          "Reconstruction error across tested radii",
          "Radius",
          "Maximum sampled error",
          [
            line(
              "Current Taylor order",
              radiusErrors.map(([r, e]) => [r, Math.max(1e-16, e)]),
            ),
            line(
              "Accuracy target 10⁻³",
              [
                [0.01, 0.001],
                [0.5, 0.001],
              ],
              { dashed: true },
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
        metric("Largest tested radius with error ≤.001", useful),
        metric("Finite-order coefficient envelope C", C),
        metric("Finite-order coefficient envelope B", B),
      ],
      message:
        "The reported useful radius is checked against the exact slice at 61 points per radius. C Bⁿ bounds the computed coefficients through order 12; the error-radius plot directly tests Taylor recovery for the chosen order. For n=3 the partitions are {123}, {12|3}, {13|2}, {23|1}, {1|2|3}: multiplicities 1,3,1.",
    });
  },
);

const d8 = descriptor(
  8,
  "Differentiate an averaged stochastic update",
  "Where does the probability derivative appear?",
  "Expected sampled fitness differs from fitness of expected measurements; the probability term closes the derivative identity.",
  "All 81 independent assignments or all three perfect matchings are enumerated for a fixed four-walker candidate set. Fixed-order greedy starts from row 0; the shuffled variant averages the engine’s random first walker.",
  [
    range("width", "Companion width", 0.7, 0.2, 2),
    range("rho", "Localization width", 0.7, 0.2, 2),
    select("law", "Joint assignment law", "independent", [
      { value: "independent", label: "Independent rows" },
      { value: "matching", label: "Ideal matching" },
      { value: "greedy", label: "Fixed-order greedy" },
      { value: "shuffled_greedy", label: "Shuffled greedy (engine law)" },
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
        "Cloud = (x₀, −.4, .4, 1.2), δ=.1, σ=.15, α=β=1; self-inclusive local statistics. Matching weights are proportional to exp(−Σpaired d²/(2ε²)). Fixed-order greedy starts at 0; shuffled greedy averages the first walker uniformly over all four, matching the engine’s random permutation law. Both use the same frozen cloud as IV-05.",
    });
  },
);

const d9 = descriptor(
  9,
  "A nonconvex potential can still have a controlled Gibbs law",
  "What changes when a bounded ripple creates wells?",
  "Negative local curvature can coexist with a confining Gaussian envelope; barriers slow movement between wells.",
  "The named potential U(x)=κx²/2 + a cos(2πx/ℓ) has bounded perturbation oscillation 2a. Four independently seeded random-walk Metropolis ensembles target its quadrature-normalized Gibbs law exactly.",
  [
    range("amplitude", "Ripple amplitude a", 0.5, 0, 2),
    thetaControl,
    range("wavelength", "Ripple wavelength ℓ", 2, 0.5, 2),
    range("k", "Confinement κ", 1, 0.5, 2),
  ],
  async ({ params: p, seed }) => {
    const randoms = Array.from({ length: 4 }, (_, i) =>
        rng((Number(seed) + 7919 * i) >>> 0),
      ),
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
    let groups = Array.from({ length: 4 }, (_, i) =>
        Array(128).fill(i < 2 ? -2 : 2),
      ),
      step = 0,
      accept = 0,
      proposals = 0;
    const histories = Array.from({ length: 4 }, () => []),
      entropy = Array.from({ length: 4 }, () => []),
      crossed = Array.from({ length: 4 }, () => Array(128).fill(false));
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
          groups = groups.map((g, groupIndex) =>
            g.map((x, walker) => {
              const random = randoms[groupIndex];
              const y = x + 0.6 * Math.sqrt(p.theta) * random.normal();
              proposals++;
              if (Math.log(random()) < logw(y) - logw(x)) {
                accept++;
                if (Math.sign(y) !== Math.sign(groupIndex < 2 ? -1 : 1))
                  crossed[groupIndex][walker] = true;
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
            histories.map((v, i) =>
              line(`${i < 2 ? "Left" : "Right"} seed ${(i % 2) + 1}`, v),
            ),
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
          metric("Independent seeds", 4),
          metric(
            "Fraction ever crossing x=0",
            mean(crossed.flat().map(Number)),
          ),
          metric(
            "Left/right start mean separation",
            Math.abs(
              mean(groups.slice(0, 2).flat()) - mean(groups.slice(2).flat()),
            ),
          ),
          metric(
            "Full kinetic reference LSI bound",
            Math.max(
              p.theta,
              (p.theta / p.k) * Math.exp((2 * p.amplitude) / p.theta),
            ),
          ),
        ],
        message:
          "Metropolis Gibbs reference: one time unit is one proposal per particle, with four independent RNG seeds. Mean separation and the fraction crossing x=0 distinguish equilibration from barrier trapping at the displayed horizon. Kinetic entropy rates use a different generator; this clock measures Metropolis mixing. The density quadrature uses 801 nodes; Gaussian-tail envelope chooses L from exp(−κL²/(2θ)+2a/θ) ≤ e⁻²⁶. Histogram KL uses 40 bins with integrated reference masses (16 midpoint samples per bin); both distributions are normalized in [−L,L]. Finite-sample bias remains visible.",
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
    const horizon = Math.max(28, 24 / Math.min(p.gamma, 2 / p.gamma)),
      totalSteps = Math.ceil(horizon / p.h),
      batchSteps = Math.ceil(totalSteps / 350);
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
        for (let i = 0; i < batchSteps && step < totalSteps; i++) {
          states = states.map((s) => gaussianStep(s, kernel));
          step++;
        }
        update();
      },
      () => ({
        step,
        time: step * p.h,
        done: step >= totalSteps,
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
            "Stationary KL discrepancy: fourth order in h",
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
          metric("Physical horizon", totalSteps * p.h),
          metric(
            "Velocity variance bias (order h²)",
            (p.theta * p.h * p.h) / 4,
          ),
          metric("KL leading term (order h⁴)", p.h ** 4 / 64),
          metric(
            "Remaining maximum KL",
            Math.max(
              ...states.map((s) => gaussianKL(s.mean, s.covariance, target)),
            ),
          ),
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
          "Kinetic harmonic BAOAB, Gaussian innovations, no selection or killing. The h² velocity-variance bias and h⁴ stationary Gaussian KL are distinct observables. The physical horizon expands at weak friction; each frame advances a bounded batch of exact covariance updates.",
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
        const covarianceSE = D.map((row, i) =>
            row.map((_, j) =>
              samples.length
                ? Math.sqrt((D[i][i] * D[j][j] + D[i][j] ** 2) / samples.length)
                : 0,
            ),
          ),
          maxZ = samples.length
            ? Math.max(
                ...observed.flatMap((row, i) =>
                  row.map((v, j) => Math.abs(v - D[i][j]) / covarianceSE[i][j]),
                ),
              )
            : 0;
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
              "Largest covariance-entry SE",
              Math.max(...covarianceSE.flat()),
            ),
            metric("Maximum covariance discrepancy / SE", maxZ),
            metric(
              "Relative max covariance error",
              Math.max(...madd(observed, mscale(D, -1)).flat().map(Math.abs)) /
                Math.max(...D.flat().map(Math.abs)),
            ),
            metric(
              "Covariance max absolute error",
              Math.max(...madd(observed, mscale(D, -1)).flat().map(Math.abs)),
            ),
          ],
          message:
            "Each sample is v_after / [(1−h²/4) √((1−e^(−2h))/2)] with h=.1, γ=1, zero initial state, U=|x|²/2. Noise factor is Σ, so these normalized kicks have covariance D. Entrywise Gaussian sampling SE = √((Dii Djj + Dij²)/n); absolute error grows with covariance scale while standardized error stays comparable.",
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
      gapDiagnostic = spectralGapDiagnostic(K),
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
          metric(
            "Normalized spectral gap",
            gapDiagnostic.value ?? "Below numerical resolution",
          ),
          metric("Two-cluster Rayleigh upper bound", gapDiagnostic.upperBound),
          metric("Gap resolution threshold", gapDiagnostic.resolution),
          metric("Weighted mean", weighted(v)),
          metric("Initial weighted mean", center),
          metric(
            "Weighted dissipation identity residual",
            history[3].at(-1)[1],
          ),
        ],
        message:
          "Frozen positions, zero self-edges. If the gap is below resolution, the positive Rayleigh upper bound describes the weak link. The ledger checks dE/dt = −ν/2 Σᵢⱼ Kᵢⱼ(vᵢ−vⱼ)² directly against the computed ODE derivative.",
      }),
    );
  },
);

export function harmonicBudget(N, h, T, gamma = 1) {
  const steps = Math.max(1, Math.round(T / h));
  h = T / steps;
  const kernel = baoab(h, 1, gamma, 1);
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
    h,
    sampling: Math.sqrt(cosineGaussianVariance(0, kernel.stationary[1][1]) / N),
    currentSampling: Math.sqrt(Math.max(0, second - mu * mu) / N),
    discretization: Math.abs(numericalStationary - target),
    transient: Math.min(
      2,
      Math.hypot(
        numerical.mean[1],
        Math.sqrt(numerical.covariance[1][1]) -
          Math.sqrt(kernel.stationary[1][1]),
      ),
    ),
    meanTransient: Math.abs(mu - numericalStationary),
  };
}
const d13 = descriptor(
  13,
  "Spend effort on the largest error term",
  "Which change most improves the bounded observable cos(v)?",
  "Stationary sampling noise falls with N; the timestep and an absolute-error mixing bound respond to h and duration.",
  "The chapter budget uses stationary variance and a Gaussian coupling bound for the absolute-error observable. Independent finite-time replicas measure the expected absolute error and its uncertainty.",
  [
    select("N", "Walkers", 2048, [32, 128, 512, 2048]),
    range("h", "Requested timestep h", 0.5, 0.01, 0.5, 0.01),
    range("T", "Physical duration", 15, 1, 30, 1),
  ],
  async ({ params: p, seed }) => {
    const b = harmonicBudget(p.N, p.h, p.T),
      random = rng(seed),
      bound = b.sampling + b.discretization + b.transient,
      diagnosticBand = Math.abs(b.mu - b.target) + 2 * b.currentSampling;
    let step = 0;
    const observed = [],
      errors = [],
      errorHistory = [],
      running = [],
      mixingDifferences = [];
    return model(
      async () => {
        let current = 0,
          stationary = 0;
        for (let i = 0; i < p.N; i++) {
          const z = random.normal();
          current += Math.cos(
            b.numerical.mean[1] + Math.sqrt(b.numerical.covariance[1][1]) * z,
          );
          stationary += Math.cos(Math.sqrt(b.kernel.stationary[1][1]) * z);
        }
        current /= p.N;
        stationary /= p.N;
        observed.push(current);
        errors.push(Math.abs(current - b.target));
        mixingDifferences.push(
          Math.abs(current - b.target) - Math.abs(stationary - b.target),
        );
        step++;
        pushBounded(errorHistory, [step, errors.at(-1)]);
        pushBounded(running, [step, mean(errors)]);
      },
      () => ({
        step,
        time: p.T,
        done: step >= 128,
        charts: [
          chart(
            "Chapter expected absolute-error budget",
            "Population / stationary sampling / timestep / absolute-error mixing",
            "Magnitude",
            [
              {
                name: "Theorem terms",
                style: "bars",
                points: [
                  [0, 0],
                  [1, b.sampling],
                  [2, b.discretization],
                  [3, b.transient],
                ],
              },
            ],
          ),
          chart(
            "Observed expected error across independent replicas",
            "Completed replicas",
            "Mean absolute error",
            [
              line("Mean measured |estimate−target|", running),
              line(
                "Chapter upper bound on expected error",
                [
                  [0, bound],
                  [Math.max(1, step), bound],
                ],
                { dashed: true },
              ),
            ],
          ),
          chart(
            "Individual replica fluctuations",
            "Replica",
            "Absolute observable error",
            [
              scatter("Measured replica error", errorHistory),
              line(
                "Finite-time |bias| + 2 SE diagnostic",
                [
                  [0, diagnosticBand],
                  [Math.max(1, step), diagnosticBand],
                ],
                { dashed: true },
              ),
            ],
          ),
          chart(
            "Population tradeoff at the same physical time",
            "N",
            "Theorem budget",
            [
              line(
                "Stationary SE + bias + mixing bound",
                [32, 64, 128, 256, 512, 1024, 2048].map((N) => [
                  N,
                  harmonicBudget(N, p.h, p.T).sampling +
                    b.discretization +
                    b.transient,
                ]),
              ),
            ],
            { xScale: "log", yScale: "log" },
          ),
        ],
        metrics: [
          metric("Target E cos(v)", b.target),
          metric("Numerical E cos(v)", b.mu),
          metric("Population interaction bias", 0),
          metric("Stationary sampling standard error", b.sampling),
          metric("Finite-time sampling standard error", b.currentSampling),
          metric("Absolute-error mixing bound", b.transient),
          metric(
            "Measured coupled absolute-error difference",
            step ? mean(mixingDifferences) : 0,
          ),
          metric("Mean measured absolute error", step ? mean(errors) : 0),
          metric(
            "SE of measured mean absolute error",
            step > 1 ? Math.sqrt(variance(errors) / (step - 1)) : 0,
          ),
          metric("Replica mean", step ? mean(observed) : 0),
          metric("Whole replicas", step),
          metric(
            "Observed 2-SE diagnostic coverage",
            step ? errors.filter((e) => e <= diagnosticBand).length / step : 0,
          ),
          metric("Effective timestep", b.h),
        ],
        message:
          "The stationary variance uses cos(V∞), V∞~N(0,Cₕ,vv). For F=|N⁻¹Σcos(vᵢ)−target|, synchronous Gaussian coupling gives |EμF−EπₕF|≤min(2,√(mᵥ²+(σₜ−σₕ)²)). The separate 2-SE curve is a finite-time fluctuation diagnostic; its empirical coverage is reported, while the theorem bounds expected error. h is adjusted to T/round(T/h) to preserve the exact physical duration.",
        experiment: {
          observable: "cos(v)",
          target: "continuous Gibbs θ=1",
          N: p.N,
          requestedH: p.h,
          h: b.h,
          steps: b.steps,
          physicalTime: p.T,
          mixingObservable: "absolute empirical error",
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
    range("cap", "Cloning saturation denominator", 1, 0.1, 4, 0.1),
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
        "Reference walker is at 0; candidates span (0,D]. Fi=1 and clone ε=10⁻⁶. A pointwise target requires (N−1)p*≤1. Equality requires uniform donor probabilities; N=2 always has probability 1. The inverse-width formula uses strict inequality. Lowering the saturation denominator raises accepted probability, up to 1. The thermal formula uses diffusion factor Σ, not its covariance.",
    });
  },
);

const d16 = descriptor(
  16,
  "Build a reproducible experiment from the lecture’s question",
  "Do independent seeded replicas show the predicted relaxation?",
  "The harmonic reference approaches E|x|²=2 with fluctuations shrinking as N grows. An absorbing box produces measurable losses against an unbounded matched-seed control.",
  "Each comparison uses independently seeded replicas within two groups. Mean observables and measured standard errors separate finite-population fluctuations from their theoretical target.",
  [
    select("card", "Experiment card", "kinetic", [
      { value: "kinetic", label: "Conservative kinetic reference" },
      { value: "operators", label: "Kinetic versus selection" },
      { value: "survival", label: "Unbounded versus absorbing" },
      { value: "population", label: "N versus 2N at fixed time" },
    ]),
    select("N", "Base walkers", 64, [64, 128, 256]),
    range("h", "Timestep h", 0.04, 0.01, 0.1, 0.01),
    gammaControl,
    select("replicas", "Independent replicas per group", 4, [4, 8]),
  ],
  async ({ params: p, seed, engine }) => {
    const started = performance.now(),
      R = p.replicas,
      configs = [],
      labels =
        p.card === "survival"
          ? ["Unbounded control", "Absorbing ±0.35"]
          : p.card === "operators"
            ? ["Kinetic only", "Selection + kinetic"]
            : p.card === "population"
              ? [`N=${p.N}`, `N=${2 * p.N}`]
              : ["Independent group A", "Independent group B"];
    for (let group = 0; group < 2; group++)
      for (let replica = 0; replica < R; replica++) {
        const runSeed =
          (Number(seed) +
            7919 * replica +
            (p.card === "kinetic" ? 104729 * group : 0)) >>>
          0;
        const c = await kineticConfig(
          engine,
          runSeed,
          p.card === "population" ? p.N * (group + 1) : p.N,
          p.h,
          p.gamma,
          1,
        );
        if (p.card === "survival" || (p.card === "operators" && group === 1)) {
          c.gas.fitness.reward_exponent = 1;
          c.gas.fitness.diversity_exponent = 1;
        }
        if (p.card === "survival") {
          c.initial_lower = -0.2;
          c.initial_upper = 0.2;
          if (group === 1)
            c.gas.boundary = {
              kind: "absorbing_box",
              field: "positions",
              domain: { lower: [-0.35, -0.35], upper: [0.35, 0.35] },
            };
        }
        configs.push(c);
      }
    const runs = [];
    try {
      for (const c of configs) runs.push(await engine.create(c));
    } catch (error) {
      runs.forEach((run) => run.free());
      throw error;
    }
    const initializationMs = performance.now() - started,
      kernel = baoab(p.h, 1, p.gamma, 1);
    let reference = {
        mean: [0, 0],
        covariance: [
          [1 / 3, 0],
          [0, 0],
        ],
      },
      frames = runs.map((run) => run.snapshot()),
      step = 0,
      firstStepMs = null,
      warmStepsMs = 0;
    const histories = Array.from({ length: 2 }, () =>
        Array.from({ length: 9 }, () => []),
      ),
      referenceHistory = [],
      everLost = Array(2 * R).fill(false),
      revivals = Array(2 * R).fill(0);
    let summaries = [];
    const summarize = () => {
      summaries = [0, 1].map((group) => {
        const indices = Array.from({ length: R }, (_, i) => group * R + i),
          rows = indices.map((i) => {
            const m = frameMetrics(frames[i], configs[i]),
              ps = positions(frames[i]).filter((_, j) =>
                alive(
                  frames[i].population.validity[j],
                  configs[i].gas.include_truncated,
                ),
              ),
              observable = ps.length
                ? mean(ps.map((x) => x.reduce((s, v) => s + v * v, 0)))
                : null;
            return {
              observable,
              alive: m.alive / configs[i].walkers,
              meanReward: m.mean,
              best: m.best,
              cloud: ps.length
                ? variance(ps.map((x) => x[0])) + variance(ps.map((x) => x[1]))
                : null,
              evaluations: m.evaluations,
            };
          }),
          finite = rows.map((row) => row.observable).filter(Number.isFinite),
          se =
            finite.length > 1
              ? Math.sqrt(variance(finite) / (finite.length - 1))
              : null;
        return {
          mean: finite.length ? mean(finite) : null,
          contributingReplicas: finite.length,
          se,
          alive: mean(rows.map((row) => row.alive)),
          swarmSurvival: rows.filter((row) => row.alive > 0).length / R,
          meanReward: mean(
            rows.map((row) => row.meanReward).filter(Number.isFinite),
          ),
          best: mean(rows.map((row) => row.best).filter(Number.isFinite)),
          cloud: mean(rows.map((row) => row.cloud).filter(Number.isFinite)),
          evaluations: rows.reduce((s, row) => s + row.evaluations, 0),
          everLost: indices.filter((i) => everLost[i]).length / R,
          revivals: indices.reduce((s, i) => s + revivals[i], 0),
          replicaObservables: rows.map((row) => row.observable),
        };
      });
      summaries.forEach((a, i) =>
        [
          a.meanReward,
          a.alive,
          a.cloud,
          a.mean,
          a.se,
          a.evaluations,
          a.swarmSurvival,
          a.everLost,
          a.revivals,
        ].forEach((value, j) =>
          pushBounded(histories[i][j], [step * p.h, value]),
        ),
      );
      pushBounded(referenceHistory, [
        step * p.h,
        2 * reference.covariance[0][0],
      ]);
    };
    summarize();
    return {
      ...model(
        async () => {
          const start = performance.now();
          for (let i = 0; i < runs.length; i++)
            if (frameMetrics(frames[i], configs[i]).alive > 0) {
              frames[i] = await runs[i].step(1);
              everLost[i] ||=
                frameMetrics(frames[i], configs[i]).alive < configs[i].walkers;
              revivals[i] += frames[i].report?.revivals ?? 0;
            }
          const elapsed = performance.now() - start;
          if (firstStepMs === null) firstStepMs = elapsed;
          else warmStepsMs += elapsed;
          step++;
          reference = gaussianStep(reference, kernel);
          summarize();
        },
        () => ({
          step,
          time: step * p.h,
          done: step >= 300 || summaries.every((a) => a.swarmSurvival === 0),
          charts: [
            chart(
              "Mean raw reward across independent replicas",
              "Physical time",
              "Mean raw reward",
              histories.map((h, i) => line(labels[i], h[0])),
            ),
            chart(
              "Absorption and revival ledger",
              "Physical time",
              "Fraction",
              [
                ...histories.map((h, i) =>
                  line(`${labels[i]} alive-walker fraction`, h[1]),
                ),
                ...histories.map((h, i) =>
                  line(`${labels[i]} replicas ever losing walkers`, h[7], {
                    dashed: true,
                  }),
                ),
                ...histories.map((h, i) =>
                  line(`${labels[i]} full-swarm survival`, h[6], {
                    dashed: true,
                  }),
                ),
              ],
            ),
            chart(
              "Observable |x|²: surviving-replica mean and two standard errors",
              "Physical time",
              "Mean radius² conditional on replica survival",
              [
                ...histories.flatMap((h, i) => [
                  line(labels[i], h[3]),
                  line(
                    `${labels[i]} +2 SE`,
                    h[3].map(([t, v], j) => [
                      t,
                      v === null || h[4][j][1] === null
                        ? null
                        : v + 2 * h[4][j][1],
                    ]),
                    { dashed: true },
                  ),
                  line(
                    `${labels[i]} −2 SE`,
                    h[3].map(([t, v], j) => [
                      t,
                      v === null || h[4][j][1] === null
                        ? null
                        : v - 2 * h[4][j][1],
                    ]),
                    { dashed: true },
                  ),
                ]),
                ...(["kinetic", "population"].includes(p.card)
                  ? [
                      line("Exact BAOAB finite-time moment", referenceHistory),
                      line(
                        "Gibbs target",
                        [
                          [0, 2],
                          [Math.max(0.1, step * p.h), 2],
                        ],
                        { dashed: true },
                      ),
                    ]
                  : []),
              ],
            ),
            chart(
              "Observable under the total reward budget",
              "Submitted reward rows across replicas",
              "Mean radius² conditional on replica survival",
              histories.map((h, i) =>
                line(
                  labels[i],
                  h[3].map((point, j) => [h[5][j][1], point[1]]),
                ),
              ),
            ),
            chart(
              "Observed uncertainty under population refinement",
              "Walkers per replica",
              "SE conditional on replica survival",
              summaries.map((a, i) =>
                scatter(labels[i], [[configs[i * R].walkers, a.se]]),
              ),
            ),
          ],
          metrics: [
            metric("Independent replicas per group", R),
            ...summaries.flatMap((a, i) => [
              metric(
                `${labels[i]} surviving-replica mean radius²`,
                a.mean ?? "Extinct",
              ),
              metric(
                `${labels[i]} surviving-replica SE`,
                a.se ?? "Requires 2 survivors",
              ),
              metric(
                `${labels[i]} contributing replicas`,
                a.contributingReplicas,
              ),
              metric(`${labels[i]} cumulative revivals`, a.revivals),
              metric(`${labels[i]} total reward rows`, a.evaluations),
            ]),
            metric("Initialization", initializationMs, "ms"),
            metric("First full replica sweep", firstStepMs ?? 0, "ms"),
            metric(
              "Mean subsequent replica sweep",
              step > 1 ? warmStepsMs / (step - 1) : 0,
              "ms",
            ),
          ],
          table: {
            columns: ["Group", "Replica", "Seed", "N", "Current radius²"],
            rows: configs.map((c, i) => [
              labels[Math.floor(i / R)],
              (i % R) + 1,
              c.gas.seed,
              c.walkers,
              summaries[Math.floor(i / R)].replicaObservables[i % R] ??
                "Extinct",
            ]),
          },
          message: `${R} independent replicas per group, CPU WASM f64, U=|x|²/2, Σ=√(2γ)I. ${p.card === "survival" ? "Both groups begin uniformly in [−.2,.2]²; only the second has absorbing boundaries ±.35. Alive-walker loss, full-swarm survival and cumulative revival counts are distinct measurements." : p.card === "population" ? "Matched seeds compare N and 2N at the same physical time. Independent replicas quantify sampling error; total reward-row cost includes every replica." : "Kinetic groups begin with uniform [−1,1]² positions and zero velocity. The kinetic exact-moment reference propagates that initial covariance."} Radius means and their SE condition on nonextinct replicas; each replica radius averages its alive walkers. Contributing counts are shown. The survival ledger retains every replica. Two-SE curves describe measured replica spread and are not a calibrated small-sample confidence interval.`,
          experiment: {
            card: p.card,
            configs,
            seeds: configs.map((c) => c.gas.seed),
            replicasPerGroup: R,
            contributingReplicas: summaries.map((a) => a.contributingReplicas),
            groupLabels: labels,
            observable:
              "mean squared radius among alive walkers; extinct values recorded separately",
            conditioning:
              p.card === "survival"
                ? "radius mean and SE conditional on replica survival; survival ledger retains all outcomes"
                : "all alive",
            backend: "cpu WASM f64",
            initializationMs,
            firstStepMs,
            warmStepsMs,
          },
        }),
        () => runs.forEach((run) => run.free()),
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
