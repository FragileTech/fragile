# Chapter 10 validation

The source ledger assigns every expression its own exact formula hash, law, operand fields and comparison indices. Definitions and analytic assumptions receive separate dispositions. Global LSI, full native QSD derivative estimates and numerical functional defects are never inferred from finite histograms.

| Quantity or estimate | Operands and experiment | Scope |
|---|---|---|
| Temperature `θ=D/γ`, Gibbs law, relative generator | Actual positive Gaussian and quadratic-plus-cosine Gibbs densities; independently expanded forward operator, derivative scores, commutators and integration by parts | Conservative continuous kinetic reference |
| `H`, `Ix`, `Iv`, `Ixv`, positive `G`, eigenvalues | Mixed positive relative density, including nonzero position–velocity Hessian; all integrals retained | Density integrals; extensive quantities compared per coordinate or per walker |
| Exact entropy and all three Fisher derivatives | Generator derivative and independent integrated Hessian/commutator formulas, including the full mixed second-derivative quadratic form | 72 nonquadratic density fixtures with wells, saddle perturbations and a genuine linear tail tilt |
| `Cx`, `C0`, global Hessian `M` | Coordinate bounded perturbation and tensorization; source proof gives Rastrigin `Cx ≤ (θ/2) exp(20/θ)` and `M ≤ 2+40π²`, independently of dimension and population | Unbounded separable confining references, retaining the nonconvex barriers |
| `Lm`, `η`, `a=c=2η`, `b=η`, `g+=3η`, sufficient rate `r` | Full parameter grid plus exact matrix/Young bounds and all density generator dissipation comparisons | The actual coefficients use `a=c=2η`, `b=η`; positive determinant retained |
| Exponential modified entropy envelope | 81 kinetic Gaussian density trajectories, with 810,000 RK4 mean updates and invariant covariance; entropy/Fisher functions recomputed at every state | Exact Gaussian reference density family, excluding native cap/cloning |
| `AJ`, jump margin and exponential envelope | Common-invariant Gaussian refresh kernel; positive-rate kinetic jump evolution has an explicit Poisson mixture of Gaussian densities | Six complete time-resolved density laws; never substituted for native sampled-fitness cloning |
| Killing rate, QSD eigenvalue, jump Bregman dissipation and survival normalization | Exact variable-killing two-state generator; complete normalized entropy derivative, killing oscillation and chain rule | Identified full killed model, rather than an assumed native QSD |
| Outgoing boundary loss | Gaussian outgoing velocity trace integral and nonnegative trace entropy term | Boundary functional check; actual QSD domain/trace assumptions remain analytic |
| Selection covariance, Fisher derivative and `K`, `v*`, `SG` bound | Independent finite-difference first variation using strictly positive full matrix `G=I₂`; normalized multiplication `V=1+0.2 sin(x)` | Specified multiplication law; native acceptance law remains its own operator |
| Forcing floor and numerical defect floor | Full scalar solutions and recurrence traces with varied forcing, step sizes and orders | Explicit implication of the stated differential/defect assumptions, without assigning an unproved native defect |
| Discrete QSD cancellation and cubic constant `12` | Exact backward conditional kernel, nonzero survival centering, input/output entropy and independent Taylor bounds | Complete specified substochastic model |
| Survival eigenfunction, Doob minorization, reweighting prefactor and KL contraction | Exact positive left/right eigenvectors, normalized Doob transitions and 64 conditioned updates for four starting laws | The algebra transfers to native gas only with its separately established full-kernel constants |
| `C* L²/N` empirical variance | 24,576 independent complete Gaussian phase-space populations; exact chi-square confidence budget; all raw vectors and normalized means saved | Product Gaussian joint law; these draws are not counted as native engine updates |

The native full-gas experiments are saved separately by the Chapters 10–12 programme. Their bounded observables, status mass and finite coarse-grain diagnostics retain their actual law and independent-seed denominators. An empirical atomic law has infinite KL against a continuous positive density, so no such histogram statistic is presented as continuous-law KL.

Reproduction:

```bash
cargo run --offline --release -p algorithmic-gas-benchmarks \
  --bin gas-chapter10-completion -- EMPTY_REFERENCE_OUTPUT
```

From the repository root:

```bash
uv run --no-sync python algorithmic-gas/proof-validation/chapter10_completion_ledger.py \
  REFERENCE_OUTPUT EMPTY_LEDGER_OUTPUT
uv run --no-sync python algorithmic-gas/proof-validation/chapter10_plot_results.py \
  REFERENCE_OUTPUT LEDGER_OUTPUT
```

Current retained evidence:

- [Rust density/reference experiments](../outputs/convergence/chapters10-12-completion-20261005/chapter10/reference-full-v1/report.json).
- [Per-statement and per-expression table](../outputs/convergence/chapters10-12-completion-20261005/chapter10/ledger-v4/table.md).
- [Machine-readable expression ledger](../outputs/convergence/chapters10-12-completion-20261005/chapter10/ledger-v4/expression-ledger.json).
- [Scientific rate and variance figure](../outputs/convergence/chapters10-12-completion-20261005/chapter10/ledger-v4/rates.pdf).

Quadrature refinement checks are empirical numerical QA. The separately retained Gaussian-envelope tail bounds control the unbounded density outside the numerical window; they do not turn refinement into a validated quadrature interval. Rates are fitted only above the explicitly saved entropy resolution threshold.
