# Fractal Gas symmetry and QFT rigor review

Reviewed 2026-09-05 against the current working files, including uncommitted revisions.

**Verdict:** the volume supports several rigorous finite algebraic constructions and explicitly conditional analytic results. It does **not** establish that the fixed Fractal Gas algorithm generates an interacting Standard Model QFT, or uniquely selects its gauge group. It is suitable as a starting point for an explicitly specified finite field model, after the implementation errors below are addressed. It is not yet a justified specification for extracting physical couplings and particle masses from walker histories.

The detailed review covers `03_lattice_qft.md` and `04_standard_model.md`, the relevant action, continuum, transfer, and reconstruction arguments in `05_yang_mills_noether.md`, and their uses in `01_fractal_set.md`, `06_empirical_validation.md`, `07_qft_calibration_report.md`, and `09_qft_calibration.md`. Selected cloning, electroweak-link, and calibration code was inspected. This is not a certification of every convergence theorem in the volume, a rerun of the experiments, or a complete audit of the dashboard implementation. The published TOC includes the inconsistent downstream chapters.

**1. High priority: calibration promotes assumed correspondences to physical theorems.**

Locations: [calibration report, opening and inversion](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/07_qft_calibration_report.md:3), [channel calibration](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/09_qft_calibration.md:31), [parameter sieve](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/09_qft_calibration.md:1238).

The report says no new assumptions are introduced and promises parameters reproducing measured couplings. Chapter 09 calls its formulas coupling identities and uses them to eliminate parameter sets. In contrast, [the actual coupling definitions](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md:905) explicitly define dimensionless proxies, and [the matching proposition](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md:1045) requires canonical normalization and matching of observables or effective-action coefficients.

The gap hierarchy used in the sieve is also absent from the cited theorem: [the current mass-scale result](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md:1055) says that the parameters impose **no ordering** of the four scales. The report's square-root bandwidth prescription is not the conclusion of [the current time-consistency theorem](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md:1200).

**Required correction:** label these as conditional proxy-calibration ansätze; state units, the precise observation law, and every matching assumption. Remove unconditional physical rejection rules. Treat hierarchy restrictions as optional experiment constraints unless proved for the actual parameter family. The diversity moment depends on the bandwidth and the particle law; inversion is an implicit self-consistency problem, with neither existence nor uniqueness established by the displayed square root.

**2. High priority: the channel-gap proof does not bound finite-time effective masses.**

Location: [Spectral-Gap Lower Bound for Channel Masses](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/09_qft_calibration.md:1158), and its ratio-sieve descendants.

An envelope \(|C(t)|\le A e^{-\lambda t}\) does not imply
\(-h^{-1}\log[C(t+h)/C(t)]\ge\lambda\), even when both correlator values are positive. One cannot divide two upper bounds to obtain that ratio inequality.

For an explicit stationary covariance, take independent ordinary and rotating Ornstein–Uhlenbeck processes so that

\[
C(t)=e^{-|t|}(1+\tfrac12\cos t).
\]

It is positive, obeys \(C(t)\le1.5e^{-t}\) for \(t\ge0\), and comes from a stationary Gaussian process with centered semigroup contraction rate 1. At \(t=3\pi/2\), \(h=0.1\), its effective rate is approximately **0.512892**. This refutes the envelope-to-log-ratio inference. It does not refute the stronger reversible spectral result below.

A sufficient repair is an actual positive spectral representation

\[
C(t)=\int_{[\lambda,\infty)}e^{-Et}\,d\nu(E),\qquad d\nu\ge0,
\quad 0<C(t)<\infty.
\]

Then \(C(t+h)\le e^{-\lambda h}C(t)\), which proves the claimed effective-rate bound. A centered autocorrelator \(\langle f,e^{-tH}f\rangle\) for the **same** nonnegative self-adjoint transfer Hamiltonian supplies this representation. An arbitrary cross-correlator, fitted noisy data, or nonreversible sampler does not. The relevant hypotheses of the cited reversible theorem must be carried into the statement. Rate-to-energy conversion additionally multiplies by the calibrated action unit; mass is energy divided by \(c^2\).

**3. High priority: nonabelian triangle factorization omits the basepoint transport.**

Location: [Plaquette Wilson Loop Factorization](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/01_fractal_set.md:1215).

The boundary-chain identity justifies cancellation in an abelian holonomy. It does not justify multiplying nonabelian matrices based at different vertices. Using the later chapters' comparison convention \(U_{ij}:V_j\to V_i\), consider triangles \((a,b,c,a)\) and \((c,d,a,c)\):

\[
T_a=U_{ab}U_{bc}U_{ca},\qquad
T_c=U_{cd}U_{da}U_{ac}.
\]

The correct outer holonomy is

\[
U_{ab}U_{bc}U_{cd}U_{da}
=T_a\bigl(U_{ac}T_cU_{ca}\bigr).
\]

Set \(U_{ab}=U_{cd}=i\sigma_x\), \(U_{ca}=i\sigma_z\), and the other independent links to the identity. All links belong to \(SU(2)\). The untransported product has normalized trace **+1**; the true outer loop has normalized trace **−1**. The corrected product agrees exactly with the outer loop. The construction also permits identity CST links.

**Required correction:** state the basepoints and insert the conjugation, or explicitly define already-rebased triangle holonomies. Do not multiply triangle traces to get a plaquette trace. Chapters 03 and 05 already explain this restriction; it must propagate back into Chapter 01 and any implementation based on it.

**4. High priority for code reuse: the implemented “SU(2) gauge link” is a scalar proxy.**

Location: [compute_su2_gauge_link](/home/guillem/fragile/src/fragile/physics/electroweak/electroweak_spinors.py:89).

It returns \(u_{ij}=\exp[i\pi|F_j-F_i|/(2h(|F_j-F_i|+\varepsilon))]\). This has scalar output shape and satisfies \(u_{ji}=u_{ij}\), generally not \(u_{ji}=u_{ij}^{-1}\). Promoting it to \(uI_2\) gives determinant \(u^2\), generally not one. At \(h=1\), \(|F_j-F_i|=\varepsilon=1\), the determinant is \(i\). Chapter 04 [already identifies this limitation](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md:835), but the function docstring still describes a gauge link.

**Required correction for a gauge simulator:** retain or rename this as a scalar diagnostic. Introduce separate matrix-valued links with shape `[..., 2, 2]`, determinant one, unitarity, endpoint covariance, and inverse reversal. A chosen lift \(\exp(i\theta n^a\sigma^a/2)\) meets group membership when its coefficients are real, but deriving its distribution remains a separate task. A scalar angle alone does not determine the three gauge directions.

**5. Medium priority: hypercharge normalization differs between theory and calibration code.**

Locations: [report's coupling conversion](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/07_qft_calibration_report.md:37), [implemented conversion](/home/guillem/fragile/src/experiments/calibrate_fractal_gas_qft.py:100), [one-loop normalization](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md:1068).

The code/report use \(g_1=e/\cos\theta_W\), whereas Chapter 04 uses \(g_Y=e/\cos\theta_W\) and \(g_1=\sqrt{5/3}\,g_Y\). These are valid alternative conventions only if all charges, kinetic coefficients, beta functions, and proxy matching use the same one. They cannot be interchanged under the shared name `g1`.

For fixed \(\hbar_{\rm eff}\) and fixed diversity moment, using \(g_Y\) instead of the chapter's normalized \(g_1\) increases the inferred \(\epsilon_d\) by \(\sqrt{5/3}\approx1.291\). This factor describes the fixed-moment formula, not the solution of a recalibrated self-consistent particle law. Use explicit `g_y` and `g1_gut` names and record the convention in outputs.

**6. Medium priority: the asserted massive-scalar correlation law is incorrect.**

Location: [Two-Point Correlator definition](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/06_empirical_validation.md:131).

The chapter calls \(G_0e^{-r^2/\xi^2}\), \(\xi=1/m\), the massive Euclidean scalar prediction. A free massive scalar Green function instead solves \((-\Delta+m^2)G=\delta\). In three Euclidean dimensions, the decaying radial solution away from the source is \(G(r)=e^{-mr}/(4\pi r)\); in general its large-distance dependence includes \(e^{-mr}\) and a dimension-dependent power. Substituting the Gaussian into the differential equation leaves an \(r^2\)-dependent term, so it cannot be that Green function. Gaussian smoothing of empirical density can itself produce Gaussian spatial correlations.

**Required correction:** report the Gaussian fit as an empirical statistic unless a distinct model predicts it. A high fit score establishes neither a relativistic mass nor the field measure. In an inhomogeneous confining ensemble the connected subtraction is \(\mathbb E[\phi(x)\phi(y)]-\mathbb E\phi(x)\mathbb E\phi(y)\); a common global mean squared requires an additional homogeneity or centering justification. The same chapter also calls a QSD stationary without specifying survival conditioning or the Doob-transformed process.

**7. Medium priority: clock relabeling is confused with changing the simulation or recording interval.**

Location: [Ratio Invariance Under Time Rescaling and its consequence](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/09_qft_calibration.md:1118).

Multiplying only the time labels of a fixed correlation array by \(s\) divides its effective rates by \(s\). Changing the integrator timestep changes the transition kernel. Changing `record_every` changes sampled lags and finite-time contamination. Neither operation is automatically a relabeling.

For \(C_1(t)=e^{-t}+e^{-3t}\), \(C_2(t)=e^{-2t}\), the ratio of their effective rates measured from zero is approximately 0.783110 at interval 1 and 0.668749 at interval 2. Both correlators have positive spectral representations. Their asymptotic spectral thresholds are unchanged, while their finite-lag estimators differ. Limit the theorem to a fixed array and require convergence studies for actual timestep or recording changes.

**What the symmetry derivations establish.**

The present Chapter 04 is substantially more careful than its downstream uses. Its finite calculations distinguish the following facts correctly:

| Construction | Established conclusion | Missing conclusion for algorithmic emergence |
|---|---|---|
| Companion square-root amplitudes | Normalization and rephasing freedom | Local symmetry of the complete stochastic update and a nontrivial connection law |
| Chosen complex doublet and determinant form | An SU(2) frame action | Selection of this internal representation and physical weak dynamics |
| Isotropic real viscous force | Conditional orthogonal covariance | Internal SU(3) covariance; the componentwise phased encoding is not equivariant |
| Independent chosen internal factors | A product representation | A uniqueness principle selecting these factors and ranks |
| Weighted cloning scores | Exact antisymmetry and opposing eligibility | CAR, Pauli statistics, or fermionic particle correlators |
| Exterior algebra | Explicit CAR/Clifford representation | Spin geometry, a consistent Dirac discretization, and the particle-to-field map |
| Supplied Higgs doublet and canonical action | Classical Higgs mass matrix | Derivation of that action from the fixed algorithm, and quantum masses |

The [Spin(10) branching calculation](/home/guillem/fragile/docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md:676) explicitly supplies the familiar sixteen left-handed states, including a neutral singlet. Its exterior-power decomposition and kernel calculation are sound as representation statements. The faithful group on this content is
\([SU(3)\times SU(2)\times U(1)]/\mathbb Z_6\), with integer charge \(q=6Y\) for the displayed U(1) parameterization. This agrees with the mathematical organization in [Baez and Huerta](https://arxiv.org/abs/0904.1556). It does not infer this field content from the walker state space. Generation multiplicity remains a choice.

**A missing but checkable chiral consistency calculation.**

For the chosen left-handed generation, conventional perturbative anomaly sums do cancel:

\[
\begin{aligned}
SU(3)^3 &: 2-1-1=0,\\
SU(3)^2Y &: \tfrac12[2(1/6)-2/3+1/3]=0,\\
SU(2)^2Y &: \tfrac12[3(1/6)-1/2]=0,\\
Y^3 &: 6(1/6)^3+3(-2/3)^3+3(1/3)^3+2(-1/2)^3+1=0,\\
\mathrm{grav}^2Y &: 6(1/6)+3(-2/3)+3(1/3)+2(-1/2)+1=0.
\end{aligned}
\]

There are four weak doublets per generation, so the conventional odd-doublet SU(2) obstruction is absent. See [Witten's original anomaly result](https://www.sciencedirect.com/science/article/abs/pii/0370269382907286). The neutral singlet contributes zero to these sums. Mixed anomalies with one traceless nonabelian generator vanish by its trace; SU(2) has no perturbative cubic anomaly for this content.

These checks are necessary and useful, but they do not construct a regulated chiral fermion measure on the recorded complex or settle all global bundle choices. A vectorlike Berezin Jacobian calculation cannot replace that task. [Lüscher's abelian nonperturbative construction](https://arxiv.org/abs/hep-lat/9811032) and [the nonabelian perturbative construction](https://arxiv.org/abs/hep-lat/0006014) have explicit regulator and anomaly hypotheses; neither applies automatically to the score-based graph operator. Select a regulator, verify its locality and unwanted-mode control, and prove the required gauge-measure properties for the selected geometry.

**The remaining dynamical proof obligations.**

1. **Identify the law.** Specify the full fixed particle kernel, initial/survival law, and reconstruction map. Prove that its pushforward is the target field law, or derive an estimator with a justified change of measure and controlled variance. Node-difference phase links have identically trivial holonomy; their law cannot reproduce a finite-coupling Haar/Wilson model with fluctuating loop flux. A singular pushforward cannot be repaired merely by declaring ordinary importance weights.
2. **Identify quantum time.** Establish a positive transfer construction for the actual field correlations. A stochastic simulation clock is not automatically Euclidean physical time. Chapter 05's reversible factorization theorem is valid under its hypotheses, but its rotating-Gaussian counterexample shows why a QSD or static LSI is insufficient. Relativistic reconstruction additionally needs the full covariance and growth conditions of [Osterwalder–Schrader II](https://doi.org/10.1007/BF01608978).
3. **Control geometry and the joint limit.** Verify actual face orientation coverage, area weights, spin transport, same-sample quadrature, shrinking bandwidth errors, boundary conditions, and the applicable joint-law estimates. Pointwise scalar consistency and the Wilson expansion on fixed smooth test connections do not prove convergence of fluctuating quantum measures. Retain confining moment/tail estimates on unbounded domains.
4. **Obtain nontrivial fields.** A law-of-large-numbers limit of intensive empirical densities factorizes. A fluctuation scaling needs its own limit theorem and is not automatically an interacting quantum theory. Prove survival of nonzero connected observables and the intended matter content.
5. **Match physical parameters.** Fix action and charge normalization, a renormalization scheme, an observable channel, and a time/length calibration before interpreting a decay threshold as a particle mass. The conditional self-adjoint gap theorem does not prove a physical Yang–Mills mass gap. The distinction between classical actions and the interacting quantum construction is explicit in [Jaffe and Witten's problem formulation](https://www.claymath.org/wp-content/uploads/2022/06/yangmills.pdf).

**A defensible implementation sequence.**

| Stage | Implementable object | Acceptance evidence |
|---|---|---|
| 1 | Finite pure-gauge field model on a declared complex | Oriented independent group links; inverse reversal; local covariance; correct based loop products; finite Haar/Wilson measure |
| 2 | A sampler for that explicitly chosen measure | Invariance or a proved weighted estimator; convergence conditions; small-system reference integrals; uncertainty accounting |
| 3 | Scalar and vectorlike matter benchmarks | Coercive scalar action; declared fermion operator and boundary conditions; determinant/sign treatment; known free-field spectra and controlled discretization errors |
| 4 | Chiral Standard Model content | Explicit integer charges and global group; anomaly checks; suitable regulator and fermion measure; spin transport and unwanted-mode control |
| 5 | Fractal Gas equivalence and physical predictions | Pushforward or estimator theorem; transfer/reconstruction proof where claimed; controlled limits and canonical observable matching |

Finite-model simulation does not require first proving a four-dimensional continuum existence theorem. It does require stating the finite target precisely. Stage 1 can reuse the proved finite gauge formulas after the loop correction. The Fractal Gas could eventually serve as a proposal or geometry generator if its connection to the target is justified; running it on a fitness landscape alone does not establish target sampling. Physical particle names should remain annotations of unvalidated channels until the spectral identification is supplied.

**Reproducible verification and changes made.**

[qft_rigor_checks.py](/home/guillem/fragile/fractal-gas-web/reviews/qft_rigor_checks.py) checks the nonabelian counterexample, envelope counterexample, recording-interval example, scalar-link defects, fixed-moment normalization factor, exact rational anomaly sums, score identity, and pure-gauge telescoping. All assertions passed using:

```text
UV_CACHE_DIR=/tmp/fragile-qft-review-uv uv run --no-project --python /home/guillem/fragile/.venv/bin/python python reviews/qft_rigor_checks.py
```

The checks evaluate the displayed formulas; they do not run the Torch pipeline or establish analytic convergence. Only this review and its companion calculation script were added. The book and simulator source were not modified.
