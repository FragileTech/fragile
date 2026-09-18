# Euclidean Gas mean-field derivation audit

This engineering record identifies the proof changes and implementation checks.
The lecture chapters present one algorithm and its current mathematical results.

## Provenance and identified errors

The two mean-field chapters were substantially rewritten in commit `cad50b91`
(September 5). Their immediately preceding text is available in commit
`b416d3a98112b360e6ae7f6655fb47497916cfcb` under
`docs/source/3_fractal_gas/convergence_program/`. Neither chapter was changed
by the subsequent Rust-demo commits through `aac2ee7a`.

| Claim or operation | Defective inference | Repair |
|---|---|---|
| Continuous cloning generator | The earlier derivation both multiplied per-step acceptance by the timestep and divided the same acceptance by the timestep to define a rate. A finite-difference identity was identified with a differential equation. | Derive the exact nonlinear fixed-step map. Differential limits must follow from its iterates with the stated parameters. |
| Immediate revival | A freely specified finite revival rate leaves positive reservoir mass between attempts. | Revive every dead recipient at the scheduled stage using its actual current-donor law, retained coordinates, jitter, and component collision. |
| Exchangeability implies deterministic empirical convergence | A symmetric mixture was treated as independent samples from its first marginal. | Prove one-step empirical concentration by measurement normalization and two-root graph exploration, then iterate. Preserve random directing laws for correlated initial data. |
| Mean-field theorem application | The fixed-step propagation theorem assumed its one-step consistency estimate. | Construct the marked rooted-component law and prove consistency for the canonical kernel. |
| Independent group rotations conserve momentum | Independent rotations of zero-sum vectors need not retain their zero sum. | Apply one shared Haar orthogonal matrix to the whole component. |
| Overlapping donor groups | Multiple groups prescribed distinct outputs for the same slot; the Python implementation overwrote by donor order. | The user selected disjoint connected components of the accepted graph and simultaneous frozen-input updates. |
| Independent collision outputs | Component members share their center of mass and rotation. | Retain all cross-walker covariance terms and prove component exploration bounds. |
| Compact feature geometry | Bounded comparison features were conflated with bounded physical coordinates. | Use a bounded squashed phase-space distance while retaining physical moments and tails. |
| Stationary uniqueness | A contraction argument for an independently specified finite-rate reaction model was used as the proposed algorithm's stationary machinery. | Prove stationary existence for the actual bounded-domain map; identify the missing attraction/concentration estimate precisely instead of transferring the other model's uniqueness. |

## Provenance of the stationary-concentration argument

The pre–September 5 text **did contain proofs claiming stationary concentration
and an N-uniform QSD LSI**. Calling this merely an absent theorem fails to
describe the repair obligation. The relevant source is commit
`b416d3a98112b360e6ae7f6655fb47497916cfcb`, under
`docs/source/3_fractal_gas/convergence_program/`:

- `12_qsd_exchangeability_theory.md:405`,
  `thm-mixing-variance-corrected`, claims empirical concentration using the
  product-relative-entropy estimate from Chapter 13.
- `13_quantitative_error_bounds.md:187`, `lem-quantitative-kl-bound`, claims
  total joint entropy `H_N <= C_int/N`. Its Step 2 invokes an LSI relative to
  the QSD to control entropy relative to the product mean-field law. These
  are different reference laws; the required dissipation comparison is not
  supplied. Step 3 asserts an `O(1/N)` interaction remainder. The subsequent
  proposition at line 312 says exchangeability cancels the leading term, but
  does not derive that cancellation. The displayed remainder is an unnormalized
  sum over N recipients. Applying its stated Lipschitz and diameter bounds
  directly gives an O(N) upper bound, not the asserted O(1/N) estimate.
  This observation identifies a missing cancellation argument; it does not
  assert that the correctly derived remainder must grow like N.
- `15_kl_convergence.md:2121`, `cor-n-uniform-lsi`, explicitly claims an
  N-uniform inequality for the full gas. The primary route uses cloning
  contraction and smoothing assertions. The alternative cloning curvature
  argument at line 7712 infers pointwise gradient bounds from bounded fitness
  and Wasserstein contraction without deriving the kernel derivatives.
  In particular, its use of Kantorovich–Rubinstein duality to obtain a
  sup-norm contraction is not the Lipschitz-seminorm conclusion of that
  duality.
- `10_kl_hypocoercive.md:606` retains a positive square-root gradient term
  `a sqrt(D)` in the cloning estimate. The combined decay assertion at lines
  646–651 then omits this term. A bound `a sqrt(D) <= C D` cannot hold near
  zero when a is positive. A cancellation in the full stationary calculation
  or a different estimate is necessary. That chapter also uses the explicit
  finite-rate growth/death operator at line 527; identifying it with the
  complete simultaneous cloning update is a separate missing step.

The finite-population identity in the older Chapter 12 also drops the diagonal
term when equating empirical variance with distinct-particle covariance.
The exact identity is
`Var(L_N g) = Var(g(Z_1))/N + (N-1) Cov(g(Z_1),g(Z_2))/N`.
This algebraic defect is locally repairable and does not invalidate the
entropy-to-concentration strategy.

Commit `cad50b91` replaced the earlier Chapter 12/13/15 applications with
conditional estimates. These three files have no subsequent committed changes
through HEAD and no changes in this repair. The current repair rewrites
Chapter 09 and removes its finite-rate-model attraction argument; it retains
the joint-LSI/Poincaré and product-entropy routes to stationary concentration.

The reporting error was to describe concentration as an outstanding obligation
without first acknowledging and tracing these existing claimed proofs. The
mathematical obligation is to repair their defective application steps for the
actual full transition. The evidence above does not show that stationary
concentration is false, or justify discarding the valid functional-inequality
and empirical-measure arguments.

## Executed algorithm and derived quantities

`GasConfig::euclidean` specifies independent current Gaussian-weighted donor
roles, regularized global fitness, frozen component cloning, Gaussian BAOAB,
independent final position diffusion, final radial velocity capping, and terminal
absorption. The force provider supplies the declared potential gradient.
Library extensions have explicit configurations and do not acquire this
canonical proof merely by using the same engine.

Canonical revival uses the cloning companion law, including its dependence on
the dead recipient's retained coordinates. Its acceptance probability is one.
All accepted recipients receive the configured position jitter. Revived dead
velocities participate in the frozen component center of mass. Component
momentum conservation concerns all slots; the alive-only balance retains
revival and boundary source terms.

For frozen measurement marks, accepted live edges strictly increase fitness.
Each recipient has at most one outgoing edge and dead vertices cannot be
targets. The graph is therefore a forest. Bounded comparison weights and a
positive alive fraction give an edge probability bound `C/N`. Counting the
monotone legs of a tagged path yields

\[
\mathbb E|C_N(i)|\le e^{2C},\qquad
\Pr(\operatorname{rad}(C_N(i))\ge r)\le (2C)^r/r!.
\]

The incoming Poisson point process in the rooted population construction is
the limit of rare **label hits as population increases**. It does not randomize
the update clock, add cloning attempts, or introduce a continuous-time rate.
Incoming children already have their outgoing edge fixed to their parent.

The terminal independent Gaussian position step supplies a positive alive-mass
bound and exponentially small finite-population failure probability. This proof
uses the actual terminal-only boundary schedule. Substep absorption is a
different explicitly selectable execution schedule.

## Validation contract

Exact tests check the component graph, frozen inputs, transported relabelings,
Haar moments, momentum, restitution energy, revival, and kinetic stage order.
Conditional field tests integrate component rotations with independent
quadrature and retain within-component covariance.

`mean_field_validation` runs actual trajectories at fixed timestep, recording
separate gate-count, component-stress, and final-position martingale residuals
and their respective conditional variances. Population-size comparisons use
independent complete trajectories and bounded full-slot observables. Their
finite-population differences are not labeled deterministic discretization bias.

The canonical study uses `N=16,32,64,128,256`, 128 independent seeds and
observations at updates `1,4,16,64`. Boundary-stress and shared-initial-law
controls test actual revival and the distinction between chaos and empirical
mixtures. Further seed batches are recorded separately.

## Upstream drift and stationary proof repairs

The universal negative positional cloning drift is false for the canonical
configuration itself. For four walkers at `(0,0,0,0.1)` with zero velocities,
exact integration over every measurement pattern and gate gives positional
variance `0.00297011744615149` after cloning, including jitter, from input
variance `0.001875`. The independent Rust test agrees within one standard error.
The error was the inference from an upper bound on a fitness-weighted coupling
quantity to a negative physical displacement drift. Chapter 03 now derives the
actual row-law variance, positional and structural reset estimates, and an
explicit global Foster moment inequality for the complete update.

Finite-N QSD convergence has a direct proof for the actual quadratic-force
kernel. Alive positions, bounded dead-position features and capped velocities
form an exact compact input representation; finite dead coordinates are
recoverable and full physical outputs remain unbounded. Conditional BAOAB and
final position noise produce nondegenerate Gaussian phase-space densities for
`h != 2`. The canonical `h = 0.04` satisfies this condition. Total-variation
continuity gives a compact positive operator, its positive eigenfunction defines
a Doob kernel, and an explicit common target density supplies minorization.
This proves unique finite-N QSD and geometric conditioned convergence. It does
not rely on negative positional cloning drift. At `h = 2`, the actual quadratic
BAOAB velocity degeneracy is derived and covered by an operator regression.

## Remaining mathematical obligation

Finite-horizon chaos, unique finite-N QSDs, and existence of a nonlinear
stationary law do not imply uniqueness or global attraction of the nonlinear
population law, or its identification with concentrated finite-swarm QSDs.
Those conclusions require a dissipative/coupling estimate for the actual
component map or a verified empirical concentration estimate for its QSDs.
Positive Gaussian smoothing alone supplies neither estimate. The repair must
not report this obligation as completed by a theorem for another process.

## Proof dependency map

| Result | Verified input from the actual transition | Replacement proof |
|---|---|---|
| Sampled fitness law | Independent weighted measurement companions; regularized empirical mean/variance; fitness retained through the gate | `lem-chaos-sampled-marks` proves marked empirical convergence and handles self-exclusion before acceptance. |
| Finite collision neighborhoods | Frozen strict fitness ordering, one outgoing accepted edge, weighted mandatory revival | `lem-chaos-component-truncation` proves the forest and factorial path bounds. |
| Rooted population law | Incoming label hits, outgoing assignments, sampled fitness and one component rotation | `thm-chaos-rooted-collision-limit` proves one-root convergence and two-root independence. Incoming children retain their already-used outgoing edge. |
| Stability under changed innovations | Recomputed global statistics, changed gate probabilities and merged/split components | `lem-chaos-canonical-innovation-replacement` bounds the actual replacement influence; it is not misused as a sharp variance theorem. |
| Full-step consistency | Rooted collision limit followed by actual BAOAB, final Gaussian position noise, smooth cap and terminal marking | `thm-chaos-canonical-one-step` proves empirical L2 consistency; `lem-chaos-canonical-map-continuity` supplies its continuity modulus. |
| Finite-horizon chaos | Kernel equivariance, one-step consistency, positive alive mass and propagated moments | `thm-chaos-finite-time-consistency` iterates the actual map and derives fixed-marginal chaos by sampling without replacement. |
| Position and full-state control | Frozen eligible position copies, accepted-row jitter, component velocity bound, complete kinetic noise | `thm-positional-variance-contraction` proves the reset estimate; `thm-canonical-full-step-reset-drift` proves a global Foster moment bound. |
| Finite-N QSD convergence | Exact state-coordinate conjugacy, quadratic BAOAB phase-space density and terminal survival | `thm-chaos-canonical-finite-n-qsd` verifies compactness, positive eigenfunction and Doob minorization for the executed kernel. |
| Resonant kinetic behavior | N=1, quadratic force, h=2, no accepted self-clone, unchanged full BAOAB schedule | `rem-chaos-canonical-baoab-resonance` derives the deterministic capped sign flip and the failure of total-variation convergence from nonzero velocity. |
| Stationary population limits | Actual discrete QSD identity, vanishing extinction probability, tightness and full-step consistency | Chapter09 identifies invariant distributions over population laws. Collapsing such a distribution to one deterministic fixed point still requires a population-level concentration or attraction proof. |

The state-coordinate conjugacy in the finite-N proof is explicit: each original
physical dead position is recovered by the inverse feature map, and coupled
primitive innovations give the same complete transition and extinction event.
No frozen companion, collision member, shared rotation, retained velocity,
kinetic stage, or boundary contribution is dropped in this representation.


## Recovery implementation and reviewed repairs

The baseline has been recovered into `proof-recovery/pre-september-5.tar.gz`,
with content hashes and 805 label locations in `proof-recovery/manifest.json`.
This contains 35 source files, including the standalone proofs. The archive
is reproducible with `tools/recover_mean_field_proofs.py`. The label index is
an inventory, not a claim that all 805 labels have been independently proved.

The original geometric fitness/error clusters remain the collective proof
strategy. Accepted collision components are a different partition determined
by the executed graph. Their moment bounds below are auxiliary estimates
for fresh random innovations and finite-population consistency.

| Intended claim and recovered hypotheses | Dependency or defective inference | Implemented replacement and review |
|---|---|---|
| Actual population evolution and empirical consistency, original 08/09 with regularized fitness and companion kernels | Shared sampled fitness, global normalizers, incoming recipients, self-exclusion and shared rotation must survive the population passage. | 09 now proves all collision-component moments uniformly in N, squared innovation influence, conditional variance O(1/N), and bounded-observable conditional bias O(1/sqrt(N)) against the actual map at its empirical input. Parent independently reviewed the marked exploration and constants. |
| Collective cloning control through the geometric error clusters, original 03/04 | Within-swarm variance proxies can be positive on identical coupled inputs; their affine drift does not control an incremental discrepancy at zero. | 03 `thm-cloning-incremental-cluster-balance` gives exact signed within/between-cluster positional fluxes, gate-jitter mismatch cost, and the full correlated velocity discrepancy. Both formulas vanish on identical inputs/plans. Stationary agent independently reviewed the algebra and coupling. |
| Kinetic contraction, original 05 under confining-force hypotheses | An additive drift offset was used as though it were a strict law-distance contraction. General confinement and the quadratic special case must be distinguished. | 05 proves exact discrete quadratic BAOAB/cap contraction with explicit positive constants and a separate actual terminal-mark coupling bound. Two independent reviews and Rust execution checks passed. This is the quadratic-stage application, not a replacement theorem for every confining potential. |
| Full entropy balance and quantitative joint KL, original 13/15 | QSD and product-reference entropies were interchanged; the interaction remainder cancellation was asserted. | 15 derives the exact survivor chain rule; 13 iterates it with every backward, interaction, and survival contribution retained. Product-reference extinction conditioning is explicit. Population agent independently verified signs, normalization and reference changes. |
| Hypocoercive forcing cancellation, original 10/15 | A positive square-root term was omitted from the final decay estimate. | 10 proves exact first-order cancellation at the actual QSD and bounds the cubic entropy remainder by 12 times the cubed perturbation amplitude. Global finite-N entropy decay is proved through the same full kernel's Doob conjugacy and verified minorization. Parent and population agent reviewed; independent finite-kernel algebra tests passed. |
| Cluster-based control of output entropy | Conditional component outputs were factored into independent walkers, and the kinetic reference law was substituted for the full-law target. | 15 proves the actual Gaussian channel entropy cost and decomposes it exactly over a common refinement of the original geometric error clusters. Cap, status marking and survival conditioning are included by exact entropy inequalities. Shared-rotation multi-information is retained. Parent independently reviewed the Gaussian and survival calculation. |
| Stationary empirical concentration and deterministic QSD limit | Finite-N convergence, per-update concentration and a one-sided variance proxy do not provide the small-error collective contraction. | 09 now derives the exact survivor-tilted QSD variance budget with O(1/N) fresh innovation term. The remaining estimate is the signed, incremental geometric-cluster contribution across the full update. It is not replaced by an assumed contraction or a standard counting argument. |
| Quantitative observable error, original 12/13 | The finite-population covariance identity dropped its diagonal term; entropy reference normalization was inconsistent. | The exact diagonal identity is retained; 13 proves the bounded-observable mean-square bound directly from total product entropy, including marked statuses and extinction conditioning. Independent review passed. |
| Dimension-independent empirical W2 squared error C/N, original 13 without a dimension restriction | This empirical-distribution rate conflicts with deterministic finite-support resolution in dimensions greater than two. | 13 proves W2 squared at least c N^(-2/d) for every N-point output compared with any actual stationary population law, using its actual final Gaussian convolution. Thus that recovered rate is false for d>2. The bounded-observable rate is a separate claim and is not contradicted. Independent review passed. |
| Strong diffusion closes the recovered fixed-point contraction | The original resolvent estimate asserted decreasing constants while retaining a fixed bottleneck, and self-mapping was inferred from fixed-point bounds. | 09 records an exact bounded output observable retaining an OU-amplitude-independent input component. It prevents an invalid one-step noise-washout shortcut. A proof of the original small-error cluster closure remains required; no parameter change or new assumption was inserted to certify it. |

### Keystone-specific baseline review

An independent review of baseline commit
`b416d3a98112b360e6ae7f6655fb47497916cfcb` confirms that the Keystone lemma
already states the intended N-uniform collective selection-pressure
estimate. This review does not certify all of its upstream probability
estimates. In baseline `03_cloning.md`, lines 5050–5065 state
`N^{-1} sum_i (p_{1,i}+p_{2,i}) |Delta delta_{x,i}|^2 >= chi V_struct-g_max`.
Lines 5079–5089 cover the low-error regime by choosing
`g_max >= chi R_spread^2`, making the lower bound nonpositive there. Thus the
lemma must not be described as lacking its collective N-uniform mechanism.

The separate defective inference is in its drift application, baseline
lines 6976–6984: copying a donor position and the triangle inequality are
asserted to give a negative term
`-p_{k,i}|Delta delta_{x,i}|^2/4` in a walker's within-swarm variance drift,
with only an `O(sigma_x^2)` remainder. The proof does not derive this sign
from the actual donor distribution and moving barycenter. Baseline line
7218 additionally invokes `V_struct >= c_struct V_Var,x`, whereas the cited
lemma at lines 674–688 supplies an upper comparison. Identical spread swarms
have zero structural discrepancy and a positive variance proxy, so that
reverse comparison cannot hold generally. Baseline `04_wasserstein_contraction.md`,
lines 493–521, itself distinguishes proxy control from closed discrepancy
drift and notes that structural dominance fails near alignment.

These observations identify application errors; they do not show that the
cluster strategy cannot close. Global strict squared-distance contraction is
not a necessary condition for stationary convergence: a verified local
coupling or entropy argument combined with the collective drift is another
possible closure. Its constants must be checked for population uniformity.
Likewise, the measured positive positional drift of the four-walker example
does not by itself refute an affine bound with a positive jitter/Keystone
offset. That example refutes monotone variance decrease, and must not be
presented as a contradiction of every offset-bearing Keystone conclusion.

### Implemented repair of the Keystone application

The individual reset inequality is false even at a positive-probability
measurement event of the canonical gas. For positions `(-a,0,a)`, the two
endpoints can both measure the center. The retained diversity measurements
then coincide, the center has the largest fitness, and it never clones.
Nevertheless its expected squared centered output is
`[2 a² q(1-q)+2 j² q]/9 > 0`, because the other rows move the barycenter.
A remainder multiplied only by the center's own cloning probability is
zero and cannot cover this term. This is a contradiction of the specific
row inference, not of the collective strategy: the exact collective drift
on the same event is `-2 a² q(4-q)/9+4 q j²/9`, which is negative at the
tested canonical parameters.

The replacement arguments are now in the chapters:

| Location | Proved replacement | Independent review and application |
|---|---|---|
| 03, `lem-cloning-individual-centered-displacement` | Exact centered row identity, including deterministic barycenter movement and the stochastic barycenter variance. | Parent and flux agent verified; independent donor/gate enumeration checks the identity. |
| 03, `lem-keystone-contraction-alive` and `thm-cloning-signed-cluster-fitness-flux` | Signed incoming/outgoing donor flux in the same geometric clusters. Both directions are bounded from retained fitness gaps and overlap, including the actual weighted gap covariance and row normalizers. Lower bounds remain valid when negative. | Parent and closure agent derived the inequalities; flux agent independently reviewed every sign and normalization. No inward orientation is assumed. |
| 03, `thm-cloning-canonical-barycenter-concentration` | The previously separated outer-fitness variance is bounded by `k Dx² B0²/(2N²)`. Total positional barycenter variance is `C/N` for every nonempty alive pool; full-slot velocity barycenter is deterministic through cloning. | All canonical constants are discharged; parent derivation and independent flux review agree. No alive-fraction floor or component counting is used. |
| 03, `thm-cloning-revival-backbone-coupling` | Rotations are coupled through unchanged alive backbones. One changed revival donor has exact total velocity discrepancy `|U|+|W|+|U-W| <= (6+8 alpha) V`, with zero momentum discrepancy. | Closure agent derivation, parent and flux reviews, and 63 production-operator cases through N=1024 verify cancellation of component size. |
| 05, `thm-kinetic-bounded-transport-smoothing` | Translated OU coupling matches the actual intermediate positions and final force evaluations. Averaged cap derivatives give a bounded marked-transport estimate proportional to `1/B`. | General globally Lipschitz force proof independently reviewed, including the nonconvex case. Its global force and timestep requirements are explicitly verified for the implemented quadratic and Rastrigin potentials. A force bound only inside the valid domain is not silently extended outside it. |
| 05, `cor-kinetic-full-cluster-smoothing`; 03, `cor-cloning-revival-full-step-cost` | Conditional application after the full correlated collision law, with normalized geometric-cluster errors. M changed revival donors cost at most `[Ax Dx+Av(6+8 alpha)V] E M/(Nq)`. | No independence of collision outputs is assumed. Actual kinetic tests include matched terminal exits and nonconvex final-force evaluations. |

The three-walker checks execute 8,192 full Rust steps in two disjoint seed
batches. Another 640 full steps cover N=16,32,64,128,256 with one alive donor
and an entire-population collision component. The latter verify the exact
positional barycenter variance `j²(N-1)/N²`, with all estimates within 1.93
Gaussian variance standard errors, and full-slot velocity-mean conservation
to `1.91e-17`. The kinetic coupling checks execute 36,864 paired row
comparisons across 18 configurations. These are separate studies; their
errors are not pooled as independent samples across reused seeds.

This repair does not assert that a large OU amplitude by itself closes the
stationary theorem. Although the unconditioned transport coefficient
decreases as `1/B`, normalizing the alive donor law can introduce
B-dependent constants. The signed cluster-flow lower bound likewise
quantifies adverse flow rather than asserting it absent. The unresolved
stationary step is the combined estimate for changes of accepted alive
backbones, signed cluster drift and alive-law normalization with a verified
population-uniform margin. Revival leaf changes and the cloning barycenter
variance are now proved inputs to that step, not assumed remainders.

### Actual measurement-averaged Keystone repair

The target probability must be averaged with respect to the algorithm's
retained measurement law. Requiring a positive acceptance floor on every
realized fitness vector excludes legitimate tie outcomes. The repair now
derives a probability bound before imposing any favorable target condition:

| Claim | Replacement derivation | Review and scope |
|---|---|---|
| Actual near/far measurement probabilities | `lem-keystone-geometric-measurement-events` derives the far mass from feature variance and the near mass from the same geometric cluster, retaining weighted self-exclusion. | Parent and stationary agent independently checked the tail inequality and the two independent measurement innovations. Global fitness statistics remain shared. |
| Averaged target cloning probability | `thm-keystone-averaged-cluster-pressure` derives the fitness contrast on the near/far event using the actual regularized scale, reward oscillation and configured positive map. It then averages actual nonlinear acceptance. | Every constant is derived from the entering state and maps. The existing valid-cluster size rule supplies the population-independent nonself mass `0.04`; no collision-component counting is used. |
| Error captured by selected clusters | `thm-keystone-averaged-error-capture` decomposes actual paired positional error over the common refinement of the unchanged geometric partitions. | Exact uncovered error, velocity contribution and common-alive scope remain explicit. A zero certificate is not reported as zero actual cloning probability. |
| Discharged structural Keystone application | `cor-keystone-canonical-balanced-structural` proves averaged pressure at least `0.30251915319 V_struct`, with zero offset, for canonical balanced swarms at radii in `[0.5,2)`, every even `N >= 4`, and every dimension. | Actual optimal matching gives `V_struct=(a-b)^2`. Every measurement outcome, including ties, is retained. Parent and both mathematical agents independently reviewed the coefficient. Independent integration checks population sizes 4 through 256 and several radii. |
| Complete averaged cloning contribution | `thm-cloning-unconditional-collective-balance` integrates the signed positional flux, shared-fitness barycenter variance, jitter, revival and component relative energy. | Parent and both mathematical agents reviewed the identities. Neither fitness nor the correlated velocity outputs are replaced by independent averages. |
| Unequal fitness versus signed drift | `prop-cloning-two-cluster-noise-balance` proves positive internal-variance drift at some active-selection inputs. `lem-cloning-coupled-error-not-internal-variance` retains the cross-swarm term of the actual coupled structural increment. | These statements do not refute the offset-bearing Keystone pressure bound or full-algorithm convergence. They prevent a false inference from selection activity to a negative internal-variance increment. |

The distinction matters in the same canonical family: every row has positive
averaged cloning probability, the structural-pressure bound has zero offset,
and internal positional variance can increase. These facts coexist because
outgoing selection pressure, incoming donor flux, and coupled structural
error are different terms of the complete calculation. Shared rotations and
retained dead velocities remain in their prescribed component balance.

Verification adds 13,568 complete Rust engine updates, exact finite
measurement/donor/gate integration, exact Gaussian fourth moments for the
three-slot observable, and independent binomial integration of the balanced
cluster family. The machine-readable evidence is
`validation/keystone-measurement-average.json`; the detailed comparisons are
in `MEAN_FIELD_VALIDATION.md`. The four targeted tests and strict Clippy pass.

This closes the measurement averaging and supplies a fully discharged
population-uniform structural example. It does **not** complete the two
general application gates: the intended broader regime still needs a bound
on the uncovered target error and a useful dissipative margin for the full
coupled increment. The exact signed identities do not assume that margin.
The theorem checklist retains these obligations instead of treating either
the canonical example or a sufficient geometric certificate as their proof.

### Overall completion status

The recovery and listed exact/quantitative repairs are implemented. The entire
requested proof repair is **not complete**: the population-uniform stationary
concentration estimate has not been derived. The combined stationary estimate
must still close the signed cluster-flow, alive-backbone and survival terms.
The entropy route likewise needs a population-uniform control of its actual
interaction and survival source relative to backward entropy dissipation.
The new one-step bounds have constants independent of N, while the proved
finite-N Doob entropy rate retains its actual N-dependence.
Neither is relabeled as the missing stationary estimate.


## Complete geometric coverage and component-growth diagnostics

The coverage calculation is proved without an assumed captured-error
fraction. The structural-growth calculations test stronger intermediate
contraction claims. They do not contradict the original combined affine
Lyapunov argument, which explicitly allows cloning to expand inter-swarm
distance and retains additive offsets. Treating these diagnostics as a gate
for that argument was a proof-strategy deviation; the recovery audit below
corrects it.

| Claim / dependency | Exact defect or missing step | Derived result and review |
|---|---|---|
| Averaged Keystone pressure outside valid clusters; actual weighted companion laws, global regularized fitness, geometric error decomposition | The certified-cluster sum omitted an uncontrolled error mass. Validity of some clusters did not imply that they captured the error. | Chapter 03, `lem-keystone-complete-coverage-constants` through `thm-keystone-discharged-averaged-pressure`, includes every original cluster. A fixed auxiliary feature cover bounds each row's actual near-neighbor mass; weighted Cauchy--Schwarz yields (3.CC8). No cluster-mass premise remains. Independently reconstructed by both mathematical agents and the parent. |
| General reward and rescaling applicability | Replacing the configured reward by a quadratic example, or assuming a strictly positive global derivative, would restrict the argument. | The actual reward's proved Lipschitz constant is used on the alive region. The powered diversity transform has a positive finite-increment minimum on its compact standardized-score interval by strict monotonicity and compactness. This discharges its positive contrast for the stated active-diversity regime. Canonical numbers are a specialization. |
| Population uniformity and self-exclusion | A population-dependent number of groups or an omitted singleton correction could destroy the estimate. | All counts refer to a fixed geometric cover independent of N. Every N has the explicit bound (3.CC11a), with an N-independent positive coefficient and an N^-2 correction. A stronger linear bound holds above the derived population threshold. The constants can be very conservative; no practical-size lower rate is inferred from their positivity. |
| Conversion to full structural error and unequal alive laws | Positional error cannot silently absorb pure velocity error or unmatched alive mass. | Equations (3.CC12)--(3.CC14) retain the actual velocity discrepancy, alive normalization and unmatched-label contributions. Retained dead coordinates are never bounded by the alive region. This is an exact interface, not an imposed alive-fraction floor. |
| Positive pressure implies negative coupled cloning increment | Incoming donor displacement, the moving barycenter, and the cross-swarm covariance have a sign not supplied by the pressure lower bound. | `prop-cloning-macroscopic-structural-expansion` gives two nonconstant four-slot swarms with unequal fitness in every measurement outcome. Every coupling of their actual cloning kernels increases expected structural error by more than 0.12. The certificate enumerates all 81 actual measurement vectors per swarm using directed rational intervals, with exact Gaussian moment integration. Both agents and the parent independently reviewed it. |
| A globally useful affine cloning contraction can absorb the counterexample in a small offset | An unspecified offset could conceal that the claimed negative-drift region is empty. | `prop-cloning-no-global-affine-structural-contraction` proves that on the zero-velocity one-dimensional class any global N-uniform affine bound must have C >= 4*kappa, while the sharp maximum entering structural error is 4. Balanced clouds approaching that maximum still have positive averaged cloning drift for an explicit sufficiently large N. Small nonconstant reference perturbations retain the contradiction. |

### Actual mechanism and independent execution

For the finite counterexample, A has positions `(-1.5,-1.5,-1.5,1)` and
B has positions `0.01*A`; all velocities are zero. The fitter minority of A
is farther from A's barycenter. Copying that minority increases the balance
of the two spatial groups and therefore increases spread. The exact entering
structural error is `1.1485546875`. The output structural-error expectation
is greater than `1.27648593291590` under every cross-swarm coupling. Shared
component rotations preserve the zero velocities exactly; they have not been
removed from the transition. Every actual measurement outcome has unequal
fitness in each swarm. Equal-fitness exceptions cannot account for this result.

The independent Rust integration in
`crates/benchmarks/tests/keystone_collective_expansion.rs` aggregates the same
geometric groups by their binomial measurement counts. It agrees with the
separate 81-vector rational enumeration at N=4. Its full-engine comparisons
use N=4,16,32,64,128,256, two initial scales, and two disjoint batches of 128
seeds per case: 3,072 complete canonical updates. All 24 cohort means lie
within 1.84 independent-run standard errors of their exact cloning-stage
predictions. All runs have unequal retained fitness. Completed-step positions
and eligibility are also read from the Rust engine; their variances are
recorded separately from cloning-stage predictions.

The reproducible artifacts are `validation/keystone-collective-expansion.json`,
`validation/certify_keystone_structural.py`, and
`validation/keystone-structural-intervals.json`. Exact interval error, floating
integration error and independent-run fluctuation are separate. No trajectory
comes from the independent integration code.

### Consequence for the repair scope

The coverage gap is repaired within the bounded alive-domain, active-diversity
regime already stated in Chapter 03. The original Chapter 03 Sections 12.2--12.4
require bounded cloning expansion of the full inter-swarm observable, followed
by a combined affine drift. They do not require a negative cloning increment
of the structural component. The exact signed balances and component-growth
examples must be used at those precise points in the original argument; they
do not justify demanding a different global contraction theorem.

### Complete-step counterexample with terminal survival retained

`prop-canonical-fullstep-structural-expansion` continues the same N=4 inputs
through every actual canonical quadratic stage at h=.04, gamma=B=1,
position-noise factor .1, cap two, and terminal box absorption. For the original
quadratic structural metric with lambda_v=1 and cross coefficient b=.1,
every coupling of the separately survival-conditioned full-step marginals
has expected structural increment greater than .08. The certificate's lower
bound is greater than .09091348.

The derivation uses the exact BAOAB positional identity
`x_pre = t*X_clone + c*q*xi + s*zeta`, with
`t=1-c^2*(1+exp(-h))`, c=h/2, q^2=(1-exp(-2h))/2 and s=.1*sqrt(h).
Both shared collision velocities start and remain zero before kinetics.
Gaussian-mixture tail and fourth-moment bounds control actual terminal exits.
The alive-normalized lower and upper moments are divided by the correct
survival probabilities; no all-alive conditioning is substituted for
nonextinction. The cap is retained, and the pointwise inequality
`Q(dx,dv) >= .9975*dx^2` transfers the positional lower bound to the actual
phase-space cost. Both mathematical agents and the parent independently
reconstructed the complete proof and its constants.

This contradicts zero-offset one-step structural contraction at canonical
B=1. It does not contradict the separate sufficiently-strong-diffusion regime
or an affine bound whose verified offset permits this expansion. No change to
the original algorithm or downstream applicability assumptions was made.


## Recovery of the original combined proof target

Comparison against the requested baseline commit confirms that the latest
structural-contraction acceptance criterion was stronger than the argument
being repaired. This was an assistant strategy error.

- Original Chapter 03 Section 12.2 (`thm-complete-wasserstein-drift`) requires
  bounded inter-swarm cloning expansion. Section 12.4 combines it with the
  kinetic contribution and all additive offsets.
- The original Keystone offset is
  `g_max=max(p_u*g_err, chi*R_spread^2)`. Its positional drift also retains
  `C_x`, persistence and revival contributions. Neither a zero offset nor
  `C < 4*kappa` for structural cloning is an original application condition.
- Original Chapter 06's principal TV functional omits `V_W`, using internal
  variance, velocity-barycenter energy and boundary exposure. Imposing
  contraction of an arbitrary two-swarm structural distance on that route
  changes its dependency chain.
- The original bounded-expansion proof underestimated donor rearrangement by
  assigning it only a jitter-sized cost. Frozen eligible source positions and
  exact full-slot component energy dissipation give a direct, population-
  uniform moment bound for the same required operator. This repairs that
  input without replacing its role by a contraction claim.
- The chapter defines `V_Var=V_Var,x+lambda_v*V_Var,v`; its Section 12
  bookkeeping had dropped `lambda_v`. The repaired comparison vector retains
  this weight, including revival and collision-energy contributions.

The original operator-composition strategy is retained. For backward operators,
cloning followed by kinetics is `Q=P_C P_K`; its exact increment is
`(Q-I)V=(P_C-I)V+P_C(P_K-I)V`. The conditional kinetic contribution is evaluated
at the actual cloned state. Shared-component dissipation and additive offsets
are propagated with their kinetic coefficients before being bounded. The
original weights are not absorbed into relative contraction rates.

The complete strong-diffusion/entropy application is not certified by these
local repairs alone. Its outstanding obligations must be traced to the
original functional and dependencies, rather than replaced by an assumed or
unnecessarily stronger Wasserstein contraction.


### Baseline application boundaries checked

The canonical quadratic growth examples were not verified against the
baseline's full landscape and parameter hypotheses. In particular, opposite
points in the symmetric quadratic domain have equal reward, so they do not
discharge a nonvacuous EG-3 separation condition. Their exact arithmetic
remains valid for the canonical engine; they must not be presented as
counterexamples within the original theorem's entire hypothesis class.

The available baseline separately contains two locally defective comparisons:
Section 10.3.6 reverses the upper structural-to-variance estimate, and Section
10.5 transfers contraction of an upper bound to the bounded quantity. The
combined argument in Section 12 uses bounded inter-swarm expansion instead.
The repair follows that dependency and retains the established geometric
cluster estimates; it does not impose separation of the two swarm laws to
force the reversed inequality.

Three standalone references in baseline Chapter 03 are absent from that
commit: `proof_20251025_0130_thm_complete_boundary_drift.md`,
`proof_20251025_0148_prop_coupling_constant_existence.md`, and
`proof_20251025_0227_thm_main_results_summary.md`. This records availability
in the selected commit only, rather than asserting that no other source
contains them. Their corresponding claims must be checked from the recovered
chapter arguments and any available complete proof, not filled with invented
statements.


### Verified local changes in the combined operator inputs

Chapter 03 `cor-cloning-actual-inter-swarm-expansion` now discharges the
required moment bound explicitly:

`C_W=4[(1+eta)(B_x^2+d*j^2)+(lambda_v+b^2/(4*eta))*V_max^2]`.

Its proof uses the actual frozen eligible positions, exact Gaussian second
moments, and full-slot energy dissipation for the shared component rotations.
Revived rows and retained dead velocities are included. No bound is placed
on retained dead positions, and no component-count factor is introduced.

Equations (3.AC1)--(3.AC9) restore the prescribed velocity weight and retain
the exact signed positional integral before upper bounding it. The original
affine positional input has the explicit two-swarm offset
`C_x=D_x^2+2*d*j^2`; the velocity input has
`C_v=8*lambda_v*V_max^2`, with the sharper one-step zero value on all-alive
inputs. Terminal deaths prevent silently iterating that sharper value across
arbitrary subsequent states. These offsets supply the original component
assembly; they do not assert that each bounded component decreases.

The composition retains the nonnegative residuals of the proved component
bounds as `-A_K*D_C-P_C*D_K`. In particular the full-slot collision
dissipation contributes
`-c_V*(1-kappa_v)*lambda_v*(1-alpha^2)*E_component` in the diagonal case.
The offsets are propagated as `w^T(A_K*b_C+b_K)`, so cloning expansion is
weighted by the subsequent kinetic coefficients. Three independent reviews
checked these identities and their normalization.

The relevant Rust cluster/kinetic targets pass: four cluster checks and one
kinetic coupling check. They exercise exact donor/collision accounting and
the actual quadratic/Rastrigin BAOAB/cap mechanism. Production engine code
was not changed by these proof corrections.


### Entropy-route scope check

The baseline Chapter 15 entropy/transport theorem also allows expansion in
the transport component. Its valid intermediate estimate is
`V' <= a*H + (c*K-b)*W^2 + C`, with nonnegative `H,W^2`, `0<a<1`,
`b>0`, and finite `K>=0`. Choosing `c=b/(K+1)` yields
`c*K-b=-c`, hence `V'<=a*V+C` for `V=H+c*W^2`. No separate `K<1`
condition is necessary. The baseline weight-division argument at lines
1737--1744 had the wrong sign. The current Chapter 15 theorem
`thm-entropy-transport-contraction`, equation (15.54), already has the correct
general coefficient. This check preserves that original route; it does not
establish its analytic entropy-transfer premise. Chapter 15 was not edited
in this scoped correction.
