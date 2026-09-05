(sec-fg-faq)=
# Frequently Asked Questions

:::{div} feynman-prose
This FAQ explains the algorithm, the objects studied by its convergence analysis,
and its geometric and physical interpretations. The volume proves finite-particle
QSD convergence, population-uniform functional inequalities under structural
hypotheses, and mean-field well-posedness and attraction in an explicit regime.
Start with the question that interests you, then follow the linked definition or chapter. An
answer about one transition rule applies to another only when the relevant
hypotheses have been checked.

The {doc}`volume introduction <../intro_fractal_gas>` gives the reading order.
The {doc}`latent specification <../1_the_algorithm/02_fractal_gas_latent>` separates
identities of the update rule from the additional assumptions needed for
long-time behavior and population limits.
:::

(rb-fg-mapping)=
::::{div} feynman-added
| Familiar question | Where it enters this volume |
|---|---|
| How is a search population updated? | {doc}`Algorithms and foundations <../parts/01_foundations>` |
| Does the population relax to a limiting law? | {doc}`Finite-particle convergence <../parts/02_convergence>` |
| What can a density approximate? | {doc}`Mean-field limits and equilibrium <../parts/03_mean_field>` |
| How accurate is a finite computation? | {doc}`Entropy, regularity, and bounds <../parts/04_entropy_regularity>` |
| What does a recorded interaction graph represent? | {doc}`Fractal Set and continuum limits <../parts/05_continuum>` |
| How are field observables defined and tested? | {doc}`Fields <../parts/06_fields>` and {doc}`experiments <../parts/07_experiments>` |
::::

(sec-fg-faq-algorithm-fundamentals)=
## Algorithm fundamentals

(sec-fg-faq-walker-velocity)=
### Why give a walker velocity as well as position?

:::{div} feynman-prose
Velocity is part of the kinetic models studied here. It supports persistent
motion, friction, a momentum thermostat, and velocity-dependent interactions.
The chosen companion distance can also compare velocities. These are modeling
choices with consequences for the transition kernel. Position-only algorithms
are possible, but their analysis concerns a different process.

See the {doc}`single-particle analysis <../convergence_program/04_single_particle>`
and {prf:ref}`def-latent-fractal-gas-state`. In the latent specification,
coordinate subtraction of velocities is chart-dependent; using a metric in the
kinetic step does not itself make that subtraction intrinsic.
:::

(sec-fg-faq-soft-kernel)=
### Why sample companions with a soft kernel?

:::{div} feynman-prose
A soft kernel assigns positive probability to each eligible companion in exact
arithmetic. Nearby companions can be favored without making the choice
completely deterministic. On a set with bounded algorithmic diameter, one can
bound these probabilities uniformly from below; see
{prf:ref}`lem-latent-fractal-gas-companion-doeblin`.

That bound concerns a draw from the current finite list. Accessibility and mixing
of the full swarm also depend on cloning, motion, noise, and killing. On an
unbounded state space, the finite-list bound need not have a positive uniform
floor over all swarm configurations. Hard neighbor selection and truncated
kernels require their own analysis.
:::

(sec-fg-faq-dual-fitness)=
### What do reward and diversity contribute to fitness?

:::{div} feynman-prose
Reward scores task performance; diversity measures separation using the chosen
companion rule. The regularized product of their transformed values lets both
channels influence cloning. The exponents and normalizations determine their
relative effect.

A diversity channel can oppose concentration, but a lower bound on population
spread requires an estimate for the dynamics. Setting one exponent to zero
removes that channel's contribution; it does not by itself imply immediate
collapse or a pure random walk. Neither a Gibbs law nor an exact power of reward
follows from the product formula alone. See
{prf:ref}`def-latent-fractal-gas-fitness`.
:::

(sec-fg-faq-sampled-fitness)=
### Is sampled fitness the same as expected fitness?

:::{div} feynman-prose
At a fixed swarm state, fitness can still depend on the randomly drawn distance
companion. Holding that assignment fixed gives the sampled-fitness derivative.
Differentiating its expectation also differentiates the companion probabilities.
The volume proves regularity for both quantities, as well as for the separate
surrogate obtained by inserting expected measurements into the fitness formula;
see {prf:ref}`thm-unified-cinf-regularity-both-mechanisms`.

These three functions generally have different values. The distinction survives
a population limit when each walker still draws one companion: taking a nonlinear
function and taking an expectation need not commute. The limiting model must
preserve the algorithm's order of sampling and evaluation. See
{prf:ref}`rem-mean-field-fitness-field-latent`.
:::

(sec-fg-faq-greedy-regularity)=
### Is smoothness proved for the actual greedy companion algorithm?

:::{div} feynman-prose
Yes. {prf:ref}`lem-greedy-ideal-equivalence` proves a derivative bound for the
actual sequential greedy history law, with a pivot rule independent of the
continuous state. The proof sums over all pairing histories. Once the moving
walker has been paired, it stops influencing the remaining choices; this closes
an induction whose constant does not grow with the number of pairs.

Combining that history bound with the normalized measurement and fitness
calculus proves smoothness of the full expected sampled fitness. Under fixed
positive scales and regularizers, bounded pair distances, and uniform analytic
reward bounds, {prf:ref}`thm-main-complete-cinf-geometric-gas-full` gives the
factorial estimate $\|D^nF\|\le C_FB_F^nn!$, with $C_F,B_F$ independent of
population size. It applies to derivatives in a specified walker-position block
on a fixed alive, candidate, and branch stratum. Smooth inputs give smoothness;
the stronger factorial hypotheses give analyticity.

Independent sampling, idealized matching, and greedy matching each have a proved
regularity calculation. Their probability laws differ, as the four-walker example
in {prf:ref}`rem-observation-common-kernel-structure` shows. Smoothness therefore
supplies coefficient estimates for each model without identifying their dynamics.
On unbounded families, use the theorem's normalized derivative-ratio and joint-law
majorants; the bounded-distance argument does not impose compactness on every
algorithmic model.
:::

(sec-fg-faq-constant-population)=
### Why keep the number of walker slots fixed?

:::{div} feynman-prose
A fixed number of slots makes the computational budget and swarm state space
explicit. Cloning reallocates those slots among states. The number currently
marked alive can still change during an update.

Variable-population methods are mathematically possible. They require a state
space and update rule that account for population changes, followed by estimates
for that process. A fixed population is a choice of this algorithm family, rather
than a general requirement of transport geometry or mean-field analysis.
:::

(sec-fg-faq-dead-walkers)=
### What happens when a walker is marked dead?

:::{div} feynman-prose
The alive mask and replacement rule specify its treatment. In the nondegenerate
latent update, a dead walker draws an alive companion and is assigned a cloning
update. Jitter can subsequently place the copied position outside the alive
region; the next mask evaluation detects that outcome.

The stopping rule matters. The mathematical latent specification stops when
fewer than two walkers are alive. The implementation discussed there allows a
lone survivor to select itself. These are different kernels at that boundary.
A dead slot inside a surviving swarm is also distinct from the cemetery state of
the entire killed process. See {prf:ref}`def-latent-fractal-gas-companions` and
{prf:ref}`def-latent-fractal-gas-cloning`.
:::

(sec-fg-faq-cloning-dynamics)=
(sec-fg-faq-kinetics)=
## Cloning and kinetic dynamics

(sec-fg-faq-cloning-asymmetry)=
### What does the direction of cloning establish?

:::{div} feynman-prose
For the stored fitness values used in a step, a positive cloning score directs
replacement toward a fitter companion. The frozen-score alignment identity is
proved in {prf:ref}`lem-latent-fractal-gas-selection-alignment`.

That identity concerns copied scores. After jitter, collision, motion, and a new
fitness measurement, both the walker states and the population statistics may
have changed. A Lyapunov inequality for the complete transition needs estimates
for those operations as well. Directed selection alone gives no such inequality.
:::

(sec-fg-faq-fermionic-antisymmetry)=
### Does asymmetric cloning imply fermionic statistics?

:::{div} feynman-prose
The sign of a fitness difference records which direction of copying is favored.
The connection to fermionic operators is made through the law and transition
operator of the complete algorithm. In
{prf:ref}`thm-lqft-record-fock-reconstruction`, centered square-integrable
observables of the stationary conservative process form the one-particle
space. Exterior multiplication creates an observable mode, contraction removes
it, and their canonical anticommutation relations follow by an explicit
calculation. The lifted transition has a one-particle vacuum matrix element
equal to the connected covariance measured in that same stochastic process.
For a finite run away from equilibrium,
{prf:ref}`prop-lqft-finite-record-transfer` gives the corresponding construction
from the joint law at two recorded times, including the empirical law used to
compute sample covariances; stationarity is unnecessary for that identity.

What do the higher-particle sectors represent? Take independent copies of the
complete process and antisymmetrize their observable amplitudes.
{prf:ref}`thm-lqft-replica-isomorphism` identifies the resulting centered
antisymmetric replica sector isometrically with the exterior space and
intertwines their evolution. Its propagators are determinants of the
one-particle propagators, with the exchange sign fixed by antisymmetrization.
Thus the reconstruction supplies both the fermionic algebra and an exact
stochastic meaning for its matrix elements.

The copies here are entire swarms, each retaining its internal interactions.
They are not the interacting walkers within one recorded swarm. Likewise,
{prf:ref}`thm-lqft-product-obstruction` proves why scalar multiplication
observables cannot themselves be identified multiplicatively with the full
fermionic operator algebra. These distinctions specify precisely which
statistics the reconstruction computes.
:::

(sec-fg-faq-momentum-conservation)=
### What does the inelastic collision conserve?

:::{div} feynman-prose
Within one collision group with unit coordinate masses, the update preserves
the mean velocity and shrinks deviations from it by the restitution coefficient.
The resulting momentum and relative-energy identities have direct proofs in
{prf:ref}`lem-latent-fractal-gas-collision-energy`.

For disjoint groups, these identities sum over the population. When groups write
to overlapping recipients using the original velocities, global conservation
needs separate analysis of those writes. On a variable metric, summing coordinate
velocities at different positions also requires care before interpreting the
sum as physical momentum. Momentum conservation alone implies neither
symplecticity nor convergence to equilibrium.
:::

(sec-fg-faq-revival-guarantee)=
### Can the whole swarm become extinct?

:::{div} feynman-prose
Yes, when the specified killing rule allows extinction. A replacement rule that
uses surviving companions needs an eligible source. A lower bound on companion
probabilities cannot supply one when that set is empty. The latent mathematical
variant also has the stricter stopping rule described above.

Restarting after termination defines an additional transition. Its stationary
law must be analyzed for that restarted process. Conditioning a killed process
on survival is a different operation. See
{prf:ref}`def-latent-fractal-gas-qsd`.
:::

(sec-fg-faq-baoab)=
### What does the Boris–BAOAB splitting guarantee?

:::{div} feynman-prose
The splitting separates force kicks, position updates, a thermostat, and any
rotation. Some individual steps admit exact identities: the frozen-metric Boris
rotation preserves its kinetic norm, and the thermostat has explicit conditional
moments. See {prf:ref}`lem-latent-fractal-gas-boris-energy` and
{prf:ref}`lem-latent-fractal-gas-ou-moments`.

The stochastic, dissipative, capped full update requires its own accuracy and
stability analysis. In particular, the force coefficients in the displayed latent
sequence must be compared with ordinary BAOAB before importing an integrator
result; see {prf:ref}`rem-latent-fractal-gas-splitting-normalization`.
Claims about invariant-measure bias also require the actual boundary rule,
regularity, and target measure.
:::

(sec-fg-faq-three-timescales)=
### How do long-time, population, and continuum limits differ?

:::{div} feynman-prose
At fixed population and step size, long-time analysis studies the repeated
transition kernel. A population limit changes the number of walkers and studies
marginals or empirical measures. A small-step limit changes the discretization in
time. A geometric continuum limit additionally changes the graph or kernel
resolution.

Each limit needs its own estimates, and exchanging their order requires
uniformity. See {doc}`mean-field analysis <../parts/03_mean_field>` and
{doc}`continuum conditions <../convergence_program/16_continuum_discharge>`.
:::

(sec-fg-faq-wfr-metric)=
### Why use Wasserstein–Fisher–Rao or Hellinger–Kantorovich geometry?

:::{div} feynman-prose
These geometries describe motion together with local gain or loss of mass. Fixed
walker slots are compatible with local selection: replacing one alive walker
redistributes the empirical measure, while its total alive mass changes only
through the specified death and revival bookkeeping.

The {doc}`HK chapter <../convergence_program/11_hk_convergence>` proves a direct
convergence route. It separates the error in square roots of total masses from
the Hellinger error of normalized shapes. Relative entropy controls the latter,
and the transportation inequality controls the Wasserstein term; see
{prf:ref}`thm-hk-convergence-main-assembly`. This gives a quantitative bound for
the chapter's additive mass-and-shape diagnostic without a global density-ratio
assumption. The canonical dynamic HK metric has its own normalization and bounds.

Apply these comparisons to probability laws or explicitly smoothed empirical
measures. An atomic empirical cloud and a diffuse density are mutually singular
in Hellinger distance. See {prf:ref}`rem-hk-empirical-singularity` and
{doc}`WFR geometry in Volume I <../../1_agent/05_geometry/02_wfr_geometry>`.
:::

(sec-fg-faq-constraints)=
## Convergence, constraints, and error bounds

(sec-fg-faq-five-constraints)=
### Which parameter constraints establish convergence?

:::{div} feynman-prose
The {doc}`composed analysis <../convergence_program/06_convergence>` combines
cloning and kinetic drift estimates and proves QSD convergence for the actual
killed transition under its survival and comparison hypotheses. In particular,
{prf:ref}`thm-main-convergence` gives a unique QSD and geometric convergence when
a surviving block of the kernel has the stated two-sided measure bound.

Use the {doc}`parameter chapter <../1_the_algorithm/03_parameter_constraints>`
to connect confinement, forces, noise, companion probabilities, regularization,
and step size to the estimates in that theorem. A block comparison constant,
an entropy dissipation rate, and a variance diagnostic are different quantities.
The proofs specify which combination yields each conclusion and whether its
constant is uniform in population size.
:::

(sec-fg-faq-classical-analysis)=
### Can the original algorithm-specific results use classical tools?

:::{div} feynman-prose
Yes; the volume supplies direct proofs. The
{doc}`cloning chapter <../convergence_program/03_cloning>` establishes its
selection, variance, and boundary estimates, and the
{doc}`kinetic chapter <../convergence_program/05_kinetic_contraction>` develops
confinement and motion estimates. Their composition is analyzed at the level of
the specified swarm transition in {prf:ref}`thm-main-convergence`.

The same approach continues through the book: normalized derivative calculus
proves coefficient regularity, functional inequalities control entropy and
fluctuations, and gain-loss estimates give a well-posed mean-field equation.
Each implication has a complete argument with its model and parameter hypotheses
stated. The latent specification links these results to the update quantities;
its abstract {prf:ref}`prop-latent-fractal-gas-conditional-qsd` is one summary
criterion, alongside the detailed component and composition proofs.
:::

(sec-fg-faq-mean-field-wellposedness)=
### Are mean-field existence, uniqueness, and convergence proved?

:::{div} feynman-prose
Yes, for the specified gain-loss equation. Under its positive, mass-preserving
kinetic semigroup and bounded Lipschitz gain-loss assumptions,
{prf:ref}`thm-chaos-mild-wellposedness` constructs a unique global positive
probability-density solution. The proof uses a damped integral equation that
preserves positivity and mass at every iteration. Kinetic smoothing extends it
to singular initial laws in {prf:ref}`cor-chaos-measure-initial-data`. Kernels
normalized by alive mass use the positive-alive estimates in
{prf:ref}`rem-chaos-positive-alive-localization` for continuation.

There is also a proved stationary-state and attraction theorem. If the kinetic
semigroup contracts zero-mass differences with bound $Ke^{-at}$ and the reaction
has Lipschitz constant $L_{\mathcal R}$, the regime
$b=a-KL_{\mathcal R}>0$ gives a unique stationary law and exponential attraction
at rate $b$; see {prf:ref}`thm-uniqueness-uniqueness-stationary-solution`.
This condition compares the rate at which motion removes differences with the
rate at which the nonlinear reaction can amplify them.

The population-limit step is proved separately.
{prf:ref}`thm-uniqueness-of-qsd` combines finite-time consistency, vanishing
scaled extinction probability, and global attraction to obtain stationary
propagation of chaos. {prf:ref}`cor-chaos-lsi-stationary-limit` gives the other
route through joint-law concentration and stationary-equation identification.
These results retain the actual collision, sampling, and time-scaling conditions
needed to connect a finite swarm to that mean-field equation.
:::

(sec-fg-faq-qsd-invariant)=
### Is a QSD the same as an invariant measure?

:::{div} feynman-prose
An invariant probability measure is unchanged by a conservative transition.
A QSD is unchanged by a killed transition after conditioning on survival and
renormalizing. Existence of either measure concerns the particular kernel and
state space being studied.

A full-swarm QSD, a one-particle marginal of that QSD, a killed single-particle
law, and a mean-field stationary density are distinct objects. An exact Gibbs
formula or a power-law expression in reward requires a separate identification
of the appropriate law. See {prf:ref}`rem-latent-fractal-gas-stationary-objects`
and {doc}`discrete QSD structure <../convergence_program/07_discrete_qsd>`.
:::

(sec-fg-faq-variance-contraction)=
### Does decreasing population variance prove Wasserstein contraction?

:::{div} feynman-prose
Population variance measures spread within one swarm. Wasserstein contraction
compares two evolving probability laws. Two swarms can each have zero positional
variance while being concentrated at different locations. Their variance says
nothing about the distance between them.

A contraction proof needs a coupling or another estimate that controls that
distance for the full transition. Likewise, an observed cloning frequency is a
frequency, rather than a positional contraction rate. See
{doc}`Wasserstein control <../convergence_program/04_wasserstein_contraction>` and
{prf:ref}`rem-latent-fractal-gas-derived-scales`.
:::

(sec-fg-faq-error-floor)=
### Is there a universal error proportional to the inverse square root of population size?

:::{div} feynman-prose
For a smooth empirical average there is a proved population-uniform bound. If
the actual joint law satisfies the full-gradient LSI of
{prf:ref}`cor-n-uniform-lsi`, then its Poincaré inequality gives
$\operatorname{Var}(N^{-1}\sum_i\varphi(Z_i))\le C_*L^2/N$ for an
$L$-Lipschitz observable in the stated form domain; see
{prf:ref}`cor-quantitative-lsi-final`. Thus its standard deviation is at most
$\sqrt{C_*}L/\sqrt N$, even though the walkers can be dependent.

Without that structural estimate, the exact variance includes every pair
covariance. Exchangeability alone does not remove those terms. Discrete alive
indicators also require a form controlling status changes, and an average over
alive walkers includes its random denominator.

An empirical measure in Wasserstein distance has a separate sampling error,
which depends on dimension and moment assumptions. Comparing an empirical
average with a limiting density also includes marginal bias and finite-time
relaxation. The {doc}`quantitative bounds <../convergence_program/13_quantitative_error_bounds>`
keep those contributions and entropy normalization explicit.
:::

(sec-fg-faq-n-uniform-lsi)=
### Where is the population-uniform logarithmic Sobolev inequality proved?

:::{div} feynman-prose
{prf:ref}`cor-n-uniform-lsi` assembles four complete analytical routes for the
specified continuous joint law: a product kinetic reference, a uniformly bounded
whole-law density tilt of that reference, a uniform lower bound on the full
joint potential Hessian, and an invariant law of a contractive additive-noise
flow. Each route gives the same convention
$\operatorname{Ent}_{\pi_N}(f^2)\le2C_*\int\sum_i(|\nabla_{x_i}f|^2+
|\nabla_{v_i}f|^2)\,d\pi_N$, with $C_*$ independent of $N$.

The reference-measure proof includes nonconvex confining potentials on unbounded
space; see {prf:ref}`thm-nonconvex-main` and {prf:ref}`thm-kinetic-lsi`.
Tensorization retains the one-particle constant. Joint perturbation and
contractive-flow arguments then treat the corresponding interacting families.
For an invariant swarm law or a QSD, use the structural criterion satisfied by
that particular law. A law with discrete status variables also needs the status
entropy term stated in the chapter.

The full gradient matters. A function depending only on positions has zero
velocity gradient, so velocity dissipation alone cannot control its static
entropy. Hypocoercivity uses transport to connect those position variations to
the directions in which noise acts.
:::

(sec-fg-faq-hypocoercive)=
### Does mixing automatically give an entropy decay rate?

:::{div} feynman-prose
The volume proves entropy decay by combining a functional inequality with the
full evolution equation. In {prf:ref}`thm-villani-hypocoercivity`, the kinetic
calculation uses a modified entropy consisting of relative entropy plus a
positive quadratic form in position and velocity derivatives. Its mixed terms
transfer velocity dissipation to position variations. The resulting differential
inequality gives an explicit exponential rate.

For the complete normalized swarm evolution,
{prf:ref}`thm-kl-convergence-euclidean` gives
$H(t)\le\Phi(t)\le e^{-rt}\Phi(0)$ with
$r=\delta/(C/2+g_+)$, where $C$ is the actual joint-law LSI constant,
$g_+$ bounds the modification matrix, and $\delta$ is the full dissipation
constant. Uniform bounds on these quantities give a rate independent of $N$.
The kinetic and common-target cloning cases have explicit dissipation proofs.

Killing is included in the normalized equation.
{prf:ref}`prop-kl-conditioned-entropy` proves the exact diffusion-jump entropy
identity with the survival-normalization term; the boundary version retains the
outgoing flux. Thus a QSD is handled through its eigenmeasure identity.
For a finite-step algorithm, {prf:ref}`lem-discrete-entropy-decay` propagates the
proved one-step entropy defect and its error floor. A mixing estimate in another
metric becomes part of this argument only through an established comparison.
:::

(sec-fg-faq-fractal-set)=
(sec-fg-faq-causal-sets)=
## The Fractal Set and continuum geometry

(sec-fg-faq-simplicial-complex)=
### Why record interactions as a directed complex?

:::{div} feynman-prose
An interaction involves a companion, a receiving walker before an update, and
that walker afterward. The Fractal Set records these incidences as interaction
triangles, with distinct temporal, interaction, and attribution edges. This is a
specified organization of the recorded data. It does not determine the dimension
of a continuum spacetime.

Exact reconstruction depends on which states, random increments, parameters,
and sampling times are stored. Consult the
{doc}`Fractal Set definition <../2_fractal_set/01_fractal_set>` and compare its
storage requirements with the available run history.

For fermionic reconstruction, the orientation also fixes which matrix entry
describes a transfer from one recorded mode to another.
{prf:ref}`thm-lqft-edge-second-quantization` converts that directed matrix into
creation-and-annihilation bilinears and calculates their action on exterior
states. The edge weights specify the coefficients; the exterior construction
specifies the exchange algebra. The correspondence with the actual stochastic
transition is given separately, and explicitly, in
{prf:ref}`thm-lqft-record-fock-reconstruction`.
:::

(sec-fg-faq-spinors)=
### What does a spinor representation add to the stored data?

:::{div} feynman-prose
A vector already has a transformation law under a change of frame. A spinor
encoding introduces a particular representation and maps between that
representation and the quantities being stored. Its reconstruction, frame
conventions, and covariance must be specified and checked.

The fermionic reconstruction in
{prf:ref}`thm-lqft-record-fock-reconstruction` starts directly from observables
and the transition of the stationary conservative algorithm. Its exterior
operators obey the canonical anticommutation relations, and
{prf:ref}`thm-lqft-replica-isomorphism` identifies their many-particle evolution
with the centered antisymmetric sector of independent algorithm replicas.
This construction needs no Dirac matrices. Spinor coordinates can provide
another representation of a chosen finite set of modes; recovering a vector
from those coordinates is a separate inverse-map calculation.

A Dirac equation describes a particular form of spacetime evolution. Proving
that form requires identifying its differential operator and continuum limit,
beyond the operator algebra and stochastic covariance identities already
established here. See {doc}`the Fractal Set <../2_fractal_set/01_fractal_set>`
and {doc}`lattice QFT <../2_fractal_set/03_lattice_qft>`.
:::

(sec-fg-faq-ia-edges)=
### Do backward attribution edges imply backward causation?

:::{div} feynman-prose
An attribution edge points from an output to an earlier input used in its update.
Its direction records how to trace an influence. The update itself still uses
the prescribed time order.

The causal order is defined using the selected forward-time relation, rather
than every oriented edge of the interaction complex. Attribution edges can close
loops used for geometric observables without becoming extra causal arrows.
:::

(sec-fg-faq-blms)=
### In what sense is the record a causal set?

:::{div} feynman-prose
Starting from a relation that strictly increases the discrete time index, its
strict transitive closure is irreflexive and transitive. With finitely many
walkers at each step, a bounded discrete-time interval contains finitely many
recorded events. These facts support the order-theoretic construction.

A locally finite order alone does not identify a Lorentzian manifold, its metric,
or a sampling law. Those require additional geometry. See
{doc}`causal-set constructions <../2_fractal_set/02_causal_set_theory>`.
:::

(sec-fg-faq-adaptive-sprinkling)=
### What is assumed when the walker record is used as a spatial sample?

:::{div} feynman-prose
Walker locations can be dependent and nonuniform, and the estimates use their
actual observation law. The joint LSI gives a direct way to control this
dependence. For the smooth kernel estimator in
{prf:ref}`lem-cst-poincare-variance`, the resulting Poincaré inequality gives
variance at most $C/(N\varepsilon^{D+4})$ under its derivative, density, and form
domain conditions. The bandwidth scaling must make that bound tend to zero;
the deterministic kernel bias is controlled separately.

This is stronger than applying a fixed-observable law of large numbers to a
shrinking neighborhood. It tracks how derivatives grow as the bandwidth shrinks.
The {doc}`continuum conditions <../convergence_program/16_continuum_discharge>`
assemble these sampling, normalization, and geometric estimates for each
specified continuum construction.
:::

(sec-fg-faq-emergent-spacetime)=
### When is a continuum spacetime interpretation justified?

:::{div} feynman-prose
Specify the candidate manifold, metric, causal neighborhoods, and correspondence
with the discrete record. Then establish convergence of the distances, operators,
or energies used in the interpretation under a compatible scaling regime.

Propagation of chaos concerns particle laws. It does not supply global
hyperbolicity, a Lorentzian signature, or a continuum differential operator by
itself. Unproved geometric identifications remain assumptions or conjectures in
the resulting statement. See {doc}`Part V <../parts/05_continuum>`.
:::

(sec-fg-faq-lattice-qft)=
(sec-fg-faq-standard-model)=
## Gauge constructions and physical claims

(sec-fg-faq-gauge-emergence)=
### How can I tell whether a proposed gauge symmetry is present?

:::{div} feynman-prose
Specify the transformation of every variable, the observables it preserves, and
its action on the transition or field dynamics. Then check the claimed invariance.
A global change of basis and an independent change of frame at each node are
different operations.

The precise formula matters. Adding a common constant to the stored fitness
values generally changes the normalized cloning score through its denominator,
even though their differences stay fixed. Thus a claimed fitness-shift symmetry
must be checked for the actual update. See
{doc}`lattice field constructions <../2_fractal_set/03_lattice_qft>` and
{prf:ref}`def-latent-fractal-gas-cloning`.
:::

(sec-fg-faq-three-gauge-groups)=
### Do the algorithm's three mechanisms determine three gauge groups?

:::{div} feynman-prose
The chapter constructs the representations explicitly. A phase fiber supplies
$U(1)$; a chosen Hermitian doublet with a determinant volume form supplies
$SU(2)$; and a chosen Hermitian color fiber with its determinant form supplies
$SU(n)$. The normalization and generator identities have direct proofs in
{prf:ref}`thm-sm-su2-emergence` and {prf:ref}`thm-sm-su3-emergence`.
Their actions on separate tensor factors commute, giving the product
representation in {prf:ref}`cor-sm-gauge-group`. Its faithful group divides out
the subgroup acting trivially on the supplied matter representations.

Those fiber structures and comparison links are part of the field construction.
The number of algorithmic mechanisms alone does not select them. The viscous
force has a proved orthogonal covariance, while its nonlinear componentwise
color encoding needs the representation map stated in the theorem. Relating the
constructed gauge action to the swarm dynamics and measured physical couplings
uses the realization and calibration conditions in the
{doc}`Standard Model chapter <../2_fractal_set/04_standard_model>`.
:::

(sec-fg-faq-wilson-action)=
### Can Wilson observables be defined on triangles?

:::{div} feynman-prose
Given compatible link transports, one can compose them around a triangular loop.
Under a gauge transformation, the based holonomy transforms by conjugation; its
trace is invariant. The holonomy matrix itself is generally not invariant.

For noncommuting transports, combining neighboring loops requires compatible
base points, order, and orientations. An action built from such loops then has
its own weights and normalization. Convergence to a continuum Yang–Mills action
requires a consistency and scaling argument for that construction. See
{doc}`Wilson-loop computation <../3_fitness_manifold/08_voronoi_wilson_loops>` and
{doc}`Yang–Mills analysis <../2_fractal_set/05_yang_mills_noether>`.
:::

(sec-fg-faq-dirac-fermions)=
### What would establish a Dirac limit?

:::{div} feynman-prose
The direct fermionic reconstruction already has a specified evolution:
{prf:ref}`thm-lqft-record-fock-reconstruction` lifts the actual stationary
conservative transition to exterior sectors, and
{prf:ref}`thm-lqft-replica-isomorphism` identifies those sectors with
antisymmetric replica amplitudes. Its one-particle matrix elements reproduce
the algorithm's connected covariances. A Dirac limit asks the further question
of whether the relevant continuum evolution has the Dirac differential form.

The algebraic part is explicit: {prf:ref}`thm-sm-dirac-isomorphism` constructs
the Clifford action on the four-dimensional spinor space
$\Lambda^*\mathbb C^2$ and proves that its complex Clifford algebra is
$M_4(\mathbb C)$. On a supplied spin manifold, a frame converts those matrices
into the corresponding metric-dependent Clifford relations.

A Dirac continuum limit then uses a spin connection, consistent directional
differences, quadrature convergence, and control of extra lattice modes.
These are operator and limiting conditions beyond the finite-dimensional
algebra calculation. See {doc}`lattice QFT <../2_fractal_set/03_lattice_qft>`
and the {doc}`twistor formulation <../2_fractal_set/08_twistor_formulation>`.
:::

(sec-fg-faq-gauge-group)=
### Does choosing three latent dimensions establish the Standard Model?

:::{div} feynman-prose
Choosing a three-dimensional space is a model input. Even when an $SU(3)$
representation is constructed, identifying the Standard Model also requires
its matter content, representations, dynamics, and observable predictions.
A match of group names or dimensions establishes only that specific match.

The same caution applies to a proposed correspondence between latent dimension
and the number of particle generations. That correspondence needs its own
argument and evidence; it is not fixed by the dimension choice alone.
:::

(sec-fg-faq-higgs-bifurcation)=
### Does a bifurcation in the optimization dynamics establish a Higgs mechanism?

:::{div} feynman-prose
The volume proves the quartic normal-form calculation and the gauge-field mass
matrix for a supplied electroweak doublet in
{prf:ref}`thm-sm-higgs-isomorphism`. The normal form determines its minima,
radial curvature, and barrier. With the stated gauge representation, covariant
derivative, and canonical kinetic normalization, expansion around the chosen
vacuum gives the displayed Higgs, W, Z, and photon masses.

Using that result for a swarm requires deriving its order-parameter equation
and relating its coefficients and observables to the supplied field action.
A curvature of an optimization potential is a deterministic linearized
relaxation quantity; the field-action calculation identifies a mass only with
its kinetic normalization and operator interpretation in place.
:::

(sec-fg-faq-cp-violation)=
### Do different companion ranges establish CP violation?

:::{div} feynman-prose
The proved test is {prf:ref}`thm-sm-cp-violation`: for a specified CP
transformation, a nonzero mean of a CP-odd observable rules out invariance of
that law. {prf:ref}`prop-sm-cp-magnitude` constructs a rephasing-invariant
quartet from a supplied unitary mixing matrix and bounds its magnitude.

Different real companion ranges change sampling probabilities. Their ratio has
zero complex argument and cannot itself supply a CP phase. A bandwidth-dependent
bound for the quartet requires the theorem's exchange symmetry and derivative
estimate for that family. These give a concrete test of a proposed mapping from
algorithmic parameters to CP observables.
:::

(sec-fg-faq-neutrino-mass)=
### Can repeated ancestry by itself explain neutrino mass?

:::{div} feynman-prose
An ancestry record can define coefficients in a proposed field action. The
Majorana theorem, {prf:ref}`thm-sm-majorana-mass`, gives the exact gauge-invariance
condition for its chosen bilinear: $R(g)^{\mathsf T}MR(g)=M$. A neutral singlet
can satisfy it, while a single field with nonzero unbroken Abelian charge cannot
have that constant mass term.

For the supplied two-state neutral mass matrix, {prf:ref}`prop-sm-seesaw` proves
the eigenvalues and the controlled light-mass approximation when the heavy
scale dominates the mixing. Identifying those coefficients with a swarm record
requires localization, spinor transport, and action normalization. Repeated
walker labels in a forward-time genealogy remain distinct events; they supply
neither a causal cycle nor those mass coefficients by themselves.
:::

(sec-fg-faq-cross-volume)=
(sec-fg-faq-implementation)=
## Implementation and connections to the agent

(sec-fg-faq-fragile-connection)=
### How does the Fractal Gas connect to Volume I?

:::{div} feynman-prose
An agent can supply a learned state space, metric, model, or score in which a
swarm searches. The swarm can supply candidate trajectories and measurements
for planning or learning. This is a possible integration of the two volumes.

When learned quantities change during the run, the transition changes with them.
Applying a fixed-kernel estimate then requires controlling that adaptation or
including the additional variables in the analyzed state. See the
{doc}`Fragile Mechanics introduction <../../1_agent/intro_agent>` and
{doc}`latent model <../1_the_algorithm/02_fractal_gas_latent>`.
:::

(sec-fg-faq-future-work)=
### What about distributed optimization or economic applications?

:::{div} feynman-prose
A distributed application must specify the objective, communication pattern,
work measurement, and incentive rules. Convergence of a particle algorithm
addresses only the hypotheses and conclusions of its own model. Consensus and
incentive properties require their own analysis.

Volume I's {doc}`economics chapter <../../1_agent/09_economics/01_pomw>` develops
that separate application setting.
:::

(sec-fg-faq-computational-cost)=
### What determines the computational cost?

:::{div} feynman-prose
Direct evaluation of all pairwise distances among $N$ walkers in dimension $d$
costs on the order of $N^2d$ arithmetic operations. Materializing all pairwise
weights uses quadratic storage; streaming or blocking changes the memory cost.
Fitness, force, Hessian, environment, and recording costs must be counted as well.

Restricting companions to a local neighborhood changes the sampling law. Its
accuracy and accessibility need analysis, especially when a theorem uses a
positive probability for every eligible companion. GPU parallelism can reduce
wall-clock time without changing the number of pairwise operations. Measure the
configured implementation on the task of interest.
:::

(sec-fg-faq-hyperparameters)=
### How do the constraints help with tuning?

:::{div} feynman-prose
An explicit inequality relates parameters to a stated property. It can guide a
choice when all quantities in the inequality are known or bounded in the
application. A default configuration is a starting point for a particular
implementation, rather than a universal feasible point for every objective.

Record the units, step size, noise, companion range, regularization, and stopping
rule. Compare diagnostics with the estimates they actually measure. See
{doc}`parameter constraints <../1_the_algorithm/03_parameter_constraints>` and
{doc}`computational proxies <../3_fitness_manifold/07_computational_proxies>`.
:::

(sec-fg-faq-when-to-use)=
### How should I compare it with a gradient method or another population method?

:::{div} feynman-prose
Specify the objective and the computational budget first. Compare methods using
the same environment interactions or objective evaluations, and also report
wall-clock time when computation differs. Distinguish finding a high-scoring
state from sampling a specified probability law.

A population can explore several candidate regions, but superiority on a task
is an empirical question. A gradient-based kinetic variant also uses derivatives;
calling the whole algorithm gradient-free would omit that cost and requirement.
See {doc}`computation and experiments <../parts/07_experiments>`.
:::

(sec-fg-faq-foundations)=
## Interpretation, evidence, and mathematical status

(sec-fg-faq-why-fractal)=
### What does the name “Fractal Gas” imply?

:::{div} feynman-prose
The name identifies the algorithm family and its connection to the Fractal Set
record. The actual specification includes length and time scales: companion
bandwidths, jitter, smoothing, and a time step. Exact scale invariance or a
particular fractal dimension requires a separate statement and analysis.

Use the {doc}`literature guide <literature>` for historical sources and the
{doc}`algorithm introduction <../1_the_algorithm/01_algorithm_intuition>` for the
operational definition.
:::

(sec-fg-faq-physics-from-optimization)=
### What kind of connection between optimization and physics is being studied?

:::{div} feynman-prose
One can construct fields and geometric observables from an algorithm's record,
then study their algebra and dynamics. A claimed mathematical equivalence must
specify the objects, maps, and properties preserved. A physical identification
additionally concerns agreement with measurements and the range of that agreement.

Neither a visual resemblance nor a shared group name establishes the full
connection. The field chapters provide the constructions to inspect; unproved
parts remain conditional or conjectural.
:::

(sec-fg-faq-falsifiability)=
### What can experiments test?

:::{div} feynman-prose
Experiments can test a specified prediction for a specified implementation and
parameter regime. Examples include an update identity, a conditional relaxation
curve, an observable's population-size dependence, or a geometric estimator's
behavior under refinement. State the predicted quantity, sampling procedure,
uncertainty, and calibration inputs before interpreting agreement.

A discrepancy can reveal a coding error, a violated hypothesis, an inaccurate
approximation, or a false conjecture. Distinguishing these requires checking the
model and measurement together. Agreement on finite runs supports the tested
comparison; it is not a proof of a universal continuum or physical identification.
See {doc}`empirical studies <../2_fractal_set/06_empirical_validation>` and the
{doc}`calibration guide <../2_fractal_set/09_qft_calibration>`.
:::

(sec-fg-faq-calibration-evidence)=
### What does a successful coupling or mass fit establish?

:::{div} feynman-prose
The calibration report solves a declared dictionary between physical targets,
algorithmic parameters, and measured pair statistics. Its inversion is exact
for the fixed inputs. A new simulation can change those statistics, so a
self-consistent calibration also remeasures them and tests the resulting
correlations. See the {doc}`calibration report <../2_fractal_set/07_qft_calibration_report>`.

The active channel definitions matter before fitting. Under the implemented
same-frame masks, the LR fraction and coupling-magnitude channels are exactly
zero; {prf:ref}`prop-qft-ew-chirality-realization` proves this from the role partition.
The routine named `compute_su2_gauge_link` returns a scalar phase, whose
representation properties are stated in {prf:ref}`prop-qft-ew-spinor-realization`.
A fitted channel uses the actual operator, not the interpretation suggested by
its name. {prf:ref}`thm-qft-ew-active-pipeline` identifies the correlator keys
that enter the mass fit.

For fixed correlator data, changing the time unit rescales every fitted rate
and preserves rate ratios. Changing the time step or dynamics produces new
data and need not preserve those ratios. Report the nonzero channels, fit
windows, lag units, sampling uncertainty, and calibration inputs. Agreement
then tests the declared correspondence at the measured resolution; continuum
and physical-field identifications retain their own hypotheses.
:::

(sec-fg-faq-rigor-classification)=
### How should I read the mathematical status of a claim?

:::{div} feynman-prose
Follow the statement to its hypotheses and proof. The volume contains complete
analytical chains: component drift and surviving-block bounds to finite-particle
QSD convergence; normalized derivative calculus through actual greedy sampling
to smooth fitness; structural LSI and full-generator dissipation to entropy
convergence; and gain-loss well-posedness, contraction, and consistency to
mean-field attraction and stationary chaos.

A later geometric or physical application uses the specific law, generator,
representation, and scaling in that chain. Its remaining identifications are
stated at the point where they enter. This distinguishes a proved mathematical
implication from the task of verifying its inputs for a particular run.
:::

::::{div} feynman-added
| Type of statement | What to check |
|---|---|
| Definition or construction | The specified data, domain, and operations |
| Proved implication | The complete argument under the stated hypotheses |
| Conditional application | The hypotheses that remain to be verified for the chosen algorithm |
| Conjecture | The precise proposed conclusion and the argument still missing |
| Numerical result | Implementation, estimator, uncertainty, and calibration |
::::

:::{div} feynman-prose
The {doc}`latent analysis <../1_the_algorithm/02_fractal_gas_latent>` and
{doc}`continuum conditions <../convergence_program/16_continuum_discharge>` make
these distinctions explicit. Use the {doc}`reference material <../parts/08_reference>`
and {doc}`volume introduction <../intro_fractal_gas>` to locate the surrounding
arguments.
:::
