# Complete execution data and parameter regimes

(sec-native-complete-execution-data)=
## Complete execution data

:::{prf:definition} Complete native execution record
:label: def-native-complete-execution-record

For an existing gas instance, retain the data

$$
\mathfrak P=(\mathcal V,\mathcal C,\mathcal L,\mu_0,
 \mathcal E,\mathcal O,\mathcal A).
$$

Here $\mathcal V$ is its twelve-component tuple in
{prf:ref}`def-gas-variant`. The configuration $\mathcal C$ contains every
field of the instantiated configuration objects, recursively, including
integrator, stage and update schedules, population size, dimension, noise,
fitness, donor, history, elite, boundary, geometry and enabled-feature fields.
The landscape/provider data $\mathcal L$ contain the configured reward,
potential, gradient, metric, curl, environment and domain maps, their parameters
and any internal state read by the update. The initial law $\mu_0$ includes all
retained state, donor-history, clock-residue and cached-geometry coordinates.
The execution convention $\mathcal E$ records arithmetic, backend, random-stream
law or fixed seed, allocation/error/termination policies and code branch
conventions. The observation record $\mathcal O$ contains the configured stages,
channels, masks, weights, localization, reconstruction and edge-coefficient
recipes. The calibration record $\mathcal A$ contains the time, length, speed
and action units consumed by the claimed physical readout.

For Rust execution, $\mathcal C$ includes the complete nested `GasConfig` and,
when used, `RunConfig`. Its top-level `GasConfig` fields are `n_elite`, `backend`,
`precision`, `seed`, `boundary`, `distance_donors`, `cloning_donors`, `reducer`,
`fitness`, `clone_decision`, `clone_transform`, `kinetic`, `qft`, `geometry`,
`include_truncated`, `invalid_reward`, `max_batch_elements`, `max_memory_bytes`.
`RunConfig` additionally retains `benchmark`, `potential`, `physics_metric`,
`geometry_reward`, `reward_shift`, `walkers`, `dimensions`, `initial_lower`,
`initial_upper` and its nested `gas`. Every tagged enum payload and provider is
retained, including the full `QftExecutionConfig` with dense viscosity, graph
viscosity, innovation shifts and curl, and the geometry pipeline.

For Python execution, $\mathcal C$ includes all instantiated `EuclideanGas`,
`KineticOperator`, `CloneOperator`, `FitnessOperator` and both
`CompanionSelection` configuration fields, the actual bounds, callable objects
and runtime overrides. In particular the enable flags, frozen-best rule,
periods, periodic-distance policy, kinetic substeps, thermostat mode, derivative
detach modes, metric/noise branch and graph-weight policies remain explicit.
Rust and Python tags retain their respective update semantics.

The real-coordinate analytic instance records independent innovations with
their configured full distributions as its execution convention. A finite-
precision fixed-seed execution records its actual finite stream. These are
distinct execution records. The narrower tuples
{prf:ref}`def-slc-parameter-register`,
{prf:ref}`def-cgd-parameter-register`,
{prf:ref}`def-cg-mf-parameter-register` and
{prf:ref}`def-lqft-edge-spectral-parameters` specify named restrictions of this
record; their additional excluded features retain their declared values.
:::

:::{prf:proposition} Native history laws and parameter restriction
:label: prop-native-complete-parameter-law

Fix $\mathfrak P$ and a finite horizon at which its existing staged update is
defined. Let $\Xi_\ell$ be the random input of stage $\ell$, with its actual
conditional law $K_{\mathfrak P,\ell}$ given the preceding record. Its full
recorded history law is the ordered iterated integral of those laws and its
deterministic stage maps, starting from $\mu_0$. Consequently every bounded
history observable, finite reflected matrix, covariance, source response and
declared finite-edge matrix is a function of the full execution record.

A proof using a restricted tuple applies to a full execution record when
its consumed stages agree with those of that restriction and its derived
parameter tests hold. Equality of variant names alone supplies no such transfer.
If a changed field is used only by a passive readout, the state history law is
unchanged and the joint readout law is its actual pushforward. If a changed
field enters a force, noise, fitness, donor or scheduling stage, it remains in
that stage's conditional kernel or deterministic map.

For fixed-seed deterministic execution and a deterministic initial state, every
bounded recorded observable is deterministic. Its sampling variance is zero.
Absolute-continuity or nondegeneracy assertions for independent continuous
innovations therefore apply to that analytic convention, with numerical
execution requiring a separately proved comparison.
:::

:::{prf:proof}
At the first stage integrate the initial state against $\mu_0$ and its actual
random input against $K_{\mathfrak P,1}$. Apply its configured deterministic
map and retain the specified fields. Repeat with the next conditional law.
Induction on the finite list of stages gives the iterated integral for every
bounded cylinder, including the terminal cylinder of the entire record.
Equality of the consumed maps and conditional kernels makes each integral
identical by the same induction. Passive recording composes the resulting
history with its recorded measurable map and hence gives the pushforward law.
Feedback instead changes a map or kernel in the induction. A fixed seed and
deterministic initial state determine every input and output in that induction,
so the law is a point mass and its centered observables vanish. These arguments
preserve the actual stage order and randomness convention.
:::

:::{prf:definition} Derived regime and discharge register
:label: def-native-derived-regime-register

For a claimed property $T$, a derived sufficient regime is a subset
$\mathcal R_T$ of existing complete execution records, specified by evaluated
configuration equalities and inequalities between profiles calculated from
their original maps and parameters. A divergent profile or failed test gives
no positive certificate from that route. A record outside $\mathcal R_T$ is
classified as failing $T$ only when an actual counterexample or necessary
condition proves that failure. Otherwise its status under this route remains
unresolved. An unknown law-specific LSI, gap or convergence property retained
as a premise is an inherited conditional result, rather than a discharge of
that property.
:::
