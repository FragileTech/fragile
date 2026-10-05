# Chapter 5 native experiments

Run from `algorithmic-gas`:

```bash
cargo run --offline -p algorithmic-gas-benchmarks --bin gas-proof-experiments -- run proof-validation/chapter05-native-full.json --chapter 5 --output outputs/convergence/chapter05-native-FRESH --strict
```

The output directory must be fresh. Each gzip CBOR chunk contains complete native
populations, both force evaluations, all BAOAB stages, actual Gaussian innovations,
terminal coordinates and cap outputs. Coupling plans and reference calculations
are gzip JSON. SHA256 checksums are committed after every archive. Native capped
engines also save resumable checkpoints at chunk boundaries. Archive chunks are
limited to 128 steps and 128 MiB before compression.

| Experiment | Measured quantity | Prediction and scope |
|---|---|---|
| Capped quadratic, N=4,16,64 and d=1,2 | Optimal normalized empirical physical Q transport | Fixed cap theorem coefficient delta, independent of N and d |
| Explicit admissible coupling | Coupling cost and its difference from optimal transport | Optimal transport cannot exceed a chosen coupling; the initial translation coupling is checked to be optimal |
| First cap application | Squared coupled velocity difference minus `(1-eta)` times pre-cap difference | Exact quadratic cap dissipation bound |
| Actual OU innovations | Second Gaussian moment averaged per complete replicate | Unit standardized covariance |
| Terminal box, near-boundary live inputs | Actual paired alive/dead mismatch event | Conditional `min(1,2|Delta x|_1/(s sqrt(2 pi)))`, computed from native pre-diffusion A2 coordinates |
| Quadratic timestep refinement | Paired quadratic moment bias and mean-square path error | Independent analytic BAOAB and exact linear SDE calculations |
| Nonconvex cosine perturbation | Native nonlinear-versus-quadratic displacement | Global bounded-force perturbation certificate; approximate nonlinear reference has an explicit RMS error bound |

The capped experiment uses actual complete native updates with constant fitness,
zero positional clone jitter and no accepted cloning edges. Its kinetic parameters
are the canonical ones. The boundary is unbounded to isolate physical-coordinate
contraction. It does not replace the full canonical selected/killed swarm. Its
principal observable minimizes over all assignments of empirical atoms; temporary
representative pairing only constructs a coupling. Tests independently permute
both representatives and duplicate empirical atoms without changing the distance.

The input position, velocity and mixed perturbations are fixed before sampling.
The independent replicate seed supplies the uncertainty unit. Interacting rows
are never treated as independent samples for the swarm trajectory.

## Exact quadratic reference

For `dz=A z dt + g dW`, with

```text
A = [[0, 1], [-curvature, -gamma]],  g = [0, diffusion],
```

the implementation evaluates the analytic two-by-two exponential `E=exp(A t)`.
It obtains the covariance from `C_t=C_infinity-E C_infinity E^T`, where
`C_infinity=diag(diffusion^2/(2 gamma curvature), diffusion^2/(2 gamma))`.
The semigroup and stationary covariance are tested independently.

Each Brownian leaf jointly generates its exact SDE increment and OU increment.
Their cross covariance is the analytic integral
`diffusion^2 integral_0^h exp((A-gamma I)u)e_v du`. Coarse OU draws are weighted
sums of disjoint fine increments. The native engine receives these unit-variance
Gaussian draws through an explicitly saved source schedule. Consequently every
level retains the correct native Gaussian marginal and shares a Brownian path
with the exact SDE reference. Native substage records preserve both the original
addressed draw and its applied schedule.

The fixed horizon is 0.16, with h=0.04,0.02,0.01. Refinement uses the uncapped
extension with no final position diffusion. Reapplying the canonical cap every
smaller step would change the transition family and is not called SDE refinement.
The exact weak bias follows from separately propagating the BAOAB mean/covariance
and the exact SDE mean/covariance. The exact strong mean-square error additionally
propagates their Brownian cross covariance.

## Controlled nonquadratic reference

The nonlinear potential is `U(x)=x^2/2+0.1(1-cos(4x))`, with global force
Lipschitz bound 2.6 and Hessian lower bound -0.6. Its force differs from the
quadratic force by at most M=0.4 on the whole unbounded physical space.

With common Brownian motion the two SDEs satisfy
`D'=A D+[0,r(x)]`, where `|r|<=M`. Gronwall gives the pathwise bound

```text
|D(T)| <= M (exp(||A|| T)-1)/||A|| = B_continuous.
```

For each native BAOAB step, put `c=h/2,a=exp(-h)`. The difference obeys

```text
D_next = A_h D + [c^2(1+a), c(a-c^2(1+a))] r_B1 + [0,c] r_B2.
```

Both perturbations have magnitude at most M. The geometric operator-norm sum
therefore gives `B_discrete`, checked against every native nonlinear displacement.
Combining both bounds with the independently calculated linear strong error gives

```text
RMS(nonlinear_native - nonlinear_SDE)
    <= B_continuous + B_discrete + sqrt(exact_linear_MSE).
```

This is a conservative, globally controlled numerical reference. It does not claim
an exact nonlinear solution or identify the chapter's unspecified general weak-error
prefactors. Matching stability, barrier alignment and quantitative minorization
remain separate analytic obligations.

## Fixed input directions and uncertainty

The three perturbation directions are assigned deterministically by replicate
index modulo three. Their proportions are fixed, including the unequal counts
when the number of replicates is not divisible by three. For a replicate
observable, the reported mean retains these actual design weights. Its noise
standard error is

```text
SE = sqrt(sum_direction n_direction * sample_variance_direction) / samples.
```

This removes between-direction differences from the uncertainty of the fixed
design mean. Independent complete simulations within each direction supply the
sample variance. The OU standardized moments have the same Gaussian law across
directions and retain the ordinary iid replicate estimate.

Saved cap plans contain every replicate observable needed to recompute this
uncertainty. A separate review preserves the raw archive and its original report:

```bash
python3 proof-validation/reanalyze_chapter05_stratified_uncertainty.py \
  outputs/convergence/chapters04-06-experiments/full-20261004/chapter05 \
  outputs/convergence/chapter05-full-stratified-uncertainty.json
```

The command verifies the saved plan checksums and requires the complete six-case
matrix. It keeps every observed mean and analytic bound unchanged.

## Exact Gaussian cubature and a uniform quadratic weak coefficient

The additional `gas-kinetic-cubature` binary integrates the quadratic refinement
observables exactly. The common Brownian construction has 16 independent leaf
vectors with three standard Gaussian coordinates each. In this 48-dimensional
Gaussian space, the 96 nodes `+/-sqrt(48)e_j`, each with weight `1/96`, integrate
constants, linear functions and all quadratic coordinate products exactly.
The uncapped native quadratic BAOAB output and exact SDE output are affine in
these coordinates. Their squared norms, weak moment differences and coupled
squared errors are therefore integrated exactly up to floating-point roundoff.
These weighted nodes are a cubature rule, not independent Gaussian replicates.
Every native stage, bridge schedule, input/output checkpoint, node weight and
exact reference leaf increment is archived.

```bash
cargo run --offline -p algorithmic-gas-benchmarks --bin gas-kinetic-cubature -- \
  outputs/convergence/chapters04-06-experiments/full-20261004/kinetic-cubature
```

A separate analytic bound applies to `f(z)=|x|²+|v|²`, input `(0.8,0.4)`,
unit quadratic force, friction and velocity diffusion one, horizon `T=.16`,
and any `0<h<=H=.04` dividing T. It applies to the uncapped extension with no
additional position diffusion or accepted cloning. It has the same constant
for a normalized average over any number of particles with these input moments.

Let `A=[[0,1],[-1,-1]]`, `L=||A||`, and `S=3`, the sum of the norms of the
force-kick, position-drift and friction generators. Expanding the palindromic
BAOAB product gives derivatives at zero `I,A,A²`; the exact matrix exponential
has the same derivatives. Product differentiation bounds the third derivative
of the native matrix by `S³ exp(SH)` and that of the exact exponential by
`L³ exp(LH)`. Taylor's integral remainder therefore gives

```text
||A_h-exp(Ah)|| <= C_A h³,
C_A = (S³ exp(SH)+L³ exp(LH))/6.
```

The native noise covariance is `q(h)u(h)u(h)^T`, where
`q=(1-exp(-2h))/2`, `u=[h/2,1-h²/4]`. At zero its first two derivatives are
`J=diag(0,1)` and `AJ+JA^T=[[0,1],[1,-2]]`, matching the exact covariance
`integral_0^h exp(As)J exp(A^T s) ds`. On `[0,H]` define

```text
U=sqrt((H/2)²+(1+H²/4)²), U1=sqrt(1/4+(H/2)²), U2=1/2.
```

These bound `|u|,|u'|,|u''|`; also `q<=H`, `|q'|<=1`, `|q''|<=2`,
`|q'''|<=4`. Differentiating the covariance three times bounds its third
derivative by `4U²+12UU1+6(U1²+UU2)+6HU1U2`. The exact covariance's third
derivative is bounded by `4L² exp(2LH)`. Thus

```text
||Q_h-Q_exact(h)|| <= C_Q h³,
C_Q = (4U²+12UU1+6(U1²+UU2)+6H U1U2+4L²exp(2LH))/6.
```

For the exact uncentered second-moment matrix, its trace is bounded uniformly
by `M2=exp(2LT)(|z0|²+T)`. Write D for the difference between native and exact
second-moment matrices. Their recursions imply the nuclear-norm estimate

```text
||D_next||_1 <= exp(2Sh)||D||_1
  + h³ [C_A(exp(SH)+exp(LH))*M2 + 2C_Q].
```

Here the factor two converts the covariance operator norm to its nuclear norm
in phase dimension two. Summing `T/h` steps and bounding every propagation
factor by `exp(2ST)` gives the coefficient, without consulting the simulation:

```text
C_weak = T exp(2ST) [C_A(exp(SH)+exp(LH))*M2 + 2C_Q],
|E f(native)-E f(exact)| <= C_weak h².
```

The cubature report compares this analytic upper bound with the exact native
integrals at all three timesteps. This conservative coefficient has the scope
just stated; it does not identify the general nonlinear weak-error constants.
