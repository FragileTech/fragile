# Chapter 6 native survivor-law experiments

The experiment runs independent native Rust trajectories from two initial swarm distributions. Each starts with independently drawn positions around either `-0.45 L` or `+0.45 L`, with half-width `0.08 L`, and zero velocity. The complete matrix uses populations 4, 16 and 64 in dimensions 1 and 2, plus a population-2 terminal-box hazard experiment. Canonical cases retain the actual quadratic reward, competitive cloning, revival, accepted-component collisions, Gaussian jitter, capped BAOAB and terminal absorption. The hazard case declares its smaller box and larger positional diffusion explicitly.

Every completed native update is saved in a full CBOR archive chunk. These retain intermediate operator populations, validity, donor decisions, original component velocities, rotations, rewards and realized noises. Each chunk also has a resumable checkpoint. The SHA256 index is committed during recording. Each trajectory's JSON manifest links its chunks and checkpoints, records the independent seed, and retains every extinction prediction and checkpoint phase sample.

| Quantity | Measurement and interpretation |
|---|---|
| Native absorption | First completed state with no alive walkers. Singleton native swarms continue through mandatory revival. |
| Chapter 6 cemetery | First completed state with fewer than two alive walkers. Subsequent native states are excluded from this externally stopped law; the algorithm is unchanged. |
| Conditional moments | Mean over each initial distribution's own surviving trajectories. Zero survivors gives an unavailable conditional law. |
| Whole-swarm state cost | Minimum over all permutations of the average marked phase cost. Dead positions are discarded; original dead velocities remain because the native collision uses them. |
| Whole-swarm empirical-law distance | Exact outer optimal transport for N=2/4. For larger N, an explicit block coupling preserves every survivor's uniform mass and supplies a coupling upper bound for the complete empirical laws. |
| Alive marginal law | Independently select one uniform alive walker from each surviving swarm, then compare these empirical samples by optimal transport. This is the swarm-first alive sampling law. |
| Empirical decay | Initial and terminal empirical-law costs and their ratio. Independent finite empirical samples retain a sampling floor; their distance is not distance to a known QSD. |
| Barrier | Alive-only `phi(x) = -sum_j log(1-(x_j/L)^2)`, averaged with denominator N. Log evaluation floors its positive argument at the smallest positive floating-point value. |
| Safe fraction | Alive walkers inside the declared smaller box, divided by N. The barrier counting inequality is tested in this normalized form. |
| One-step absorption | Reconstruct the mean before the final independent position Gaussian by subtracting its recorded noise. Gaussian box probabilities yield per-row survival probabilities and the exact Poisson-binomial k=0 and k<2 absorption predictions conditional on preparation. |
| Rare-event precision | Independent trajectory counts, Wilson 95% intervals and the one-sided zero-event upper bound. Zero observed extinctions does not mean zero hazard. |

Conditional absorption comparisons retain the preceding cloning and collision randomness. Independence is used only for the final positional Gaussian after that preparation has been recorded. The CDF implementation has a stated absolute numerical error bound, propagated through the product law. Comparison tolerances use conditional Bernoulli variance and an explicit Bernstein error budget over times and conventions.

The integrable barrier has box integral `(2L)^d * 2d(1-log 2)`. Final positional Gaussian density is bounded uniformly in its prior mean by `(2 pi sigma_pos^2 h)^(-d/2)`. Their product gives an **N-independent** expected normalized barrier bound, which is compared with unconditional native and stopped-chain measurements. Conditional barrier means retain their own survival denominator; they are not assigned an equilibrium moment without an invariant or QSD certificate.

For the harmonic native kernel, the report reconstructs the joint kinetic covariance and a positive all-alive target minorization. Restrict each used jitter coordinate to `|Z| <= 1`, use the actual accepted-component velocity bound `(1+2 alpha) Vmax`, lower-bound the joint Gaussian on an interior position/velocity target, and apply the inverse radial-cap Jacobian, which is at least one. The resulting epsilon is stored logarithmically. Its hypotheses include bounded original velocities in every slot and live positions inside the box; arbitrary entering dead positions are overwritten by revival. This certificate depends on N and establishes positivity. It is **not** advertised as an N-uniform mixing rate or the two-sided surviving-block hypothesis needed to identify a QSD.

Global QSD eigenvalues, two-sided surviving-block constants, joint-law LSI and analytic sensitivity endpoints remain separate required certificates. The fresh data allow future reanalysis without rerunning the simulations.
