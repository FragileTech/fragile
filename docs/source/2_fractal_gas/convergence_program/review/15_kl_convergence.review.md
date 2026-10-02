# Mathematical Review: 15_kl_convergence.md

## Metadata

- Reviewed file: [15_kl_convergence.md](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/convergence_program/15_kl_convergence.md).
- Review date: 2026-10-02.
- Reviewer: Codex, adversarial exact-algorithm review.
- Scope: Complete chapter, read in source order; particular attention to whether its analytic tools characterize the executed, moving, cloning gas. This is not a machine-checked proof certification or a new verification of every external reference.
- Framework anchors: `def-kl-finite-particle-laws`, `cor-n-uniform-lsi`, `thm-kl-convergence-euclidean`, `thm-kl-canonical-cluster-entropy-bridge`, the exact marked-step entropy identities, and Chapters 18–19.

## Executive summary

- Critical: 0 newly identified local errors.
- Major: 0 newly identified local errors.
- Moderate: 0 newly identified local errors.
- Minor: 0 newly identified local errors.
- Notes: 3 exact-target obligations below.
- Primary theme: The implications are not a discharge of their hypotheses for the full coupled gas. The chapter generally states this distinction explicitly.

## Error log

| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| N-001 | `cor-n-uniform-lsi`, `thm-kl-convergence-euclidean` | Note | Scope restriction | The actual joint law, its LSI and the complete entropy derivative still require identification. |
| N-002 | `lem-meanfield-cloning-dissipation-hybrid`, frozen-field and frozen-OU results | Note | Algorithm mismatch if transferred | These separately specified operators must not replace the canonical cloning/kinetic update. |
| N-003 | Continuous entropy, Gaussian smoothing and discrete defect results | Note | Scope restriction | A continuous reference calculation does not establish a fixed-step numerical entropy estimate. |

## Detailed findings

### [N-001] Joint-law and derivative provenance remain necessary

- Location: source lines 543–578 and 1163–1206.
- The product Gibbs law, bounded joint tilt, joint log-density curvature and contractive additive-noise flow are sufficient LSI criteria. None identifies the coupled cloning QSD merely because the stable potential is quadratic or marginal tails are Gaussian.
- The full entropy theorem explicitly supposes an LSI of the actual law and a strictly negative complete modified-Fisher derivative. Its short Grönwall proof is a valid implication, not a proof of those premises.
- Downstream impact: Uniform concentration, stationary fluctuation compactness and stationary limiting LSIs cannot be marked established for the recorded viscous gas on this basis. Discrete alive/dead entropy also remains separate.
- Fix guidance: Derive a criterion for the same full marked kernel/law; retain all selection, coupled B-stage, killing and normalization terms. If it does not close, leave the application unresolved. Do not assign a convenient Gibbs density to the actual QSD.
- Required new assumptions: None authorized. The missing statements must be proved from the configured algorithm and initial/landscape data.
- Validation: Match state space, reference law, generator/kernel, derivative domain and all parameter dependencies before using the estimate. Chapter 18's finite-population discrete entropy theorem is a different, correctly identified route; its nonuniform constants cannot be erased.

### [N-002] Auxiliary operators are not native substitutes

- The hybrid mean-field dissipation lemma uses its displayed Metropolis-type pair rule; the following remark does not identify it with every other cloning rule. Frozen alignment and frozen-position OU results hold for their named processes. The degree-ratio requirement of the row-normalized OU calculation is visible and cannot be inferred globally for a decaying Gaussian kernel on unbounded configurations.
- The canonical cluster entropy bridge retains actual selection/correlation but is restricted to its stated uncoupled kinetic case. The random proof partition is not a change of collision components.
- Framework-first fix direction: Calculate the contribution of the actual accepted-edge rule and both actual coupled kicks. For the moving swarm, identify its own conditional/joint law rather than inserting the frozen Gaussian. Do not change acceptance, replace component Haar rotations, add a denominator floor or freeze a dynamical field to obtain a theorem about a different kernel.
- Required new assumptions: None. These are scope boundaries already largely explicit in the chapter.

### [N-003] Continuous, partial-noise and discrete objects stay distinct

- The full Gaussian Fisher contraction is not automatically a theorem for correlated/partial cloning noise. A smooth-test weak-error estimate is not an entropy-functional defect bound. The chapter correctly states these limits.
- Doob reweighting is useful only with the exact eigenfunction and the inverse change of measure retained; its invariant law is not the killed QSD itself. A step-normalized nonlinear law is not automatically a whole-horizon survivor law.
- Validation: Use the complete marked-step entropy chain rule or an actual numerical-kernel functional estimate, not an auxiliary elliptic generator or a continuous Poisson jump realization in place of the fixed-step transition.

## Scope restrictions and clarifications

- A static LSI is a law inequality, not by itself an entropy-decay theorem for the executed algorithm.
- Marginal moments and Gaussian tails do not imply joint LSI, joint density curvature or population-uniform mixing.
- The final structural-landscape remark cites a specific closed selection regime; it does not remove that regime's displayed conditions or transfer its result to every viscosity/metric configuration.

## Proposed edits

- No mathematical or algorithm source was changed during this review. Preserve the conditional scope in every downstream use.

## Open questions

- Which proved structural criterion, if any, holds for the actual moving, cloning, coupled marked QSD?
- Can its complete fixed-step entropy constant be bounded in primitive parameters uniformly in population size without changing the transition?
