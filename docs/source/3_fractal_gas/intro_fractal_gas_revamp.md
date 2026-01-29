---
title: "The Fractal Gas"
subtitle: "Population-Based Optimization with Gauge Structure"
author: "Guillem Duran-Ballester"
---
(sec-fractal-gas-intro-revamp)=

# The Fractal Gas - Intro Revamp (Draft)

This is a working replacement for `intro_fractal_gas.md`. It is written to make the
scope explicit, and to foreground the proof strategy for the QFT layer and for
convergence (via the appendices).

(sec-fg-revamp-purpose)=
## Purpose and scope

Volume 3 does three things, with explicit proof paths and references:

- Define the Fractal Gas algorithm as a population-based optimization and sampling
  scheme, with concrete operators and a precise state space.
- Prove convergence for a concrete, fully specified instantiation (the Euclidean Gas)
  using classical probabilistic and PDE tools.
- Derive the Fractal Set, its gauge structure, and the lattice QFT layer from the
  algorithmic dynamics, and connect this to emergent geometry.

The convergence chain in this intro is explicitly restricted to the appendices and
avoids hypostructure-based proofs by design.

(sec-fg-revamp-what)=
## What we do in Volume 3

- Specify the algorithmic operators and their composition for the Fractal Gas
  ({doc}`1_the_algorithm/01_algorithm_intuition`, {doc}`1_the_algorithm/02_fractal_gas_latent`).
- Build the Fractal Set data structure (CST/IG/IA edges and interaction triangles)
  as the discrete carrier for gauge and QFT constructions
  ({doc}`2_fractal_set/01_fractal_set`).
- Derive the causal set and lattice QFT layers from Fractal Set structure
  ({doc}`2_fractal_set/02_causal_set_theory`, {doc}`2_fractal_set/03_lattice_qft`).
- Extract the Standard Model gauge group from algorithmic redundancies and
  encode Yang-Mills dynamics and Noether currents
  ({doc}`2_fractal_set/04_standard_model`, {doc}`2_fractal_set/05_yang_mills_noether`).
- Develop emergent geometry on the fitness manifold and the related physics layer
  ({doc}`3_fitness_manifold/01_emergent_geometry`, {doc}`3_fitness_manifold/02_scutoid_spacetime`,
  {doc}`3_fitness_manifold/03_curvature_gravity`).
- Provide the classical convergence theory, mean-field limit, and quantitative
  bounds in the appendices (see the next section).

(sec-fg-revamp-qft-strategy)=
## Proof strategy for the QFT layer (physicist/mathematician TL;DR)

This QFT layer is not a handwaving analogy. It is a constructive chain with
explicit carriers, operators, and convergence/discharge steps. The lattice action
and continuum limits are grounded in operator-level convergence, QSD structure,
mean-field PDE theory, hypoelliptic regularity, and LSI-based mixing. Every
continuum hypothesis used by the causal set action is discharged internally,
and the appendices provide the full audit trail.

**Strategy (compressed):**

1. **Discrete carrier (Fractal Set).** Define the Fractal Set as a directed
   2-complex with CST/IG/IA edges and interaction triangles; this is the discrete
   substrate for holonomies and action functionals
   ({doc}`2_fractal_set/01_fractal_set`).

2. **Causal structure and continuum link.** Establish the causal order and the
   continuum lift for the CST component; define the nonlocal d'Alembertian and
   action limit on the emergent causal set
   ({doc}`2_fractal_set/02_causal_set_theory`).

3. **Lattice gauge construction.** Build Wilson loops and plaquette holonomies on
   the Fractal Set lattice and formalize the lattice gauge theory layer
   ({doc}`2_fractal_set/03_lattice_qft`).

4. **Gauge group from algorithmic redundancies.** Identify the gauge symmetries
   arising from the algorithmic structure (fitness, cloning, viscous coupling),
   and map these to the Standard Model gauge group; this fixes link-variable
   semantics and the color structure used in the lattice action
   ({doc}`2_fractal_set/04_standard_model`, {doc}`1_the_algorithm/02_fractal_gas_latent`).

5. **Yang-Mills action and currents.** Derive the Yang-Mills action from lattice
   holonomies and extract the corresponding Noether currents; this is the QFT
   bridge inside the Fractal Set framework
   ({doc}`2_fractal_set/05_yang_mills_noether`).

6. **Continuum hypotheses discharge.** Discharge the continuum assumptions for
   the causal set action and nonlocal d'Alembertian using internal QSD, mean-field,
   and mixing results (see the full discharge chain in the appendices)
   ({doc}`convergence_program/16_continuum_discharge`).

**Appendices inventory (all files, for audit trail):**

- Core appendices (main proofs and constructions): `appendices/00_faq.md`,
  `convergence_program/01_fragile_gas_framework.md`, `convergence_program/02_euclidean_gas.md`,
  `convergence_program/03_cloning.md`, `convergence_program/04_single_particle.md`,
  `convergence_program/04_wasserstein_contraction.md`, `convergence_program/05_kinetic_contraction.md`,
  `convergence_program/06_convergence.md`, `convergence_program/07_discrete_qsd.md`,
  `convergence_program/08_mean_field.md`, `convergence_program/09_propagation_chaos.md`,
  `convergence_program/10_kl_hypocoercive.md`, `convergence_program/11_hk_convergence.md`,
  `convergence_program/12_qsd_exchangeability_theory.md`, `convergence_program/13_quantitative_error_bounds.md`,
  `convergence_program/14_a_geometric_gas_c3_regularity.md`,
  `convergence_program/14_b_geometric_gas_cinf_regularity_full.md`,
  `convergence_program/15_kl_convergence.md`, `convergence_program/16_continuum_discharge.md`,
  `convergence_program/17_geometric_gas.md`.

- Proofs subfolder (technical lemmas and full derivations): `convergence_program/proofs/proof_cor_exponential_qsd_companion_dependent_full.md`,
  `convergence_program/proofs/proof_lem_effective_cluster_size_bounds_full(1).md`,
  `convergence_program/proofs/14_geometric_gas_cinf_regularity_full.md`,
  `convergence_program/proofs/proof_lem_telescoping_derivatives.md`,
  `convergence_program/proofs/proof_thm_exponential_tails.md`,
  `convergence_program/proofs/proof_cor_exp_convergence.md`,
  `convergence_program/proofs/proof_prop_complete_gradient_bounds.md`,
  `convergence_program/proofs/proof_thm_backbone_convergence.md`,
  `convergence_program/proofs/13_geometric_gas_c3_regularity.md`,
  `convergence_program/proofs/proof_cor_effective_interaction_radius_full.md`,
  `convergence_program/proofs/proof_lem_greedy_ideal_equivalence.md`,
  `convergence_program/proofs/proof_lem_hormander.md`,
  `convergence_program/proofs/proof_lem_effective_cluster_size_bounds_full.md`,
  `convergence_program/proofs/proof_cor_gevrey_1_fitness_potential_full.md`,
  `convergence_program/proofs/proof_lem_effective_companion_count_corrected_full.md`,
  `convergence_program/proofs/proof_thm_faa_di_bruno_appendix.md`,
  `convergence_program/proofs/proof_lem_variance_to_gap_adaptive.md`,
  `convergence_program/proofs/proof_lem_macro_transport.md`.


(sec-fg-revamp-convergence-strategy)=
## Convergence proof strategy (appendices only, no hypostructure)

This is the end-to-end convergence chain used in Volume 3 when we restrict to the
appendices and ignore hypostructure proofs. The focus is the Euclidean Gas.

- **Framework and instantiation.** Fix the axioms and definitions for Fragile Gas systems,
  and instantiate them as the Euclidean Gas with explicit state space and operators
  ({doc}`convergence_program/01_fragile_gas_framework`, {doc}`convergence_program/02_euclidean_gas`,
  {doc}`convergence_program/04_single_particle`).
- **Cloning operator drift.** Prove N-uniform contraction of positional variance and
  boundary potential, with bounded velocity expansion (Keystone Lemma and Safe Harbor)
  ({doc}`convergence_program/03_cloning`).
- **Kinetic operator drift and minorization.** Prove velocity dissipation, bounded positional
  expansion, boundary contraction, and a small-set condition for the kinetic step; this is the
  TV-ready kinetic analysis ({doc}`convergence_program/05_kinetic_contraction`).
- **Synergistic composition and QSD convergence (TV).** Combine the cloning and kinetic drifts
  into a Foster-Lyapunov inequality for the composed operator and obtain geometric ergodicity
  with a unique quasi-stationary distribution ({doc}`convergence_program/06_convergence`).
- **QSD structure and exchangeability.** Characterize the QSD and its exchangeability structure,
  which underpins the mean-field limit and concentration arguments
  ({doc}`convergence_program/07_discrete_qsd`, {doc}`convergence_program/12_qsd_exchangeability_theory`).
- **Mean-field PDE.** Derive the McKean-Vlasov Fokker-Planck equation with cloning and revival
  terms as the continuum forward equation ({doc}`convergence_program/08_mean_field`).
- **Propagation of chaos.** Construct the mean-field QSD as the large-N limit of finite-N QSDs
  via tightness, identification, and uniqueness ({doc}`convergence_program/09_propagation_chaos`).
- **Stronger convergence modes.** Establish KL convergence via a discrete-time logarithmic Sobolev
  inequality and, independently, an unconditional hypocoercive entropy route
  ({doc}`convergence_program/15_kl_convergence`, {doc}`convergence_program/10_kl_hypocoercive`).
- **HK convergence.** Upgrade the convergence statement to the Hellinger-Kantorovich metric by
  assembling mass, transport, and shape contraction lemmas ({doc}`convergence_program/11_hk_convergence`).
- **Quantitative rates.** Convert the qualitative mean-field limit into explicit finite-N error bounds
  and rates (including O(1/sqrt(N)) estimates) ({doc}`convergence_program/13_quantitative_error_bounds`).
- **Proof pack (appendices/proofs, complete list).** These are the technical lemmas and full derivations
  that complete the convergence chain. Each file is a standalone proof with its role indicated here:
  `convergence_program/proofs/proof_thm_faa_di_bruno_appendix.md` proves Faà di Bruno bounds for high-order derivative
  control; `convergence_program/proofs/proof_lem_telescoping_derivatives.md` proves the telescoping derivative identities
  used in multi-scale estimates; `convergence_program/proofs/proof_prop_complete_gradient_bounds.md` proves complete
  gradient bounds needed for regularity and stability; `convergence_program/proofs/proof_thm_exponential_tails.md` proves
  exponential tail control; `convergence_program/proofs/proof_cor_exp_convergence.md` derives exponential convergence
  corollaries from the main drift/LSI estimates; `convergence_program/proofs/proof_cor_exponential_qsd_companion_dependent_full.md`
  proves exponential QSD convergence in the companion-dependent setting; `convergence_program/proofs/proof_lem_hormander.md`
  verifies Hörmander’s bracket condition for hypoelliptic regularity; `convergence_program/proofs/proof_lem_macro_transport.md`
  proves the macro-scale transport estimate used in mean-field/continuum coupling;
  `convergence_program/proofs/proof_thm_backbone_convergence.md` proves convergence along the backbone construction;
  `convergence_program/proofs/proof_lem_variance_to_gap_adaptive.md` derives the variance-to-gap bound for adaptive control;
  `convergence_program/proofs/proof_lem_greedy_ideal_equivalence.md` proves equivalence between greedy and ideal companion
  selection in the regime of interest; `convergence_program/proofs/proof_lem_effective_companion_count_corrected_full.md`
  proves the corrected effective companion count bound; `convergence_program/proofs/proof_lem_effective_cluster_size_bounds_full.md`
  and `convergence_program/proofs/proof_lem_effective_cluster_size_bounds_full(1).md` prove effective cluster size bounds
  (two full derivations); `convergence_program/proofs/proof_cor_effective_interaction_radius_full.md` proves the effective
  interaction radius corollary; `convergence_program/proofs/proof_cor_gevrey_1_fitness_potential_full.md` proves the Gevrey-1
  regularity corollary for the fitness potential; and the geometric gas regularity proofs
  `convergence_program/proofs/13_geometric_gas_c3_regularity.md` and `convergence_program/proofs/14_geometric_gas_cinf_regularity_full.md`
  provide the C3 and C-infinity regularity details required by the geometric gas appendix.

```{mermaid}
flowchart TD
    A[01_fragile_gas_framework] --> B[02_euclidean_gas]
    B --> C[04_single_particle]
    B --> D[03_cloning]
    B --> E[05_kinetic_contraction]
    D --> F[06_convergence]
    E --> F
    F --> G[07_discrete_qsd]
    F --> H[12_qsd_exchangeability_theory]
    F --> I[08_mean_field]
    I --> J[09_propagation_chaos]
    J --> K[15_kl_convergence]
    J --> L[10_kl_hypocoercive]
    J --> M[11_hk_convergence]
    J --> N[13_quantitative_error_bounds]
    G --> J
    H --> J
    K --> N
    L --> N
    M --> N
    P[(convergence_program/proofs/*)] -.-> D
    P -.-> E
    P -.-> F
    P -.-> J
```

(sec-fg-revamp-reading)=
## Reading order (fast path)

- QFT and gauge derivation: {doc}`2_fractal_set/01_fractal_set` ->
  {doc}`2_fractal_set/02_causal_set_theory` -> {doc}`2_fractal_set/03_lattice_qft` ->
  {doc}`2_fractal_set/04_standard_model` -> {doc}`2_fractal_set/05_yang_mills_noether`.
- Convergence (appendices only): {doc}`convergence_program/01_fragile_gas_framework` ->
  {doc}`convergence_program/02_euclidean_gas` -> {doc}`convergence_program/03_cloning` ->
  {doc}`convergence_program/05_kinetic_contraction` -> {doc}`convergence_program/06_convergence` ->
  {doc}`convergence_program/08_mean_field` -> {doc}`convergence_program/09_propagation_chaos` ->
  {doc}`convergence_program/15_kl_convergence` -> {doc}`convergence_program/13_quantitative_error_bounds`.
