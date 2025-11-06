# Dependency Layers for Geometric Gas Theorems

**Generated**: 2025-10-25

Theorems grouped by dependency depth. Layer 0 has no dependencies, layer n depends on theorems from layers 0 to n-1.

## Summary

- Total layers: 7
- Total theorems: 89

## Layer 0 (52 theorems)

**Foundation layer** - No internal dependencies. These theorems can be proven in any order.

### 11_geometric_gas.md

- `prop-limiting-regimes` (line 310)
- `thm-backbone-convergence` (line 1035)
- `cor-exp-convergence` (line 1684)
- `thm-qsd-existence` (line 2106)
- `thm-lsi-mean-field` (line 2217)
- `lem-dissipation-decomp` (line 2271)
- `lem-micro-coercivity` (line 2304)
- `lem-macro-transport` (line 2316)
- `lem-micro-reg` (line 2328)
- `lem-weight-derivatives` (line 2550)
- `thm-signal-generation-adaptive` (line 3124)
- `lem-variance-to-gap-adaptive` (line 3159)
- `lem-rho-pipeline-bounds` (line 3175)
- `lem-raw-to-rescaled-gap-rho` (line 3211)

### 13_geometric_gas_c3_regularity.md

- `lem-telescoping-derivatives` (line 199)

### 15_geometric_gas_lsi_proof.md

- `thm-fitness-third-deriv-proven` (line 802)

### 16_convergence_mean_field.md

- `thm-finite-n-lsi-preservation` (line 517)
- `thm-data-processing` (line 577)
- `lem-wasserstein-revival` (line 728)
- `thm-revival-kl-expansive` (line 962)
- `thm-joint-not-contractive` (line 986)
- `thm-stage0-complete` (line 1015)
- `thm-qsd-existence-corrected` (line 1214)
- `thm-qsd-stability` (line 1342)
- `lem-hormander` (line 1383)
- `cor-hypoelliptic-regularity` (line 1403)
- `prop-complete-gradient-bounds` (line 1881)
- `lem-drift-condition-corrected` (line 2087)
- `thm-exponential-tails` (line 2249)
- `thm-corrected-kl-convergence` (line 3084)
- `lem-fisher-bound` (line 3529)
- `lem-kinetic-energy-bound` (line 3610)
- `thm-exponential-convergence-local` (line 3958)
- `thm-main-explicit-rate` (line 4150)
- `thm-alpha-net-explicit` (line 4449)
- `thm-optimal-parameter-scaling` (line 4533)

### 17_qsd_exchangeability_geometric.md

- `thm-lsi-geometric` (line 276)

### 18_emergent_geometry.md

- `thm-main-informal` (line 261)
- `thm-main-convergence` (line 827)
- `prop-rate-metric-ellipticity` (line 2631)
- `thm-algorithmic-tunability` (line 3419)

### 19_geometric_gas_cinf_regularity_simplified.md

- `thm-c1-established-cinf` (line 369)
- `thm-c2-established-cinf` (line 377)
- `thm-c3-established-cinf` (line 385)
- `thm-c4-established-cinf` (line 395)
- `lem-telescoping-all-orders-cinf` (line 431)
- `lem-variance-cinf-inductive` (line 491)
- `prop-gevrey-regularization-cinf` (line 807)
- `cor-instantaneous-smoothing-cinf` (line 885)
- `prop-bakry-emery-gamma2-cinf` (line 964)

### 20_geometric_gas_cinf_regularity_full.md

- `lem-velocity-squashing-compact-domain-full` (line 568)
- `lem-greedy-ideal-equivalence` (line 2388)

## Layer 1 (10 theorems)

**Depends on**: Layers 0-0

### 11_geometric_gas.md

- `lem-mean-first-derivative` (line 2586) - Depends on: `lem-weight-derivatives`
- `lem-log-gap-bounds-adaptive` (line 3267) - Depends on: `thm-signal-generation-adaptive`, `lem-variance-to-gap-adaptive`, `lem-raw-to-rescaled-gap-rho`

### 15_geometric_gas_lsi_proof.md

- `thm-adaptive-lsi-main` (line 1279) - Depends on: `thm-fitness-third-deriv-proven`

### 16_convergence_mean_field.md

- `thm-qsd-smoothness` (line 1427) - Depends on: `lem-hormander`
- `thm-qsd-positivity` (line 1437) - Depends on: `lem-hormander`
- `lem-irreducibility` (line 1449) - Depends on: `lem-hormander`
- `prop-velocity-gradient-uniform` (line 1539) - Depends on: `lem-hormander`
- `lem-entropy-l1-bound` (line 3791) - Depends on: `lem-fisher-bound`

### 19_geometric_gas_cinf_regularity_simplified.md

- `lem-mean-cinf-inductive` (line 451) - Depends on: `lem-telescoping-all-orders-cinf`

### 20_geometric_gas_cinf_regularity_full.md

- `lem-fokker-planck-density-bound-conservative-full` (line 589) - Depends on: `lem-velocity-squashing-compact-domain-full`

## Layer 2 (6 theorems)

**Depends on**: Layers 0-1

### 11_geometric_gas.md

- `lem-mean-second-derivative` (line 2664) - Depends on: `lem-mean-first-derivative`, `lem-weight-derivatives`
- `thm-c1-regularity` (line 2819) - Depends on: `lem-mean-first-derivative`
- `prop-diversity-signal-rho` (line 3291) - Depends on: `thm-signal-generation-adaptive`, `lem-variance-to-gap-adaptive`, `lem-log-gap-bounds-adaptive` and 1 more

### 16_convergence_mean_field.md

- `lem-strong-max-principle` (line 1493) - Depends on: `lem-irreducibility`

### 19_geometric_gas_cinf_regularity_simplified.md

- `lem-z-score-cinf-inductive` (line 511) - Depends on: `lem-mean-cinf-inductive`, `lem-variance-cinf-inductive`

### 20_geometric_gas_cinf_regularity_full.md

- `lem-qsd-density-bound-with-cloning-full` (line 635) - Depends on: `lem-fokker-planck-density-bound-conservative-full`

## Layer 3 (5 theorems)

**Depends on**: Layers 0-2

### 11_geometric_gas.md

- `thm-lsi-adaptive-gas` (line 1834) - Depends on: `thm-lsi-mean-field`, `thm-c1-regularity`, `thm-fitness-third-deriv-proven`
- `lem-variance-gradient` (line 2742) - Depends on: `lem-mean-first-derivative`, `lem-mean-second-derivative`
- `thm-c2-regularity` (line 2932) - Depends on: `lem-mean-first-derivative`, `thm-c1-regularity`, `lem-mean-second-derivative`
- `prop-reward-bias-rho` (line 3351) - Depends on: `prop-diversity-signal-rho`

### 19_geometric_gas_cinf_regularity_simplified.md

- `thm-inductive-step-cinf` (line 531) - Depends on: `lem-mean-cinf-inductive`, `lem-variance-cinf-inductive`, `lem-z-score-cinf-inductive`

## Layer 4 (6 theorems)

**Depends on**: Layers 0-3

### 11_geometric_gas.md

- `prop:bounded-adaptive-force` (line 562) - Depends on: `thm-c1-regularity`, `thm-c2-regularity`
- `cor-geometric-ergodicity-lsi` (line 2033) - Depends on: `thm-lsi-mean-field`, `thm-lsi-adaptive-gas`
- `lem-variance-hessian` (line 2788) - Depends on: `lem-mean-first-derivative`, `lem-mean-second-derivative`, `lem-variance-gradient`
- `cor-axioms-verified` (line 3053) - Depends on: `thm-c1-regularity`, `thm-c2-regularity`
- `thm-stability-condition-rho` (line 3401) - Depends on: `thm-signal-generation-adaptive`, `prop-reward-bias-rho`, `prop-diversity-signal-rho` and 1 more

### 19_geometric_gas_cinf_regularity_simplified.md

- `thm-cinf-regularity` (line 697) - Depends on: `thm-inductive-step-cinf`, `lem-telescoping-all-orders-cinf`, `thm-c1-established-cinf` and 1 more

## Layer 5 (8 theorems)

**Depends on**: Layers 0-4

### 11_geometric_gas.md

- `thm-keystone-adaptive` (line 3429) - Depends on: `thm-signal-generation-adaptive`, `thm-stability-condition-rho`, `lem-raw-to-rescaled-gap-rho`

### 19_geometric_gas_cinf_regularity_simplified.md

- `cor-derivatives-continuous-cinf` (line 725) - Depends on: `thm-cinf-regularity`
- `cor-compact-bounds-cinf` (line 733) - Depends on: `thm-cinf-regularity`
- `thm-gevrey-1-cinf` (line 767) - Depends on: `thm-cinf-regularity`
- `thm-essential-self-adjoint-cinf` (line 842) - Depends on: `thm-cinf-regularity`
- `thm-hypoellipticity-cinf` (line 869) - Depends on: `thm-cinf-regularity`
- `prop-gaussian-tail-bounds-cinf` (line 893) - Depends on: `thm-cinf-regularity`
- `prop-talagrand-cinf` (line 915) - Depends on: `thm-cinf-regularity`

## Layer 6 (2 theorems)

**Depends on**: Layers 0-5

### 19_geometric_gas_cinf_regularity_simplified.md

- `cor-wasserstein-convergence-cinf` (line 932) - Depends on: `thm-cinf-regularity`, `prop-talagrand-cinf`
- `prop-brascamp-lieb-cinf` (line 946) - Depends on: `thm-cinf-regularity`, `prop-talagrand-cinf`

## Critical Path Analysis

The critical path is the longest dependency chain. The depth of the graph determines the minimum number of sequential proof steps.

- **Maximum depth**: 6
- **Theorems at maximum depth** (2): `cor-wasserstein-convergence-cinf`, `prop-brascamp-lieb-cinf`

### Example Critical Path (to `cor-wasserstein-convergence-cinf`)

1. `lem-telescoping-all-orders-cinf` (19_geometric_gas_cinf_regularity_simplified.md:431)
2. `lem-mean-cinf-inductive` (19_geometric_gas_cinf_regularity_simplified.md:451)
3. `lem-z-score-cinf-inductive` (19_geometric_gas_cinf_regularity_simplified.md:511)
4. `thm-inductive-step-cinf` (19_geometric_gas_cinf_regularity_simplified.md:531)
5. `thm-cinf-regularity` (19_geometric_gas_cinf_regularity_simplified.md:697)
6. `prop-talagrand-cinf` (19_geometric_gas_cinf_regularity_simplified.md:915)
7. `cor-wasserstein-convergence-cinf` (19_geometric_gas_cinf_regularity_simplified.md:932)

