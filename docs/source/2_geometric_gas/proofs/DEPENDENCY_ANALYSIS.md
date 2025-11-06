# Dependency Analysis for Geometric Gas Theorems

**Generated**: 2025-10-25

## Summary

- **Total theorems needing proof**: 89
- **Topological sort**: ✓ Successful (no circular dependencies)
- **Root theorems** (no internal dependencies): 52
- **Internal dependencies**: 84 theorem→theorem dependencies
- **Missing dependencies**: 13 unique labels

## Execution Order

The following order ensures all dependencies are proven before dependent theorems:

### Phase 1: Foundations (Theorems 1-20)

1. `lem-greedy-ideal-equivalence` (20_geometric_gas_cinf_regularity_full.md)
2. `prop-complete-gradient-bounds` (16_convergence_mean_field.md)
3. `lem-macro-transport` (11_geometric_gas.md)
4. `thm-data-processing` (16_convergence_mean_field.md)
5. `thm-c2-established-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
6. `thm-stage0-complete` (16_convergence_mean_field.md)
7. `thm-c4-established-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
8. `prop-rate-metric-ellipticity` (18_emergent_geometry.md)
9. `thm-fitness-third-deriv-proven` (15_geometric_gas_lsi_proof.md)
10. `thm-exponential-tails` (16_convergence_mean_field.md)
11. `lem-wasserstein-revival` (16_convergence_mean_field.md)
12. `lem-velocity-squashing-compact-domain-full` (20_geometric_gas_cinf_regularity_full.md)
13. `thm-backbone-convergence` (11_geometric_gas.md)
14. `thm-main-informal` (18_emergent_geometry.md)
15. `thm-qsd-stability` (16_convergence_mean_field.md)
16. `lem-telescoping-derivatives` (13_geometric_gas_c3_regularity.md)
17. `thm-main-explicit-rate` (16_convergence_mean_field.md)
18. `lem-hormander` (16_convergence_mean_field.md)
19. `thm-signal-generation-adaptive` (11_geometric_gas.md)
20. `thm-qsd-existence` (11_geometric_gas.md)

### Phase 2: Core Regularity (Theorems 21-40)

21. `prop-gevrey-regularization-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
22. `lem-micro-coercivity` (11_geometric_gas.md)
23. `lem-mean-first-derivative` (11_geometric_gas.md)
24. `lem-kinetic-energy-bound` (16_convergence_mean_field.md)
25. `lem-fisher-bound` (16_convergence_mean_field.md)
26. `thm-finite-n-lsi-preservation` (16_convergence_mean_field.md)
27. `thm-exponential-convergence-local` (16_convergence_mean_field.md)
28. `lem-drift-condition-corrected` (16_convergence_mean_field.md)
29. `thm-c3-established-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
30. `thm-c1-established-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
31. `cor-instantaneous-smoothing-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
32. `cor-derivatives-continuous-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
33. `thm-alpha-net-explicit` (16_convergence_mean_field.md)
34. `prop-talagrand-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
35. `prop-gaussian-tail-bounds-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
36. `prop-brascamp-lieb-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
37. `prop-bakry-emery-gamma2-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
38. `lem-z-score-cinf-inductive` (19_geometric_gas_cinf_regularity_simplified.md)
39. `lem-variance-to-gap-adaptive` (11_geometric_gas.md)
40. `lem-variance-gradient` (11_geometric_gas.md)

### Phase 3: Advanced Results (Theorems 41-60)

41. `lem-telescoping-all-orders-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
42. `lem-strong-max-principle` (16_convergence_mean_field.md)
43. `lem-qsd-density-bound-with-cloning-full` (20_geometric_gas_cinf_regularity_full.md)
44. `lem-micro-reg` (11_geometric_gas.md)
45. `lem-mean-second-derivative` (11_geometric_gas.md)
46. `lem-mean-cinf-inductive` (19_geometric_gas_cinf_regularity_simplified.md)
47. `lem-irreducibility` (16_convergence_mean_field.md)
48. `lem-fokker-planck-density-bound-conservative-full` (20_geometric_gas_cinf_regularity_full.md)
49. `lem-entropy-l1-bound` (16_convergence_mean_field.md)
50. `lem-dissipation-decomp` (11_geometric_gas.md)
51. `cor-wasserstein-convergence-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
52. `cor-hypoelliptic-regularity` (16_convergence_mean_field.md)
53. `cor-compact-bounds-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
54. `thm-qsd-smoothness` (16_convergence_mean_field.md)
55. `thm-qsd-positivity` (16_convergence_mean_field.md)
56. `thm-optimal-parameter-scaling` (16_convergence_mean_field.md)
57. `thm-main-convergence` (18_emergent_geometry.md)
58. `thm-lsi-mean-field` (11_geometric_gas.md)
59. `thm-joint-not-contractive` (16_convergence_mean_field.md)
60. `thm-hypoellipticity-cinf` (19_geometric_gas_cinf_regularity_simplified.md)

### Phase 4: Convergence & LSI (Theorems 61-80)

61. `thm-gevrey-1-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
62. `thm-essential-self-adjoint-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
63. `thm-corrected-kl-convergence` (16_convergence_mean_field.md)
64. `thm-algorithmic-tunability` (18_emergent_geometry.md)
65. `prop-velocity-gradient-uniform` (16_convergence_mean_field.md)
66. `prop-limiting-regimes` (11_geometric_gas.md)
67. `prop:bounded-adaptive-force` (11_geometric_gas.md)
68. `lem-weight-derivatives` (11_geometric_gas.md)
69. `lem-variance-hessian` (11_geometric_gas.md)
70. `lem-variance-cinf-inductive` (19_geometric_gas_cinf_regularity_simplified.md)
71. `lem-rho-pipeline-bounds` (11_geometric_gas.md)
72. `lem-raw-to-rescaled-gap-rho` (11_geometric_gas.md)
73. `lem-log-gap-bounds-adaptive` (11_geometric_gas.md)
74. `thm-revival-kl-expansive` (16_convergence_mean_field.md)
75. `thm-qsd-existence-corrected` (16_convergence_mean_field.md)
76. `thm-lsi-geometric` (17_qsd_exchangeability_geometric.md)
77. `thm-inductive-step-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
78. `thm-cinf-regularity` (19_geometric_gas_cinf_regularity_simplified.md)
79. `thm-c2-regularity` (11_geometric_gas.md)
80. `thm-c1-regularity` (11_geometric_gas.md)

### Phase 5: Final Results (Theorems 81-89)

81. `thm-adaptive-lsi-main` (15_geometric_gas_lsi_proof.md)
82. `prop-reward-bias-rho` (11_geometric_gas.md)
83. `prop-diversity-signal-rho` (11_geometric_gas.md)
84. `cor-geometric-ergodicity-lsi` (11_geometric_gas.md)
85. `cor-exp-convergence` (11_geometric_gas.md)
86. `cor-axioms-verified` (11_geometric_gas.md)
87. `thm-stability-condition-rho` (11_geometric_gas.md)
88. `thm-lsi-adaptive-gas` (11_geometric_gas.md)
89. `thm-keystone-adaptive` (11_geometric_gas.md)

## Dependency Statistics

### Theorems by Dependency Count

The following theorems have the most dependencies (and should be proven last):

1. `prop-diversity-signal-rho` - 4 dependencies
2. `thm-stability-condition-rho` - 4 dependencies
3. `thm-cinf-regularity` - 4 dependencies
4. `thm-lsi-adaptive-gas` - 3 dependencies
5. `lem-variance-hessian` - 3 dependencies
6. `thm-c2-regularity` - 3 dependencies
7. `lem-log-gap-bounds-adaptive` - 3 dependencies
8. `thm-keystone-adaptive` - 3 dependencies
9. `thm-inductive-step-cinf` - 3 dependencies
10. `prop:bounded-adaptive-force` - 2 dependencies

### Root Theorems (No Internal Dependencies)

52 theorems have no dependencies on other theorems in the pipeline. These can be proven in any order and serve as the foundation:

- `lem-greedy-ideal-equivalence`
- `prop-complete-gradient-bounds`
- `lem-macro-transport`
- `thm-data-processing`
- `thm-c2-established-cinf`
- `thm-stage0-complete`
- ... (and 46 more)

## Missing Dependencies

The following dependencies are referenced but not found in the current pipeline or glossary:

### Missing Definitions (4)

These definitions need to be added or their labels need to be corrected:

- `def-adaptive-generator-cinf` - Referenced by C∞ regularity theorems
- `def-localized-mean-field-fitness` - Referenced by `prop-limiting-regimes`
- `def-localized-mean-field-moments` - Referenced by `prop-limiting-regimes`, `prop:bounded-adaptive-force`
- `def-unified-z-score` - Referenced by `prop-limiting-regimes`, `prop-rate-metric-ellipticity`

### Missing Assumptions (2)

- `assump-cinf-primitives` - Referenced by all C∞ regularity theorems in 19_geometric_gas_cinf_regularity_simplified.md
- `assump-uniform-density-full` - Referenced by density bound lemmas in 20_geometric_gas_cinf_regularity_full.md

### Document Cross-References (3)

These are references to other chapters and should be treated as satisfied externally:

- `doc-02-euclidean-gas` - Chapter 1 (Euclidean Gas)
- `doc-03-cloning` - Chapter 1 (Cloning theory)
- `doc-13-geometric-gas-c3-regularity` - Within Chapter 2 (C³ regularity)

### Other Missing (4)

- `lem-conditional-gaussian-qsd` - Referenced by `thm-fitness-third-deriv-proven`
- `rem-concentration-lsi` - Referenced by `cor-geometric-ergodicity-lsi`
- `thm-ueph-proven` - Referenced by `thm-adaptive-lsi-main`, `thm-fitness-third-deriv-proven`
- Malformed reference: ` \infty$ independent of N...` (appears to be a parsing error)

## Recommendations

1. **Add missing definitions**: The 4 missing definitions should be added as proper `{prf:definition}` blocks with the specified labels.

2. **Add missing assumptions**: The 2 missing assumptions should be formalized as `{prf:axiom}` or `{prf:assumption}` blocks.

3. **Resolve label mismatches**: Some references like `thm-ueph-proven` may exist with different labels (e.g., `thm-ueph`). These should be reconciled.

4. **Fix parsing errors**: The malformed reference should be corrected in the source document.

5. **Document cross-references**: Consider adding explicit labels for document cross-references or marking them clearly as external dependencies.

## Usage

The full dependency graph is available in `theorem_dependencies.json` with the following structure:

```json
{
  "metadata": {
    "total_theorems_needing_proof": 89,
    "topological_sort_successful": true,
    "cycles_detected": []
  },
  "execution_order": ["label1", "label2", ...],
  "dependency_graph": [
    {
      "label": "theorem-label",
      "document": "filename.md",
      "line": 123,
      "type": "Theorem",
      "title": "Theorem Title",
      "status": "needs_proof",
      "dependencies": [...],
      "truly_missing_deps": [...]
    }
  ]
}
```

This can be used to:
- Drive the automated proof pipeline
- Track progress through the proof dependencies
- Identify critical path theorems
- Detect and resolve missing dependencies
