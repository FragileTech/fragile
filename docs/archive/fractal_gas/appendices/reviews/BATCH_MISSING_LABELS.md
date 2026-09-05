# Batch Analysis: Missing Cross-Reference Labels

**Total labels defined:** 2650
**Missing labels:** 39



## 🔴 Critical: Corrupted Label References (Need Manual Fix)

These labels contain corrupted text from previous edits:

| Label (corrupted) | File | Issue |
|-------------------|------|-------|
| `def-perturbation-m boundaryE=\mathcal{X}...` | 01_fragile_gas_framework.md | Corrupted prf:ref |
| `lem-borel-image-of-the-projected-swarm ({prf:ref}` | 01_fragile_gas_framework.md | Corrupted prf:ref |
| `swarm-and-state raw valueh alive set ({prf:ref}` | 01_fragile_gas_framework.md | Corrupted prf:ref |
| `thm-inter-swarm ({prf:ref}` | 03_cloning.md | Corrupted prf:ref |
| `axiom-bounded-algorithmic ({prf:ref}` | 01_fragile_gas_framework.md | Corrupted prf:ref |
| `thm-entropy-transport-contraction$...` | 15_kl_convergence.md | Corrupted prf:ref |



## 🟡 Missing Label Definitions (May need to be created)

### Used in 03_cloning.md
- `def-euclidean-gas`
- `def-foster-lyapunov`
- `def-fragile-gas-axioms` (also in 04_single_particle.md)
- `def-geometric-ergodicity`
- `def-hypocoercive-distance`
- `def-hypocoercive-lyapunov`
- `def-kinetic-operator`
- `def-langevin-operator`
- `def-markov-kernel`
- `def-wasserstein-distance`
- `def-cloning-operator`
- `thm-bounded-velocity-expansion-cloning` (also in 06_convergence.md)

### Used in 01_fragile_gas_framework.md
- `02_euclidean_gas` (doc ref, not label)
- `alive-set-potential-operator`
- `proof-lem-sub-unify-holder-terms`
- `proof-thm-total-error-status-bound`
- `swarm-update-procedure`

### Used in 02_euclidean_gas.md
- `def-assumption-instep-independence`
- `def-distance-to-companion`

### Used in 06_convergence.md
- `thm-velocity-variance-contraction`

### Used in 11_hk_convergence.md
- `ax-confining-potential`
- `ax-uniform-density-bound-hk`
- `axiom-local-perturbation`
- `thm-main-contraction-full` (also in 15_kl, 13_quantitative)

### Used in 14_a_geometric_gas_c3_regularity.md
- `lem-variance-gradient`
- `lem-variance-hessian`
- `thm-c1-regularity`
- `thm-c2-regularity`

### Used in 15_kl_convergence.md
- `13_quantitative_error_bounds` (doc ref)
- `thm-propagation-chaos-ips`
- `thm-qsd-riemannian-volume-main`

### Used in 13_quantitative_error_bounds.md
- `thm-energy-bounds`
- `thm-entropy-production-discrete`



## Recommendations

1. **Priority 1:** Fix the 6 corrupted labels in 01_fragile_gas_framework.md - these are broken syntax
2. **Priority 2:** Create missing common definitions (def-fragile-gas-axioms, def-euclidean-gas, etc.)
3. **Priority 3:** Add the missing theorem labels or update references to existing labels
