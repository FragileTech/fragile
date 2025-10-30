# Pipeline Data Validation Report

**Pipeline Directory**: `docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data`
**Total Files**: 216
**Valid Files**: 164
**Files with Errors**: 52

## Summary by Entity Type

| Entity Type | Valid | Errors | Total | Success Rate |
|-------------|-------|--------|-------|--------------|
| axioms | 20 | 2 | 22 | 90.9% |
| objects | 56 | 0 | 56 | 100.0% |
| parameters | 52 | 3 | 55 | 94.5% |
| theorems | 36 | 47 | 83 | 43.4% |

## Validation Errors

Found **52 files** with validation errors:

### Axioms (2 errors)

#### 1. `axioms/axiom-well-behaved-rescale-function.json`

```
  - statement: String should have at least 1 character
    Input: 
  - mathematical_expression: String should have at least 1 character
    Input: 
```

#### 2. `axioms/def-axiom-rescale-function.json`

```
  - label: String should match pattern '^axiom-[a-z0-9-]+$'
    Input: def-axiom-rescale-function
  - statement: String should have at least 1 character
    Input: 
  - mathematical_expression: String should have at least 1 character
    Input: 
```

### Parameters (3 errors)

#### 1. `parameters/param-l-sigma'-reg.json`

```
  - label: String should match pattern '^param-[a-z0-9-]+$'
    Input: param-l-sigma'-reg
```

#### 2. `parameters/param-\delta-{x-i}.json`

```
  - label: String should match pattern '^param-[a-z0-9-]+$'
    Input: param-\delta-{x-i}
```

#### 3. `parameters/param-\mathcal{x}.json`

```
  - label: String should match pattern '^param-[a-z0-9-]+$'
    Input: param-\mathcal{x}
```

### Theorems (47 errors)

#### 1. `theorems/lem-probabilistic-bound-perturbation-displacement.json`

```
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 2. `theorems/lem-potential-boundedness.json`

```
  - document_id: Input should be a valid string
  - file_path: Input should be a valid string
```

#### 3. `theorems/lem-perturbation-positional-bound.json`

```
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 4. `theorems/lem-boundary-heat-kernel.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: General Result
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Lipschitz constant bounded by C_d'*Per(E)/sigma
  - relations_established -> 1: Input should be a valid dictionary or instance of Relationship
    Input: Heat-kernel smoothed probability P_sigma(x) is Lipschitz continuous
  - relations_established -> 2: Input should be a valid dictionary or instance of Relationship
    Input: In algorithmic metric: L_death <= (Per(phi(E))/sigma)*L_phi
  - assumptions -> 0: Input should be a valid dictionary or instance of DualStatement
    Input: $E=\mathcal{X}_{\mathrm{invalid}}$ has finite perimeter
  - assumptions -> 1: Input should be a valid dictionary or instance of DualStatement
    Input: $p_{\sigma^2}$ is heat kernel at scale $\sigma$
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: $|P_\sigma(x)-P_\sigma(y)| \le C_d'\,\frac{\mathrm{Per}(E)}{\sigma}\, d_{\mathcal X}(x,y)$
```

#### 5. `theorems/lem-cubic-patch-derivative-bounds.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Explicit bound: L_P = 1 + (3log(2)-2)^2/(3(2log(2)-1)) ~= 1.0054
  - relations_established -> 1: Input should be a valid dictionary or instance of Relationship
    Input: Cubic patch derivative P'(z(s)) is uniformly bounded: 0 <= P'(z(s)) <= L_P
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 6. `theorems/lem-unify-holder-terms.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: General Result
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Unified Hölder bound for multiple terms
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 7. `theorems/lem-component-potential-lipschitz.json`

```
  - document_id: Input should be a valid string
  - file_path: Input should be a valid string
```

#### 8. `theorems/lem-sigma-patch-derivative-bound.json`

```
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 9. `theorems/lem-single-walker-structural-error.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Bounds structural error in expected distance for walker i due to companion selection measure change
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 10. `theorems/lem-stable-walker-error-decomposition.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Stable walker error decomposes into positional, structural, and own-status components
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 11. `theorems/lem-lipschitz-variance-functional.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: Lipschitz Continuity
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 12. `theorems/cor-pipeline-continuity-margin-stability.json`

```
  - label: String should match pattern '^(thm|lem|prop)-[a-z0-9-]+$'
    Input: cor-pipeline-continuity-margin-stability
  - statement_type: Input should be 'theorem', 'lemma' or 'proposition'
    Input: TheoremBox
  - lemma_dag_edges -> 0: Input should be a valid tuple
  - lemma_dag_edges -> 1: Input should be a valid tuple
    Input: {'source': 'cor-pipeline-continuity-margin-stability', 'target': 'def-axiom-margin-stability'}
```

#### 13. `theorems/lem-single-walker-own-status-error.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Change in expected distance from walker's own status change bounded by D_Y
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 14. `theorems/cor-chain-rule-sigma-reg-var.json`

```
  - label: String should match pattern '^(thm|lem|prop)-[a-z0-9-]+$'
    Input: cor-chain-rule-sigma-reg-var
  - statement_type: Input should be 'theorem', 'lemma' or 'proposition'
    Input: TheoremBox
  - lemma_dag_edges -> 0: Input should be a valid tuple
    Input: {'source': 'cor-chain-rule-sigma-reg-var', 'target': 'lem-lipschitz-variance-functional'}
  - lemma_dag_edges -> 1: Input should be a valid tuple
    Input: {'source': 'cor-chain-rule-sigma-reg-var', 'target': 'lem-sigma-patch-derivative-bound'}
  - lemma_dag_edges -> 2: Input should be a valid tuple
    Input: {'source': 'cor-chain-rule-sigma-reg-var', 'target': 'lem-empirical-aggregator-properties'}
```

#### 15. `theorems/lem-stats-value-continuity.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: Lipschitz Continuity
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Mean and regularized standard deviation are Lipschitz continuous with respect to raw value vector
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 16. `theorems/lem-cubic-patch-coefficients.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Coefficients A, B, C, D of cubic patch P(z) are uniquely determined
  - relations_established -> 1: Input should be a valid dictionary or instance of Relationship
    Input: Determined by boundary conditions and parameter z_max > 1
  - assumptions -> 0: Input should be a valid dictionary or instance of DualStatement
    Input: z_max > 1
```

#### 17. `theorems/lem-stable-structural-error-bound.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Structural error for stable walkers bounded by status changes
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 18. `theorems/lem-validation-heat-kernel.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Heat kernel satisfies boundary regularity conditions
  - assumptions -> 0: Input should be a valid dictionary or instance of DualStatement
    Input: Polish metric measure space $(\mathcal{X}, d_{\mathcal{X}}, \mu)$
  - assumptions -> 1: Input should be a valid dictionary or instance of DualStatement
    Input: Heat kernel $p_t(x, \cdot)$ with uniformly bounded second moment
  - assumptions -> 2: Input should be a valid dictionary or instance of DualStatement
    Input: Sufficiently regular boundary of $\mathcal{X}_{\mathrm{valid}}$
  - conclusion: Input should be a valid dictionary or instance of DualStatement
```

#### 19. `theorems/lem-subadditivity-power.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: General Result
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Subadditivity property for fractional powers with 0 < α ≤ 1
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 20. `theorems/lem-final-status-change-bound.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: General Result
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Final status change bounded by initial displacement squared
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 21. `theorems/lem-final-positional-displacement-bound.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: General Result
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Final positional displacement bounded by initial displacement squared
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 22. `theorems/lem-polynomial-patch-monotonicity.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: General Result
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 23. `theorems/lem-total-squared-error-stable.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Total squared error from stable walkers is bounded
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 24. `theorems/lem-bounded-differences-favg.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Applies to normalized functional f_avg = (1/N)Delta_pert^2(S_in)
  - relations_established -> 1: Input should be a valid dictionary or instance of Relationship
    Input: Establishes McDiarmid bounded-difference constants c_i = D_Y^2/N
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 25. `theorems/lem-inequality-toolbox.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: General Result
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 26. `theorems/lem-empirical-moments-lipschitz.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: Lipschitz Continuity
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Empirical mean is Lipschitz: L_{mu,M} = 1/sqrt(k)
  - relations_established -> 1: Input should be a valid dictionary or instance of Relationship
    Input: Empirical second moment is Lipschitz: L_{m2,M} = 2V_max/sqrt(k)
```

#### 27. `theorems/lem-total-clone-prob-structural-error.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Bounds structural component of cloning probability error by number of status changes
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: Structural error bounded by status changes
```

#### 28. `theorems/cor-closed-form-lipschitz-composite.json`

```
  - label: String should match pattern '^(thm|lem|prop)-[a-z0-9-]+$'
    Input: cor-closed-form-lipschitz-composite
  - statement_type: Input should be 'theorem', 'lemma' or 'proposition'
    Input: TheoremBox
  - lemma_dag_edges -> 0: Input should be a valid tuple
    Input: {'source': 'cor-closed-form-lipschitz-composite', 'target': 'cor-chain-rule-sigma-reg-var'}
  - lemma_dag_edges -> 1: Input should be a valid tuple
    Input: {'source': 'cor-closed-form-lipschitz-composite', 'target': 'thm-rescale-function-lipschitz'}
  - lemma_dag_edges -> 2: Input should be a valid tuple
    Input: {'source': 'cor-closed-form-lipschitz-composite', 'target': 'lem-empirical-aggregator-properties'}
```

#### 29. `theorems/thm-canonical-logistic-validity.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: PROPERTY
```

#### 30. `theorems/lem-validation-uniform-ball.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: Lipschitz Continuity
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Uniform ball distribution satisfies boundary regularity conditions
  - assumptions -> 0: Input should be a valid dictionary or instance of DualStatement
    Input: Uniform probability measure over ball $B(x, \sigma)$
  - assumptions -> 1: Input should be a valid dictionary or instance of DualStatement
    Input: Sufficiently regular boundary (Lipschitz or finite perimeter)
  - conclusion: Input should be a valid dictionary or instance of DualStatement
```

#### 31. `theorems/lem-sigma-reg-derivative-bounds.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Bounds on derivatives of regularized standard deviation sigma'_reg
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 32. `theorems/thm-deterministic-potential-continuity.json`

```
  - document_id: Input should be a valid string
  - file_path: Input should be a valid string
```

#### 33. `theorems/lem-single-walker-positional-error.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Bound on error in expected distance measurement from positional displacement
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 34. `theorems/thm-mean-square-standardization-error.json`

```
  - relations_established -> 0 -> relationship_type: Input should be 'equivalence', 'embedding', 'approximation', 'reduction', 'extension', 'generalization', 'specialization' or 'other'
    Input: OTHER
  - assumptions -> 0: Input should be a valid dictionary or instance of DualStatement
    Input: The number of alive walkers, k_1 = |A(S_1)|, is large
  - assumptions -> 1: Input should be a valid dictionary or instance of DualStatement
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: \mathbb{E}[\| \mathbf{z}_1 - \mathbf{z}_2 \|_2^2] \in O(E_{V,ms}^2(k_1)) + O(E_{S,ms}^2(k_1))
```

#### 35. `theorems/lem-total-clone-prob-value-error.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Bounds value component of cloning probability error
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: Value error bounded by expected potential error
```

#### 36. `theorems/lem-empirical-aggregator-properties.json`

```
  - relations_established -> 0 -> label: String should match pattern '^rel-[a-z0-9]+(-[a-z0-9]+)*-(equivalence|embedding|approximation|reduction|extension|generalization|specialization|other)$'
    Input: rel-empirical-aggregator-is-valid-swarm-aggregator
  - relations_established -> 0 -> relationship_type: Input should be 'equivalence', 'embedding', 'approximation', 'reduction', 'extension', 'generalization', 'specialization' or 'other'
    Input: INSTANTIATION
  - relations_established -> 0 -> bidirectional: Field required
  - relations_established -> 0 -> target_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: empirical-measure-aggregator
  - relations_established -> 0 -> established_by: Field required
  - relations_established -> 0 -> expression: Field required
  - relations_established -> 1 -> label: String should match pattern '^rel-[a-z0-9]+(-[a-z0-9]+)*-(equivalence|embedding|approximation|reduction|extension|generalization|specialization|other)$'
    Input: rel-empirical-aggregator-value-lipschitz-mean
  - relations_established -> 1 -> relationship_type: Input should be 'equivalence', 'embedding', 'approximation', 'reduction', 'extension', 'generalization', 'specialization' or 'other'
    Input: PROPERTY
  - relations_established -> 1 -> bidirectional: Field required
  - relations_established -> 1 -> source_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: empirical-measure-aggregator
  - relations_established -> 1 -> target_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: L_mu_M
  - relations_established -> 1 -> established_by: Field required
  - relations_established -> 2 -> label: String should match pattern '^rel-[a-z0-9]+(-[a-z0-9]+)*-(equivalence|embedding|approximation|reduction|extension|generalization|specialization|other)$'
    Input: rel-empirical-aggregator-value-lipschitz-m2
  - relations_established -> 2 -> relationship_type: Input should be 'equivalence', 'embedding', 'approximation', 'reduction', 'extension', 'generalization', 'specialization' or 'other'
    Input: PROPERTY
  - relations_established -> 2 -> bidirectional: Field required
  - relations_established -> 2 -> source_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: empirical-measure-aggregator
  - relations_established -> 2 -> target_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: L_m2_M
  - relations_established -> 2 -> established_by: Field required
  - relations_established -> 3 -> label: String should match pattern '^rel-[a-z0-9]+(-[a-z0-9]+)*-(equivalence|embedding|approximation|reduction|extension|generalization|specialization|other)$'
    Input: rel-empirical-aggregator-structural-lipschitz-mean
  - relations_established -> 3 -> relationship_type: Input should be 'equivalence', 'embedding', 'approximation', 'reduction', 'extension', 'generalization', 'specialization' or 'other'
    Input: PROPERTY
  - relations_established -> 3 -> bidirectional: Field required
  - relations_established -> 3 -> source_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: empirical-measure-aggregator
  - relations_established -> 3 -> target_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: L_mu_S
  - relations_established -> 3 -> established_by: Field required
  - relations_established -> 4 -> label: String should match pattern '^rel-[a-z0-9]+(-[a-z0-9]+)*-(equivalence|embedding|approximation|reduction|extension|generalization|specialization|other)$'
    Input: rel-empirical-aggregator-structural-lipschitz-m2
  - relations_established -> 4 -> relationship_type: Input should be 'equivalence', 'embedding', 'approximation', 'reduction', 'extension', 'generalization', 'specialization' or 'other'
    Input: PROPERTY
  - relations_established -> 4 -> bidirectional: Field required
  - relations_established -> 4 -> source_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: empirical-measure-aggregator
  - relations_established -> 4 -> target_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: L_m2_S
  - relations_established -> 4 -> established_by: Field required
  - relations_established -> 5 -> label: String should match pattern '^rel-[a-z0-9]+(-[a-z0-9]+)*-(equivalence|embedding|approximation|reduction|extension|generalization|specialization|other)$'
    Input: rel-empirical-aggregator-variance-deviation-factor
  - relations_established -> 5 -> relationship_type: Input should be 'equivalence', 'embedding', 'approximation', 'reduction', 'extension', 'generalization', 'specialization' or 'other'
    Input: PROPERTY
  - relations_established -> 5 -> bidirectional: Field required
  - relations_established -> 5 -> source_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: empirical-measure-aggregator
  - relations_established -> 5 -> target_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: kappa_var
  - relations_established -> 5 -> established_by: Field required
  - relations_established -> 6 -> label: String should match pattern '^rel-[a-z0-9]+(-[a-z0-9]+)*-(equivalence|embedding|approximation|reduction|extension|generalization|specialization|other)$'
    Input: rel-empirical-aggregator-range-variance-factor
  - relations_established -> 6 -> relationship_type: Input should be 'equivalence', 'embedding', 'approximation', 'reduction', 'extension', 'generalization', 'specialization' or 'other'
    Input: PROPERTY
  - relations_established -> 6 -> bidirectional: Field required
  - relations_established -> 6 -> source_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: empirical-measure-aggregator
  - relations_established -> 6 -> target_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: kappa_range
  - relations_established -> 6 -> established_by: Field required
  - relations_established -> 7 -> label: String should match pattern '^rel-[a-z0-9]+(-[a-z0-9]+)*-(equivalence|embedding|approximation|reduction|extension|generalization|specialization|other)$'
    Input: rel-empirical-aggregator-structural-growth-exponents
  - relations_established -> 7 -> relationship_type: Input should be 'equivalence', 'embedding', 'approximation', 'reduction', 'extension', 'generalization', 'specialization' or 'other'
    Input: PROPERTY
  - relations_established -> 7 -> bidirectional: Field required
  - relations_established -> 7 -> source_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: empirical-measure-aggregator
  - relations_established -> 7 -> target_object: String should match pattern '^obj-[a-z0-9-]+$'
    Input: structural_growth_exponents
  - relations_established -> 7 -> established_by: Field required
  - assumptions -> 0: Input should be a valid dictionary or instance of DualStatement
    Input: Raw values bounded: |v_i| <= V_max for all i
  - assumptions -> 1: Input should be a valid dictionary or instance of DualStatement
    Input: Alive set non-empty: k = |A(S)| >= 1
  - assumptions -> 2: Input should be a valid dictionary or instance of DualStatement
    Input: Bounded relative collapse holds: k_2 >= c_min * k_1 for axiom verification
  - conclusion: Input should be a valid dictionary or instance of DualStatement
```

#### 37. `theorems/lem-stats-structural-continuity.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: Lipschitz Continuity
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Mean and regularized standard deviation are continuous with respect to swarm structure changes
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 38. `theorems/lem-total-squared-error-unstable.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Total squared error from unstable walkers bounded by status changes
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 39. `theorems/lem-rescale-monotonicity.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: General Result
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 40. `theorems/thm-distance-operator-satisfies-bounded-variance-axiom.json`

```
  - document_id: Input should be a valid string
  - file_path: Input should be a valid string
```

#### 41. `theorems/lem-cubic-patch-derivative.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: General Result
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Derivative of cubic patch P(z) is P'(z) = 3Az^2 + 2Bz + C
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: 
```

#### 42. `theorems/thm-pipeline-continuity-margin-stability.json`

```
  - document_id: Input should be a valid string
  - file_path: Input should be a valid string
```

#### 43. `theorems/lem-cubic-patch-uniqueness.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Cubic polynomial with 4 boundary conditions has unique solution via confluent Vandermonde matrix
  - assumptions -> 0: Input should be a valid dictionary or instance of DualStatement
    Input: z_max > 1
```

#### 44. `theorems/lem-boundary-uniform-ball.json`

```
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
    Input: Uniform ball death probability P_sigma(x) is Lipschitz continuous
  - relations_established -> 1: Input should be a valid dictionary or instance of Relationship
    Input: Lipschitz bound: |P_sigma(x) - P_sigma(y)| <= C_d*Per(E)/sigma*d_X(x,y)
  - assumptions -> 0: Input should be a valid dictionary or instance of DualStatement
    Input: $E=\mathcal{X}_{\mathrm{invalid}}$ has finite perimeter (BV boundary)
  - assumptions -> 1: Input should be a valid dictionary or instance of DualStatement
    Input: $\mathcal P_\sigma(x,\cdot)$ is uniform law on $B(x,\sigma)$
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: $|P_\sigma(x)-P_\sigma(y)| \le C_d\,\frac{\mathrm{Per}(E)}{\sigma}\, d_{\mathcal X}(x,y)$
```

#### 45. `theorems/lem-potential-stable-error-mean-square.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: Lipschitz Continuity
```

#### 46. `theorems/thm-cloning-transition-operator-continuity-recorrected.json`

```
  - document_id: Input should be a valid string
  - file_path: Input should be a valid string
```

#### 47. `theorems/lem-cloning-probability-lipschitz.json`

```
  - output_type: Input should be 'Property', 'Relation', 'Existence', 'Construction', 'Classification', 'Uniqueness', 'Impossibility', 'Embedding', 'Approximation', 'Equivalence', 'Decomposition', 'Extension', 'Reduction', 'Bound', 'Convergence' or 'Contraction'
    Input: Lipschitz Continuity
  - relations_established -> 0: Input should be a valid dictionary or instance of Relationship
  - relations_established -> 1: Input should be a valid dictionary or instance of Relationship
    Input: Conditional cloning probability pi(v_c, v_i) is Lipschitz continuous
  - conclusion: Input should be a valid dictionary or instance of DualStatement
    Input: The cloning probability function is Lipschitz continuous with explicit constants
```

## Common Error Patterns

### Invalid object type (35 files)

- `theorems/lem-boundary-heat-kernel.json`
- `theorems/lem-boundary-uniform-ball.json`
- `theorems/lem-bounded-differences-favg.json`
- `theorems/lem-cloning-probability-lipschitz.json`
- `theorems/lem-cubic-patch-coefficients.json`
- `theorems/lem-cubic-patch-derivative-bounds.json`
- `theorems/lem-cubic-patch-derivative.json`
- `theorems/lem-cubic-patch-uniqueness.json`
- `theorems/lem-empirical-moments-lipschitz.json`
- `theorems/lem-final-positional-displacement-bound.json`
- ... and 25 more

### Other validation errors (8 files)

- `theorems/lem-component-potential-lipschitz.json`
- `theorems/lem-potential-boundedness.json`
- `theorems/lem-potential-stable-error-mean-square.json`
- `theorems/thm-canonical-logistic-validity.json`
- `theorems/thm-cloning-transition-operator-continuity-recorrected.json`
- `theorems/thm-deterministic-potential-continuity.json`
- `theorems/thm-distance-operator-satisfies-bounded-variance-axiom.json`
- `theorems/thm-pipeline-continuity-margin-stability.json`

### Invalid label format (7 files)

- `parameters/param-\delta-{x-i}.json`
- `parameters/param-\mathcal{x}.json`
- `parameters/param-l-sigma'-reg.json`
- `theorems/cor-chain-rule-sigma-reg-var.json`
- `theorems/cor-closed-form-lipschitz-composite.json`
- `theorems/cor-pipeline-continuity-margin-stability.json`
- `theorems/lem-empirical-aggregator-properties.json`

### Empty string fields (2 files)

- `axioms/axiom-well-behaved-rescale-function.json`
- `axioms/def-axiom-rescale-function.json`
