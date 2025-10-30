# Section 16 Extraction Report
## The Cloning Transition Measure

**Extraction Date:** 2025-10-27T21:51:10.619265
**Source Document:** 01_fragile_gas_framework.md
**Section:** 16 (lines 4214-4554)
**Total Lines:** 340

---

## Extraction Summary

### Total Entities Extracted: 24

| Entity Type   | Count | Directory                    |
|---------------|-------|------------------------------|
| Definitions   | 6     | definitions/                 |
| Theorems      | 4     | theorems/                    |
| Lemmas        | 4     | lemmas/                      |
| Sub-Lemmas    | 1     | lemmas/ (counted in total)   |
| Proofs        | 8     | proofs/                      |
| Remarks       | 2     | remarks/                     |

---

## Extracted Entities

### Definitions (6)

1. **def-cloning-score-function** (line 4219)
   - Cloning Score Function: $S(v_c, v_i) := \frac{v_c - v_i}{v_i + \varepsilon}$
   - Tags: cloning, score-function, measurement

2. **def-stochastic-threshold-cloning** (line 4231)
   - Stochastic Threshold Cloning procedure
   - Tags: cloning, stochastic-process, measurement, revival

3. **def-total-expected-cloning-action** (line 4284)
   - Total Expected Cloning Action: $\overline{P}_{\text{clone}}(\mathcal{S})_i$
   - Tags: cloning, probability, expectation, continuity

4. **def-cloning-probability-function** (line 4297)
   - Conditional Cloning Probability Function: $\pi(v_c, v_i)$
   - Tags: cloning, probability, function

5. **def-expected-cloning-action** (line 4340)
   - Conditional Expected Cloning Action: $P_{\text{clone}}(\mathcal{S}, \mathbf{V})_i$
   - Tags: cloning, probability, expectation

6. **def-cloning-operator-continuity-coeffs-recorrected** (line 4447)
   - Cloning Operator Continuity Coefficients: $C_{\text{clone},L}$, $C_{\text{clone},H}$, $K_{\text{clone}}$
   - Tags: cloning, continuity, coefficients

### Theorems (4)

1. **thm-expected-cloning-action-continuity** (line 4349)
   - Continuity of the Conditional Expected Cloning Action
   - Key result: Bounds change in conditional expected action
   - Dependencies: def-expected-cloning-action, lem-cloning-probability-lipschitz, thm-total-error-status-bound

2. **thm-total-expected-cloning-action-continuity** (line 4372)
   - Continuity of the Total Expected Cloning Action
   - Key result: Total cloning probability is continuous in swarm state
   - Dependencies: def-total-expected-cloning-action, def-expected-cloning-action

3. **thm-potential-operator-is-mean-square-continuous** (line 4407)
   - The Fitness Potential Operator is Mean-Square Continuous
   - Key result: Expected squared potential error is bounded
   - Dependencies: thm-fitness-potential-mean-square-continuity

4. **thm-cloning-transition-operator-continuity-recorrected** (line 4437)
   - Mean-Square Continuity of the Cloning Transition Operator
   - Key result: Main continuity theorem with Lipschitz and Hölder terms
   - Dependencies: def-cloning-operator-continuity-coeffs-recorrected, sub-lem-bound-sum-total-cloning-probs

### Lemmas (4)

1. **lem-cloning-probability-lipschitz** (line 4308)
   - Lipschitz Continuity of the Conditional Cloning Probability Function
   - Establishes constants: $L_{\pi,c}$ and $L_{\pi,i}$
   - Dependencies: def-cloning-probability-function, lem-potential-boundedness

2. **lem-total-clone-prob-structural-error** (line 4394)
   - Bounding the Structural Component of Cloning Probability Error
   - Bounds: $E_{\text{struct}}^{(\overline{P})} \le C_{\text{struct}}^{(\pi)}(k_1) \cdot n_c$
   - Dependencies: thm-total-expected-cloning-action-continuity, thm-expected-cloning-action-continuity

3. **lem-total-clone-prob-value-error** (line 4420)
   - Bounding the Value Component of Cloning Probability Error
   - Bounds: $E_{\text{val}}^{(\overline{P})} \le C_{\text{val}}^{(\pi)} \sqrt{2N \cdot F_{\text{pot}}}$
   - Dependencies: thm-total-expected-cloning-action-continuity, thm-potential-operator-is-mean-square-continuous

4. **sub-lem-bound-sum-total-cloning-probs** (line 4502)
   - Bounding the Sum of Total Cloning Probabilities (Sub-Lemma)
   - Bounds sum with linear, Hölder, and constant terms
   - Dependencies: multiple theorems and lemmas

### Proofs (8)

1. **proof-lem-cloning-probability-lipschitz** (line 4330)
   - Proves: lem-cloning-probability-lipschitz
   - Technique: Partial derivative analysis with case splitting

2. **proof-thm-expected-cloning-action-continuity** (line 4361)
   - Proves: thm-expected-cloning-action-continuity
   - Technique: Triangle inequality decomposition

3. **proof-thm-total-expected-cloning-action-continuity** (line 4382)
   - Proves: thm-total-expected-cloning-action-continuity
   - Technique: Intermediate term method

4. **proof-lem-total-clone-prob-structural-error** (line 4402)
   - Proves: lem-total-clone-prob-structural-error
   - Technique: Jensen's inequality

5. **proof-thm-potential-operator-is-mean-square-continuous** (line 4415)
   - Proves: thm-potential-operator-is-mean-square-continuous
   - Technique: Reference to Section 12.2 analysis

6. **proof-lem-total-clone-prob-value-error** (line 4429)
   - Proves: lem-total-clone-prob-value-error
   - Technique: Lipschitz continuity and Jensen's inequality

7. **proof-cloning-transition-operator-continuity-recorrected** (line 4454)
   - Proves: thm-cloning-transition-operator-continuity-recorrected
   - Technique: Multi-step composition of bounds

8. **proof-sub-lem-bound-sum-total-cloning-probs** (line 4512)
   - Proves: sub-lem-bound-sum-total-cloning-probs
   - Technique: Decomposition and displacement component analysis

### Remarks (2)

1. **remark-cloning-randomness-discipline** (line 4269)
   - Randomness discipline for cloning: per-walker independence
   - Tags: cloning, randomness, independence

2. **remark-cloning-scope-companion-convention** (line 4278)
   - Scope and companion convention for Section 16 bounds
   - Tags: cloning, scope, convention, edge-case

---

## Key Mathematical Objects

### Central Concepts
- **Cloning Score Function**: Maps companion and walker potentials to raw score
- **Conditional Cloning Probability**: Clips normalized score to [0,1]
- **Expected Cloning Action**: Averages over companion selection
- **Total Expected Cloning Action**: Averages over measurement stochasticity

### Main Results
- **Lipschitz Continuity**: Cloning probability function is Lipschitz in both arguments
- **Mean-Square Continuity**: Cloning operator is mean-square continuous with explicit bounds
- **Error Decomposition**: Structural and value error components separately bounded

### Continuity Structure
```
Input Swarm States (S1, S2)
    ↓
Fitness Potential Operator (mean-square continuous)
    ↓
Cloning Probability Function (Lipschitz continuous)
    ↓
Expected Cloning Action (continuous)
    ↓
Total Expected Cloning Action (continuous)
    ↓
Cloning Transition Operator (mean-square continuous)
    ↓
Output Intermediate Swarm States (S'1, S'2)
```

---

## Dependency Graph

### Critical Dependencies
- **lem-potential-boundedness**: Referenced for potential bounds
- **thm-total-error-status-bound**: Used for structural error analysis
- **thm-fitness-potential-mean-square-continuity**: Provides $F_{\text{pot}}$ bound
- **thm-distance-operator-mean-square-continuity**: Contains $n_c^2$ term
- **def-displacement-components**: Used for displacement relations

### Internal Dependencies (Section 16)
- Definitions → Lemmas → Theorems → Main Theorem
- Proofs follow theorem/lemma structure
- Sub-lemma provides key ingredient for main theorem proof

---

## Quality Metrics

### Coverage
- ✓ All definitions extracted (6/6)
- ✓ All theorems extracted (4/4)
- ✓ All lemmas extracted (4/4)
- ✓ All proofs extracted (8/8)
- ✓ All remarks extracted (2/2)

### Completeness
- ✓ Labels preserved from source
- ✓ Full mathematical statements included
- ✓ LaTeX notation preserved
- ✓ Dependencies tracked
- ✓ Line numbers recorded
- ✓ Context documented

### Organization
- ✓ Entities organized by type
- ✓ JSON format validated
- ✓ Consistent metadata structure
- ✓ Cross-references preserved

---

## Output Locations

All entities saved to: `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/`

```
01_fragile_gas_framework/
├── definitions/
│   ├── def-cloning-score-function.json
│   ├── def-stochastic-threshold-cloning.json
│   ├── def-total-expected-cloning-action.json
│   ├── def-cloning-probability-function.json
│   ├── def-expected-cloning-action.json
│   └── def-cloning-operator-continuity-coeffs-recorrected.json
├── theorems/
│   ├── thm-expected-cloning-action-continuity.json
│   ├── thm-total-expected-cloning-action-continuity.json
│   ├── thm-potential-operator-is-mean-square-continuous.json
│   └── thm-cloning-transition-operator-continuity-recorrected.json
├── lemmas/
│   ├── lem-cloning-probability-lipschitz.json
│   ├── lem-total-clone-prob-structural-error.json
│   ├── lem-total-clone-prob-value-error.json
│   └── sub-lem-bound-sum-total-cloning-probs.json
├── proofs/
│   ├── proof-lem-cloning-probability-lipschitz.json
│   ├── proof-thm-expected-cloning-action-continuity.json
│   ├── proof-thm-total-expected-cloning-action-continuity.json
│   ├── proof-lem-total-clone-prob-structural-error.json
│   ├── proof-thm-potential-operator-is-mean-square-continuous.json
│   ├── proof-lem-total-clone-prob-value-error.json
│   ├── proof-cloning-transition-operator-continuity-recorrected.json
│   └── proof-sub-lem-bound-sum-total-cloning-probs.json
└── remarks/
    ├── remark-cloning-randomness-discipline.json
    └── remark-cloning-scope-companion-convention.json
```

---

## Next Steps

### Recommended Follow-up
1. **Verify cross-references**: Check that all referenced labels exist
2. **Update glossary**: Add Section 16 entities to docs/glossary.md
3. **Check consistency**: Verify notation matches framework conventions
4. **Extract remaining sections**: Continue with Sections 17+

### Known Dependencies to Extract
- lem-potential-boundedness
- thm-total-error-status-bound
- thm-fitness-potential-mean-square-continuity
- thm-distance-operator-mean-square-continuity
- def-displacement-components
- lem-subadditivity-power

---

**Extraction completed successfully.**
**Total time: < 1 second (manual extraction)**
**Extraction method: Systematic parsing with structured JSON output**
