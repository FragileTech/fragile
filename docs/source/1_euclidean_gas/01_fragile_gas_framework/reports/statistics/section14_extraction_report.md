# Section 14 Extraction Report: The Perturbation Operator

**Source Document**: `01_fragile_gas_framework.md`
**Section**: 14. The Perturbation Operator
**Line Range**: 3937-4118 (182 lines)
**Extraction Date**: 2025-10-27

---

## Summary

Successfully extracted **15 mathematical entities** from Section 14 (The Perturbation Operator):

| Entity Type | Count | Files Created |
|------------|-------|---------------|
| Definitions | 2 | `definitions/` |
| Axioms | 1 | `axioms/` |
| Lemmas | 3 | `lemmas/` |
| Theorems | 2 | `theorems/` |
| Proofs | 3 | `proofs/` |
| Objects | 2 | `objects/` |
| Remarks | 2 | `remarks/` |
| **TOTAL** | **15** | **8 directories** |

---

## Extracted Entities

### 1. Definitions (2)

#### def-perturbation-operator
- **Name**: Perturbation Operator
- **Notation**: $\Psi_{\text{pert}}: \Sigma_N \to \mathcal{P}(\Sigma_N)$
- **File**: `definitions/def-perturbation-operator.json`
- **Key Properties**:
  - Stochastic operator updating positions only
  - Preserves walker status
  - Product measure of N independent processes
- **Tags**: perturbation, operator, stochastic, exploration

#### def-perturbation-fluctuation-bounds-reproof
- **Name**: Perturbation Fluctuation Bounds
- **Components**:
  - Mean Displacement Bound: $B_M(N) := N \cdot M_{\text{pert}}^2$
  - Stochastic Fluctuation Bound: $B_S(N, \delta') := D_{\mathcal{Y}}^2 \sqrt{\frac{N}{2} \ln(\frac{2}{\delta'})}$
- **File**: `definitions/def-perturbation-fluctuation-bounds.json`
- **Tags**: definition, perturbation, bound, concentration

---

### 2. Axioms (1)

#### def-axiom-bounded-second-moment-perturbation
- **Name**: Axiom of Bounded Second Moment of Perturbation
- **Statement**: The expectation of the squared displacement caused by the Perturbation Measure is uniformly bounded
- **Axiomatic Parameter**: $M_{\text{pert}}^2$ (Maximum Expected Squared Displacement)
- **Condition**: $M_{\text{pert}}^2 \ge \sup_{x_{\text{in}} \in \mathcal{X}} \mathbb{E}[d_{\mathcal{Y}}(\varphi(x_{\text{out}}), \varphi(x_{\text{in}}))^2]$
- **File**: `axioms/axiom-bounded-second-moment-perturbation.json`
- **Framework Role**: Ensures statistical boundedness of perturbation-induced displacement
- **Tags**: axiom, perturbation, bounded-moment, concentration, continuity

---

### 3. Lemmas (3)

#### sub-lem-perturbation-positional-bound-reproof
- **Name**: Bounding the Output Positional Displacement
- **Statement**:
  $$\Delta_{\text{pos}}^2(\mathcal{S}'_1, \mathcal{S}'_2) \le 3\Delta_{\text{pos}}^2(\mathcal{S}_1, \mathcal{S}_2) + 3\Delta_{\text{pert}}^2(\mathcal{S}_1) + 3\Delta_{\text{pert}}^2(\mathcal{S}_2)$$
- **File**: `lemmas/lem-perturbation-positional-bound.json`
- **Proof Strategy**: Triangle inequality + $(a+b+c)^2 \le 3(a^2 + b^2 + c^2)$
- **Tags**: lemma, perturbation, displacement, bound, algebraic

#### lem-bounded-differences-favg
- **Name**: Bounded differences for $f_{\text{avg}}$
- **Statement**: McDiarmid bounded-difference constants are $c_i=D_{\mathcal{Y}}^2/N$
- **File**: `lemmas/lem-bounded-differences-favg.json`
- **Tags**: lemma, concentration, bounded-differences, McDiarmid

#### sub-lem-probabilistic-bound-perturbation-displacement-reproof
- **Name**: Probabilistic Bound on Total Perturbation-Induced Displacement
- **Statement**:
  $$\Delta_{\text{pert}}^2(\mathcal{S}_{\text{in}}) \le B_M(N) + B_S(N, \delta')$$
  with probability at least $1-\delta'$
- **File**: `lemmas/lem-probabilistic-bound-perturbation-displacement.json`
- **Proof Strategy**: McDiarmid's Inequality + axiomatic mean bound
- **Tags**: lemma, perturbation, concentration, probabilistic-bound

---

### 4. Theorems (2)

#### thm-mcdiarmids-inequality
- **Name**: McDiarmid's Inequality (Bounded Differences Inequality)
- **Statement**: For independent random variables with bounded differences:
  $$P(|f(X_1, \dots, X_N) - \mathbb{E}[f]| \ge t) \le 2\exp\left(\frac{-2t^2}{\sum_{i=1}^N c_i^2}\right)$$
- **File**: `theorems/thm-mcdiarmids-inequality.json`
- **External Reference**: Boucheron–Lugosi–Massart
- **Tags**: theorem, concentration, probability, bounded-differences, external

#### thm-perturbation-operator-continuity-reproof
- **Name**: Probabilistic Continuity of the Perturbation Operator
- **Statement**: For any $\delta \in (0, 1)$, with probability at least $1-\delta$:
  $$d_{\text{Disp},\mathcal{Y}}(\mathcal{S}'_1, \mathcal{S}'_2)^2 \le 3 \frac{\Delta_{\text{pos}}^2(\mathcal{S}_1, \mathcal{S}_2)}{N} + \lambda_{\mathrm{status}} \frac{n_c(\mathcal{S}_1, \mathcal{S}_2)}{N} + \frac{6}{N} \left( B_M(N) + B_S(N, \delta/2) \right)$$
- **File**: `theorems/thm-perturbation-operator-continuity.json`
- **Main Result**: YES (Section's primary theorem)
- **Proof Strategy**: Compose algebraic and probabilistic bounds using union bound
- **Tags**: theorem, perturbation, continuity, probabilistic, main-result

---

### 5. Proofs (3)

#### proof-sub-lem-perturbation-positional-bound-reproof
- **Proves**: `sub-lem-perturbation-positional-bound-reproof`
- **Type**: Direct proof
- **File**: `proofs/proof-perturbation-positional-bound.json`
- **Key Techniques**: Triangle inequality, Algebraic inequality
- **Tags**: proof, algebraic, triangle-inequality

#### proof-sub-lem-probabilistic-bound-perturbation-displacement-reproof
- **Proves**: `sub-lem-probabilistic-bound-perturbation-displacement-reproof`
- **Type**: Constructive proof
- **File**: `proofs/proof-probabilistic-bound-perturbation-displacement.json`
- **Key Techniques**: McDiarmid's Inequality, Bounded differences, Concentration of measure
- **Tags**: proof, concentration, McDiarmid, probabilistic

#### proof-thm-perturbation-operator-continuity-reproof
- **Proves**: `thm-perturbation-operator-continuity-reproof`
- **Type**: Synthetic proof (composition of lemmas)
- **File**: `proofs/proof-perturbation-operator-continuity.json`
- **Key Techniques**: Metric decomposition, Union bound, Lemma composition
- **Tags**: proof, synthesis, union-bound, composition

---

### 6. Objects (2)

#### obj-mean-displacement-bound
- **Name**: Mean Displacement Bound
- **Symbol**: $B_M(N)$
- **Formula**: $B_M(N) := N \cdot M_{\text{pert}}^2$
- **File**: `objects/obj-mean-displacement-bound.json`
- **Properties**: Deterministic bound, Linear in N
- **Tags**: bound, perturbation, mean, displacement

#### obj-stochastic-fluctuation-bound
- **Name**: Stochastic Fluctuation Bound
- **Symbol**: $B_S(N, \delta')$
- **Formula**: $B_S(N, \delta') := D_{\mathcal{Y}}^2 \sqrt{\frac{N}{2} \ln(\frac{2}{\delta'})}$
- **File**: `objects/obj-stochastic-fluctuation-bound.json`
- **Properties**: Probabilistic bound (holds with probability $\ge 1-\delta'$), Scales as $\sqrt{N}$
- **Tags**: bound, perturbation, concentration, fluctuation

---

### 7. Remarks (2)

#### remark-randomness-discipline-perturbation
- **Context**: Randomness discipline for perturbation
- **Content**: Implementation note on per-walker PRNG streams and Assumption A enforcement
- **File**: `remarks/remark-randomness-discipline-perturbation.json`
- **Related**: Assumption A
- **Tags**: remark, implementation, randomness, independence

#### remark-scope-and-assumptions
- **Context**: Scope and assumptions for perturbation operator continuity
- **Content**: Notes on Assumption A and with-replacement sampling requirement
- **File**: `remarks/remark-scope-and-assumptions.json`
- **Related**: `thm-perturbation-operator-continuity-reproof`, Assumption A
- **Tags**: remark, scope, assumptions, implementation

---

## Key Mathematical Concepts

1. **Perturbation Operator** - Stochastic exploration operator
2. **Bounded Second Moment of Perturbation** - Axiomatic constraint on displacement
3. **McDiarmid's Inequality** - Concentration inequality for functions of independent variables
4. **Probabilistic Continuity** - High-probability bound on operator output distance
5. **Concentration of Measure** - Statistical deviation control
6. **Union Bound** - Simultaneous probabilistic guarantees
7. **Mean Displacement Bound** - Deterministic expected displacement bound
8. **Stochastic Fluctuation Bound** - Probabilistic deviation bound

---

## Mathematical Techniques Used

- Triangle inequality decomposition
- McDiarmid concentration inequality
- Bounded differences method
- Union bound for simultaneous events
- Metric decomposition (positional + status)
- Concentration of measure theory
- Algebraic inequality manipulation

---

## Dependencies

### Internal Framework Dependencies
- N-Particle Displacement Metric
- Perturbation Measure
- Projection Map
- Algorithmic Space
- Assumption A (in-step independence)
- Axiom of Bounded Algorithmic Diameter

### External References
- Boucheron–Lugosi–Massart (McDiarmid's Inequality, Appendix B)

---

## File Structure

```
docs/source/1_euclidean_gas/01_fragile_gas_framework/
├── axioms/
│   └── axiom-bounded-second-moment-perturbation.json
├── definitions/
│   ├── def-perturbation-operator.json
│   └── def-perturbation-fluctuation-bounds.json
├── lemmas/
│   ├── lem-perturbation-positional-bound.json
│   ├── lem-bounded-differences-favg.json
│   └── lem-probabilistic-bound-perturbation-displacement.json
├── theorems/
│   ├── thm-mcdiarmids-inequality.json
│   └── thm-perturbation-operator-continuity.json
├── proofs/
│   ├── proof-perturbation-positional-bound.json
│   ├── proof-probabilistic-bound-perturbation-displacement.json
│   └── proof-perturbation-operator-continuity.json
├── objects/
│   ├── obj-mean-displacement-bound.json
│   └── obj-stochastic-fluctuation-bound.json
├── remarks/
│   ├── remark-randomness-discipline-perturbation.json
│   └── remark-scope-and-assumptions.json
└── section14_extraction_report.json
```

---

## Verification

All 15 entities successfully extracted and saved to JSON files:
- 2 definitions → `definitions/`
- 1 axiom → `axioms/`
- 3 lemmas → `lemmas/`
- 2 theorems → `theorems/`
- 3 proofs → `proofs/`
- 2 objects → `objects/`
- 2 remarks → `remarks/`

Total: 15 JSON files created across 7 subdirectories.

---

## Notes

- **Section Numbering Discrepancy**: Document header says "Section 14" but internal subsections labeled as "13.x" (likely from prior refactoring). Labels preserved as-is.
- **Main Result**: `thm-perturbation-operator-continuity-reproof` is the section's primary theorem, establishing probabilistic continuity of the Perturbation Operator.
- **External Dependencies**: McDiarmid's Inequality cited from Boucheron–Lugosi–Massart reference.
- **Implementation Notes**: Two remarks provide critical implementation guidance on randomness handling and sampling policy requirements.
