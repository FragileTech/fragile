# Section 18 Extraction Report

## Summary

**Document**: `docs/source/1_euclidean_gas/01_fragile_gas_framework.md`  
**Section**: 18. Swarm Update Operator: A Composition of Measures  
**Lines**: 4627-5086 (460 lines)  
**Extraction Date**: 2025-10-27

## Entities Extracted

### Total: 14 mathematical entities

| Type | Count |
|------|-------|
| Definitions | 4 |
| Theorems | 1 |
| Lemmas | 5 |
| Propositions | 3 |
| Proofs | 1 |

## Entity Details

### Definitions (4)

1. **def-swarm-update-procedure**
   - Name: Swarm Update Procedure
   - File: `definitions/def-swarm-update-procedure.json`

2. **def-final-status-change-coeffs**
   - Name: Final Status Change Bound Coefficients
   - Subsection: 17.2.3.1
   - File: `objects/def-final-status-change-coeffs.json`

3. **def-composite-continuity-coeffs-recorrected**
   - Name: Composite Continuity Coefficients
   - Subsection: 17.2.4.1
   - File: `objects/def-composite-continuity-coeffs-recorrected.json`

4. **def-w2-output-metric**
   - Name: Wasserstein-2 on the output space (quotient)
   - Subsection: 17.2.4.4
   - File: `definitions/def-w2-output-metric.json`

### Theorems (1)

1. **thm-swarm-update-operator-continuity-recorrected**
   - Name: Continuity of the Swarm Update Operator
   - Subsection: 17.2.4
   - File: `theorems/thm-swarm-update-operator-continuity-recorrected.json`

### Lemmas (5)

1. **lem-final-positional-displacement-bound**
   - Name: Bounding the Final Positional Displacement (unconditional)
   - Subsection: 17.2.2
   - File: `theorems/lem-final-positional-displacement-bound.json`

2. **lem-final-status-change-bound**
   - Name: Bounding the Expected Final Status Change
   - Subsection: 17.2.3.2
   - File: `theorems/lem-final-status-change-bound.json`

3. **lem-inequality-toolbox**
   - Name: Inequality Toolbox
   - Subsection: 17.2.4.0
   - File: `theorems/lem-inequality-toolbox.json`

4. **lem-subadditivity-power**
   - Name: Subadditivity of Fractional Powers
   - Subsection: 17.2.4.2a
   - File: `theorems/lem-subadditivity-power.json`

5. **sub-lem-unify-holder-terms**
   - Name: Unifying Multiple Hölder Terms (global, with case split)
   - Subsection: 17.2.4.3
   - File: `theorems/sub-lem-unify-holder-terms.json`

### Propositions (3)

1. **prop-w2-bound-no-offset**
   - Name: W2 continuity bound without offset (for k≥2)
   - Subsection: 17.2.4.4
   - File: `theorems/prop-w2-bound-no-offset.json`

2. **prop-psi-markov-kernel**
   - Name: The Swarm Update defines a Markov kernel
   - Subsection: 17.2.4.5
   - File: `theorems/prop-psi-markov-kernel.json`

3. **prop-coefficient-regularity**
   - Name: Boundedness and continuity of composite coefficients
   - Subsection: 17.2.4.6
   - File: `theorems/prop-coefficient-regularity.json`

### Proofs (1)

1. **proof-composite-continuity-bound-recorrected**
   - Name: Proof of the Composite Continuity Bound
   - Subsection: 17.2.4.2
   - File: `proofs/proof-composite-continuity-bound-recorrected.json`

## Output Structure

```
docs/source/1_euclidean_gas/01_fragile_gas_framework/
├── definitions/
│   ├── def-swarm-update-procedure.json
│   └── def-w2-output-metric.json
├── objects/
│   ├── def-final-status-change-coeffs.json
│   └── def-composite-continuity-coeffs-recorrected.json
├── theorems/
│   ├── lem-final-positional-displacement-bound.json
│   ├── lem-final-status-change-bound.json
│   ├── lem-inequality-toolbox.json
│   ├── lem-subadditivity-power.json
│   ├── sub-lem-unify-holder-terms.json
│   ├── thm-swarm-update-operator-continuity-recorrected.json
│   ├── prop-w2-bound-no-offset.json
│   ├── prop-psi-markov-kernel.json
│   └── prop-coefficient-regularity.json
├── proofs/
│   └── proof-composite-continuity-bound-recorrected.json
└── section_18_extraction_summary.json
```

## Files Created

Total files created: **14 JSON files** + 1 summary file

### By Subdirectory:
- `definitions/`: 2 files
- `objects/`: 2 files
- `theorems/`: 9 files
- `proofs/`: 1 file

## Next Steps

These extracted entities are now ready for:
1. Cross-reference analysis (using `cross-referencer` agent)
2. Dependency mapping
3. Proof validation
4. Integration with the global mathematical framework

## Notes

- All JSON files contain metadata including:
  - `label`: Unique identifier
  - `type`: Entity type (definition, theorem, lemma, proposition, proof)
  - `name`: Human-readable name
  - `section`: Section number (18)
  - `subsection`: Subsection identifier (where applicable)
  - `source_file`: Source markdown file
  - `source_lines`: Line range in source
  - `extraction_date`: Date of extraction
  - `status`: Extraction status

- Coefficient definitions were placed in `objects/` subdirectory
- All theorems, lemmas, and propositions were placed in `theorems/` subdirectory
- The main swarm update definition captures the complete 4-stage algorithm
