# Enrichment Summary: thm-expected-raw-distance-k1

## Manual Refinement Process
**Date:** 2025-10-28
**Agent:** document-refiner (manual mode)
**Source:** docs/source/1_euclidean_gas/01_fragile_gas_framework.md (lines 2609-2645)

## Raw Input
- **Source File:** `raw_data/theorems/thm-expected-raw-distance-k1.json`
- **Theorem Type:** theorem
- **Section:** §10.2.8 - Discontinuous Behavior of the Expected Raw Distance Vector at k=1

## Enrichments Applied

### 1. Source Location
Added complete source metadata:
```json
{
  "document_id": "01_fragile_gas_framework",
  "file_path": "docs/source/1_euclidean_gas/01_fragile_gas_framework.md",
  "section": "§10.2.8",
  "directive_label": "thm-expected-raw-distance-k1",
  "line_range": [2609, 2645]
}
```

### 2. Output Type Classification
**Classified as:** `Property`
**Reasoning:** This theorem establishes a property of the expected raw distance vector when exactly one walker is alive (k=1). It characterizes the deterministic behavior of the system in this special case, making it a property statement rather than a bound, convergence result, or other type.

### 3. Input Objects
Identified 4 mathematical objects used in the theorem:
- `obj-swarm-and-state-space`: The swarm state S
- `obj-distance-to-companion-measurement`: The distance measurement d(S)
- `obj-raw-value-operator`: The raw value operator V
- `obj-companion-selection-measure`: The companion selection measure C_i

### 4. Input Axioms
**Result:** No axioms directly required
**Note:** This theorem relies on definitions rather than axioms. It follows from the metric property of d_alg and the definition of the companion selection measure.

### 5. Input Parameters
Identified 1 parameter:
- `param-N`: The total number of walkers (dimension of the expected distance vector)

### 6. Attributes Required
**Result:** No specific attributes required on objects
**Reasoning:** The theorem applies to any swarm state with |A(S)| = 1, without requiring additional properties on the objects.

### 7. Uses Definitions
Identified 3 critical definitions:
- `def-companion-selection-measure`: Defines how companions are selected (self-loop when k=1)
- `def-raw-value-operator`: Defines that dead walkers have raw value 0
- `def-distance-to-companion-measurement`: Defines the distance measurement process

### 8. Natural Language Statement
Enriched with complete prose statement:
"Let S be a swarm state with exactly one alive walker, |A(S)| = 1. The N-dimensional vector of expected raw distances, E[d(S)], is deterministically the zero vector. Any transition between a state S₁ with |A(S₁)| ≥ 2 and a state S₂ with |A(S₂)| = 1 induces a discontinuous change in the expected raw distance vector. The magnitude of this change is not governed by the Lipschitz bounds derived for the k ≥ 2 regime, but is instead given by the norm of the vector in the k ≥ 2 state: ||E[d(S₁)] - E[d(S₂)]||₂² = ||E[d(S₁)]||₂²."

### 9. Chapter and Document Metadata
Added hierarchical location:
- `chapter`: "1_euclidean_gas"
- `document`: "01_fragile_gas_framework"

## Validation
✓ **Pydantic validation passed**
- Label pattern: `thm-*` ✓
- Statement type auto-detected: `theorem` ✓
- All required fields present ✓
- Output type valid enum value ✓

## Mathematical Significance

### Key Insights
1. **Boundary Discontinuity:** This theorem formally establishes that the k=1 regime is fundamentally different from k≥2, with a discontinuous jump in the expected distance measurement.

2. **Revival Mechanism Connection:** The discontinuity at k=1 justifies the need for special revival dynamics rather than relying on continuity-based analysis.

3. **Metric Property Application:** The proof relies on the fact that d_alg(x_j, x_j) = 0 for any point, using the metric property that distance from a point to itself is zero.

4. **Self-Loop Behavior:** When only one walker survives, it becomes its own companion (self-loop in companion selection), leading to deterministically zero distance.

### Framework Context
This theorem appears in Section 10 (Abstract Raw Value Measurement) and establishes a critical boundary case that affects:
- Continuity analysis of the distance operator
- Revival mechanism design
- Treatment of single-survivor scenarios

The discontinuity is NOT a flaw but a fundamental property that guides the algorithm's behavior when the swarm nearly collapses.

## LLM Assistance
**Model Used:** Gemini 2.5 Pro
**Task:** Semantic enrichment and classification
**Result:** Provided initial output_type classification and identified key objects/definitions

## Next Steps
This refined theorem is ready for:
1. Integration into the Registry
2. Cross-reference resolution with Stage 3 (cross-referencer)
3. Proof sketch attachment (if needed)
4. Relationship inference with other theorems

## File Locations
- **Raw:** `docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/theorems/thm-expected-raw-distance-k1.json`
- **Refined:** `docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/theorems/thm-expected-raw-distance-k1.json`
- **Source Document:** `docs/source/1_euclidean_gas/01_fragile_gas_framework.md` (§10.2.8)
