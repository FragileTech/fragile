# Section 6 Extraction Report

**Agent**: Document Parser (Raw Extraction - Stage 1)
**Source**: `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md`
**Section**: §6 - Algorithm Space and Distance Measurement
**Line Range**: 1212-1241 (30 lines)
**Date**: 2025-10-27

---

## Extraction Summary

- **Total Entities Extracted**: 3
- **Total Files Created**: 3
- **Extraction Method**: Manual extraction by Claude Code
- **Reason**: ANTHROPIC_API_KEY not available for LLM-based extraction

### Entity Breakdown

| Entity Type  | Count |
|--------------|-------|
| Definitions  | 3     |
| Theorems     | 0     |
| Axioms       | 0     |
| Proofs       | 0     |
| Equations    | 0     |
| Parameters   | 0     |
| Remarks      | 0     |
| Citations    | 0     |

---

## Output Directory Structure

```
/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/
├── raw_data/
│   └── definitions/
│       ├── raw-def-001.json  (Algorithmic Space)
│       ├── raw-def-002.json  (Distance Between Positional Measures)
│       └── raw-def-003.json  (Algorithmic Distance)
└── statistics/
    └── section6_raw_statistics.json
```

---

## Extracted Entities

### 1. Definition: Algorithmic Space
- **Label**: `def-algorithmic-space-generic`
- **File**: `raw-def-001.json`
- **Subsection**: 5.1 Specification of the Algorithmic Space
- **Content**: An **algorithmic space** is a pair $(\mathcal{Y}, d_{\mathcal{Y}})$ consisting of a real vector space $\mathcal{Y}$ and a true metric $d_{\mathcal{Y}}$ on $\mathcal{Y}$.
- **Parameters**: `\mathcal{Y}`, `d_{\mathcal{Y}}`

### 2. Definition: Distance Between Positional Measures
- **Label**: `def-distance-positional-measures`
- **File**: `raw-def-002.json`
- **Subsection**: 5.2 Distance Between Positional Measures
- **Content**: Let two walkers, $i$ and $j$, have their positions represented by the Dirac positional measures $\delta_{x_i}$ and $\delta_{x_j}$. The distance between them in the algorithmic space is the **1-Wasserstein distance ($W_1$)** between their **projected positional measures**, with $d_{\mathcal{Y}}$ as the ground metric.
- **Formula**:
  $$d(\varphi_* \delta_{x_i}, \varphi_* \delta_{x_j}) := W_1(\delta_{\varphi(x_i)}, \delta_{\varphi(x_j)})$$
- **Parameters**: `i`, `j`, `x_i`, `x_j`, `\delta_{x_i}`, `\delta_{x_j}`, `W_1`, `d_{\mathcal{Y}}`, `\varphi`, `\varphi_*`

### 3. Definition: Algorithmic Distance
- **Label**: `def-alg-distance`
- **File**: `raw-def-003.json`
- **Subsection**: 5.3 Algorithmic Distance
- **Content**: The **algorithmic distance** $d_{\text{alg}}\colon\mathcal{X}\times\mathcal{X}\to\mathbb{R}_{\ge0}$ is the distance between the projected positional measures of two walkers. In practice, this is the distance or semidistance function $d_{\mathcal{Y}}$ applied to the projected points in the algorithmic space.
- **Formula**:
  $$\boxed{d_{\text{alg}}(x_1, x_2) := d_{\mathcal{Y}}(\varphi(x_1), \varphi(x_2))}$$
- **Purpose**: This is the practical implementation of the Wasserstein distance between the walkers' projected Dirac measures and serves as the ground distance for all subsequent calculations.
- **Parameters**: `d_{\text{alg}}`, `\mathcal{X}`, `\mathbb{R}_{\ge0}`, `d_{\mathcal{Y}}`, `\varphi`, `x_1`, `x_2`

---

## Mathematical Context

Section 6 establishes the foundational framework for measuring distances between walkers in the Fragile Gas algorithm. The three definitions form a logical progression:

1. **Algorithmic Space** establishes the geometric framework $(\mathcal{Y}, d_{\mathcal{Y}})$
2. **Distance Between Positional Measures** defines the theoretical foundation using 1-Wasserstein distance
3. **Algorithmic Distance** provides the practical implementation formula

### Key Mathematical Concepts

- **Projection Map** $\varphi$: Maps from state space $\mathcal{X}$ to algorithmic space $\mathcal{Y}$
- **Pushforward Measure** $\varphi_*$: Transports Dirac measures through the projection
- **Wasserstein Distance** $W_1$: Optimal transport distance between probability measures
- **Ground Metric** $d_{\mathcal{Y}}$: The metric on the algorithmic space used for distance calculations

### Simplification for Dirac Measures

For Dirac measures $\delta_{x_i}$ and $\delta_{x_j}$, the Wasserstein distance simplifies:
$$W_1(\delta_{\varphi(x_i)}, \delta_{\varphi(x_j)}) = d_{\mathcal{Y}}(\varphi(x_i), \varphi(x_j))$$

This simplification is the basis for the practical **algorithmic distance** formula.

---

## Next Steps

### Stage 2: Document Refiner (Enrichment)

The raw entities extracted here should be processed through the **document-refiner** agent for semantic enrichment:

```bash
# Refine Section 6 entities
Load document-refiner agent.
Refine: docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/
Mode: section6
```

### Expected Enrichment Tasks

1. **Cross-reference resolution**: Link to related definitions (e.g., valid state space, projection map)
2. **Dependency analysis**: Identify prerequisite definitions
3. **Type inference**: Classify as structural definitions vs. operational definitions
4. **Notation standardization**: Ensure consistent use of $\mathcal{Y}$, $d_{\mathcal{Y}}$, etc.
5. **Validation**: Verify against Pydantic schemas for enriched models

---

## Files Generated

### Raw Entity Files
- `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/definitions/raw-def-001.json`
- `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/definitions/raw-def-002.json`
- `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/definitions/raw-def-003.json`

### Statistics
- `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/statistics/section6_raw_statistics.json`

### Reports
- This file: `SECTION6_EXTRACTION_REPORT.md`

---

## Status

✅ **Complete** - All entities from Section 6 successfully extracted and saved to individual JSON files following the staging_types schema.
