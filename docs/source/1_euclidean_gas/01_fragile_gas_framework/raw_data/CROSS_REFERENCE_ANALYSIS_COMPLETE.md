# Cross-Reference Analysis Report

**Date**: 2025-10-27
**Framework**: Fragile Gas Framework (01_fragile_gas_framework)
**Total Entities**: 81 theorem-like mathematical statements

---

## Executive Summary

A comprehensive cross-reference analysis infrastructure has been created to fill dependency fields for all 81 theorem-like entities in the Fragile Gas Framework. The system uses both pattern-matching and LLM-based analysis (Gemini 2.5 Pro) to identify:

- **input_objects**: Mathematical objects each result depends on
- **input_axioms**: Required axioms
- **input_parameters**: Parameters appearing in bounds
- **output_type**: Classification of result type
- **relations_established**: Specific mathematical relationships proven

**Status**: Infrastructure complete, ready for systematic processing.

---

## Entity Inventory

### Distribution by Type

| Type | Count | Directory |
|------|-------|-----------|
| Theorems | 30 | `raw_data/theorems/` |
| Lemmas | 45 | `raw_data/lemmas/` |
| Propositions | 3 | `raw_data/propositions/` |
| Corollaries | 3 | `raw_data/corollaries/` |
| **TOTAL** | **81** | |

### Statement Availability

- **With valid statements**: 69 entities
- **Missing statements**: 12 entities (skipped in LLM analysis)

---

## Framework Context

### Available Entities (for cross-referencing)

| Category | Count | Example Labels |
|----------|-------|----------------|
| Objects | 37 | `obj-algorithmic-space-generic`, `obj-perturbation-measure`, `obj-cloning-measure` |
| Axioms | 5 | `axiom-raw-value-mean-square-continuity`, `axiom-rescale-function` |
| Parameters | 2 | `param-kappa-variance`, `param-F-V-ms` |
| Definitions | 31 | `def-perturbation-measure`, `def-statistical-properties-measurement` |

**Total framework entities**: 75

---

## Analysis Phases

### Phase 1: Pattern-Matching Cross-Reference ✅ COMPLETE

**Script**: `scripts/cross_reference_raw_data.py`

**Method**:
- Extract explicit dependencies from existing `dependencies` and `uses_definitions` fields
- Map definition labels to corresponding object labels
- Perform keyword matching in statement text
- Infer `output_type` from statement keywords (Lipschitz, bounded, continuous, etc.)

**Results**:
- **Entities processed**: 81
- **Objects filled**: 22
- **Axioms filled**: 7
- **Parameters filled**: 0
- **Output types inferred**: 81
- **Validation errors**: 7 (missing dependencies)

**Limitations**:
- Misses implicit dependencies
- Keyword matching is inexact
- Cannot identify subtle mathematical relationships

### Phase 2: LLM-Based Dependency Detection 🔄 IN PROGRESS

**Script**: `scripts/gemini_batch_processor.py`

**Method**:
- For each entity, create comprehensive prompt with:
  - Full mathematical statement
  - Complete list of available framework entities
  - Instructions to identify ALL dependencies
- Query Gemini 2.5 Pro for analysis
- Parse JSON response with validated labels
- Fill all dependency fields

**Infrastructure**:
- ✅ All 69 valid entities organized into 9 batches
- ✅ Prompts generated for each entity
- ✅ Batch metadata (batch_info.json) created
- ✅ Application script ready (`apply_cross_references.py`)

**Batch Distribution**:

| Batch | Entities | Status |
|-------|----------|--------|
| batch_000 | 10 | Ready for processing |
| batch_001 | 6 | Ready for processing |
| batch_002 | 10 | Ready for processing |
| batch_003 | 9 | Ready for processing |
| batch_004 | 6 | Ready for processing |
| batch_005 | 9 | Ready for processing |
| batch_006 | 10 | Ready for processing |
| batch_007 | 9 | Ready for processing |
| batch_008 | 0 | (all processed in earlier batches) |
| **TOTAL** | **69** | |

---

## Sample Analysis

### Example: lem-boundary-heat-kernel

**Label**: `lem-boundary-heat-kernel`
**Name**: Heat-kernel death probability is Lipschitz with constant ≲ 1/σ

**Statement** (excerpt):
> Let $E=\mathcal{X}_{\mathrm{invalid}}\subset\mathcal X$ have finite perimeter and let $p_{\sigma^2}$ be the heat kernel at scale $\sigma$. Define $P_\sigma(x)=\int \chi_E(y)\,p_{\sigma^2}(x,\mathrm dy)$. Then $|P_\sigma(x)-P_\sigma(y)| \le C_d'\,\frac{\mathrm{Per}(E)}{\sigma}\, d_{\mathcal X}(x,y)$...

**Gemini 2.5 Pro Analysis**:
```json
{
  "input_objects": [
    "obj-algorithmic-space-generic",
    "obj-alg-distance",
    "obj-smoothed-gaussian-measure",
    "obj-cemetery-state-measure",
    "obj-boundary-regularity-constants",
    "obj-perturbation-constants"
  ],
  "input_axioms": [],
  "input_parameters": [],
  "output_type": "Lipschitz",
  "relations_established": [
    "Establishes that the heat-kernel-smoothed probability of entering an invalid set (P_sigma(x)) is Lipschitz continuous.",
    "Bounds the Lipschitz constant of the death probability by a term proportional to Per(E)/σ.",
    "Relates the Lipschitz constant of the death probability (L_death) in the algorithmic metric to the perimeter of the invalid set and the Lipschitz constant of the coordinate map (L_phi)."
  ]
}
```

**Quality Assessment**:
- ✅ All labels valid (exist in framework)
- ✅ Comprehensive dependency identification (6 objects)
- ✅ Specific, concrete relations (not generic)
- ✅ Correct output_type classification

---

## Scripts Created

### 1. cross_reference_raw_data.py
**Purpose**: Pattern-matching cross-referencer
**Status**: ✅ Complete and tested
**Usage**:
```bash
python scripts/cross_reference_raw_data.py \
  /home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data
```
**Output**: Updated entity JSONs, CROSS_REFERENCE_REPORT.md

### 2. llm_cross_reference.py
**Purpose**: LLM prompt generation
**Status**: ✅ Complete
**Usage**:
```bash
python scripts/llm_cross_reference.py \
  /path/to/raw_data \
  --output-prompts /output/dir \
  --limit 5
```
**Output**: Individual prompt files with metadata

### 3. batch_cross_reference.py
**Purpose**: Batch prompt generation with metadata
**Status**: ✅ Complete
**Usage**:
```bash
python scripts/batch_cross_reference.py \
  /path/to/raw_data \
  --save-prompts /dir
```
**Output**: Batch of prompts with metadata

### 4. gemini_batch_processor.py
**Purpose**: Full batch processing system
**Status**: ✅ Complete and tested
**Usage**:
```bash
# Check total batches
python scripts/gemini_batch_processor.py /path/to/raw_data

# Generate specific batch
python scripts/gemini_batch_processor.py /path/to/raw_data --generate-batch 0
```
**Output**: Numbered batches with prompts and metadata

### 5. apply_cross_references.py
**Purpose**: Apply JSON analysis results to entity files
**Status**: ✅ Complete
**Usage**:
```bash
python scripts/apply_cross_references.py \
  /tmp/gemini_batches/batch_000_results.json \
  /path/to/raw_data
```
**Output**: Updated entity JSON files, statistics summary

---

## Processing Workflow

### For Each Batch

1. **Load batch info**:
   ```bash
   cat /tmp/gemini_batches/batch_000/batch_info.json
   ```

2. **For each entity in batch**:
   - Read prompt: `cat /tmp/gemini_batches/batch_000/<label>.txt`
   - Query Gemini 2.5 Pro (via Claude Code MCP):
     ```
     mcp__gemini-cli__ask-gemini(
       model="gemini-2.5-pro",
       prompt=<prompt_content>
     )
     ```
   - Parse JSON response
   - Validate labels
   - Save result: `/tmp/gemini_batches/batch_000/<label>.result.json`

3. **Consolidate results**:
   ```json
   {
     "batch_num": 0,
     "processed": 10,
     "results": {
       "lem-boundary-heat-kernel": { ... },
       ...
     }
   }
   ```
   Save as: `/tmp/gemini_batches/batch_000_results.json`

4. **Apply to entity files**:
   ```bash
   python scripts/apply_cross_references.py \
     /tmp/gemini_batches/batch_000_results.json \
     /path/to/raw_data
   ```

5. **Review statistics and validate**

---

## Validation Criteria

For each analysis result:

| Check | Requirement |
|-------|-------------|
| Label validity | All `input_objects` exist in `refined_data/objects/` |
| | All `input_axioms` exist in `raw_data/axioms/` |
| | All `input_parameters` exist in `raw_data/parameters/` |
| Completeness | ALL mathematical objects mentioned in statement identified |
| | Implicit dependencies (e.g., spaces, measures) included |
| Specificity | `relations_established` are concrete, not generic |
| | Each relation describes a specific mathematical result |
| Correctness | `output_type` matches theorem category |
| | Dependencies logically necessary for the result |

---

## File Locations

### Input
- **Raw entity files**: `raw_data/{theorems,lemmas,propositions,corollaries}/*.json`
- **Framework context**: `raw_data/{objects,axioms,parameters,definitions}/*.json`

### Generated
- **Batch prompts**: `/tmp/gemini_batches/batch_NNN/<label>.txt`
- **Batch metadata**: `/tmp/gemini_batches/batch_NNN/batch_info.json`
- **Batch results**: `/tmp/gemini_batches/batch_NNN_results.json`

### Output
- **Updated entities**: `raw_data/{theorems,lemmas,propositions,corollaries}/*.json` (modified in place)
- **Reports**: `raw_data/CROSS_REFERENCE_*.md`

### Scripts
- **All scripts**: `/home/guillem/fragile/scripts/`

---

## Statistics

### Current (After Pattern-Matching)

| Metric | Value |
|--------|-------|
| Entities processed | 81 |
| Objects filled | 22 |
| Axioms filled | 7 |
| Parameters filled | 0 |
| Output types filled | 81 |
| Relations filled | 0 |
| Missing dependencies | 7 |

### Projected (After LLM Analysis)

| Metric | Estimated Value |
|--------|-----------------|
| Entities with full analysis | 69 |
| Objects filled | 200-300 |
| Axioms filled | 30-50 |
| Parameters filled | 10-20 |
| Output types filled | 69 |
| Relations established | 150-200 |
| Average objects per entity | 3-4 |
| Average relations per entity | 2-3 |

---

## Time Estimates

| Task | Entities | Time per Entity | Total Time |
|------|----------|-----------------|------------|
| **Batch 000** | 10 | 2-3 min | 20-30 min |
| **Batches 001-007** | 59 | 2-3 min | 2-3 hours |
| **Validation & Review** | 69 | 30 sec | 30-40 min |
| **Final Report** | - | - | 15 min |
| **TOTAL** | 69 | - | **3-4 hours** |

---

## Next Actions

1. **Process batch_000** (10 entities)
   - Demonstrates complete workflow
   - Validates prompt quality
   - Establishes baseline quality

2. **Review batch_000 results**
   - Check label accuracy
   - Verify completeness
   - Assess relation quality

3. **Process batches 001-007** (59 entities)
   - Apply same workflow systematically
   - Track statistics per batch

4. **Generate final report**
   - Aggregate all statistics
   - Validate all dependencies
   - Create dependency visualization

5. **Integration with proof system**
   - Load enriched entities into MathematicalRegistry
   - Validate theorem-proof consistency
   - Generate dependency graphs

---

## Conclusion

**Infrastructure**: ✅ Complete
**Pattern-matching**: ✅ Complete (22 objects, 7 axioms filled)
**LLM batch system**: ✅ Ready (69 entities, 9 batches)
**Application pipeline**: ✅ Ready
**Sample analysis**: ✅ Validated (lem-boundary-heat-kernel)

**Next Step**: Systematically process batches 000-007 using Gemini 2.5 Pro to complete full cross-reference analysis.

All infrastructure, scripts, and workflows are in place. The system is ready for comprehensive dependency analysis of the Fragile Gas Framework.

---

**Report Generated**: 2025-10-27
**Location**: `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/`
**Scripts**: `/home/guillem/fragile/scripts/`
**Batches**: `/tmp/gemini_batches/`
