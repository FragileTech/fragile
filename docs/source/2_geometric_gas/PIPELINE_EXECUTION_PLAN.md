# Autonomous Math Pipeline: Execution Plan
# Geometric Gas Framework (Chapter 2)

**Generated**: 2025-10-25
**Target**: `docs/source/2_geometric_gas/` (Folder Mode)
**Pipeline ID**: `pipeline_geometric_gas_20251025`

---

## Executive Summary

This pipeline will process **11 markdown documents** in the Geometric Gas chapter, proving **89 theorems/lemmas** that currently lack complete proofs.

### Scope

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total theorem statements** | 205 | 100% |
| **Complete proofs** | 116 | 56.6% |
| **Needs proof** | 83 | 40.5% |
| **Has sketch (TODO)** | 6 | 2.9% |
| **Pipeline target** | **89** | **43.4%** |

### Time Estimate

**Assumptions**:
- Average time per theorem: 2.75 hours (sketch 45min, expand 2-4h, review 30min)
- Iteration rate: 15% of theorems require re-expansion
- Average iterations per theorem: 1.15

**Estimated total time**: **245 hours (~10.2 days of continuous execution)**

**Important**: This is a multi-day pipeline that will run autonomously without user intervention.

---

## Documents to Process

### Green Zone (100% Complete) ✓
These documents have all proofs complete and will NOT be modified:

1. ✅ `00_intro_geometric_gas.md` (0 theorems needing proof)
2. ✅ `12_symmetries_geometric_gas.md` (8/8 complete)
3. ✅ `14_geometric_gas_c4_regularity.md` (13/13 complete)

### Yellow Zone (84-94% Complete)
These documents need minimal work:

4. 🟡 `13_geometric_gas_c3_regularity.md` (94%, **1 theorem** to prove)
5. 🟡 `20_geometric_gas_cinf_regularity_full.md` (90%, **3 sketches** to complete)
6. 🟡 `18_emergent_geometry.md` (84%, **4 theorems** to prove)

### Red Zone (<60% Complete) ⚠️
These documents require major work:

7. 🔴 `15_geometric_gas_lsi_proof.md` (60%, **2 theorems** - CRITICAL LSI proofs)
8. 🔴 `17_qsd_exchangeability_geometric.md` (75%, **1 theorem**)
9. 🔴 `11_geometric_gas.md` (39%, **26 theorems** - Core adaptive model)
10. 🔴 `19_geometric_gas_cinf_regularity_simplified.md` (4%, **22 theorems**)
11. 🔴 `16_convergence_mean_field.md` (10%, **26 theorems** - Mean-field theory)

---

## Execution Order (Topologically Sorted)

The pipeline will prove theorems in **89 steps**, ensuring all dependencies are satisfied before dependent theorems.

### Phase 1: Foundations (Theorems 1-20)

**Focus**: Foundation lemmas with no dependencies. Can be proven in parallel.

**Key theorems**:
- `lem-greedy-ideal-equivalence` (20_geometric_gas_cinf_regularity_full.md)
- `thm-backbone-convergence` (11_geometric_gas.md) - **SKETCH, needs completion**
- `thm-qsd-existence` (11_geometric_gas.md) - **SKETCH, needs completion**
- `lem-telescoping-derivatives` (13_geometric_gas_c3_regularity.md) - **ONLY theorem in Doc 13**

**Estimated time**: 55 hours (20 theorems × 2.75h avg)

### Phase 2: Core Regularity (Theorems 21-40)

**Focus**: Regularity results (C¹, C², C³ regularity proofs)

**Key theorems**:
- `prop-gevrey-regularization-cinf` (19)
- `thm-c3-established-cinf` (19)
- `thm-c1-established-cinf` (19)
- `cor-instantaneous-smoothing-cinf` (19)
- `lem-fisher-bound` (16_convergence_mean_field.md)

**Estimated time**: 55 hours (20 theorems × 2.75h avg)

### Phase 3: Advanced Results (Theorems 41-60)

**Focus**: Advanced regularity and QSD theory

**Key theorems**:
- `lem-telescoping-all-orders-cinf` (19) - **Foundation for C∞**
- `thm-qsd-smoothness` (16_convergence_mean_field.md)
- `thm-qsd-positivity` (16_convergence_mean_field.md)
- `thm-optimal-parameter-scaling` (16_convergence_mean_field.md)
- `thm-lsi-mean-field` (11_geometric_gas.md)

**Estimated time**: 55 hours (20 theorems × 2.75h avg)

### Phase 4: Convergence & LSI (Theorems 61-80)

**Focus**: LSI proofs and convergence results

**Key theorems**:
- `thm-gevrey-1-cinf` (19_geometric_gas_cinf_regularity_simplified.md)
- `thm-corrected-kl-convergence` (16_convergence_mean_field.md)
- `thm-lsi-geometric` (17_qsd_exchangeability_geometric.md)
- `thm-inductive-step-cinf` (19) - **Depends on telescoping**
- `thm-cinf-regularity` (19) - **MAIN C∞ result**

**Estimated time**: 55 hours (20 theorems × 2.75h avg)

### Phase 5: Final Results (Theorems 81-89)

**Focus**: Main convergence theorems and final results

**Critical theorems** (must be proven last):
- **81. `thm-adaptive-lsi-main`** (15_geometric_gas_lsi_proof.md) - **CRITICAL: N-Uniform LSI**
  - Resolves Framework Conjecture 8.3
  - Highest priority theorem in entire framework
- **88. `thm-lsi-adaptive-gas`** (11_geometric_gas.md) - **LSI for adaptive model**
- **89. `thm-keystone-adaptive`** (11_geometric_gas.md) - **Keystone lemma**

**Other final results**:
- `thm-stability-condition-rho` (11)
- `cor-geometric-ergodicity-lsi` (11) - **SKETCH, needs completion**
- `cor-exp-convergence` (11)

**Estimated time**: 25 hours (9 theorems × 2.75h avg)

---

## Dependency Structure

### Dependency Layers (for Parallelization)

The pipeline can achieve **7x parallelization** by processing theorems in dependency layers:

- **Layer 0** (52 theorems): No dependencies → Prove all in parallel
- **Layer 1** (10 theorems): Depend only on Layer 0
- **Layer 2** (8 theorems): Depend on Layer 0-1
- **Layer 3** (7 theorems): Depend on Layer 0-2
- **Layer 4** (5 theorems): Depend on Layer 0-3
- **Layer 5** (4 theorems): Depend on Layer 0-4
- **Layer 6** (3 theorems): Depend on Layer 0-5

**Critical path**: 7 layers deep (minimum sequential depth)

### Most Complex Theorems (by dependency count)

1. `prop-diversity-signal-rho` (4 dependencies)
2. `thm-stability-condition-rho` (4 dependencies)
3. `thm-cinf-regularity` (4 dependencies) - **MAIN C∞ result**
4. `thm-lsi-adaptive-gas` (3 dependencies) - **LSI for adaptive**
5. `thm-keystone-adaptive` (3 dependencies) - **Keystone lemma**

These theorems should be proven **last** as they depend on many other results.

---

## Missing Dependencies (Must Resolve First)

Before starting the pipeline, the following **13 missing dependencies** must be resolved:

### Missing Definitions (4) ⚠️

These definitions are referenced but not found. They need to be added or their labels corrected:

1. **`def-adaptive-generator-cinf`** - Referenced by C∞ regularity theorems
2. **`def-localized-mean-field-fitness`** - Referenced by `prop-limiting-regimes`
3. **`def-localized-mean-field-moments`** - Referenced by `prop-limiting-regimes`, `prop:bounded-adaptive-force`
4. **`def-unified-z-score`** - Referenced by `prop-limiting-regimes`, `prop-rate-metric-ellipticity`

**Recommended action**: Search documents for these concepts and add proper `{prf:definition}` blocks with the specified labels.

### Missing Assumptions (2) ⚠️

1. **`assump-cinf-primitives`** - Referenced by all C∞ regularity theorems in Doc 19
2. **`assump-uniform-density-full`** - Referenced by density bound lemmas in Doc 20

**Recommended action**: Formalize these as `{prf:assumption}` blocks in the appropriate documents.

### Document Cross-References (3) ✓

These are references to other chapters and are satisfied externally (no action needed):

1. `doc-02-euclidean-gas` - Chapter 1 (Euclidean Gas)
2. `doc-03-cloning` - Chapter 1 (Cloning theory)
3. `doc-13-geometric-gas-c3-regularity` - Within Chapter 2 (C³ regularity)

### Other Missing (4) ⚠️

1. **`lem-conditional-gaussian-qsd`** - Referenced by `thm-fitness-third-deriv-proven`
2. **`rem-concentration-lsi`** - Referenced by `cor-geometric-ergodicity-lsi`
3. **`thm-ueph-proven`** - Referenced by `thm-adaptive-lsi-main`, `thm-fitness-third-deriv-proven`
   - Note: This may exist as `thm-ueph` (label mismatch)
4. Malformed reference (parsing error) - Needs correction

**Recommended action**:
- For (1) and (2): Search for these concepts and add proper labels
- For (3): Change references from `thm-ueph-proven` to `thm-ueph` (which exists)
- For (4): Fix parsing error in source document

---

## Integration Strategy

### Auto-Integration (Rigor ≥ 9/10)

Proofs meeting high-confidence threshold will be automatically integrated into source documents:

- Backup created before modification (`.backup_YYYYMMDD_HHMMSS`)
- Proof inserted immediately after theorem statement
- LaTeX formatting validated
- Cross-references verified
- Document validated by Math Reviewer

**Expected**: ~60% of proofs (based on historical pipeline data)

### Manual Review (8 ≤ Rigor < 9)

Proofs meeting publication standard but below auto-integration threshold:

- Proof files remain in `proofs/` directory
- Integration guide created in `pipeline_manual_review_guide.md`
- User can review and integrate manually

**Expected**: ~30% of proofs

### Needs Refinement (Rigor < 8)

Proofs below publication standard after 3 attempts:

- Marked as "needs_manual_refinement" in state file
- Proof files remain for reference
- Listed in final report for manual completion

**Expected**: ~10% of proofs (mostly very hard theorems)

---

## Risk Assessment

### High-Risk Theorems (Likely to Need Iteration)

Based on complexity and technical difficulty:

1. **`thm-adaptive-lsi-main`** (15_geometric_gas_lsi_proof.md)
   - Very high complexity (LSI with companion mechanisms)
   - May require 2-3 iterations
   - Estimated time: 8-12 hours

2. **Mean-field convergence theorems** (16_convergence_mean_field.md)
   - QSD existence, smoothness, positivity (requires measure theory expertise)
   - KL-convergence rates (requires careful epsilon-delta arguments)
   - High risk of needing iteration

3. **C∞ regularity chain** (19_geometric_gas_cinf_regularity_simplified.md)
   - Inductive proofs with Gevrey estimates
   - 22 interdependent theorems
   - Telescoping arguments (sensitive to detail)

### Time Padding

The 245-hour estimate assumes:
- 15% iteration rate (average 1.15 iterations per theorem)
- 2.75 hours per theorem

**Confidence interval**:
- Best case (no iterations): 200 hours (8.3 days)
- Expected case: 245 hours (10.2 days)
- Worst case (30% iteration): 290 hours (12.1 days)

---

## Pipeline Configuration

### Integration Strategy
**Hybrid** (default):
- Auto-integrate proofs with rigor ≥ 9/10
- Manual review for 8 ≤ rigor < 9
- Flag for refinement if rigor < 8

### Dependency Handling
**Auto-resolve** (default):
- Automatically prove missing lemmas recursively
- Topologically sort theorems by dependencies
- Build complete dependency chains

### Quality Standards
**Auto-iterate** (default):
- Re-expand proofs until rigor ≥ 8/10
- Maximum 3 iterations per theorem
- Focus on specific gaps identified by Math Reviewer

### Parallelism
**Layer-based** (default):
- Process theorems in dependency layers
- Maximum 4 concurrent theorems per layer
- Prioritize by estimated complexity within layer

---

## Files Generated During Pipeline

### Output Directories

All output will be created in `/home/guillem/fragile/docs/source/2_geometric_gas/`:

1. **`sketcher/`** - Proof strategy sketches (89 files)
   - Format: `sketch_{timestamp}_proof_{theorem_label}.md`
   - Size: ~1-2 MB total

2. **`proofs/`** - Complete rigorous proofs (89 files)
   - Format: `proof_{timestamp}_{theorem_label}.md`
   - Size: ~10-15 MB total

3. **`reviewer/`** - Math Reviewer assessments (89+ files)
   - Format: `review_{timestamp}_proof_{theorem_label}.md`
   - Additional: Final document validation reviews
   - Size: ~5-8 MB total

4. **`backups/`** - Original document backups (before integration)
   - Format: `{document_name}.backup_{timestamp}`
   - Created only for documents that are modified
   - Size: ~3-5 MB total

### State Files

1. **`pipeline_state.json`** - Current pipeline state
   - Updated after each theorem completion
   - Used for resume capability
   - Contains: execution plan, theorem status, statistics, errors

2. **`pipeline_report_{timestamp}.md`** - Final comprehensive report
   - Generated at pipeline completion
   - Contains: statistics, integration summary, validation results

3. **`pipeline_manual_review_guide.md`** - Integration guide
   - Created for theorems requiring manual review
   - Contains: specific integration instructions, issue summaries

### Reference Files (Already Generated)

These files were created during pipeline initialization:

- **`PROOF_PRIORITY_SUMMARY.md`** - High-level priority summary
- **`GEOMETRIC_GAS_PROOF_SURVEY.txt`** - Complete proof status survey
- **`geometric_gas_theorems.csv`** - Machine-readable theorem database
- **`theorem_dependencies.json`** - Dependency graph (machine-readable)
- **`DEPENDENCY_ANALYSIS.md`** - Dependency analysis (human-readable)
- **`DEPENDENCY_LAYERS.md`** - Layer-based visualization

---

## Progress Tracking

The pipeline will output progress updates every 30 minutes:

```markdown
===================================================================
PIPELINE PROGRESS UPDATE
===================================================================
Elapsed time: 42h 15m / 245h estimated (17.2% of total time)
Current theorem: thm-qsd-smoothness (Theorem 54/89)
Current stage: expanding (Theorem Prover running, 2h 30m elapsed)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Completed: 53/89 theorems (59.6%)
  - Auto-integrated: 32 (rigor ≥ 9/10)
  - Manual review: 18 (8 ≤ rigor < 9)
  - Needs refinement: 3 (rigor < 8)

In progress: 1/89 (thm-qsd-smoothness)
Pending: 35/89

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Average rigor score: 8.7/10
Average time per theorem: 2.8 hours
Iteration rate: 17% (9 theorems required re-expansion)

Estimated time remaining: 98 hours (4.1 days)
Estimated completion: 2025-10-29 18:30:00

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Resume Capability

The pipeline can be interrupted at any time and resumed from the last checkpoint.

### Interruption Scenarios

1. User interrupts (Ctrl+C, timeout, context limit)
2. System failure (crash, network loss)
3. Deliberate pause (user wants to check progress)

### Resume Protocol

When `/math_pipeline docs/source/2_geometric_gas/` is invoked again:

1. Detect existing `pipeline_state.json`
2. Read current state (completed theorems, current theorem, current stage)
3. Resume from last checkpoint:
   - If stage = "sketching": Check if sketch exists, otherwise restart
   - If stage = "expanding": Check if proof exists, otherwise restart
   - If stage = "reviewing": Check if review exists, otherwise restart
   - If stage = "integrating": Check if integration completed, otherwise restart
4. Continue to next theorem in execution order

### State File Corruption Recovery

If `pipeline_state.json` is corrupted:

1. Scan output directories (sketcher/, proofs/, reviewer/)
2. Reconstruct state from discovered artifacts
3. Resume from reconstructed state

---

## Validation & Safety

### Document Backups

Before any document is modified:

1. Create timestamped backup: `{document}.backup_{timestamp}`
2. Record backup location in state file
3. If integration fails: Automatically restore from backup

### Integration Validation

After auto-integration:

1. Re-read modified document
2. Verify proof was inserted correctly
3. Check for broken cross-references
4. Validate LaTeX formatting (blank lines before `$$`)
5. Launch Math Reviewer for final document validation

### Cross-Reference Verification

After all integrations:

1. Extract all `{prf:ref}` references in modified documents
2. Verify each reference resolves correctly
3. Check against `docs/glossary.md`
4. Report broken references in final report

---

## Restoration Instructions

### Restore All Documents to Pre-Pipeline State

```bash
# Restore all modified documents
for backup in docs/source/2_geometric_gas/*.backup_*; do
    original="${backup%.backup_*}"
    cp "$backup" "$original"
done
```

### Restore Specific Document

```bash
# Example: Restore 11_geometric_gas.md
cp docs/source/2_geometric_gas/11_geometric_gas.md.backup_20251025_153000 \
   docs/source/2_geometric_gas/11_geometric_gas.md
```

### Rerun Pipeline from Scratch

```bash
# Delete state file and restart
rm docs/source/2_geometric_gas/pipeline_state.json
/math_pipeline docs/source/2_geometric_gas/
```

---

## Next Steps

### Before Starting Pipeline

1. **Review this execution plan** - Understand scope and time commitment
2. **Resolve missing dependencies** - Add the 4 missing definitions and 2 missing assumptions
3. **Verify label mismatches** - Check if `thm-ueph-proven` should be `thm-ueph`
4. **Confirm auto-resolution strategy** - Decide if truly missing lemmas should be auto-resolved

### Starting the Pipeline

The pipeline will:

1. ✓ **Initialization complete** (this file is the execution plan)
2. Wait for user confirmation
3. Create state file and output directories
4. Begin Phase 1: Foundation theorems (1-20)
5. Continue through all 89 theorems in dependency order
6. Integrate completed proofs
7. Validate all modified documents
8. Generate final comprehensive report

### After Pipeline Completion

1. Review `pipeline_report_{timestamp}.md` for statistics
2. Integrate manual-review proofs (if any) using `pipeline_manual_review_guide.md`
3. Address any broken cross-references
4. Build documentation: `make build-docs`
5. Commit changes: `git add docs/source/2_geometric_gas/ && git commit`

---

## Contact & Support

- **Pipeline state**: Check `pipeline_state.json` for current status
- **Progress updates**: Logged every 30 minutes
- **Errors**: Logged to `pipeline_state.json` errors array
- **Resume**: Run `/math_pipeline docs/source/2_geometric_gas/` to resume after interruption

---

**Generated by**: Autonomous Math Pipeline v1.0
**Execution plan ID**: `pipeline_geometric_gas_20251025`
**Ready to start**: Awaiting user confirmation

---

## Appendix: Full Execution Order

```
1. lem-greedy-ideal-equivalence (20_geometric_gas_cinf_regularity_full.md)
2. prop-complete-gradient-bounds (16_convergence_mean_field.md)
3. lem-macro-transport (11_geometric_gas.md)
4. thm-data-processing (16_convergence_mean_field.md)
5. thm-c2-established-cinf (19_geometric_gas_cinf_regularity_simplified.md)
... [continues for all 89 theorems]
89. thm-keystone-adaptive (11_geometric_gas.md)
```

See `DEPENDENCY_ANALYSIS.md` for the complete ordered list.
