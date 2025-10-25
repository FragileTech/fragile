# Geometric Gas Framework: Proof Survey & Priority Guide

This directory contains a comprehensive survey of all mathematical statements (theorems, lemmas, propositions, corollaries) in the Geometric Gas framework documentation, with their proof status and priority rankings.

## Survey Results

**Total Statements:** 205 across 11 documents  
**Survey Date:** 2025-10-25  
**Coverage:** All main documents in `docs/source/2_geometric_gas/` (excludes backups and generated files)

### Quick Summary

| Status | Count | % |
|--------|-------|-----|
| Complete Proofs | 116 | 56.6% |
| Needs Proofs | 83 | 40.5% |
| Sketches/TODOs | 6 | 2.9% |

---

## Files in This Survey

### 1. `PROOF_PRIORITY_SUMMARY.md` 📋
**START HERE** for a quick overview.

Contains:
- Top 5 highest priority items
- Document completion status (GREEN/YELLOW/RED zones)
- 3-week recommended workflow
- What's already done (don't redo!)
- Phase-based action plan

**Best for:** Quick decisions, getting started, understanding priorities

---

### 2. `GEOMETRIC_GAS_PROOF_SURVEY.txt` 📊
**FULL REFERENCE** - comprehensive analysis document.

Contains:
- Executive summary and statistics
- Document-by-document breakdown (11 sections)
- Detailed proof status for each document
- Document completion matrix
- Proof priority matrix (Category A/B/C)
- Actionable recommendations (Phase 1/2/3)
- Statistics by document class
- Notes on missing proofs (common patterns)
- Quality assurance recommendations
- Full statement index

**Best for:** Understanding all the details, reference, Gemini reviews

---

### 3. `geometric_gas_theorems.csv` 📈
**DATA FORMAT** - machine-readable data for tracking.

Contains:
- 205 rows (one per theorem/lemma/proposition/corollary)
- Columns: Document, Label, Type, Title, Line, Status
- Easy to sort/filter in spreadsheet software
- Import into tracking systems

**Best for:** 
- Filtering by document or status
- Tracking updates
- Generating reports
- Automated processing

**How to use:**
```bash
# View in terminal
column -t -s',' geometric_gas_theorems.csv | head -20

# Sort by status
sort -t',' -k6 geometric_gas_theorems.csv | grep needs_proof

# Count by document
cut -d',' -f1 geometric_gas_theorems.csv | sort | uniq -c
```

---

## Navigation Guide

### If you want to...

**Understand the overall status:**
→ Read `PROOF_PRIORITY_SUMMARY.md` (5 min read)

**Know what to work on next:**
→ Check PHASE 1 section in `PROOF_PRIORITY_SUMMARY.md`

**Get all the details:**
→ Read `GEOMETRIC_GAS_PROOF_SURVEY.txt` (comprehensive reference)

**Track completion programmatically:**
→ Use `geometric_gas_theorems.csv` in Excel/Google Sheets/Python

**Find a specific theorem:**
→ Search `geometric_gas_theorems.csv` for the label or title

**Understand why a proof is missing:**
→ Search `GEOMETRIC_GAS_PROOF_SURVEY.txt` for the document name

---

## Critical Items Summary

### The "Must Do" List

1. **Document 15, Line 1279:** `thm-adaptive-lsi-main`
   - N-Uniform Log-Sobolev Inequality (BLOCKER)
   - Resolves Framework Conjecture 8.3
   - Highest priority item

2. **Document 11, Multiple:** 26 core stability proofs
   - Including: signal generation, keystone lemma for adaptive case
   - Without these: no convergence guarantees
   - 61.9% of document missing

3. **Document 16, Multiple:** 26 mean-field convergence proofs
   - Most demanding section (89.7% missing)
   - Requires QSD theory and hypoellipticity expertise
   - Depends on items #1 and #2

4. **Document 19:** Decision point
   - 95.7% empty despite "simplified" label
   - Option A: Consolidate with Doc 20 (recommended)
   - Option B: Complete the simplified proof

5. **Document 20:** 3 sketches with TODOs
   - Otherwise 90% complete
   - Quick wins (LOW-MEDIUM effort)

---

## Document Status at a Glance

```
✓ COMPLETE (100%)
  - 00_intro_geometric_gas.md (introduction only)
  - 12_symmetries_geometric_gas.md (all 8 results proven)
  - 14_geometric_gas_c4_regularity.md (all 13 results proven)

🟡 NEARLY DONE (84-94%)
  - 13_geometric_gas_c3_regularity.md (94%, 1 item missing)
  - 18_emergent_geometry.md (84%, 4 items need verification)
  - 20_geometric_gas_cinf_regularity_full.md (90%, 3 sketches)

🟠 PARTIALLY DONE (60-84%)
  - 17_qsd_exchangeability_geometric.md (75%, 1 item)
  - 15_geometric_gas_lsi_proof.md (60%, 2 critical items)

🔴 MAJOR WORK NEEDED (<40%)
  - 11_geometric_gas.md (39%, 26 missing)
  - 16_convergence_mean_field.md (10%, 26 missing)
  - 19_geometric_gas_cinf_regularity_simplified.md (4%, 22 missing)
```

---

## How to Update This Survey

The survey was generated automatically by analyzing all theorem statements. To update:

1. Make changes to the markdown documents as you complete proofs
2. Run the analysis script (see source code in GEOMETRIC_GAS_PROOF_SURVEY.txt)
3. Update the CSV and summary files
4. Commit with message: "Update proof survey after completing [document/items]"

The analysis script:
- Finds all `:::{prf:theorem}`, `:::{prf:lemma}`, etc.
- Checks for `:::{prf:proof}` blocks
- Detects TODO/SKETCH/INCOMPLETE markers
- Classifies as: needs_proof / has_sketch / has_complete_proof

---

## Standards & Conventions

All proofs should follow CLAUDE.md standards:
- Use `{prf:proof}` blocks immediately after theorem statements
- No TODO/SKETCH markers in final version
- Cross-references use Jupyter Book `{prf:ref}` directives
- Mathematical notation matches `docs/glossary.md`
- Use dual-review protocol (Gemini + Codex) from CLAUDE.md

---

## Key Facts

### Completed Documents (No work needed)
- **Symmetries (Doc 12):** All 8 results complete
- **C⁴ Regularity (Doc 14):** All 13 results complete
- **Introduction (Doc 00):** Overview only (no theorems)

### Nearly Ready (Quick wins)
- **C³ Regularity (Doc 13):** 16/17 complete, 1 telescoping identity
- **C∞ Regularity Full (Doc 20):** 35/39 complete, 3 sketches with TODOs
- **Emergent Geometry (Doc 18):** 21/25 complete, 4 cross-refs

### Critical Path (Blockers)
- **LSI Proof (Doc 15):** 3/5 complete, 2 critical items missing
- **Core Model (Doc 11):** 13/42 complete, 26 missing
- **Mean-Field (Doc 16):** 3/29 complete, 26 missing (hardest)

### Decision Point (Needs triage)
- **C∞ Simplified (Doc 19):** 1/23 complete, 22 missing
  - Recommend: Consolidate with Doc 20 instead of completing separately

---

## Effort Estimates

| Category | Items | Est. Time |
|----------|-------|-----------|
| Low (cross-refs, minor fixes) | 10 | 2-4 hours |
| Medium (lemmas, bounds) | 35 | 1-2 weeks |
| High (convergence, LSI) | 38 | 2-4 weeks |
| Very High (mean-field theory) | 26 | 4-8 weeks |

**Total to 100% completion:** 6-12 weeks (depending on expertise)

---

## Document Details

### By Difficulty Level

**Easy (mostly complete, few gaps):**
- Doc 13: C³ Regularity → 1 telescoping
- Doc 14: C⁴ Regularity → All done ✓
- Doc 20: C∞ Full → 3 TODOs

**Medium (substantial work but well-defined):**
- Doc 11: Core Model → 26 proofs (reference Part I heavily)
- Doc 12: Symmetries → All done ✓
- Doc 18: Emergent Geometry → 4 cross-refs

**Hard (research-level, requires expertise):**
- Doc 15: LSI Proof → 2 deep theory results
- Doc 16: Mean-Field → 26 QSD/convergence results
- Doc 17: QSD Properties → 1 LSI result
- Doc 19: C∞ Simplified → 22 (mostly reference docs)

---

## How to Read the Full Report

The main survey report (`GEOMETRIC_GAS_PROOF_SURVEY.txt`) is organized as:

1. **EXECUTIVE SUMMARY** - Quick stats (2 min read)
2. **DOCUMENT-BY-DOCUMENT BREAKDOWN** - Each of 11 docs (30 min read)
3. **DOCUMENT COMPLETION MATRIX** - Visual summary
4. **PROOF PRIORITY MATRIX** - Categories A/B/C
5. **ACTIONABLE RECOMMENDATIONS** - Phased workflow
6. **STATISTICS BY DOCUMENT CLASS** - Summary by type
7. **NOTES ON MISSING PROOFS** - Common patterns
8. **RECOMMENDATIONS FOR COMPLETION** - Next steps
9. **APPENDIX** - Full theorem index

---

## Integration with Your Workflow

### For Gemini Reviews
Use `GEOMETRIC_GAS_PROOF_SURVEY.txt` as context when asking Gemini to:
- Review a specific proof
- Suggest approaches for missing proofs
- Verify cross-references
- Check consistency with Part I

### For Tracking
Update `geometric_gas_theorems.csv` as you complete proofs:
- Change status to "has_complete_proof"
- Re-generate summary stats
- Commit to git with proof completion message

### For Documentation
When writing proofs:
- Check the full survey for related work
- Reference completed proofs in same document
- Use labels consistently
- Run formatting tools (from `src/tools/`)

---

## Questions & Answers

**Q: Should I complete Doc 19 or consolidate with Doc 20?**
A: Consolidate (Option A). Doc 20 already has complete C∞ proof. Doc 19 is 95% empty.

**Q: Which document should I start with?**
A: Start with Doc 15 (LSI proof) - it's the highest priority and shortest (5 items).

**Q: What if I get stuck on a proof?**
A: Use the full survey to find related work in other documents, then use Gemini for review.

**Q: Can I work on Doc 16 (mean-field) first?**
A: No - it depends on completing Doc 15 and Doc 11 first.

**Q: How do I know if a "sketch" is actually incomplete?**
A: Search for TODO, SKETCH, INCOMPLETE keywords in the proof block.

---

## Contact & Maintenance

This survey was generated on: **2025-10-25**

To regenerate after updates:
```python
# Use the analysis script embedded in GEOMETRIC_GAS_PROOF_SURVEY.txt
# Or contact the person who created this survey
```

---

**Total Time Investment:** ~6-12 weeks to 100% completion  
**Current Status:** 56.6% complete (116/205 proofs)  
**Critical Path:** Document 15 → Document 11 → Document 16  

Start with `PROOF_PRIORITY_SUMMARY.md` for your next action!
