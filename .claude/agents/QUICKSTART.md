# Math Reviewer Agent - Quick Start Guide

## Simplest Usage (Copy-Paste Ready)

### Single Document Review

Just paste this into Claude:

```
Please load the math-reviewer agent from .claude/agents/math-reviewer.md
and review the document at:

docs/source/2_geometric_gas/20_geometric_gas_cinf_regularity_full.md

Use thorough depth and focus on non-circularity and k-uniformity.
```

### Parallel Review (Multiple Documents)

```
Please load the math-reviewer agent and run 3 parallel reviews:

1. docs/source/1_euclidean_gas/03_cloning.md (focus: Keystone Principle)
2. docs/source/2_geometric_gas/19_geometric_gas_cinf_regularity_simplified.md
3. docs/source/3_brascamp_lieb/eigenvalue_gap_complete_proof.md (exhaustive depth)

Provide separate comprehensive reports for each.
```

---

## What Happens

1. Agent reads the document strategically (extracts 4-6 key sections)
2. Submits identical prompts to Gemini 2.5 Pro + Codex in parallel
3. Waits for both reviews to complete
4. **Critically compares** both reviews:
   - Identifies consensus issues (high confidence)
   - Identifies contradictions (investigates)
   - Cross-validates against framework docs
5. Makes **evidence-based judgments** about who is correct
6. Produces comprehensive report with:
   - Issue summary table (compact overview)
   - Detailed issue analysis table
   - Detailed analysis
   - Proposed fixes
   - Implementation checklist
   - Your decision points
7. **Writes report to file**: `reviewer/review_{timestamp}_{filename}.md`

---

## Expected Output Format

You'll receive a report like this:

```markdown
# Dual Review Summary for {filename}

## Comparison Overview
- Consensus Issues: 4 (both agree)
- Gemini-Only Issues: 1
- Codex-Only Issues: 2
- Contradictions: 1

## Issue Summary Table
| # | Issue | Severity | Location | Gemini | Codex | Claude | Status |
|---|-------|----------|----------|--------|-------|--------|--------|
| 1 | Non-circular density | CRITICAL | §2.3.5, lines 450-465 | CRITICAL - Insufficient | CRITICAL - Not compact | ✅ Verified | ✗ Contradicts |

## Issue Analysis Table
| # | Issue | Severity | Gemini | Codex | Claude | Verification |
|---|-------|----------|--------|-------|--------|--------------|
| 1 | Non-circular density | CRITICAL | Insufficient proof | Not compact | ✅ Verified CRITICAL | ✗ Contradicts |

## Detailed Issues and Proposed Fixes

### Issue #1: Non-Circular Density Bound (CRITICAL)
- **Location**: §2.3.5, lines 447-609
- **Gemini's Analysis**: "Insufficient proof that doc-13 avoids density assumptions"
- **Codex's Analysis**: "Velocity squashing doesn't make domain compact - fatal flaw"
- **My Assessment**: ✅ VERIFIED CRITICAL - Codex correct about mechanism

**Evidence**:
- Checked doc-02: velocity squashing is post-processing, NOT dynamical
- SDE evolves unsquashed v which can grow unboundedly
- Breaks compactness → density bound fails

**Proposed Fix**:
```
Reformulate dynamics in terms of ψ(v) using Itô's lemma, OR
Replace compactness with moment bound approach
```

**Consensus**: AGREE with both - Codex identifies mechanism, Gemini identifies gap

---

[... full report continues ...]

## Final Verdict

**Recommendation**: MAJOR REVISIONS REQUIRED

The document contains:
- 1 CRITICAL flaw (non-compact velocity)
- 4 MAJOR issues (k_eff inconsistency, etc.)
- 5 MINOR issues

**User, would you like me to**:
1. Implement fixes for Issues #1, #2?
2. Draft revised proofs?
3. Create detailed action plan?

---

✅ Review written to: docs/source/2_geometric_gas/reviewer/review_20251024_1430_20_geometric_gas_cinf_regularity_full.md
```

---

## Customization Options

### Depth Levels

**Quick** (~10 min):
```
Depth: quick
```
Reviews abstract + main theorems only

**Thorough** (~30-45 min, DEFAULT):
```
Depth: thorough
```
Reviews key sections including proofs

**Exhaustive** (~1-2 hours):
```
Depth: exhaustive
```
Complete document analysis

### Focus Areas

Be specific:
```
Focus on:
- Section 2.3.5 (non-circular density proof)
- Lemma 6.4 (k-uniformity via telescoping)
- Appendix A (Faà di Bruno combinatorics)
```

---

## Real Example

I'll demonstrate with the document you have open:

```
Load math-reviewer agent.

Review: docs/source/2_geometric_gas/20_geometric_gas_cinf_regularity_full.md

Depth: thorough

Focus on:
- §2.3.5: Non-circular density bound logical chain
- §5.7: Statistical equivalence of companion mechanisms
- §6.3-6.4: Telescoping identity for k-uniformity
- Appendix A: Faà di Bruno factorial bounds

Produce comprehensive report with issue table, verification status, and implementation checklist.
```

**What the agent will do**:
1. Extract those 4 sections (~1000 lines total)
2. Submit to Gemini 2.5 Pro + Codex with identical prompts
3. Compare both reviews critically
4. Verify claims against glossary.md and source docs
5. Produce ~6000-word report with specific fixes

**Time**: ~40 minutes

---

## Parallel Execution Example

Review 3 documents simultaneously:

```
Launch 3 math-reviewer agents in parallel:

Agent 1: Review docs/source/1_euclidean_gas/03_cloning.md
  - Focus: Keystone Principle (§8), Companion Selection (§5.1)
  - Depth: thorough

Agent 2: Review docs/source/2_geometric_gas/19_geometric_gas_cinf_regularity_simplified.md
  - Depth: quick (sanity check)

Agent 3: Review docs/source/3_brascamp_lieb/eigenvalue_gap_complete_proof.md
  - Focus: LSI constants, spectral gap bounds
  - Depth: exhaustive

Provide 3 separate comprehensive reports.
```

All 3 will run independently and complete around the same time.

---

## Comparison: Agent vs Slash Command

| Feature | Math Reviewer Agent | /dual_review Command |
|---------|-------------------|----------------------|
| **Autonomy** | Fully autonomous | Interactive guidance |
| **Parallelizable** | ✅ Yes | ❌ No |
| **Strategic extraction** | ✅ Automatic | Manual |
| **Large docs (>2000 lines)** | ✅ Handles well | ⚠️ May struggle |
| **Critical evaluation** | ✅ Makes judgments | Shows both views |
| **Framework verification** | ✅ Automatic | Manual |
| **Output format** | Standardized template | Flexible |
| **Best for** | Complex documents | Quick checks |

**Recommendation**: Use agent for serious reviews, slash command for quick feedback.

---

## Tips

1. **Start with thorough depth** - it's the sweet spot for most documents

2. **Be specific about focus** - agent will prioritize those sections

3. **Check file path first**:
   ```
   ls -lh docs/source/path/to/file.md
   ```

4. **For huge documents** (>5000 lines), review in parts:
   ```
   Review Part I: lines 1-2000
   Review Part II: lines 2000-4000
   etc.
   ```

5. **After review**, you can ask agent to implement specific fixes

---

## Next Steps After Review

1. **Read the issue table** - sorted by severity
2. **Start with CRITICAL issues** - these break the proof
3. **Verify agent's verification** - double-check important claims
4. **Implement fixes** - use proposed fixes as starting point
5. **Re-run review** - verify fixes with another pass

---

That's it! Just copy-paste the simple usage above to get started.

For more details, see:
- Full agent definition: `.claude/agents/math-reviewer.md`
- Complete docs: `.claude/agents/README.md`
- Framework context: `CLAUDE.md`
