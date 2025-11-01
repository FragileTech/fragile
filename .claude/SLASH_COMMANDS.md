# Available Slash Commands for Mathematical Agents

**Last Updated**: 2025-10-26
**All Commands**: ✅ OPERATIONAL

---

## Quick Reference Card

| Command | Purpose | Runtime | Output |
|---------|---------|---------|--------|
| `/parse_doc` | Extract mathematical content | ~30 sec | `data/extraction_inventory.json` |
| `/proof_sketch` | Generate proof strategy | ~45 min | `sketcher/sketch_*.md` |
| `/math_verify` | Validate algebra | ~30 min | `verifier/*.md` + `src/proofs/` |
| `/prove` | Expand to complete proof | ~2-4 hrs | `proofs/proof_*.md` |
| `/math_review` | Dual-review quality control | ~45 min | `reviewer/review_*.md` |

---

## 1. Parse Document - `/parse_doc`

**Extract mathematical content from MyST markdown**

```
/parse_doc docs/source/1_euclidean_gas/03_cloning.md
Mode: both
```

**Options**:
- `Mode: sketch | expand | both`
- `No LLM: true` (skip relationship inference)
- `Skip proofs: true` (extract only objects/theorems)

**Output**: `docs/source/.../data/extraction_inventory.json`

---

## 2. Proof Sketch - `/proof_sketch`

**Generate proof strategy (3-7 high-level steps)**

```
/proof_sketch thm-kl-convergence-euclidean
Document: docs/source/1_euclidean_gas/09_kl_convergence.md
```

**Options**:
- `Depth: quick | thorough | exhaustive`
- `Focus: [specific topics]`
- `Theorems: [comma-separated labels]`

**Output**: `docs/source/.../sketcher/sketch_{timestamp}_proof_{doc}.md`

**Uses**: Gemini 2.5 Pro + GPT-5 Pro

---

## 3. Math Verify - `/math_verify`

**Validate algebraic manipulations using sympy**

```
/math_verify docs/source/1_euclidean_gas/03_cloning.md
Depth: thorough
```

**Options**:
- `Theorem: [specific theorem label]`
- `Depth: quick | thorough | exhaustive`
- `Focus: [algebraic categories]`

**Output**:
- Validation scripts: `src/proofs/{doc_name}/*.py`
- Report: `docs/source/.../verifier/verification_{timestamp}_{doc}.md`

**Uses**: Gemini 2.5 Pro + GPT-5 Pro + sympy

---

## 4. Prove - `/prove`

**Expand proof sketch to complete rigorous proof (Annals of Mathematics standard)**

```
/prove docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md
```

**Options**:
- `Depth: standard | maximum`
- `Focus: [specific steps or technical elements]`
- `Expand steps: [e.g., 4-5]`

**Output**: `docs/source/.../proofs/proof_{timestamp}_{theorem_label}.md`

**Uses**: Gemini 2.5 Pro + GPT-5 Pro

**Runtime**: 2-3 hours (standard), 4-6 hours (maximum)

---

## 5. Math Review - `/math_review`

**Dual-review quality control (Gemini + Codex)**

```
/math_review docs/source/2_geometric_gas/20_geometric_gas_cinf_regularity_full.md
Depth: thorough
Focus: Non-circularity, k-uniformity claims
```

**Options**:
- `Depth: quick | thorough | exhaustive`
- `Focus: [specific sections or topics]`

**Output**: `docs/source/.../reviewer/review_{timestamp}_{doc}.md`

**Uses**: Gemini 2.5 Pro + Codex GPT-5 Pro

**Runtime**: 10 min (quick), 45 min (thorough), 2 hours (exhaustive)

---

## Complete Workflow Examples

### New Theorem (Full Pipeline)

```bash
# 1. Parse document structure
/parse_doc docs/source/1_euclidean_gas/09_kl_convergence.md

# 2. Generate proof strategy
/proof_sketch thm-kl-convergence-euclidean
Document: docs/source/1_euclidean_gas/09_kl_convergence.md

# 3. Validate strategy algebra (optional but recommended)
/math_verify docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md

# 4. Expand to complete proof
/prove docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md

# 5. Validate proof algebra
/math_verify docs/source/1_euclidean_gas/mathster/proof_20251024_1630_thm_kl_convergence_euclidean.md

# 6. Final quality control
/math_review docs/source/1_euclidean_gas/mathster/proof_20251024_1630_thm_kl_convergence_euclidean.md
Depth: exhaustive
```

**Total time**: ~4-6 hours
**Output**: Publication-ready proof with dual validation

---

### Existing Document Review

```bash
# 1. Validate computational correctness
/math_verify docs/source/1_euclidean_gas/03_cloning.md
Depth: thorough

# 2. Semantic + logical review
/math_review docs/source/1_euclidean_gas/03_cloning.md
Depth: thorough
Focus: Keystone Principle, companion selection
```

**Total time**: ~1-2 hours
**Output**: Comprehensive validation report

---

### Quick Strategy Check

```bash
# Just get proof strategy (no expansion)
/proof_sketch thm-keystone-lemma
Document: docs/source/1_euclidean_gas/03_cloning.md
Depth: quick
```

**Total time**: ~15 minutes
**Output**: High-level proof approach

---

## Parallel Execution

All commands support parallel execution:

```
Launch 3 reviews in parallel:

/math_review docs/source/1_euclidean_gas/03_cloning.md
/math_review docs/source/2_geometric_gas/11_geometric_gas.md
/math_review docs/source/3_brascamp_lieb/eigenvalue_gap.md
```

Each command runs independently and produces separate output files.

---

## Common Patterns

### Pattern 1: Sketch → Prove → Review
```
/proof_sketch [theorem]
/prove [sketch-file]
/math_review [proof-file]
```

### Pattern 2: Verify → Review (Existing Docs)
```
/math_verify [document]
/math_review [document]
```

### Pattern 3: Parse → Sketch (Structure Analysis)
```
/parse_doc [document]
/proof_sketch [document]
```

### Pattern 4: Dual Validation (Maximum Rigor)
```
/proof_sketch [theorem]
/math_verify [sketch-file]
/prove [sketch-file]
/math_verify [proof-file]
/math_review [proof-file]
```

---

## Output File Structure

After running commands, your document directory will have:

```
docs/source/1_euclidean_gas/03_cloning/
├── 03_cloning.md                    # Original document
├── data/
│   ├── extraction_inventory.json    # From /parse_doc
│   └── statistics.json
├── sketcher/
│   └── sketch_20251024_1530_*.md    # From /proof_sketch
├── verifier/
│   └── verification_20251024_*.md   # From /math_verify
├── proofs/
│   └── proof_20251024_1630_*.md     # From /prove
└── reviewer/
    └── review_20251024_1430_*.md    # From /math_review
```

Plus validation scripts:
```
src/proofs/03_cloning/
├── thm_keystone_lemma.py            # From /math_verify
├── lem_variance_decomp.py
└── ...
```

---

## Tips

1. **Always sketch before proving**: `/proof_sketch` first, then `/prove`
2. **Use thorough depth by default**: Good balance of coverage vs. time
3. **Verify algebra early**: Run `/math_verify` on sketches to catch errors early
4. **Focus areas speed up review**: Specific focus = faster, deeper analysis
5. **Parallel execution**: Launch multiple commands at once for efficiency

---

## Getting Help

- **Full agent docs**: `.claude/agents/README.md`
- **Quick start**: `.claude/agents/QUICKSTART.md`
- **Agent definitions**: `.claude/agents/{agent-name}.md`
- **This reference**: `.claude/SLASH_COMMANDS.md`
- **Framework guide**: `CLAUDE.md`

---

**All commands ready! Use `/agents` or type `/` to see available commands.**
