# How to Run Mathematical Agents

**Quick Answer**: Use the **slash commands** - they're already configured and ready!

---

## ✅ All 5 Agents Are Ready

| Slash Command | Agent | What It Does |
|---------------|-------|--------------|
| `/parse_doc` | document-parser | Extract mathematical content from MyST markdown |
| `/proof_sketch` | proof-sketcher | Generate proof strategies (3-7 steps) |
| `/math_verify` | math-verifier | Validate algebra using sympy |
| `/prove` | theorem-prover | Expand sketches to complete proofs |
| `/math_review` | math-reviewer | Dual-review quality control |

---

## Usage Examples

### Review a Document
```
/math_review docs/source/2_geometric_gas/20_geometric_gas_cinf_regularity_full.md
Depth: thorough
Focus: Non-circularity, k-uniformity
```

### Sketch a Proof
```
/proof_sketch thm-kl-convergence-euclidean
Document: docs/source/1_euclidean_gas/09_kl_convergence.md
```

### Validate Algebra
```
/math_verify docs/source/1_euclidean_gas/03_cloning.md
```

### Expand to Complete Proof
```
/prove docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md
```

### Parse Document Structure
```
/parse_doc docs/source/1_euclidean_gas/03_cloning.md
Mode: both
```

---

## Why Only "theorem-prover" Shows in `/agents`?

Good question! Here's why:

1. **`theorem-prover`** is registered as a built-in subagent type in Claude Code
2. **The other 4 agents** are custom agents accessed via slash commands
3. **Slash commands are actually better** because they:
   - Are easier to use (just type `/command`)
   - Have autocomplete
   - Are properly configured with all settings
   - Are documented in `.claude/commands/`

So you're actually set up with the **best** invocation method already!

---

## How to See All Available Commands

Type `/` in the chat to see all slash commands, or run:
```
cat .claude/SLASH_COMMANDS.md
```

---

## Complete Workflow Example

**Goal**: Create a publication-ready proof for a new theorem

```bash
# 1. Generate strategy (~45 min)
/proof_sketch thm-my-theorem
Document: docs/source/1_euclidean_gas/my_document.md

# 2. Validate strategy algebra (~20 min, optional)
/math_verify docs/source/1_euclidean_gas/sketcher/sketch_*.md

# 3. Expand to complete proof (~2-4 hours)
/prove docs/source/1_euclidean_gas/sketcher/sketch_*.md

# 4. Validate proof algebra (~30 min)
/math_verify docs/source/1_euclidean_gas/proofs/proof_*.md

# 5. Final quality control (~1 hour)
/math_review docs/source/1_euclidean_gas/proofs/proof_*.md
Depth: exhaustive
```

**Total**: ~4-6 hours for a complete, validated, publication-ready proof

---

## Quick References

- **Full docs**: `.claude/agents/README.md`
- **Quick start**: `.claude/agents/QUICKSTART.md`
- **Slash commands**: `.claude/SLASH_COMMANDS.md`
- **Agent status**: `.claude/agents/AGENT_STATUS.md`
- **This guide**: `.claude/agents/HOW_TO_RUN.md`

---

## Summary

✅ **You don't need to do anything special** - just use the slash commands!

✅ **All agents are working** and accessible via:
- `/parse_doc`
- `/proof_sketch`
- `/math_verify`
- `/prove`
- `/math_review`

✅ **Type `/` to see all available commands**

✅ **Agents run autonomously** - just provide input and they handle everything

---

**Ready to use! Try running any of the commands above.**
