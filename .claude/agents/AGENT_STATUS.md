# Agent System Status

**Last Verified**: 2025-10-26
**All Systems**: ✅ OPERATIONAL
**Parallel Execution**: ✅ ENABLED (all agents registered globally)

---

## Available Agents (5)

| Agent | Lines | Status | Global | Parallel | Color | Output Location |
|-------|-------|--------|--------|----------|-------|-----------------|
| **document-parser** | 456 | ✅ Ready | ✅ | ✅ | 🔵 Cyan | `docs/source/.../data/` |
| **math-reviewer** | 939 | ✅ Ready | ✅ | ✅ | 🔵 Blue | `docs/source/.../reviewer/` |
| **math-verifier** | 1115 | ✅ Ready | ✅ | ✅ | 🟠 Orange | `docs/source/.../verifier/` + `src/proofs/` |
| **proof-sketcher** | 1025 | ✅ Ready | ✅ | ✅ | 🟢 Green | `docs/source/.../sketcher/` |
| **theorem-prover** | 1426 | ✅ Ready | ✅ | ✅ | 🟣 Purple | `docs/source/.../proofs/` |

**Global**: Registered in `~/.claude/agents/` (available via `@agent-` calls)
**Parallel**: Can run multiple instances simultaneously

---

## MCP Server Status

| Server | Status | Purpose |
|--------|--------|---------|
| `mcp__gemini-cli__ask-gemini` | ✅ Authorized | Gemini 2.5 Pro API access |
| `mcp__codex__codex` | ✅ Authorized | GPT-5 Codex API access |

---

## Quick Invocation Reference

**PARALLEL EXECUTION** (NEW): Use `@agent-` mentions:

### Via `@agent-` Mentions (Parallel-Ready)
```
@agent-math-reviewer review docs/source/1_euclidean_gas/03_cloning.md
Depth: thorough
```

**Run Multiple in Parallel**:
```
Launch 3 agents:
1. @agent-math-reviewer review doc1.md
2. @agent-proof-sketcher sketch thm-A
3. @agent-math-verifier validate doc2.md
```

---

**ALTERNATIVE**: Use slash commands (sequential):

### Document Parser
```
/parse_doc docs/source/1_euclidean_gas/03_cloning.md
Mode: both
```

### Proof Sketcher
```
/proof_sketch thm-kl-convergence-euclidean
Document: docs/source/1_euclidean_gas/09_kl_convergence.md
```

### Math Verifier
```
/math_verify docs/source/1_euclidean_gas/03_cloning.md
Depth: thorough
```

### Theorem Prover
```
/prove docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md
```

### Math Reviewer
```
/math_review docs/source/2_geometric_gas/20_geometric_gas_cinf_regularity_full.md
Depth: thorough
Focus: Non-circularity, k-uniformity
```

---

**Alternative** (verbose, manual loading):
```
Load the [agent-name] agent from .claude/agents/[agent-name].md
[task description]
```

---

## Parallel Execution Example

All agents support parallel execution:

```
Launch 3 agents in parallel:

1. Load math-reviewer → Review docs/source/1_euclidean_gas/03_cloning.md
2. Load proof-sketcher → Sketch thm-keystone-lemma
3. Load math-verifier → Validate docs/source/1_euclidean_gas/04_convergence.md
```

---

## Agent Workflow Pipeline

**Recommended workflow for new theorems**:

```
1. Proof Sketcher    → Generate strategy (3-7 steps)
2. Math Verifier     → Validate algebra in strategy
3. Theorem Prover    → Expand to complete proof
4. Math Verifier     → Validate proof algebra
5. Math Reviewer     → Semantic validation (final QC)
```

**For existing documents**:

```
1. Math Verifier     → Check computational correctness
2. Math Reviewer     → Semantic + logic review
```

---

## Verification Tests

Run these to verify agents are working:

### Test 1: Document Parser
```
Load document-parser agent.

Parse: docs/source/1_euclidean_gas/03_cloning.md
Mode: sketch
```
Expected: Creates `docs/source/1_euclidean_gas/03_cloning/data/extraction_inventory.json`

### Test 2: Math Reviewer
```
Load math-reviewer agent.

Review: docs/source/1_euclidean_gas/03_cloning.md
Depth: quick
```
Expected: Creates `docs/source/1_euclidean_gas/reviewer/review_{timestamp}_03_cloning.md`

### Test 3: Math Verifier
```
Load math-verifier agent.

Validate: docs/source/1_euclidean_gas/03_cloning.md
Depth: quick
```
Expected: Creates validation scripts in `src/proofs/03_cloning/` and report in `verifier/`

### Test 4: Proof Sketcher
```
Load proof-sketcher agent.

Sketch proof for: thm-keystone-lemma
Document: docs/source/1_euclidean_gas/03_cloning.md
Depth: quick
```
Expected: Creates `sketcher/sketch_{timestamp}_proof_03_cloning.md`

### Test 5: Theorem Prover
(Requires proof sketch from Test 4 first)
```
Load theorem-prover agent.

Expand proof sketch: [path from Test 4]
```
Expected: Creates `proofs/proof_{timestamp}_thm_keystone_lemma.md`

---

## Configuration Files

| File | Purpose | Status |
|------|---------|--------|
| `.claude/agents/document-parser.md` | Agent prompt | ✅ 450 lines |
| `.claude/agents/math-reviewer.md` | Agent prompt | ✅ 933 lines |
| `.claude/agents/math-verifier.md` | Agent prompt | ✅ 1109 lines |
| `.claude/agents/proof-sketcher.md` | Agent prompt | ✅ 1019 lines |
| `.claude/agents/theorem-prover.md` | Agent prompt | ✅ 1420 lines |
| `.claude/agents/README.md` | Comprehensive guide | ✅ 930 lines |
| `.claude/agents/QUICKSTART.md` | Quick reference | ✅ 267 lines |
| `.claude/settings.local.json` | Permissions | ✅ Configured |

---

## Common Issues

### "Agent not found"
**Solution**: Use exact path:
```
Load the math-reviewer agent from .claude/agents/math-reviewer.md
```

### "MCP server unavailable"
**Check**: Run `grep mcp__gemini /home/guillem/fragile/.claude/settings.local.json`
**Status**: ✅ Both servers authorized

### "Document not found"
**Solution**: Use absolute path or verify file exists:
```bash
ls -lh docs/source/path/to/document.md
```

---

## Documentation

- **Full Guide**: `.claude/agents/README.md`
- **Quick Start**: `.claude/agents/QUICKSTART.md`
- **Framework Integration**: `CLAUDE.md` § Mathematical Proofing
- **Agent Definitions**: `.claude/agents/{agent-name}.md`

---

**Status Summary**: ✅ All 5 agents operational and ready for use
