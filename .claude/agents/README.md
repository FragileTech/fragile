# Mathematical Document Processing Agents

**Framework**: Claude Code Sub-Agents + Document Processing Pipeline
**Last Updated**: October 27, 2025

---

## Overview

This directory contains 7 specialized Claude Code agents for mathematical document processing and proof development. All agents are registered as Claude Code sub-agents and can be invoked using the Task tool.

---

## Agent Categories

### 1. Proof Development Agents

**proof-sketcher** - Generate proof strategy
- Dual validation: Gemini 2.5 Pro + GPT-5
- Output: Proof sketch with framework verification
- Invocation: `Use the proof-sketcher agent to...`

**theorem-prover** - Expand sketch to full proof
- Annals of Mathematics-level rigor
- Dual expansion with critical comparison
- Invocation: `Use the theorem-prover agent to...`

**math-reviewer** - Review mathematical documents
- Dual independent review
- Framework cross-validation
- Invocation: `Use the math-reviewer agent to...`

**math-verifier** - Validate algebraic steps
- Symbolic computation with sympy
- Executable validation scripts
- Invocation: `Use the math-verifier agent to...`

### 2. Document Processing Agents

**document-parser** (Claude Code agent + Python CLI)
- Stage 1: Raw extraction from MyST markdown
- Implementation: `src/fragile/agents/raw_document_parser.py`
- Invocation: `Use the document-parser agent to...`
- CLI: `python -m fragile.proofs.pipeline extract <document>`

**cross-referencer** (Claude Code agent)
- Stage 1.5: Relationship discovery
- Fills dependencies and typed relationships
- Invocation: `Use the cross-referencer agent to...`

**document-refiner** (Claude Code agent)
- Stage 2: Semantic enrichment
- Transforms raw JSON to validated entities
- Invocation: `Use the document-refiner agent to...`

---

## Quick Start

### Using Claude Code Agents

All agents with YAML frontmatter can be invoked via the Task tool:

```
Task: Use the proof-sketcher agent to generate a proof sketch for theorem thm-kl-convergence in docs/source/1_euclidean_gas/09_kl_convergence.md
```

### Complete Proof Development Workflow

```
1. Generate sketch:
   Task: Use the proof-sketcher agent to sketch proof for <theorem>

2. Expand to full proof:
   Task: Use the theorem-prover agent to expand <sketch_file>

3. Review proof:
   Task: Use the math-reviewer agent to review <proof_file>

4. Validate algebra:
   Task: Use the math-verifier agent to validate <proof_file>
```

### Document Processing Pipeline

```
1. Raw extraction (Task tool OR CLI):
   Task: Use the document-parser agent to extract <document.md>
   # OR
   python -m fragile.proofs.pipeline extract <document.md>

2. Relationship discovery:
   Task: Use the cross-referencer agent to analyze <document_dir>

3. Semantic enrichment:
   Task: Use the document-refiner agent to refine <raw_data_dir>
```

---

## Agent Files

### Specifications
- `proof-sketcher.md` - Proof strategy generator
- `theorem-prover.md` - Proof expander
- `math-reviewer.md` - Document reviewer
- `math-verifier.md` - Symbolic validator
- `cross-referencer.md` - Relationship analyzer
- `document-refiner.md` - Semantic enricher
- `document-parser.md` - Raw extractor (Python)

### Documentation
- `AGENT_STATUS.md` - Implementation status
- `README.md` - This file
- `*-QUICKSTART.md` - Quick start guides
- `*-README.md` - Detailed documentation

---

## Technical Details

### Agent Configuration

All Claude Code agents use this structure:

```yaml
---
name: agent-name
description: When this agent should be invoked
tools: Read, Grep, Glob, Bash, Write, mcp__gemini-cli__ask-gemini, mcp__codex__codex
model: sonnet
---
```

### Models Used

- **Primary**: Claude Sonnet 4.5 (agent orchestration)
- **Validation**: Gemini 2.5 Pro (strategic reasoning)
- **Expansion**: GPT-5 with high reasoning effort (constructive proofs)
- **Verification**: sympy (symbolic computation)

### Parallel Execution

All agents support parallel execution. Multiple instances can run simultaneously on different theorems/documents.

---

## Getting Started

1. **For proof development**: Start with `proof-sketcher-QUICKSTART.md`
2. **For document processing**: Start with `document-parser-QUICKSTART.md`
3. **For full details**: See `AGENT_STATUS.md`

---

## Support

For issues or questions:
- Check agent-specific QUICKSTART guides
- Review AGENT_STATUS.md for current implementation status
- Consult CLAUDE.md for project-level guidance
