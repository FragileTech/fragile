# Agent Implementation Status

**Last Updated**: October 27, 2025

---

## Claude Code Agents (Task Tool)

All agents below are registered as Claude Code sub-agents and can be invoked via the Task tool.

### Mathematical Proof Pipeline

#### 1. proof-sketcher
**Status**: ✅ **READY**
**Type**: Proof Strategy Generator
**Invocation**: `Use the proof-sketcher agent to generate proof sketches for <theorem>`
**Models**: Gemini 2.5 Pro + GPT-5 (pinned)
**Output**: `sketcher/sketch_{timestamp}_proof_{filename}.md`

#### 2. theorem-prover
**Status**: ✅ **READY**
**Type**: Proof Expander
**Invocation**: `Use the theorem-prover agent to expand proof sketch <file>`
**Models**: Gemini 2.5 Pro + GPT-5 (pinned)
**Output**: `proofs/proof_{timestamp}_{theorem_label}.md`
**Rigor**: Annals of Mathematics standard

#### 3. math-reviewer
**Status**: ✅ **READY**
**Type**: Document Reviewer
**Invocation**: `Use the math-reviewer agent to review <document>`
**Models**: Gemini 2.5 Pro + Codex (pinned)
**Output**: `reviewer/review_{timestamp}_{filename}.md`

#### 4. math-verifier
**Status**: ✅ **READY**
**Type**: Symbolic Validator
**Invocation**: `Use the math-verifier agent to validate <document>`
**Models**: Gemini 2.5 Pro + GPT-5 (pinned)
**Output**: Validation scripts + verification report
**Engine**: sympy

---

### Document Processing Pipeline

#### 5. document-parser
**Status**: ✅ **READY** (Python CLI + Task tool)
**Type**: Raw Extractor (Stage 1)
**Invocation (Task)**: `Use the document-parser agent to extract <document>`
**Invocation (CLI)**: `python -m fragile.proofs.pipeline extract <document>`
**Implementation**: `src/fragile/agents/raw_document_parser.py`
**Input**: MyST markdown documents
**Output**: `raw_data/` directory with JSON files

#### 6. cross-referencer
**Status**: ✅ **READY**
**Type**: Relationship Analyzer (Stage 1.5)
**Invocation**: `Use the cross-referencer agent to analyze <document_dir>`
**Input**: document-parser output
**Output**: Enhanced JSON + `relationships/` directory

#### 7. document-refiner
**Status**: ✅ **READY** (specification complete, implementation pending)
**Type**: Semantic Enricher (Stage 2)
**Invocation**: `Use the document-refiner agent to refine <raw_data_dir>`
**Input**: `raw_data/` from document-parser
**Output**: `refined_data/` with enriched JSON

---

## Removed/Deprecated

### math_document_parser.py
**Status**: ❌ **DEPRECATED** (moved to `.deprecated`)
**Reason**: Mixed Stage 1/2 logic, replaced by two-stage pipeline

---

## Usage Examples

### Proof Development Workflow
```bash
# 1. Generate proof sketch
Task: Use the proof-sketcher agent to sketch proof for theorem thm-kl-convergence in docs/source/1_euclidean_gas/09_kl_convergence.md

# 2. Expand to full proof
Task: Use the theorem-prover agent to expand the generated sketch

# 3. Review the proof
Task: Use the math-reviewer agent to review the proof document

# 4. Validate algebra
Task: Use the math-verifier agent to validate algebraic steps
```

### Document Processing Workflow
```bash
# 1. Extract raw data (Task tool OR Python CLI)
Task: Use the document-parser agent to extract docs/source/1_euclidean_gas/03_cloning.md
# OR
python -m fragile.mathster.pipeline extract docs/source/1_euclidean_gas/03_cloning.md

# 2. Discover relationships
Task: Use the cross-referencer agent to analyze docs/source/1_euclidean_gas/03_cloning

# 3. Enrich with semantics
Task: Use the document-refiner agent to refine docs/source/1_euclidean_gas/03_cloning/raw_data
```
