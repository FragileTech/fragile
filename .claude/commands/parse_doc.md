# Parse Document Command

Invoke the Document Parser agent for autonomous extraction of mathematical content from MyST markdown.

## Instructions

You are now acting as the Document Parser agent. Follow the complete protocol defined in `.claude/agents/document-parser.md`.

**CRITICAL**: You MUST read the agent definition file first:

```
Read: .claude/agents/document-parser.md
```

Then execute the Document Parser protocol on the document specified by the user.

## Expected Input Format

The user will provide input in one of these formats:

### Format 1: Single Document
```
/parse_doc docs/source/1_euclidean_gas/03_cloning.md
Mode: both
```

### Format 2: Entire Directory
```
/parse_doc docs/source/1_euclidean_gas/
Output: docs/source/1_euclidean_gas/
```

### Format 3: With Custom Options
```
/parse_doc docs/source/2_geometric_gas/11_geometric_gas.md
Mode: sketch
No LLM: true
```

### Format 4: Extract Only (No Proofs)
```
/parse_doc docs/source/1_euclidean_gas/05_mean_field.md
Skip proofs: true
```

## Parameters

- **source** (required): Path to document or directory
- **mode** (optional): `sketch` | `expand` | `both` (default: `both`)
- **no_llm** (optional): Disable LLM processing (default: `false`)
- **output_dir** (optional): Custom output directory (default: auto-detected)

## Agent Protocol

After reading the agent definition, you MUST follow this autonomous workflow:

### Phase 1: MyST Directive Extraction
1. Parse all `{prf:...}` blocks using regex
2. Report counts by type (definition, theorem, lemma, axiom, etc.)
3. Create DocumentInventory with full indexing

### Phase 2: Mathematical Object Creation
1. Transform `{prf:definition}` → MathematicalObject
2. Infer object types (SET, FUNCTION, MEASURE, SPACE, etc.)
3. Extract tags from content
4. Validate against Pydantic schema

### Phase 3: Theorem Creation
1. Transform `{prf:theorem}`, `{prf:lemma}`, `{prf:proposition}` → TheoremBox
2. Extract axioms separately → Axiom
3. Infer theorem output types
4. Validate labels and cross-references

### Phase 4: Relationship Extraction (Hybrid)
1. **Explicit**: Extract cross-references from `{prf:ref}` directives
2. **LLM-Assisted** (if enabled): Use Gemini 2.5 Pro to infer implicit dependencies
3. Create Relationship instances
4. Validate bidirectionality and transitivity

### Phase 5: Proof Sketch Creation (if mode includes sketch)
1. Parse `{prf:proof}` directives
2. Create ProofBox structures with SKETCHED steps
3. Map proof inputs/outputs to properties
4. Validate dataflow consistency

### Phase 6: Proof Expansion (if mode includes expand)
1. Use Gemini 2.5 Pro to expand SKETCHED steps to EXPANDED
2. Fill in mathematical derivations
3. Add techniques and references
4. Validate rigor and completeness

### Phase 7: Validation
1. Check all Pydantic constraints
2. Validate label format and uniqueness
3. Check cross-reference integrity
4. Report errors and warnings

### Phase 8: Export to JSON
1. Export to `docs/source/N_chapter/document/data/`
2. Create `extraction_inventory.json` (complete catalog)
3. Create `statistics.json` (summary metrics)
4. Update MathematicalRegistry

## Output

The agent will:
1. Execute complete parsing autonomously
2. Write structured JSON files to `{document_dir}/data/`
3. Display summary with validation status

**File Locations**:
- **Extraction Inventory**: `{document_dir}/data/extraction_inventory.json`
- **Statistics**: `{document_dir}/data/statistics.json`

## Quality Guarantees

- ✅ MyST directive extraction (all `{prf:...}` blocks)
- ✅ Type inference (SET, FUNCTION, MEASURE, etc.)
- ✅ Framework symbol integration
- ✅ Cross-reference validation
- ✅ Pydantic schema compliance
- ✅ Structured JSON export

## Notes

- Agent runs autonomously (no interruptions)
- Expected runtime: 2-5 seconds (extraction) + 10-30 seconds (LLM if enabled)
- Output is validated JSON ready for downstream processing
- Can process entire directories in batch
- Integrates with proof-sketcher and theorem-prover agents

## Integration with Proof Pipeline

**Complete Pipeline**:
```
Document Parser → Proof Sketcher → Math Verifier → Theorem Prover → Math Reviewer
```

**Standalone Use**:
```
Document Parser → [Manual analysis of extracted structure]
```

---

**Now begin the Document Parser protocol for the source provided by the user.**
