---
name: extract-and-refine
description: Complete pipeline for extracting mathematical entities from MyST markdown documents and validating them into structured JSON format. Use when processing new mathematical documents, extracting theorems/definitions, or building entity databases from markdown.
---

# Extract-and-Refine Skill

## Purpose

Complete pipeline for extracting mathematical entities from MyST markdown documents and validating them into structured JSON format ready for registry building.

**Input**: MyST markdown documents (`.md` files with `{prf:definition}`, `{prf:theorem}`, etc.)
**Output**: Validated JSON entities in `refined_data/` directories
**Pipeline**: Document Parser → Cross-Referencer → Document Refiner

---

## Agents Involved

| Agent | Role | Input | Output |
|-------|------|-------|--------|
| **document-parser** | Extract raw entities | `.md` files | `raw_data/` JSON files |
| **cross-referencer** | Link dependencies | `raw_data/` | `relationships/` + updated JSONs |
| **document-refiner** | Validate & enrich | `raw_data/` + `relationships/` | `refined_data/` JSON files |

---

## Complete Workflow

### Stage 1: Raw Extraction (document-parser)

**Purpose**: Extract all MyST directives into structured JSON

```bash
python -m fragile.agents.math_document_parser \
  docs/source/1_euclidean_gas/01_fragile_gas_framework.md \
  --no-llm
```

**What it does**:
- Finds all `{prf:definition}`, `{prf:theorem}`, `{prf:lemma}`, etc.
- Creates `MathematicalObject` instances from definitions
- Creates `TheoremBox` instances from theorems/lemmas
- Creates `Axiom` instances from axiom directives
- Exports to `raw_data/` directory

**Output structure**:
```
docs/source/1_euclidean_gas/01_fragile_gas_framework/
├── raw_data/
│   ├── objects/
│   │   └── obj-*.json
│   ├── axioms/
│   │   └── axiom-*.json
│   └── theorems/
│       └── thm-*.json
└── data/
    ├── extraction_inventory.json
    └── statistics.json
```

**Time**: ~15-30 seconds (no LLM), ~3-5 minutes (with LLM)

---

### Stage 1.5: Cross-Referencing (cross-referencer)

**Purpose**: Discover and link relationships between entities

```bash
python -m fragile.agents.cross_reference_analyzer \
  docs/source/1_euclidean_gas/01_fragile_gas_framework
```

**What it does**:
- Processes explicit cross-references from `{prf:ref}` tags
- Discovers implicit dependencies via LLM analysis (optional)
- Traces mathematical symbols to definitions
- Fills `input_objects`, `input_axioms`, `input_parameters` fields
- Creates typed `Relationship` objects
- Validates all relationships against registry

**Output structure**:
```
docs/source/1_euclidean_gas/01_fragile_gas_framework/
├── theorems/               # (Modified - filled dependencies)
│   └── thm-*.json
└── relationships/          # (NEW)
    ├── rel-*.json
    ├── index.json
    └── REPORT.md
```

**Time**: ~2-3 seconds (no LLM), ~5-10 minutes (with LLM)

---

### Stage 2: Refinement & Validation (document-refiner)

**Purpose**: Enrich entities and validate against strict schema

```bash
# Run via agent (recommended)
Load document-refiner agent.
Refine: docs/source/1_euclidean_gas/01_fragile_gas_framework
```

**What it does**:
- Validates all JSON files against Pydantic schemas
- Enriches with semantic information
- Adds mathematical context
- Fills missing fields via LLM
- Ensures schema compliance for pipeline

**Output structure**:
```
docs/source/1_euclidean_gas/01_fragile_gas_framework/
├── refined_data/           # (NEW - validated entities)
│   ├── objects/
│   │   └── obj-*.json
│   ├── axioms/
│   │   └── axiom-*.json
│   ├── theorems/
│   │   └── thm-*.json
│   └── parameters/
│       └── param-*.json
└── reports/
    └── statistics/
        └── refined_statistics.json
```

**Time**: ~10-30 minutes (depends on entity count and LLM usage)

---

### Stage 3: Validation & Quality Assurance

**Purpose**: Comprehensive validation of refined entities

```bash
# Quick schema validation
python -m fragile.proofs.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --mode schema

# Complete validation (recommended)
python -m fragile.proofs.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --mode complete \
  --output-report validation_report.md
```

**What it validates**:
- ✅ Schema compliance (Pydantic validation)
- ✅ Required fields present and non-empty
- ✅ Cross-references valid (all labels exist)
- ✅ Dependency consistency (input_objects, input_axioms, etc.)
- ✅ Framework consistency (notation, axiom usage)
- ✅ Entity-specific rules (7 validators)

**Output**: Validation report with errors, warnings, and recommendations

**When validation fails**: Use [complete-partial-refinement](../complete-partial-refinement/) skill to fix incomplete entities systematically.

**Time**: ~10-20 seconds (schema), ~2-3 minutes (complete)

See [validate-refinement](../validate-refinement/) skill for detailed validation workflow.

---

## Tools & Scripts

| Tool | Purpose | Location |
|------|---------|----------|
| `math_document_parser` | Extract raw entities | `fragile.agents.math_document_parser` |
| `cross_reference_analyzer` | Link dependencies | `fragile.agents.cross_reference_analyzer` |
| `dependency_detector` | Detect implicit deps | `fragile.agents.dependency_detector` |
| `relationship_builder` | Build relationships | `fragile.agents.relationship_builder` |
| Document-refiner agent | Validate & enrich | `.claude/agents/document-refiner.md` |
| **Validation module** | **Comprehensive QA** | **`fragile.proofs.tools.validation`** |
| **find_incomplete_entities** | **Discover incomplete** | **`fragile.proofs.tools.find_incomplete_entities`** |
| **complete_refinement** | **Generate completion plans** | **`fragile.proofs.tools.complete_refinement`** |

### Validation & Completion Tools

```bash
# Validate refined entities
python -m fragile.proofs.tools.validation \
  --refined-dir PATH \
  --mode complete

# Find incomplete entities
python -m fragile.proofs.tools.find_incomplete_entities \
  --refined-dir PATH

# Generate completion plan
python -m fragile.proofs.tools.complete_refinement \
  --incomplete-file incomplete_entities.json \
  --refined-dir PATH
```

See [validate-refinement](../validate-refinement/), [complete-partial-refinement](../complete-partial-refinement/), and [refine-entity-type](../refine-entity-type/) skills for detailed workflows.

---

## Input Requirements

### Markdown Format

Documents must use MyST markdown with Jupyter Book directives:

```markdown
:::{prf:definition} Object Label
:label: obj-my-object

Content here...
:::

:::{prf:theorem} Theorem Name
:label: thm-my-theorem

Statement here...
:::
```

**Required**:
- Valid MyST directive syntax (`:::` markers)
- Unique labels (lowercase kebab-case)
- Proper directive types (definition, theorem, lemma, axiom, etc.)

### Prerequisites

- Jupyter Book-formatted markdown documents
- Python environment with fragile package installed
- Gemini 2.5 Pro API access (for LLM features, optional)

---

## Output Format

### Raw Data (`raw_data/`)

Minimal preprocessing, extracted directly from directives:

```json
{
  "label": "obj-euclidean-gas",
  "name": "Euclidean Gas",
  "object_type": "SPACE",
  "description": "...",
  "mathematical_expression": "..."
}
```

### Refined Data (`refined_data/`)

Fully validated and enriched:

```json
{
  "label": "obj-euclidean-gas",
  "name": "Euclidean Gas",
  "object_type": "SPACE",
  "description": "...",
  "mathematical_expression": "...",
  "source": {
    "document_id": "01_fragile_gas_framework",
    "file_path": "docs/source/...",
    "chapter": "1_euclidean_gas",
    "section": "§2.1",
    "line_start": 450,
    "line_end": 465
  },
  "tags": ["langevin", "cloning"],
  "properties_added": [...]
}
```

---

## Best Practices

### 1. Always Start with --no-llm

Fast validation before expensive LLM calls:

```bash
python -m fragile.agents.math_document_parser \
  docs/source/.../document.md \
  --no-llm
```

Check `statistics.json` for `"validation_errors": 0` before proceeding.

### 2. Process in Order

**Correct order**: parse → cross-reference → refine

```bash
# Step 1
python -m fragile.agents.math_document_parser docs/source/.../doc.md --no-llm

# Step 2
python -m fragile.agents.cross_reference_analyzer docs/source/.../section_dir

# Step 3
# Load document-refiner agent and refine
```

### 3. Validate Early and Often

Use standardized validation tools at each stage:

```bash
# After parsing - Check raw extraction
cat docs/source/.../data/statistics.json | grep validation_errors

# After cross-referencing - Verify relationships
cat docs/source/.../relationships/index.json

# After refinement - Comprehensive validation (RECOMMENDED)
python -m fragile.proofs.tools.validation \
  --refined-dir docs/source/.../refined_data/ \
  --mode complete \
  --output-report validation_report.md
```

**Validation Gates**:
- ✅ After Stage 1: Check extraction statistics
- ✅ After Stage 1.5: Verify relationship counts
- ✅ **After Stage 2: Run full validation** (catches 100% of issues)

If validation fails, use [complete-partial-refinement](../complete-partial-refinement/) skill to fix systematically.

### 4. Use Fast Mode for Iteration

During document development, use `--no-llm` for rapid iteration:

```bash
# Edit document
vim docs/source/.../document.md

# Quick re-parse
python -m fragile.agents.math_document_parser docs/source/.../document.md --no-llm

# Check results
cat docs/source/.../data/statistics.json
```

Only enable LLM features when ready for full processing.

### 5. Centralize Reports

All agent reports now go to `reports/` subdirectories:

```
section_dir/
├── reports/
│   ├── statistics/       # Extraction/refinement reports
│   └── relationships/    # Cross-reference reports
├── raw_data/             # Raw extraction
└── refined_data/         # Validated entities
```

---

## Common Pitfalls

### Pitfall 1: Incorrect Directive Syntax

**Problem**: Using `::::` (4 colons) instead of `:::` (3 colons)

**Solution**: Use exactly 3 colons:
```markdown
:::{prf:definition} My Definition
:label: obj-my-definition

Content
:::
```

### Pitfall 2: Invalid Labels

**Problem**: Labels with uppercase or underscores: `Obj-MyObject`, `obj_my_object`

**Solution**: Use lowercase kebab-case: `obj-my-object`

### Pitfall 3: Missing Prerequisites

**Problem**: Running cross-referencer before document-parser

**Solution**: Always parse first, then cross-reference, then refine

### Pitfall 4: Scattered Old Data

**Problem**: Old `data/` directories interfering with new extraction

**Solution**: Parser/cross-referencer only look in specific locations:
- Parser creates: `section_dir/raw_data/`
- Cross-referencer reads: `section_dir/raw_data/`, creates `section_dir/relationships/`
- Refiner creates: `section_dir/refined_data/`

### Pitfall 5: Schema Validation Errors

**Problem**: Entities don't match Pydantic schemas

**Solution**: Check error messages, fix at source (markdown) or during refinement

---

## Integration with Other Workflows

### Complete Pipeline Flow

```bash
# Stage 1: Parse
python -m fragile.agents.math_document_parser docs/source/.../document.md --no-llm

# Stage 1.5: Cross-reference
python -m fragile.agents.cross_reference_analyzer docs/source/.../section_dir

# Stage 2: Refine
# Load document-refiner agent and refine

# Stage 3: Validate
python -m fragile.proofs.tools.validation \
  --refined-dir docs/source/.../refined_data/ \
  --mode complete \
  --output-report validation_report.md

# Stage 3a: Fix incomplete entities if needed
python -m fragile.proofs.tools.find_incomplete_entities --refined-dir docs/source/.../refined_data/
python -m fragile.proofs.tools.complete_refinement --incomplete-file incomplete_entities.json --refined-dir docs/source/.../refined_data/
# Execute completion with Claude Code + Gemini
# Re-validate after completion

# Stage 4: Build registries
python -m fragile.proofs.tools.build_all_registries --docs-root docs/source
```

### To Validation Workflows

After refinement, always validate:

```bash
# Quick validation
Load validate-refinement skill.
Validate: docs/source/.../refined_data/
```

**If validation fails**:

```bash
# Systematic completion
Load complete-partial-refinement skill.
Complete: docs/source/.../refined_data/
```

**For entity-specific refinement**:

```bash
# Entity-type-specific guidance
Load refine-entity-type skill.
Refine theorems: raw_data/theorems/
```

### To Registry Management

After **validation passes**, build registries:

```bash
# Build all registries automatically
python -m fragile.proofs.tools.build_all_registries --docs-root docs/source
```

**IMPORTANT**: Only proceed to registry building after validation confirms 0 errors. This prevents downstream issues.

See [registry-management](../registry-management/) skill for details.

### To Proof Validation

After extraction and validation, theorems are available for proof development:

```bash
# List validated theorems
ls docs/source/.../refined_data/theorems/

# Use proof-sketcher on specific theorem
# (Load proof-sketcher agent)
```

See [proof-validation](../proof-validation/) skill for details.

---

## Performance Tips

| Scenario | Command | Time |
|----------|---------|------|
| Quick validation | `--no-llm` | ~15 sec |
| Full extraction | default (with LLM) | ~3-5 min |
| Cross-reference (fast) | `--no-llm` | ~2-3 sec |
| Cross-reference (full) | default (with LLM) | ~5-10 min |
| Refinement | via agent | ~10-30 min |

**Recommendation**: Use `--no-llm` during document development, enable LLM for final processing.

---

## Related Documentation

- **Agent Definitions**:
  - `.claude/agents/document-parser.md`
  - `.claude/agents/cross-referencer.md`
  - `.claude/agents/document-refiner.md`
- **Quick Start**: [QUICKSTART.md](./QUICKSTART.md)
- **Detailed Workflow**: [WORKFLOW.md](./WORKFLOW.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- **Examples**: [examples/](./examples/)
- **CLAUDE.md**: Mathematical document style and rigor requirements
- **Validation & Completion Skills**:
  - [validate-refinement](../validate-refinement/) - Comprehensive QA workflow
  - [complete-partial-refinement](../complete-partial-refinement/) - Systematic completion
  - [refine-entity-type](../refine-entity-type/) - Entity-specific guidance

---

**Next**: See [QUICKSTART.md](./QUICKSTART.md) for copy-paste commands.
