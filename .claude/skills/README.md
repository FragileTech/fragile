# Claude Code Skills for Fragile

This directory contains reusable workflow documentation organized as "skills" - complete, focused guides for common agent workflows in the Fragile mathematical proof system.

## What are Skills?

**Skills** are task-oriented workflow guides that consolidate scattered documentation into single-purpose modules. Each skill provides:
- Complete workflow documentation
- Copy-paste quick start commands
- Step-by-step procedures
- Troubleshooting guides
- Real-world examples

## Available Skills

### Core Pipeline Skills

### 1. [extract-and-refine](./extract-and-refine/)
**Purpose**: Mathematical entity extraction pipeline
**Agents**: document-parser, cross-referencer, document-refiner
**Workflow**: Markdown documents → Validated JSON entities
**Use when**: Processing new mathematical documents into structured format

### 2. [validate-refinement](./validate-refinement/)
**Purpose**: Comprehensive validation workflow for refined entities
**Tools**: validation module (schema, relationship, framework validators)
**Workflow**: Refined data → Validation report → Error identification → Quality assurance
**Use when**: After completing refinement, before building registries, debugging validation errors

### 3. [complete-partial-refinement](./complete-partial-refinement/)
**Purpose**: Systematic completion of incomplete entities
**Tools**: find_incomplete_entities, complete_refinement, Gemini 2.5 Pro
**Workflow**: Find incomplete → Generate completion plan → Execute with Gemini → Validate
**Use when**: Validation reveals incomplete entities, updating old refined data, filling missing fields

### 4. [refine-entity-type](./refine-entity-type/)
**Purpose**: Entity-specific refinement workflows by type
**Focus**: Theorems, axioms, objects, parameters, proofs, remarks, equations
**Workflow**: Raw entity → Entity-specific enrichment → Validation → Refined entity
**Use when**: Refining specific entity types, understanding entity-specific requirements, fixing type-specific errors

### 5. [registry-management](./registry-management/)
**Purpose**: Registry building and data transformation
**Tools**: build_all_registries (automated pipeline)
**Workflow**: Validated refined data → Per-document registries → Combined registry → Queryable database
**Use when**: Building registries, transforming data formats, querying entities

### Proof Development Skills

### 6. [proof-validation](./proof-validation/)
**Purpose**: Dual-review proof development
**Agents**: proof-sketcher, theorem-prover, math-reviewer, math-verifier
**Workflow**: Theorem → Proof sketch → Expanded proof → Dual validation
**Use when**: Developing proofs, reviewing mathematical rigor, verifying correctness

### Documentation and Verification Skills

### 7. [mathematical-writing](./mathematical-writing/)
**Purpose**: Write and publish mathematical documentation
**Tools**: Formatting tools, dual-review (Gemini + Codex), Jupyter Book
**Workflow**: Draft → Format → Dual-review → Build → Publish
**Use when**: Writing theorems/proofs, formatting math content, ensuring MyST/LaTeX compliance

### 8. [framework-consistency](./framework-consistency/)
**Purpose**: Verify content consistency with Fragile framework
**Tools**: Index cross-check, notation audit, axiom validation scripts
**Workflow**: Index check → Notation audit → Axiom validation → Contradiction detection
**Use when**: Checking notation compliance, validating axiom usage, identifying contradictions

### Algorithm Development Skills

### 9. [algorithm-development](./algorithm-development/)
**Purpose**: Develop, test, and visualize Gas algorithm variants
**Tools**: PyTorch, Pydantic, HoloViz stack, pytest
**Workflow**: Design → Implement → Test → Visualize → Document
**Use when**: Implementing new algorithms, adding features, testing correctness, creating visualizations

---

## Quick Navigation

### By Task

| I want to... | Use this skill |
|-------------|----------------|
| Extract entities from markdown documents | [extract-and-refine](./extract-and-refine/) |
| Validate refined entities (quality assurance) | [validate-refinement](./validate-refinement/) |
| Complete incomplete entities systematically | [complete-partial-refinement](./complete-partial-refinement/) |
| Refine specific entity types (theorems, axioms, etc.) | [refine-entity-type](./refine-entity-type/) |
| Build a mathematical entity registry | [registry-management](./registry-management/) |
| Develop and validate proofs | [proof-validation](./proof-validation/) |
| Write mathematical documentation | [mathematical-writing](./mathematical-writing/) |
| Check framework consistency | [framework-consistency](./framework-consistency/) |
| Implement a new Gas algorithm | [algorithm-development](./algorithm-development/) |
| Parse MyST directives from docs | [extract-and-refine](./extract-and-refine/) |
| Cross-reference mathematical objects | [extract-and-refine](./extract-and-refine/) |
| Find incomplete entities | [complete-partial-refinement](./complete-partial-refinement/) |
| Fix validation errors | [validate-refinement](./validate-refinement/) |
| Fill missing fields with Gemini | [complete-partial-refinement](./complete-partial-refinement/) |
| Understand entity-specific requirements | [refine-entity-type](./refine-entity-type/) |
| Generate proof sketches | [proof-validation](./proof-validation/) |
| Run dual-AI reviews (Gemini + Codex) | [proof-validation](./proof-validation/) |
| Format LaTeX/MyST math content | [mathematical-writing](./mathematical-writing/) |
| Verify notation against glossary | [framework-consistency](./framework-consistency/) |
| Validate axiom references | [framework-consistency](./framework-consistency/) |
| Test algorithm correctness | [algorithm-development](./algorithm-development/) |
| Create interactive visualizations | [algorithm-development](./algorithm-development/) |

### By Agent

| Agent | Primary Skill | Quick Start |
|-------|---------------|-------------|
| document-parser | [extract-and-refine](./extract-and-refine/) | [QUICKSTART](./extract-and-refine/QUICKSTART.md) |
| cross-referencer | [extract-and-refine](./extract-and-refine/) | [QUICKSTART](./extract-and-refine/QUICKSTART.md) |
| document-refiner | [extract-and-refine](./extract-and-refine/) | [QUICKSTART](./extract-and-refine/QUICKSTART.md) |
| proof-sketcher | [proof-validation](./proof-validation/) | [QUICKSTART](./proof-validation/QUICKSTART.md) |
| theorem-prover | [proof-validation](./proof-validation/) | [QUICKSTART](./proof-validation/QUICKSTART.md) |
| math-reviewer | [proof-validation](./proof-validation/) | [QUICKSTART](./proof-validation/QUICKSTART.md) |
| math-verifier | [proof-validation](./proof-validation/) | [QUICKSTART](./proof-validation/QUICKSTART.md) |

---

## Skill Structure

Each skill follows a consistent structure:

```
skill-name/
├── SKILL.md              # Complete documentation (main reference)
├── QUICKSTART.md         # Copy-paste commands (quick reference)
├── WORKFLOW.md           # Step-by-step procedures (detailed guide)
├── TROUBLESHOOTING.md    # Common issues & fixes (problem-solving)
└── examples/             # Real-world examples (learning)
    ├── example1.md
    └── example2.md
```

**Reading order**:
1. **New to the skill?** Start with `SKILL.md` for overview
2. **Need commands now?** Jump to `QUICKSTART.md`
3. **Want details?** Read `WORKFLOW.md` for step-by-step guide
4. **Having issues?** Check `TROUBLESHOOTING.md`
5. **Want to learn?** Study `examples/` for real usage

---

## Relationship to Agent Definitions

**Skills** are workflow-oriented documentation that use **agents** to accomplish tasks.

| Type | Location | Purpose | Example |
|------|----------|---------|---------|
| **Skill** | `.claude/skills/` | Workflow documentation | How to extract entities from docs |
| **Agent** | `.claude/agents/` | Agent protocol definition | document-parser agent definition |

**Key difference**:
- **Agents** define *what* the agent does and *how* it operates
- **Skills** define *when* to use agents and *how* to accomplish complete workflows

**Example**:
- **Agent**: `document-parser.md` defines the document-parser agent protocol
- **Skill**: `extract-and-refine/SKILL.md` explains how to use document-parser + cross-referencer + document-refiner together to extract and validate entities

---

## When to Use Skills vs Agent Definitions

### Use Skills When:
- ✅ You want to accomplish a specific task
- ✅ You need copy-paste commands
- ✅ You want step-by-step workflows
- ✅ You need troubleshooting help
- ✅ You want to see examples

### Use Agent Definitions When:
- ✅ You're modifying agent behavior
- ✅ You need detailed protocol specifications
- ✅ You're debugging agent execution
- ✅ You want to understand agent internals

---

## Contributing New Skills

To add a new skill:

1. **Identify the workflow**: What complete task does it accomplish?
2. **List the agents/tools**: What components are involved?
3. **Create the directory**: `.claude/skills/new-skill-name/`
4. **Write the 4 core files**: SKILL.md, QUICKSTART.md, WORKFLOW.md, TROUBLESHOOTING.md
5. **Add examples**: Real-world usage in `examples/`
6. **Update this README**: Add to navigation tables

---

## Related Documentation

- **Agent Definitions**: `.claude/agents/` - Individual agent protocols
- **Project Documentation**: `CLAUDE.md` - Overall project guidelines
- **Mathematical Framework**: `docs/glossary.md` - Mathematical entity reference
- **Tool Documentation**: `src/fragile/proofs/tools/` - Individual tool docs
- **Validation Module**: `src/fragile/proofs/tools/validation/` - Standardized validation infrastructure
- **Registry System**: `src/fragile/proofs/registry/` - Registry building and querying

---

**Questions?** Check the SKILL.md file for each specific skill, or refer to agent definitions in `.claude/agents/`.
