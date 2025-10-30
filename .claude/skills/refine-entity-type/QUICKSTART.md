# Refine Entity Type - Quick Start

Entity-specific refinement workflows by type.

---

## Quick Reference: Entity-Specific Commands

### Validate Specific Entity Type

```bash
# Theorems only
python -m fragile.proofs.tools.validation \
  --refined-dir refined_data/ \
  --entity-types theorems \
  --mode schema

# Axioms only
python -m fragile.proofs.tools.validation \
  --refined-dir refined_data/ \
  --entity-types axioms \
  --mode schema

# Multiple types
python -m fragile.proofs.tools.validation \
  --refined-dir refined_data/ \
  --entity-types theorems axioms objects \
  --mode complete
```

---

## Entity-Specific Enrichment Checklist

### THEOREMS (thm-, lem-, prop-, cor-)
✅ **Required**: label, name, statement
✅ **Enrich**: output_type, input_objects, input_axioms, properties_required, tags

```bash
# Gemini prompt for theorem
echo "Enrich theorem: {label}
Statement: {statement}
Fill: output_type, input_objects, input_axioms, properties_required, tags"
```

### AXIOMS (ax-)
✅ **Required**: label, name, statement
✅ **Enrich**: foundational_framework, core_assumption, parameters, failure_mode_analysis

### OBJECTS (obj-)
✅ **Required**: label, name, mathematical_expression
✅ **Enrich**: object_type, current_attributes, tags

### PARAMETERS (param-)
✅ **Required**: symbol
✅ **Enrich**: domain, constraints, scope, default_value

### PROOFS (proof-)
✅ **Required**: proof_id, theorem
✅ **Enrich**: steps, strategy, proof_status

### REMARKS (remark-)
✅ **Required**: label, content
✅ **Enrich**: remark_type, related_entities, key_insight

### EQUATIONS (eq-)
✅ **Required**: label, latex_content
✅ **Enrich**: equation_type, introduces_symbols, references_symbols

---

## Entity Type Classification Quick Reference

### Output Types (Theorems)
- property, bound, convergence, existence, uniqueness, equivalence, characterization

### Object Types (Objects)
- SPACE, OPERATOR, MEASURE, FUNCTION, SET, METRIC, DISTRIBUTION, PROCESS, ALGORITHM, CONSTANT

### Equation Types (Equations)
- definition, identity, evolution, constraint, property

### Remark Types (Remarks)
- note, observation, intuition, example, warning, historical

---

## Workflow per Entity Type

### Process Theorems

```bash
# 1. Validate raw theorems
python -m fragile.proofs.tools.validation \
  --refined-dir raw_data/ \
  --entity-types theorems \
  --mode schema

# 2. For each theorem, use Gemini to fill:
#    - output_type
#    - input_objects
#    - input_axioms
#    - properties_required
#    - tags

# 3. Save to refined_data/theorems/

# 4. Validate refined
python -m fragile.proofs.tools.validation \
  --refined-dir refined_data/ \
  --entity-types theorems \
  --mode complete
```

### Process Axioms

```bash
# Similar workflow, focus on:
#    - foundational_framework
#    - core_assumption
#    - parameters
#    - failure_mode_analysis
```

### Process Objects

```bash
# Focus on:
#    - object_type (SPACE, OPERATOR, etc.)
#    - current_attributes
#    - tags
```

---

## Time Estimates

| Type | Per Entity | 10 Entities | 50 Entities |
|------|-----------|-------------|-------------|
| Theorems | 2-3 min | 25 min | 2 hours |
| Axioms | 1-2 min | 15 min | 1.5 hours |
| Objects | 1-2 min | 15 min | 1.5 hours |
| Parameters | 30 sec | 5 min | 25 min |
| Proofs | 3-5 min | 40 min | 4 hours |
| Remarks | 1 min | 10 min | 50 min |
| Equations | 1 min | 10 min | 50 min |

---

## Common Patterns

### Finding Dependencies (Theorems)

Look for in statement:
- "Under axiom X" → input_axioms
- "For operator Y" → input_objects
- "With parameter γ" → input_parameters

### Classifying Objects

- Describes space → SPACE
- Transforms elements → OPERATOR
- Assigns weights → MEASURE
- Maps to values → FUNCTION
- Collection of elements → SET

### Structuring Proofs

1. Break into logical steps
2. Number sequentially
3. Add justification for each step
4. Mark status (SKETCHED/EXPANDED/VERIFIED)

---

## Pro Tips

1. **Process in batches**: 10 entities at a time
2. **Validate incrementally**: After each batch
3. **Use Gemini for enrichment**: Especially dependencies and tags
4. **Check existing entities**: For naming conventions
5. **Consult glossary**: docs/glossary.md for reference
