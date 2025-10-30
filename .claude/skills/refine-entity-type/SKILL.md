---
name: refine-entity-type
description: Granular entity-specific refinement workflows for each mathematical entity type. Use when refining specific entity categories, understanding entity-specific enrichment requirements, or fixing entity-type-specific validation errors. Covers all 7 types: theorems, axioms, objects, parameters, proofs, remarks, equations.
---

# Refine Entity Type Skill

## Purpose

Granular guidance for refining individual mathematical entity types with entity-specific validation and enrichment requirements. Each entity type has unique fields, validation rules, and enrichment strategies.

**Input**: Raw JSON files from document-parser (`raw_data/{entity_type}/`)
**Output**: Enriched, validated JSON files (`refined_data/{entity_type}/`)
**Scope**: Entity-type-specific refinement with targeted guidance

---

## Supported Entity Types

| Entity Type | Input Schema | Output Schema | Primary Focus |
|-------------|-------------|---------------|---------------|
| **Theorems** | RawTheorem | TheoremBox | Dependencies, output_type, properties |
| **Axioms** | RawAxiom | Axiom | Framework, assumptions, failure modes |
| **Objects** | RawDefinition | MathematicalObject | Type classification, attributes |
| **Parameters** | RawParameter | Parameter/ParameterBox | Scope, constraints, domain |
| **Proofs** | RawProof | ProofBox | Step structure, theorem linkage |
| **Remarks** | RawRemark | RemarkBox | Type classification, related entities |
| **Equations** | RawEquation | EquationBox | Symbol tracking, dual representation |

---

## Complete Workflow for Each Entity Type

### THEOREMS (TheoremBox)

**Includes**: Theorems, Lemmas, Propositions, Corollaries

#### Required Fields
- `label`: Label with prefix (`thm-`, `lem-`, `prop-`, `cor-`)
- `name`: Short descriptive name
- `statement`: Full mathematical statement

#### Enrichment Focus
1. **Identify Dependencies**:
   - `input_objects`: What definitions does this use?
   - `input_axioms`: What axioms does this require?
   - `input_parameters`: What parameters appear?

2. **Classify Output Type**:
   - property, bound, convergence, existence, uniqueness, equivalence, characterization

3. **Specify Property Requirements**:
   - For each `input_object`, what properties does the theorem need?
   - Fill `properties_required` dictionary

4. **Add Comprehensive Tags**:
   - Mathematical domain (analysis, algebra, topology, etc.)
   - Technique (variational, spectral, probabilistic, etc.)
   - Result type (convergence, stability, regularity, etc.)

#### Example Refinement

**Raw (from document-parser)**:
```json
{
  "label": "thm-keystone-principle",
  "name": "Keystone Principle",
  "statement": "Under log-concave quasi-stationary distribution...",
  "statement_type": "theorem"
}
```

**Refined (after enrichment)**:
```json
{
  "label": "thm-keystone-principle",
  "name": "Keystone Principle",
  "statement": "Under log-concave quasi-stationary distribution...",
  "statement_type": "theorem",
  "output_type": "convergence",
  "input_objects": [
    "obj-euclidean-gas",
    "obj-qsd",
    "obj-cloning-operator"
  ],
  "input_axioms": [
    "ax-log-concave-qsd",
    "ax-lipschitz-fields"
  ],
  "input_parameters": [
    "gamma",
    "epsilon_F",
    "N"
  ],
  "properties_required": {
    "obj-qsd": ["prop-log-concave", "prop-strict-positivity"],
    "obj-cloning-operator": ["prop-lipschitz"]
  },
  "tags": [
    "cloning",
    "quasi-stationary-distribution",
    "convergence",
    "log-concavity",
    "entropy-dissipation"
  ]
}
```

#### Validation Rules
- Label must start with `thm-`, `lem-`, `prop-`, or `cor-`
- Statement must be non-empty
- If `input_objects` specified, should have `properties_required`
- `output_type` should be one of 7 valid types
- Tags should be non-empty list

---

### AXIOMS (Axiom)

#### Required Fields
- `label`: Label with prefix `ax-`
- `name`: Axiom name
- `statement`: Mathematical statement

#### Enrichment Focus
1. **Classify Framework**:
   - `foundational_framework`: Which framework? (e.g., "Fragile Gas Framework", "Euclidean Gas Dynamics")

2. **Identify Core Assumption**:
   - `core_assumption`: What does this axiom fundamentally assume?

3. **List Parameters**:
   - `parameters`: All parameters that appear in axiom

4. **Describe Failure Mode**:
   - `failure_mode_analysis`: What happens if axiom doesn't hold?

5. **Add Conditions**:
   - `condition`: When does this axiom apply?

#### Example Refinement

**Raw**:
```json
{
  "label": "ax-lipschitz-fields",
  "name": "Lipschitz Fields Axiom",
  "statement": "The potential U and fitness function..."
}
```

**Refined**:
```json
{
  "label": "ax-lipschitz-fields",
  "name": "Lipschitz Fields Axiom",
  "statement": "The potential U and fitness function V_fit are Lipschitz continuous...",
  "foundational_framework": "Fragile Gas Framework",
  "core_assumption": "Fields are Lipschitz continuous with bounded gradients",
  "parameters": ["L_U", "L_V"],
  "condition": "Throughout the domain X",
  "failure_mode_analysis": "Without Lipschitz continuity, cannot guarantee finite-time controllability or bounded energy dissipation",
  "tags": ["regularity", "lipschitz", "potential", "fitness"]
}
```

#### Validation Rules
- Label must start with `ax-`
- Should have `foundational_framework`
- Should have `core_assumption`
- Should list `parameters`

---

### OBJECTS (MathematicalObject)

**Derived from definitions**

#### Required Fields
- `label`: Label with prefix `obj-`
- `name`: Object name
- `mathematical_expression`: LaTeX expression or definition

#### Enrichment Focus
1. **Classify Object Type**:
   - SPACE, OPERATOR, MEASURE, FUNCTION, SET, METRIC, DISTRIBUTION, PROCESS, ALGORITHM, CONSTANT

2. **List Current Attributes**:
   - What properties does this object currently have?
   - Fill `current_attributes` list

3. **Add Comprehensive Tags**:
   - Domain (topology, analysis, probability)
   - Structure (vector-space, metric-space, operator)
   - Properties (complete, compact, bounded)

#### Example Refinement

**Raw**:
```json
{
  "label": "obj-euclidean-gas",
  "name": "Euclidean Gas",
  "mathematical_expression": "Algorithm combining Langevin dynamics and cloning..."
}
```

**Refined**:
```json
{
  "label": "obj-euclidean-gas",
  "name": "Euclidean Gas",
  "mathematical_expression": "$$\\Psi = \\Psi_{\\text{kin}} \\circ \\Psi_{\\text{clone}}$$",
  "object_type": "ALGORITHM",
  "current_attributes": [
    "prop-markov",
    "prop-ergodic",
    "prop-reversible-kinetic-operator",
    "prop-non-reversible-full-operator"
  ],
  "tags": [
    "langevin-dynamics",
    "cloning",
    "markov-chain",
    "quasi-stationary-distribution",
    "ergodic"
  ]
}
```

#### Validation Rules
- Label must start with `obj-`
- `mathematical_expression` must be non-empty
- `object_type` should be one of 10 valid types
- `current_attributes` should be non-empty list

---

### PARAMETERS (Parameter/ParameterBox)

#### Required Fields
- `symbol`: Parameter symbol (e.g., γ, ε_F, N)

#### Enrichment Focus
1. **Specify Domain**:
   - Mathematical domain (ℝ₊, ℕ, [0,1], etc.)

2. **Add Constraints**:
   - Restrictions on parameter (γ > 0, N ≥ 2, etc.)

3. **Define Scope** (enriched only):
   - global, local, or universal

4. **Set Default Value** (if applicable):
   - Standard or recommended value

#### Example Refinement (Enriched Parameter)

**Raw**:
```json
{
  "symbol": "gamma",
  "name": "Friction Coefficient"
}
```

**Refined (ParameterBox)**:
```json
{
  "symbol": "gamma",
  "name": "Friction Coefficient",
  "domain": "ℝ₊",
  "constraints": ["gamma > 0"],
  "scope": "global",
  "default_value": 1.0,
  "dependencies": [],
  "tags": ["langevin", "friction", "kinetic"]
}
```

#### Validation Rules
- `symbol` must be non-empty
- Should specify `domain`
- Should add `constraints` if parameter has restrictions
- Enriched parameters should specify `scope`

---

### PROOFS (ProofBox)

#### Required Fields
- `proof_id`: Unique proof identifier
- `theorem`: Back-reference to theorem label

#### Enrichment Focus
1. **Link to Theorem**:
   - `theorem` field with label reference

2. **Structure Steps**:
   - Each step with `step_number`, `content`, `justification`, `dependencies`

3. **Set Status**:
   - unproven, sketched, expanded, or verified

4. **Add Strategy**:
   - High-level proof approach

#### Example Refinement

**Raw**:
```json
{
  "proof_id": "proof-keystone-principle",
  "theorem_label": "thm-keystone-principle"
}
```

**Refined**:
```json
{
  "proof_id": "proof-keystone-principle",
  "theorem": {
    "label": "thm-keystone-principle",
    "name": "Keystone Principle"
  },
  "strategy": "Entropy dissipation via LSI, then Grönwall inequality",
  "steps": [
    {
      "step_number": 1,
      "status": "EXPANDED",
      "content": "Apply Log-Sobolev inequality to entropy functional...",
      "justification": "lem-kinetic-lsi-established",
      "dependencies": ["ax-log-concave-qsd"]
    },
    {
      "step_number": 2,
      "status": "EXPANDED",
      "content": "Use Grönwall to get exponential rate...",
      "justification": "algebraic-manipulation",
      "dependencies": []
    }
  ],
  "proof_status": "expanded"
}
```

#### Validation Rules
- `proof_id` must be non-empty
- `theorem` must reference valid theorem
- `steps` should be non-empty list
- Each step should have `step_number`, `content`, `justification`

---

### REMARKS (RemarkBox)

#### Required Fields
- `label`: Label with prefix `remark-`
- `content`: Remark content

#### Enrichment Focus
1. **Classify Type**:
   - note, observation, intuition, example, warning, historical

2. **Link Related Entities**:
   - `related_entities`: Labels of theorems/definitions discussed

3. **Extract Key Insight**:
   - `key_insight`: Summarize main point

#### Example Refinement

**Raw**:
```json
{
  "label": "remark-log-concavity-importance",
  "content": "Log-concavity is crucial for..."
}
```

**Refined**:
```json
{
  "label": "remark-log-concavity-importance",
  "content": "Log-concavity is crucial for establishing LSI...",
  "remark_type": "observation",
  "related_entities": [
    "ax-log-concave-qsd",
    "thm-kinetic-lsi-established",
    "lem-entropy-transport-dissipation"
  ],
  "key_insight": "Log-concavity enables entropy dissipation analysis"
}
```

#### Validation Rules
- Label must start with `remark-`
- `content` must be non-empty
- `remark_type` should be one of valid types
- Should link to `related_entities`

---

### EQUATIONS (EquationBox)

#### Required Fields
- `label`: Label with prefix `eq-`
- `latex_content`: LaTeX equation

#### Enrichment Focus
1. **Classify Type**:
   - definition, identity, evolution, constraint, property

2. **Track Symbols**:
   - `introduces_symbols`: New symbols defined
   - `references_symbols`: Symbols used

3. **Add Context**:
   - `context_before`: Paragraph before equation
   - `equation_number`: If numbered in source

#### Example Refinement

**Raw**:
```json
{
  "label": "eq-langevin-dynamics",
  "latex_content": "dX_t = -\\nabla U(X_t) dt + \\sqrt{2\\gamma^{-1}} dW_t"
}
```

**Refined**:
```json
{
  "label": "eq-langevin-dynamics",
  "latex_content": "$$dX_t = -\\nabla U(X_t) dt + \\sqrt{2\\gamma^{-1}} dW_t$$",
  "equation_type": "evolution",
  "introduces_symbols": [],
  "references_symbols": ["X_t", "U", "gamma", "W_t"],
  "context_before": "The kinetic operator implements overdamped Langevin dynamics:",
  "equation_number": "2.1"
}
```

#### Validation Rules
- Label must start with `eq-`
- `latex_content` must be non-empty
- Should classify `equation_type`
- Should track `introduces_symbols` or `references_symbols`

---

## Gemini-Assisted Enrichment

For each entity type, use Gemini 2.5 Pro to fill enrichment fields:

### Prompt Template

```
I need to enrich the following {entity_type} entity:

**Label**: {label}
**Name**: {name}
**Content**: {statement/expression/content}

**Missing enrichment fields**:
{list of fields to fill}

Based on the entity content and your knowledge of mathematical frameworks,
please provide the missing fields in JSON format.

Consider:
- Framework consistency (Fragile Gas Framework conventions)
- Mathematical rigor (precise terminology)
- Discoverability (appropriate tags)

Output format:
```json
{
  "field_name": "value",
  ...
}
```
```

### Example: Enriching Theorem

```
Model: gemini-2.5-pro

Prompt:
I need to enrich the following theorem entity:

**Label**: thm-keystone-principle
**Name**: Keystone Principle
**Statement**: Under log-concave quasi-stationary distribution and Lipschitz fields, the Euclidean Gas converges exponentially fast in relative entropy to the QSD.

**Missing enrichment fields**:
- output_type
- input_objects
- input_axioms
- properties_required
- tags

Based on the statement, please provide:
1. Output type (property/bound/convergence/etc.)
2. Input objects (definitions used)
3. Input axioms (axioms required)
4. Properties required (what properties each object needs)
5. Tags (for discoverability)

Output in JSON format.
```

---

## Best Practices

### 1. Process by Entity Type
Don't mix entity types - complete all theorems, then all axioms, etc.

### 2. Use Validation Incrementally
After refining each entity, validate immediately:
```bash
python -m fragile.proofs.tools.validation \
  --refined-dir refined_data/ \
  --entity-types theorems \
  --mode schema
```

### 3. Leverage Gemini for Enrichment
Use Gemini to fill:
- Dependencies (input_objects, input_axioms)
- Classifications (object_type, equation_type)
- Tags (for discoverability)

### 4. Maintain Consistency
- Check existing entities for naming conventions
- Use framework glossary (docs/glossary.md) for reference
- Follow established patterns

### 5. Validate Cross-References
After enrichment, verify all referenced labels exist:
```bash
python -m fragile.proofs.tools.validation \
  --refined-dir refined_data/ \
  --mode relationships
```

---

## Integration with Other Skills

### From extract-and-refine

After Stage 1 extraction:
```
Load extract-and-refine skill.
# Stage 1: document-parser extracts raw entities

Load refine-entity-type skill.
# Stage 2: Refine each entity type with specific guidance
Refine theorems: raw_data/theorems/ → refined_data/theorems/
Refine axioms: raw_data/axioms/ → refined_data/axioms/
...
```

### To validate-refinement

After entity-specific refinement:
```
Load refine-entity-type skill.
Refine theorems: raw_data/theorems/

Load validate-refinement skill.
Validate theorems: refined_data/theorems/
```

---

## Troubleshooting

### Issue: Can't infer dependencies

**Solution**: Consult source markdown, look for references to definitions/axioms

### Issue: Object type unclear

**Solution**: Review object definition, check similar entities in registry

### Issue: Tags too generic

**Solution**: Add specific mathematical domain tags, technique tags, result type tags

---

## Success Criteria

Entity refinement successful when:
- ✅ All required fields populated
- ✅ Schema validation passes
- ✅ Dependencies correctly identified
- ✅ Classifications appropriate
- ✅ Tags comprehensive (5+ tags per entity)
- ✅ Cross-references valid

---

## Time Estimates

| Entity Type | Time per Entity | Batch (50) |
|-------------|----------------|------------|
| Theorems | 2-3 min | ~2 hours |
| Axioms | 1-2 min | ~1.5 hours |
| Objects | 1-2 min | ~1.5 hours |
| Parameters | 30 sec | ~25 min |
| Proofs | 3-5 min | ~4 hours |
| Remarks | 1 min | ~50 min |
| Equations | 1 min | ~50 min |
