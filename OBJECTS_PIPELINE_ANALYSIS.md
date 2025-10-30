# Mathematical Objects in the Three-Stage Pipeline: Analysis & Recommendations

**Date**: 2025-10-30
**Scope**: Investigate whether "objects" need to be a distinct entity type in raw_data/ and refined_data/ stages
**Key Finding**: YES, objects ARE semantically distinct from definitions at all pipeline stages

---

## Executive Summary

After analyzing the three-stage data pipeline (raw_data → refined_data → pipeline_data), the investigation reveals that **mathematical objects are fundamentally different from definitions** and **MUST remain separate entity types** throughout all pipeline stages.

### Key Findings

1. **Definitions vs Objects**: Clear conceptual distinction
   - **Definitions** (def-*): Formal statements defining *concepts* ("what does 'Lipschitz continuous' mean?")
   - **Objects** (obj-*): Concrete *instances* of those concepts ("the Euclidean Gas algorithm")

2. **No RawObject Type**: `staging_types.py` intentionally has **no RawObject schema**
   - Objects are extracted as RawDefinition during Stage 1
   - They are **semantically distinct but structurally identical** at extraction time
   - Separation happens during Stage 2 refinement based on semantic analysis

3. **Current Pipeline Works Correctly**:
   - Stage 1 (raw_data/): Definitions only → includes both concept definitions and object instantiations
   - Stage 2 (refined_data/): Definitions + Objects → semantic separation occurs here
   - Stage 3 (pipeline_data/): Objects only → definitions have served their purpose

4. **The "objects/" Directory Anomaly**: raw_data/objects/ exists but shouldn't
   - Contains 39 already-refined MathematicalObject instances
   - These were manually created or resulted from workflow confusion
   - Should be in refined_data/objects/ or pipeline_data/objects/ instead

---

## Stage-by-Stage Analysis

### Stage 1: raw_data/ (Direct LLM Extraction)

**Schema**: `staging_types.py` defines:
- ✅ RawDefinition
- ✅ RawTheorem
- ✅ RawProof
- ✅ RawAxiom
- ✅ RawParameter
- ✅ RawEquation
- ✅ RawRemark
- ✅ RawCitation
- ❌ RawObject (intentionally absent)

**Why No RawObject?**

The document-parser agent spec (.claude/agents/document-parser.md) makes this clear:

> "You are Document Parser, a Stage 1 extraction agent specialized in performing **verbatim transcription** of mathematical content from MyST markdown documents"
>
> "Your Mission: Extract ALL mathematical entities with **ZERO interpretation**. Goal: **Completeness over semantic understanding**."

At extraction time, both concept definitions and object instantiations appear as `{prf:definition}` directives in the markdown. The LLM cannot reliably distinguish between:
- A definition: "We say a function is *Lipschitz continuous* if..."
- An object instantiation: "The *Euclidean Gas* is a tuple (X, v, U) where..."

Both are transcribed as `RawDefinition` with identical schemas:
```python
class RawDefinition(RawDataModel):
    label: str                          # e.g., "def-lipschitz" or "def-euclidean-gas"
    term_being_defined: str             # "Lipschitz continuous" or "Euclidean Gas"
    full_text: str                      # Complete verbatim text
    parameters_mentioned: List[str]     # Symbols used
    source_section: str                 # Location
```

**Current State of raw_data/objects/**:

Directory exists with 39 files, but analysis reveals these are **already-refined MathematicalObject instances**:

```bash
$ ls raw_data/objects/ | wc -l
39

$ head raw_data/objects/obj-cloning-measure.json
{
  "label": "obj-cloning-measure",              # ← obj- prefix (refined)
  "name": "Cloning Measure",
  "mathematical_expression": "$\\mathcal{Q}_\\delta(x, \\cdot)$",
  "object_type": "measure",                    # ← ObjectType enum (refined)
  "current_attributes": [],                    # ← MathematicalObject field
  "attribute_history": [],                     # ← MathematicalObject field
  "tags": ["noise", "cloning", ...],
  "chapter": "1_euclidean_gas",
  "document": "01_fragile_gas_framework",
  "definition_label": "def-cloning-measure",   # ← Links back to definition
  "source": { ... }
}
```

These match the `MathematicalObject` schema from `math_types.py`, not a raw extraction schema.

**Problem**: These files are in the wrong directory. They should be in either:
- `refined_data/objects/` (if still undergoing enrichment)
- `pipeline_data/objects/` (if ready for execution)

**Comparison with raw_data/definitions/**:

```bash
$ head raw_data/definitions/raw-def-001.json
{
  "temp_id": "raw-def-001",                    # ← temp_id (staging)
  "term_being_defined": "Algorithmic Space",   # ← RawDefinition field
  "full_text": "An **algorithmic space** is...",
  "parameters_mentioned": ["\\mathcal{Y}", ...],
  "source_section": "§6",
  "label": "def-algorithmic-space-generic",    # ← def- prefix
  "context_before": "### 5.1 Specification...",
  "source": { ... }
}
```

This correctly matches `RawDefinition` schema.

---

### Stage 2: refined_data/ (Semantic Enrichment)

**Schema**: `enriched_types.py` defines:
- ✅ EnrichedDefinition (def-* labels)
- ✅ EnrichedObject (obj-* labels)
- ✅ EnrichedTheorem
- ✅ EnrichedAxiom
- (Plus EquationBox, ParameterBox, RemarkBox)

**Semantic Separation Occurs Here**

The document-refiner agent (Stage 2) performs semantic analysis to distinguish:

1. **Concept Definitions** → EnrichedDefinition
   - Label pattern: `def-*`
   - Defines abstract concepts, properties, conditions
   - Examples: `def-lipschitz-continuous`, `def-v-porous`, `def-confining-potential`

2. **Object Instantiations** → EnrichedObject
   - Label pattern: `obj-*` (changed from `def-*`)
   - Concrete instances, specific systems, particular constructions
   - Examples: `obj-euclidean-gas`, `obj-cloning-measure`, `obj-baoab-integrator`

**Schema Comparison**:

```python
# EnrichedDefinition (refined_data format)
class EnrichedDefinition(BaseModel):
    label: str                      # Pattern: ^(def-|obj-)[a-z0-9-]+$
    name: str
    entity_type: str = "definition"
    natural_language_statement: Optional[str]
    description: Optional[str]
    formal_statement: Optional[str]
    source: SourceLocation          # MANDATORY
    # ... enrichment metadata

# EnrichedObject (refined_data format - transitional)
class EnrichedObject(BaseModel):
    label: str                      # Pattern: ^obj-[a-z0-9-]+$
    name: str
    entity_type: str = "object"
    object_type: Optional[str]      # ObjectType enum value
    mathematical_expression: Optional[str]
    description: Optional[str]
    current_attributes: List[Dict]  # Attributes from theorems
    attribute_history: List[Dict]
    source: SourceLocation          # MANDATORY
    # ... enrichment metadata
```

**Evidence from Actual Data**:

```bash
$ ls refined_data/definitions/ | wc -l
30

$ ls refined_data/objects/ | wc -l
53

$ head refined_data/definitions/def-algorithmic-space-generic.json
{
  "label": "def-algorithmic-space-generic",
  "term_being_defined": "Algorithmic Space",
  "statement_type": "definition",
  "name": "Algorithmic Space",
  "chapter": "1_euclidean_gas",
  "document": "01_fragile_gas_framework"
}

$ head refined_data/objects/obj-cloning-measure.json
{
  "label": "obj-cloning-measure",
  "name": "Cloning Measure",
  "mathematical_expression": "\\mathcal{Q}_\\delta(x, \\cdot)",
  "object_type": "measure",
  "current_attributes": [],
  "attribute_history": [],
  "tags": ["cloning", "measure", ...],
  "definition_label": "def-cloning-measure",  # ← Links back!
  "source": { ... }
}
```

**Key Insight**: The `definition_label` field creates a bidirectional link:
- Objects know which definition they instantiate
- Definitions can be queried for all instances

---

### Stage 3: pipeline_data/ (Execution-Ready)

**Schema**: `math_types.py` defines final execution types:
- ✅ MathematicalObject (obj-* labels)
- ✅ TheoremBox
- ✅ Axiom
- ✅ Parameter
- ✅ Attribute
- ✅ Relationship
- ❌ DefinitionBox (exists but rarely used in pipeline execution)

**Why Only Objects?**

The pipeline execution model treats theorems as processing boxes:

```
Input: MathematicalObject instances
       ↓
   TheoremBox (processing)
       ↓
Output: Updated MathematicalObject instances (with new attributes)
```

Definitions have served their role by creating the initial objects. The pipeline doesn't need them for execution.

**MathematicalObject Schema**:

```python
class MathematicalObject(BaseModel):
    """
    Mathematical object created by Definition directives.
    
    Objects accumulate attributes as they flow through theorems.
    Only definitions create objects; theorems only add attributes.
    """
    model_config = ConfigDict(frozen=True)
    
    label: str                              # Pattern: ^obj-[a-z0-9-]+$
    name: str
    mathematical_expression: str            # LaTeX expression
    object_type: ObjectType                 # Enum: set, function, measure, etc.
    current_attributes: List[Attribute]     # Accumulates via theorems
    attribute_history: List[AttributeEvent] # Timeline of attribute additions
    tags: List[str]
    source: SourceLocation                  # MANDATORY
    chapter: Optional[str]
    document: Optional[str]
    definition_label: Optional[str]         # Link to formal definition
    
    def has_attribute(self, attr_label: str) -> bool: ...
    def add_attribute(self, attr: Attribute, timestamp: int) -> "MathematicalObject": ...
```

**DefinitionBox Schema** (exists but less used):

```python
class DefinitionBox(BaseModel):
    """
    Represents a formal mathematical definition from a source document.
    
    A DefinitionBox captures the formal definition of a concept,
    distinguishing it from a MathematicalObject which is an instance.
    """
    model_config = ConfigDict(frozen=True)
    
    label: str                              # Pattern: ^def-[a-z0-9-]+$
    term: str                               # Term being defined
    formal_statement: Optional[DualStatement]
    applies_to_object_type: Optional[ObjectType]
    parameters: List[str]
    source: Optional[SourceLocation]
    natural_language_description: Optional[str]
    tags: List[str]
    chapter: Optional[str]
    document: Optional[str]
```

---

## Conceptual Distinction: Definitions vs Objects

### The Fundamental Difference

**Definition** = Universal Template
- Answers: "What does this concept mean?"
- Defines conditions, properties, criteria
- Universal scope (applies to all instances)
- Examples:
  - "A set E is *v-porous on lines* if for every line..."
  - "A function f is *Lipschitz continuous* if there exists L such that..."
  - "A potential U is *confining* if lim_{|x|→∞} U(x) = +∞"

**Object** = Concrete Instance
- Answers: "What is this specific thing?"
- Instantiates a template with specific choices
- Particular scope (one specific instance)
- Examples:
  - "The *Euclidean Gas* is the tuple (ℝ^d, d_Eucl, π_Eucl, Ψ_kin, Ψ_clone)"
  - "The *BAOAB integrator* is the numerical scheme..."
  - "The *cloning measure* is Q_δ(x, ·) = Normal(x, δ²I)"

### Real-World Analogy

```
Definition: "A car is a vehicle with 4 wheels, engine, steering..."
           ↓ (instantiates)
Objects:   • "My Honda Civic (VIN: 1234567)"
           • "The Tesla Model 3 in the garage"
           • "The red Ford Mustang we test-drove"
```

### In the Fragile Framework

```
Definition: def-confining-potential
  "A potential U: X → ℝ is confining if lim_{|x|→∞} U(x) = +∞"
           ↓ (instantiates)
Objects:   • obj-euclidean-gas-potential (specific U in Euclidean Gas)
           • obj-quartic-potential (U(x) = |x|^4)
           • obj-double-well-potential (U(x) = (|x|² - 1)²)
```

All three objects satisfy the *confining potential* definition, but each is a distinct mathematical entity with specific properties.

---

## Why Objects Can't Be Merged with Definitions

### 1. Different Lifecycle

**Definitions**: Static throughout pipeline
- Created once during extraction
- Never modified
- Serve as reference documentation

**Objects**: Dynamic state machines
- Created from definitions
- Accumulate attributes from theorems
- Change state through pipeline execution
- Track attribute history with timestamps

Example workflow:
```
Stage 2:
  obj-euclidean-gas created from def-euclidean-gas
  current_attributes: []
  
After thm-lipschitz-fields:
  current_attributes: [attr-lipschitz-displacement]
  
After thm-ergodic:
  current_attributes: [attr-lipschitz-displacement, attr-ergodic]
  
After thm-exponential-convergence:
  current_attributes: [..., attr-exponential-mixing]
```

### 2. Different Multiplicity

**Definitions**: One per concept
- `def-lipschitz-continuous` (the concept)
- `def-confining-potential` (the concept)

**Objects**: Many instances per definition
- `obj-euclidean-distance` (Lipschitz with L=1)
- `obj-hyperbolic-distance` (Lipschitz with different L)
- `obj-discrete-metric` (Lipschitz with L=2)

All three objects are *instances* of the Lipschitz continuous definition.

### 3. Different Theorem Interactions

**Definitions**:
- Referenced by theorems (input context)
- Never modified by theorems
- Used to check if object satisfies a concept

**Objects**:
- Processed by theorems (input/output)
- Modified by theorems (attributes added)
- Track which theorems have touched them

### 4. Different Schema Requirements

**Definitions Need**:
- Formal statement (if-and-only-if condition)
- Applicability scope (which types of objects)
- Parameters (free variables in definition)
- No mutable state

**Objects Need**:
- Current state (attributes, properties)
- Mutation history (timeline)
- Type classification (set, function, measure, etc.)
- Execution metadata (which theorems applied)

---

## Current Pipeline Status & Issues

### What Works

✅ **Stage 1 → Stage 2 Transformation**:
- RawDefinition correctly captures both concepts and instances
- Semantic analysis distinguishes them during refinement
- EnrichedDefinition and EnrichedObject schemas handle both

✅ **Stage 2 → Stage 3 Transformation**:
- EnrichedObject → MathematicalObject transformation is well-defined
- Pipeline execution only needs MathematicalObject instances
- Definitions archived for documentation

✅ **Bidirectional Links**:
- Objects reference their definitions via `definition_label`
- Enables "show me all instances of this concept" queries

### What's Broken

❌ **raw_data/objects/ Directory Exists**:
- Contains 39 MathematicalObject instances
- Should be in refined_data/objects/ or pipeline_data/objects/
- Violates Stage 1 principle ("zero interpretation")
- Creates confusion about pipeline boundaries

**Evidence**:
```bash
$ ls raw_data/objects/*.json | wc -l
39

$ head raw_data/objects/obj-cloning-measure.json
{
  "object_type": "measure",           # ← Requires semantic classification
  "current_attributes": [],           # ← MathematicalObject field
  "attribute_history": [],            # ← MathematicalObject field
  "definition_label": "def-..."       # ← Requires relationship analysis
}
```

These fields require semantic analysis that shouldn't happen in Stage 1.

❌ **Inconsistent Workflow Documentation**:
- OBJECT_REFINEMENT_SUMMARY.md describes fixing raw_data/objects/
- But raw_data/ should only contain raw extractions
- Suggests workflow confusion or manual intervention

---

## Recommendations

### Recommendation 1: Remove raw_data/objects/ Directory

**Action**:
```bash
# Move misplaced objects to correct location
mv docs/source/*/raw_data/objects/* docs/source/*/refined_data/objects/
rmdir docs/source/*/raw_data/objects/

# Update documentation
rm docs/source/*/raw_data/OBJECT_REFINEMENT_SUMMARY.md
```

**Rationale**:
- Stage 1 (raw_data/) should contain ONLY raw extractions
- Objects with obj- labels are already refined
- current_attributes, attribute_history, definition_label all require enrichment

### Recommendation 2: Clarify Pipeline Boundaries

**Update .claude/agents/document-parser.md**:

Add explicit note:
```markdown
### IMPORTANT: Objects in raw_data/

**DO NOT** create an `objects/` subdirectory in raw_data/.

Both concept definitions and object instantiations are extracted as
`RawDefinition` during Stage 1. Semantic separation happens in Stage 2.

If you encounter `{prf:object}` directives or object instantiations,
extract them as `RawDefinition` with appropriate labels.

Semantic distinction (def- vs obj-) occurs during enrichment, not extraction.
```

### Recommendation 3: Enforce Schema Validation

**Add validation check** to document-parser:

```python
def validate_raw_output(output_dir: Path):
    """Ensure raw_data/ contains only staging types."""
    objects_dir = output_dir / "objects"
    if objects_dir.exists():
        raise ValueError(
            "raw_data/objects/ should not exist. "
            "Objects are extracted as RawDefinition and separated in Stage 2."
        )
```

### Recommendation 4: Update Pipeline Documentation

**Add to docs/source/.../README.md**:

```markdown
## Pipeline Stage Boundaries

### Stage 1: raw_data/ (Extraction)
- Schema: staging_types.py (RawDefinition, RawTheorem, etc.)
- Goal: Verbatim transcription, zero interpretation
- Directories:
  - ✅ definitions/ (includes both concepts and instances)
  - ✅ theorems/
  - ✅ axioms/
  - ✅ proofs/
  - ❌ objects/ (should not exist)

### Stage 2: refined_data/ (Enrichment)
- Schema: enriched_types.py (EnrichedDefinition, EnrichedObject, etc.)
- Goal: Semantic analysis, relationship inference
- Directories:
  - ✅ definitions/ (concept definitions only)
  - ✅ objects/ (concrete instances)
  - ✅ theorems/
  - ✅ axioms/

### Stage 3: pipeline_data/ (Execution)
- Schema: math_types.py (MathematicalObject, TheoremBox, etc.)
- Goal: Theorem execution, attribute accumulation
- Directories:
  - ✅ objects/ (execution-ready instances)
  - ✅ theorems/
  - ✅ axioms/
```

### Recommendation 5: Keep Objects Separate

**DO NOT** merge objects with definitions at any stage.

**Rationale Summary**:
1. Conceptually distinct (universal template vs concrete instance)
2. Different lifecycle (static vs dynamic)
3. Different multiplicity (one-to-many relationship)
4. Different theorem interactions (context vs state machine)
5. Different schema requirements (formal logic vs execution state)

---

## Conclusion

### Final Answer to Investigation Question

**Q**: Are objects semantically distinct from definitions at extraction time?

**A**: NO at extraction time, YES after semantic analysis.

- At Stage 1 (extraction): Both appear as `{prf:definition}` directives → RawDefinition
- At Stage 2 (enrichment): Semantic analysis separates concept (def-) from instance (obj-)
- At Stage 3 (execution): Only objects flow through pipeline; definitions archived

### Current State Summary

| Stage | Directory | Definitions | Objects | Schema | Status |
|-------|-----------|-------------|---------|--------|--------|
| 1 | raw_data/definitions/ | ✅ 30 files | ✅ (as RawDefinition) | RawDefinition | ✅ CORRECT |
| 1 | raw_data/objects/ | N/A | ❌ 39 files | MathematicalObject | ❌ WRONG STAGE |
| 2 | refined_data/definitions/ | ✅ 30 files | N/A | EnrichedDefinition | ✅ CORRECT |
| 2 | refined_data/objects/ | N/A | ✅ 53 files | EnrichedObject | ✅ CORRECT |
| 3 | pipeline_data/objects/ | N/A | ✅ (when ready) | MathematicalObject | ✅ CORRECT |

### Action Items

1. ❌ **DO NOT** remove objects/ directories from refined_data/ or pipeline_data/
2. ✅ **DO** remove objects/ directory from raw_data/ (misplaced files)
3. ✅ **DO** move raw_data/objects/* to refined_data/objects/
4. ✅ **DO** update pipeline documentation to clarify stage boundaries
5. ✅ **DO** add validation to prevent future misplacement
6. ✅ **DO** keep objects as distinct entity type throughout pipeline

---

**Analysis Completed**: 2025-10-30
**Framework**: Fragile Gas (Extract-then-Enrich Pipeline)
**Recommendation**: Keep objects separate, fix directory structure
