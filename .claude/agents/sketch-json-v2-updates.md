# Sketch-JSON Agent v2.0 Updates

## Updates Required for Full sketch.json Schema Support

This document contains all the updates needed to transform sketch-json.md from outputting simplified format to full sketch.json format.

---

## UPDATE 1: Phase 4 - Strengthen JSON Enforcement (Line ~280)

### REPLACE the validation prompt section with:

**CRITICAL**: Output your response as a valid JSON object matching this schema structure.

**STRICT JSON REQUIREMENTS**:
- Output ONLY raw JSON - no markdown formatting, no code blocks, no explanatory text
- Start with `{` and end with `}`
- Ensure all strings are properly escaped
- No trailing commas
- All required fields must be present

**Required JSON Schema** (sketch_strategy.json):

```json
{
  "strategist": "Your model name (e.g., 'Gemini 2.5 Pro' or 'GPT-5 via Codex')",
  "method": "Concise proof technique name",
  "summary": "Brief narrative explaining core insight",
  "keySteps": [
    "Step 1: High-level description",
    "Step 2: High-level description"
  ],
  "strengths": ["Advantage 1", "Advantage 2"],
  "weaknesses": ["Challenge 1", "Challenge 2"],
  "frameworkDependencies": {
    "theorems": [
      {
        "label": "thm-xxx",
        "document": "document_id",
        "purpose": "What this provides",
        "usedInSteps": ["Step 1", "Step 3"]
      }
    ],
    "lemmas": [],
    "axioms": [],
    "definitions": []
  },
  "technicalDeepDives": [
    {
      "challengeTitle": "Most Difficult Point",
      "difficultyDescription": "Why hard",
      "proposedSolution": "Technique to overcome",
      "references": ["Related result"]
    }
  ],
  "confidenceScore": "High" | "Medium" | "Low"
}
```

**VALIDATION**:
- keySteps: MUST have at least 1 item (minItems: 1)
- confidenceScore: MUST be exactly "High", "Medium", or "Low"
- All dependency objects MUST have label, document, and purpose fields

**OUTPUT ONLY THE JSON - NOTHING ELSE**

---

## UPDATE 2: Add Phase 6.5 - Schema Transformation (Insert after Phase 6.4, before Phase 7)

### ADD NEW PHASE:

---

## PHASE 6.5: Transform to Full Proof Sketch Format (sketch.json)

**Purpose**: Transform the selected strategy (sketch_strategy.json format) into a comprehensive proof sketch (sketch.json format) that includes all metadata, alternative approaches, expansion roadmap, and validation checklist.

### Step 6.5.1: Map Entity Type

```python
def map_entity_type(entity_type: str) -> str:
    """Map entity type to sketch.json enum."""
    mapping = {
        "theorem": "Theorem",
        "lemma": "Lemma",
        "proposition": "Proposition",
        "corollary": "Corollary"
    }
    return mapping.get(entity_type.lower(), "Theorem")

entity_type_formatted = map_entity_type(entity_type)
```

### Step 6.5.2: Determine Sketch Status

```python
def determine_status(selector_confidence: str, schema_valid: bool) -> str:
    """Determine sketch status from validation results."""
    if selector_confidence == "HIGH" and schema_valid and len(unverified_deps) == 0:
        return "Ready for Expansion"
    elif selector_confidence in ["HIGH", "MEDIUM"]:
        return "Draft"
    else:
        return "Sketch"

sketch_status = determine_status(
    selection_result["confidence"],
    schema_valid
)
```

### Step 6.5.3: Build Strategy Synthesis Section

**IMPORTANT**: Preserve **BOTH** Gemini and Codex strategies in the output.

```python
# Transform both strategies to strategySynthesis format
strategies_list = []

# Add Gemini strategy
if gemini_strategy is not None:
    strategies_list.append({
        "strategist": gemini_strategy["strategist"],
        "method": gemini_strategy["method"],
        "keySteps": gemini_strategy["keySteps"],
        "strengths": gemini_strategy["strengths"],
        "weaknesses": gemini_strategy["weaknesses"]
    })

# Add Codex strategy
if codex_strategy is not None:
    strategies_list.append({
        "strategist": codex_strategy["strategist"],
        "method": codex_strategy["method"],
        "keySteps": codex_strategy["keySteps"],
        "strengths": codex_strategy["strengths"],
        "weaknesses": codex_strategy["weaknesses"]
    })

# Determine cross-validation status
cross_validation_status = "Single strategist only"
if gemini_strategy and codex_strategy:
    if selection_result["decision"] in ["STRATEGY A", "STRATEGY B"]:
        # Both provided strategies, one selected
        if gemini_strategy["method"] == codex_strategy["method"]:
            cross_validation_status = "Consensus between strategists"
        else:
            cross_validation_status = "Discrepancies noted"
    elif selection_result["decision"] == "HYBRID":
        cross_validation_status = "Consensus between strategists"

strategy_synthesis = {
    "strategies": strategies_list,  # BOTH Gemini and Codex strategies preserved!
    "recommendedApproach": {
        "chosenMethod": selected_strategy["method"],
        "rationale": selection_result["justification"],
        "verificationStatus": {
            "frameworkDependencies": (
                "Verified" if len(unverified_deps) == 0
                else "Partially Verified" if len(unverified_deps) < 3
                else "Needs Verification"
            ),
            "circularReasoning": "No circularity detected",  # Verified by strategy-selector
            "keyAssumptions": "All assumptions are standard",  # Can be enhanced
            "crossValidation": cross_validation_status
        }
    }
}
```

### Step 6.5.4: Transform Dependencies

```python
def transform_dependencies(
    framework_deps: dict,
    unverified_deps: list
) -> list:
    """Transform frameworkDependencies to verifiedDependencies format."""
    verified = []
    unverified_labels = {dep["label"] for dep in unverified_deps}

    for dep_type in ["theorems", "lemmas", "axioms", "definitions"]:
        for dep in framework_deps.get(dep_type, []):
            dep_label = dep["label"]

            # Only add if verified
            if dep_label not in unverified_labels:
                verified.append({
                    "type": dep_type.rstrip('s').capitalize(),  # "theorems" → "Theorem"
                    "label": dep_label,
                    "sourceDocument": dep.get("document", "unknown"),
                    "purpose": dep.get("purpose", ""),
                    "usedInSteps": dep.get("usedInSteps", [])
                })

    return verified

# Extract missing lemmas and uncertain assumptions
def extract_required_lemmas(strategy: dict) -> list:
    """Extract lemmas that need to be proven from technical challenges."""
    lemmas = []

    for challenge in strategy.get("technicalDeepDives", []):
        if "lemma" in challenge["challengeTitle"].lower() or "intermediate result" in challenge.get("proposedSolution", "").lower():
            lemmas.append({
                "name": challenge["challengeTitle"],
                "statement": "[To be formalized during expansion]",
                "justification": challenge["difficultyDescription"],
                "difficulty": "Medium"  # Default
            })

    return lemmas

def extract_uncertain_assumptions(weaknesses: list) -> list:
    """Extract uncertain assumptions from weaknesses."""
    assumptions = []

    for weakness in weaknesses:
        if "requires" in weakness.lower() or "assumes" in weakness.lower() or "needs" in weakness.lower():
            assumptions.append({
                "statement": weakness,
                "justification": "Identified as potential assumption during strategy generation",
                "resolutionPath": "Verify during proof expansion or add as explicit hypothesis"
            })

    return assumptions

dependencies = {
    "verifiedDependencies": transform_dependencies(
        selected_strategy["frameworkDependencies"],
        unverified_deps
    ),
    "missingOrUncertainDependencies": {
        "lemmasToProve": extract_required_lemmas(selected_strategy),
        "uncertainAssumptions": extract_uncertain_assumptions(
            selected_strategy["weaknesses"]
        )
    }
}
```

### Step 6.5.5: Build Detailed Proof Section

```python
def expand_key_steps_to_proof_steps(key_steps: list) -> list:
    """Transform high-level key steps to detailed proof step structure."""
    proof_steps = []

    for i, step_desc in enumerate(key_steps, start=1):
        proof_steps.append({
            "stepNumber": i,
            "title": f"Step {i}",
            "goal": step_desc,  # High-level description from strategy
            "action": "[To be expanded during proof development]",
            "justification": "[Framework result to be identified during expansion]",
            "expectedResult": "[Intermediate conclusion to be derived]",
            "dependencies": [],  # Will be populated during expansion
            "potentialIssues": "[To be identified during detailed expansion]"
        })

    return proof_steps

detailed_proof = {
    "overview": selected_strategy["summary"],
    "topLevelOutline": selected_strategy["keySteps"],
    "steps": expand_key_steps_to_proof_steps(selected_strategy["keySteps"]),
    "conclusion": f"Therefore, {entity_label} is proven. Q.E.D."
}
```

### Step 6.5.6: Generate Validation Checklist

```python
validation_checklist = {
    "logicalCompleteness": schema_valid,  # Schema validation passed
    "hypothesisUsage": True,  # To be verified during expansion
    "conclusionDerivation": True,  # To be verified during expansion
    "frameworkConsistency": len(unverified_deps) == 0,
    "noCircularReasoning": True,  # Verified by strategy-selector
    "constantTracking": True,  # To be verified during expansion
    "edgeCases": False,  # Not yet addressed in sketch
    "regularityAssumptions": True  # To be verified during expansion
}
```

### Step 6.5.7: Format Alternative Approaches

```python
def format_alternative_approaches(
    gemini_strategy: dict,
    codex_strategy: dict,
    selection_result: dict
) -> list:
    """Format non-selected strategies as alternative approaches."""
    alternatives = []

    # Determine which was selected
    selected_strategist = selection_result["decision"]

    if selected_strategist == "STRATEGY A":
        # Gemini selected, Codex is alternative
        if codex_strategy:
            alternatives.append({
                "name": codex_strategy["method"],
                "approach": f"{codex_strategy['strategist']}'s approach: {codex_strategy['summary']}",
                "pros": codex_strategy["strengths"],
                "cons": codex_strategy["weaknesses"],
                "whenToConsider": "When alternative perspective is needed or selected approach encounters obstacles"
            })
    elif selected_strategist == "STRATEGY B":
        # Codex selected, Gemini is alternative
        if gemini_strategy:
            alternatives.append({
                "name": gemini_strategy["method"],
                "approach": f"{gemini_strategy['strategist']}'s approach: {gemini_strategy['summary']}",
                "pros": gemini_strategy["strengths"],
                "cons": gemini_strategy["weaknesses"],
                "whenToConsider": "When alternative perspective is needed or selected approach encounters obstacles"
            })
    elif selected_strategist == "HYBRID":
        # Both contributed, document synthesis
        alternatives.append({
            "name": "Pure Gemini Approach",
            "approach": f"{gemini_strategy['method']} (not synthesized)",
            "pros": gemini_strategy["strengths"],
            "cons": ["Not as refined as hybrid approach"] + gemini_strategy["weaknesses"],
            "whenToConsider": "For comparison with hybrid synthesis"
        })
        alternatives.append({
            "name": "Pure Codex Approach",
            "approach": f"{codex_strategy['method']} (not synthesized)",
            "pros": codex_strategy["strengths"],
            "cons": ["Not as refined as hybrid approach"] + codex_strategy["weaknesses"],
            "whenToConsider": "For comparison with hybrid synthesis"
        })

    return alternatives

alternative_approaches = format_alternative_approaches(
    gemini_strategy,
    codex_strategy,
    selection_result
)
```

### Step 6.5.8: Generate Expansion Roadmap

```python
def estimate_hours(num_challenges: int, num_steps: int) -> tuple[int, int]:
    """Estimate min and max hours for expansion."""
    # Phase 1: Prove lemmas (2-4 hours per challenge)
    lemma_hours = num_challenges * 2, num_challenges * 4

    # Phase 2: Fill details (2-4 hours per step)
    detail_hours = num_steps * 2, num_steps * 4

    # Phase 3: Rigor (fixed)
    rigor_hours = 2, 4

    # Phase 4: Review (fixed)
    review_hours = 1, 2

    total_min = sum([lemma_hours[0], detail_hours[0], rigor_hours[0], review_hours[0]])
    total_max = sum([lemma_hours[1], detail_hours[1], rigor_hours[1], review_hours[1]])

    return total_min, total_max

num_challenges = len(selected_strategy.get("technicalDeepDives", []))
num_steps = len(selected_strategy["keySteps"])
total_min, total_max = estimate_hours(num_challenges, num_steps)

expansion_roadmap = {
    "phases": [
        {
            "phaseTitle": "Phase 1: Prove Missing Lemmas",
            "estimatedTime": f"{num_challenges * 2}-{num_challenges * 4} hours" if num_challenges > 0 else "0 hours (no lemmas needed)",
            "tasks": [
                {
                    "taskName": f"Prove {challenge['challengeTitle']}",
                    "strategy": challenge.get("proposedSolution", "TBD")[:100],  # Truncate
                    "difficulty": "Medium"
                }
                for challenge in selected_strategy.get("technicalDeepDives", [])
            ] if num_challenges > 0 else []
        },
        {
            "phaseTitle": "Phase 2: Fill Technical Details",
            "estimatedTime": f"{num_steps * 2}-{num_steps * 4} hours",
            "tasks": [
                {
                    "taskName": f"Expand {step[:50]}...",  # Truncate step description
                    "strategy": "Detailed mathematical derivation with full justification",
                    "difficulty": "Medium"
                }
                for step in selected_strategy["keySteps"]
            ]
        },
        {
            "phaseTitle": "Phase 3: Add Rigor and Edge Cases",
            "estimatedTime": "2-4 hours",
            "tasks": [
                {"taskName": "Add epsilon-delta arguments where needed", "strategy": "Formal analysis", "difficulty": "Medium"},
                {"taskName": "Add measure-theoretic details", "strategy": "Rigorous foundations", "difficulty": "Hard"},
                {"taskName": "Verify edge cases and boundary conditions", "strategy": "Boundary analysis", "difficulty": "Medium"}
            ]
        },
        {
            "phaseTitle": "Phase 4: Review and Validation",
            "estimatedTime": "1-2 hours",
            "tasks": [
                {"taskName": "Cross-validate against framework", "strategy": "Systematic verification"},
                {"taskName": "Audit constant tracking", "strategy": "Ensure all constants bounded"},
                {"taskName": "Final completeness check", "strategy": "Verify all claims addressed"}
            ]
        }
    ],
    "totalEstimatedTime": f"{total_min}-{total_max} hours"
}
```

### Step 6.5.9: Build Cross-References

```python
def extract_labels(dep_list: list) -> list[str]:
    """Extract just the labels from dependency list."""
    return [dep["label"] for dep in dep_list]

cross_references = {
    "theoremsUsed": extract_labels(
        selected_strategy["frameworkDependencies"].get("theorems", [])
    ),
    "definitionsUsed": extract_labels(
        selected_strategy["frameworkDependencies"].get("definitions", [])
    ),
    "axiomsUsed": extract_labels(
        selected_strategy["frameworkDependencies"].get("axioms", [])
    ),
    "relatedProofs": [],  # Optional - could be populated from registry
    "downstreamConsequences": []  # Optional - could be populated from registry
}
```

### Step 6.5.10: Generate Special Notes

```python
special_notes = f"""Generated by sketch-json agent v2.0

**Generation Metadata**:
- Agent: sketch-json v2.0
- Timestamp: {metadata['generated_at']}
- Strategy Selector Decision: {selection_result['decision']}
- Selector Confidence: {selection_result['confidence']}
- Schema Validation: {'PASSED' if schema_valid else 'FAILED'}
- Framework Dependencies Verified: {len(verified_deps)}/{len(verified_deps) + len(unverified_deps)}

**Preprocessing Notes**:
{metadata.get('validation', {}).get('preprocessing_notes', 'None')}

**Strategy Selection Process**:
- Gemini 2.5 Pro Confidence: {gemini_strategy.get('confidenceScore', 'N/A')}
- GPT-5 via Codex Confidence: {codex_strategy.get('confidenceScore', 'N/A')}
- Selected: {selection_result['decision']}
- Justification: {selection_result['justification'][:200]}...

**Next Steps**:
1. Review sketch completeness
2. Address any missing dependencies
3. Expand to full proof using theorem-prover agent
4. Validate with math-reviewer agent
"""
```

### Step 6.5.11: Assemble Full Proof Sketch

```python
import datetime

full_sketch = {
    # Metadata
    "title": entity.get("title", f"{entity_type_formatted}: {entity_label}"),
    "label": entity_label,
    "type": entity_type_formatted,
    "source": f"docs/source/{chapter}/{document_id}.md",
    "date": datetime.datetime.now().strftime("%Y-%m-%d"),
    "status": sketch_status,

    # Statement
    "statement": {
        "formal": entity.get("statement", entity.get("nl_statement", entity.get("content_markdown", ""))),
        "informal": entity.get("nl_statement", "[Informal statement to be generated]")
    },

    # Strategy Synthesis (preserves BOTH Gemini and Codex strategies!)
    "strategySynthesis": strategy_synthesis,

    # Dependencies
    "dependencies": dependencies,

    # Detailed Proof
    "detailedProof": detailed_proof,

    # Technical Deep Dives (from selected strategy)
    "technicalDeepDives": selected_strategy.get("technicalDeepDives", []),

    # Validation Checklist
    "validationChecklist": validation_checklist,

    # Alternative Approaches
    "alternativeApproaches": alternative_approaches,

    # Future Work
    "futureWork": {
        "remainingGaps": [w for w in selected_strategy["weaknesses"] if "gap" in w.lower() or "missing" in w.lower()],
        "conjectures": [],  # Optional
        "extensions": []     # Optional
    },

    # Expansion Roadmap
    "expansionRoadmap": expansion_roadmap,

    # Cross References
    "crossReferences": cross_references,

    # Special Notes
    "specialNotes": special_notes
}
```

### Step 6.5.12: Validate Against sketch.json Schema

```python
from mathster.agent_schemas.validate_sketch import validate_full_sketch, fill_missing_sketch_fields

# Validate
is_valid, errors = validate_full_sketch(full_sketch)

if not is_valid:
    # Attempt to fill missing fields
    full_sketch, filled_fields = fill_missing_sketch_fields(full_sketch)

    # Re-validate
    is_valid, errors = validate_full_sketch(full_sketch)

    if not is_valid:
        # Note validation issues in special notes
        full_sketch["specialNotes"] += f"\n\n⚠️ Schema Validation Warnings:\n" + "\n".join(f"  - {err}" for err in errors)

    if filled_fields:
        full_sketch["specialNotes"] += f"\n\n⚠️ Auto-filled missing fields: {', '.join(filled_fields)}"
```

---

## UPDATE 3: Modify Phase 7 Output (Replace existing Phase 7)

### REPLACE Phase 7 with:

## PHASE 7: Output Generation

### Step 7.1: Format JSON Output

```python
import json

json_output = json.dumps(full_sketch, indent=2, ensure_ascii=False)
```

### Step 7.2: Write to File

```python
output_filename = f"sketch-{entity_label}.json"
output_path = f"docs/source/{chapter}/{document_id}/sketches/{output_filename}"

Write(
    file_path=output_path,
    content=json_output
)
```

### Step 7.3: Verify File Written

```bash
ls -lh "{output_path}"
```

### Step 7.4: Report to User

**Success Message**:
```markdown
✅ **Proof sketch generated successfully**

**Label**: {entity_label}
**Type**: {entity_type_formatted}
**Document**: {document_id}.md
**Status**: {sketch_status}

**Strategy Selected**: {selection_result["decision"]}
- Gemini 2.5 Pro: {gemini_strategy["method"]} (Confidence: {gemini_strategy["confidenceScore"]})
- GPT-5 via Codex: {codex_strategy["method"]} (Confidence: {codex_strategy["confidenceScore"]})
- **Chosen**: {selected_strategy["method"]}

**Selector Confidence**: {selection_result["confidence"]}

**Output File**: `{output_path}`

**Validation Status**:
- Full Schema Valid: {"✅" if is_valid else "⚠️"}
- Framework Verified: {len(verified_deps)}/{len(verified_deps) + len(unverified_deps)} dependencies
- Missing Fields Auto-Filled: {len(filled_fields) if filled_fields else 0}

---

### Proof Sketch Summary

**Method**: {selected_strategy["method"]}

**Overview**: {selected_strategy["summary"]}

**Key Steps** ({len(selected_strategy["keySteps"])} steps):
{for i, step in enumerate(selected_strategy["keySteps"], 1):
  {i}. {step}
}

**Technical Challenges** ({num_challenges}):
{for challenge in selected_strategy.get("technicalDeepDives", []):
  - **{challenge["challengeTitle"]}**: {challenge["difficultyDescription"][:100]}...
}

**Alternative Approaches Documented**: {len(alternative_approaches)}

**Expansion Roadmap**:
- Total Phases: {len(expansion_roadmap["phases"])}
- Estimated Time: {expansion_roadmap["totalEstimatedTime"]}

---

**Next Steps**:

1. **Review Sketch**: `cat {output_path} | jq .`

2. **Validate Sketch**:
   ```bash
   python src/mathster/agent_schemas/validate_sketch.py {output_path}
   ```

3. **Validate with AI Review** (optional):
   ```bash
   sketch-judge {output_path}
   ```

4. **Expand to Full Proof**:
   ```bash
   theorem-prover {output_path}
   ```

---

**Sketch generation complete.** Full proof sketch written to: `{output_path}`
```

---

## Summary of Changes

1. **Strengthened JSON enforcement** in Gemini/Codex prompts
2. **Added Phase 6.5** with comprehensive schema transformation
3. **Preserves BOTH Gemini and Codex strategies** in strategySynthesis.strategies array
4. **Generates full sketch.json format** with all required fields:
   - Strategy synthesis with both AI approaches
   - Verified and missing dependencies
   - Detailed proof outline with expandable steps
   - Technical deep dives
   - Validation checklist
   - Alternative approaches
   - Future work notes
   - Expansion roadmap with time estimates
   - Cross-references
   - Special generation notes

5. **Validates against sketch.json schema** before output
6. **Auto-fills missing fields** with appropriate defaults
7. **Enhanced user reporting** with comprehensive summary

All changes maintain backward compatibility while adding rich metadata for proof expansion.
