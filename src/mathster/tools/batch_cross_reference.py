#!/usr/bin/env python3
"""
Batch Cross-Reference Analysis using Gemini 2.5 Pro.

Processes all theorem-like entities and fills dependency fields.
"""

import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def build_entity_context(raw_data_dir: Path) -> dict[str, list[str]]:
    """Build context of all available entities."""
    context = {
        "objects": [],
        "axioms": [],
        "parameters": [],
    }

    # Load objects
    objects_dir = raw_data_dir / "objects"
    if objects_dir.exists():
        for obj_file in objects_dir.glob("*.json"):
            with open(obj_file) as f:
                data = json.load(f)
                label = data.get("label")
                name = data.get("name", "")
                if label:
                    context["objects"].append(f"{label}: {name}")

    # Load axioms
    axioms_dir = raw_data_dir / "axioms"
    if axioms_dir.exists():
        for axiom_file in axioms_dir.glob("*.json"):
            with open(axiom_file) as f:
                data = json.load(f)
                label = data.get("label")
                name = data.get("name", "")
                if label:
                    context["axioms"].append(f"{label}: {name}")

    # Load parameters
    params_dir = raw_data_dir / "parameters"
    if params_dir.exists():
        for param_file in params_dir.glob("*.json"):
            with open(param_file) as f:
                data = json.load(f)
                label = data.get("label")
                name = data.get("name", "")
                if label:
                    context["parameters"].append(f"{label}: {name}")

    return context


def create_analysis_prompt(entity: dict, statement: str, context: dict[str, list[str]]) -> str:
    """Create prompt for Gemini 2.5 Pro."""

    label = entity.get("label", "unknown")
    name = entity.get("name", "")

    return f"""You are analyzing a mathematical theorem/lemma to identify ALL dependencies and characterize the result.

THEOREM/LEMMA LABEL: {label}
NAME: {name}

STATEMENT:
{statement}

AVAILABLE FRAMEWORK ENTITIES:

MATHEMATICAL OBJECTS (use these labels):
{chr(10).join(f"- {obj}" for obj in context["objects"])}

AXIOMS (use these labels):
{chr(10).join(f"- {axiom}" for axiom in context["axioms"])}

PARAMETERS (use these labels):
{chr(10).join(f"- {param}" for param in context["parameters"])}

TASK: Identify every mathematical entity this theorem/lemma depends on.

OUTPUT FORMAT (JSON):
{{
  "input_objects": [
    "obj-label-1",
    "obj-label-2"
  ],
  "input_axioms": [
    "axiom-label-1"
  ],
  "input_parameters": [
    "param-label-1"
  ],
  "output_type": "Bound|Property|Existence|Continuity|Lipschitz|Convergence|Equivalence|Other",
  "relations_established": [
    "Establishes Lipschitz continuity of X with constant C",
    "Bounds Y by quadratic function of Z"
  ]
}}

CRITICAL INSTRUCTIONS:
1. Use ONLY labels from AVAILABLE FRAMEWORK ENTITIES above
2. For input_objects: Include ALL mathematical objects mentioned or used (operators, measures, spaces, functions)
3. For input_axioms: Include axioms that are required assumptions
4. For input_parameters: Include parameters that appear in bounds or expressions
5. For output_type: Choose the most specific category
6. For relations_established: List concrete relationships this result proves (be specific)
7. Look for implicit dependencies (e.g., if statement mentions "swarm states", include relevant objects)
8. DO NOT hallucinate labels - only use labels from the lists above
9. Return ONLY valid JSON, no explanatory text

Analyze the statement and return the JSON:"""


def parse_gemini_response(response_text: str) -> dict | None:
    """Parse JSON response from Gemini."""
    try:
        # Look for JSON block
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        elif "{" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_text = response_text[start:end]
        else:
            return None

        return json.loads(json_text)

    except Exception as e:
        print(f"      Error parsing: {e}")
        return None


def analyze_entity_with_gemini(
    entity_file: Path, context: dict[str, list[str]], dry_run: bool = False
) -> dict | None:
    """Analyze a single entity file with Gemini 2.5 Pro."""

    # Load entity
    with open(entity_file) as f:
        entity = json.load(f)

    label = entity.get("label") or entity.get("label_text") or entity_file.stem

    # Extract statement
    statement = None
    for field in ["statement", "natural_language_statement", "full_statement_text"]:
        if entity.get(field):
            statement = entity[field]
            break

    if not statement:
        print(f"    SKIP {label}: No statement")
        return None

    # Create prompt
    prompt = create_analysis_prompt(entity, statement, context)

    if dry_run:
        print(f"    [Dry run] Would analyze {label}")
        return None

    # Query Gemini (this would be replaced with actual MCP call)
    # For now, print that we would call it
    print(f"    Analyzing {label}...")

    # Placeholder - actual implementation would call Gemini
    # response = mcp__gemini-cli__ask-gemini(model="gemini-2.5-pro", prompt=prompt)
    # analysis = parse_gemini_response(response)

    return {"prompt": prompt, "label": label, "entity_file": str(entity_file)}


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch cross-reference analysis")
    parser.add_argument("raw_data_dir", type=Path, help="Path to raw_data directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually process, just show what would be done",
    )
    parser.add_argument("--limit", type=int, help="Limit number of entities")
    parser.add_argument(
        "--save-prompts", type=Path, help="Directory to save prompts for manual processing"
    )

    args = parser.parse_args()

    raw_data_dir = Path(args.raw_data_dir)

    # Build context
    print("Building entity context...")
    context = build_entity_context(raw_data_dir)
    print(f"  Objects: {len(context['objects'])}")
    print(f"  Axioms: {len(context['axioms'])}")
    print(f"  Parameters: {len(context['parameters'])}")
    print()

    # Get all entity files
    entity_files = []
    for subdir in ["theorems", "lemmas", "propositions", "corollaries"]:
        entity_dir = raw_data_dir / subdir
        if entity_dir.exists():
            entity_files.extend(entity_dir.glob("*.json"))

    entity_files = sorted(entity_files)

    if args.limit:
        entity_files = entity_files[: args.limit]

    print(f"Processing {len(entity_files)} entities...")
    print()

    # Create prompts directory if requested
    if args.save_prompts:
        prompts_dir = Path(args.save_prompts)
        prompts_dir.mkdir(parents=True, exist_ok=True)

    # Process each entity
    processed = 0
    skipped = 0

    for entity_file in entity_files:
        result = analyze_entity_with_gemini(entity_file, context, args.dry_run)

        if result:
            processed += 1

            # Save prompt if requested
            if args.save_prompts:
                prompt_file = prompts_dir / f"{result['label']}.txt"
                with open(prompt_file, "w") as f:
                    f.write(result["prompt"])
        else:
            skipped += 1

    print()
    print("=" * 80)
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")

    if args.save_prompts:
        print(f"Prompts saved to: {prompts_dir}")


if __name__ == "__main__":
    main()
