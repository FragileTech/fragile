"""Complete partial refinements by filling missing fields.

This tool takes the output from find_incomplete_entities.py and uses Gemini 2.5 Pro
to intelligently fill in missing fields based on entity content.

NOTE: This is a CLI tool designed to be called from Claude Code with access to the
mcp__gemini-cli__ask-gemini tool. It generates prompts and instructions for Claude Code
to execute the actual Gemini calls.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def generate_completion_plan(incomplete_entities_file: Path, refined_dir: Path) -> dict[str, Any]:
    """Generate completion plan from incomplete entities report.

    Args:
        incomplete_entities_file: Path to incomplete_entities.json
        refined_dir: Path to refined_data directory

    Returns:
        Completion plan with Gemini prompts for each entity
    """
    # Load incomplete entities report
    with open(incomplete_entities_file) as f:
        report = json.load(f)

    statistics = report["statistics"]
    incomplete_entities = report["incomplete_entities"]

    print("=" * 70)
    print("COMPLETION PLAN GENERATION")
    print("=" * 70)
    print(f"Incomplete entities: {statistics['incomplete_entities']}")
    print(f"Completion rate: {statistics['completion_rate']:.1f}%")
    print()

    # Group entities by type and missing fields
    completion_tasks: dict[str, list] = {}

    for entity_type, entities in incomplete_entities.items():
        if not entities:
            continue

        print(f"\n{entity_type.upper()} ({len(entities)} incomplete)")
        print("-" * 70)

        for entity in entities:
            label = entity["label"]
            missing_fields = entity["missing_fields"]
            errors = entity["errors"]
            warnings = entity["warnings"]

            # Determine priority
            has_critical_errors = any(e["severity"] == "critical" for e in errors)
            has_missing_required = any(
                f in missing_fields for f in ["statement", "name", "mathematical_expression"]
            )

            priority = (
                "critical"
                if has_critical_errors
                else ("high" if has_missing_required else "medium")
            )

            # Generate Gemini prompt based on entity type and missing fields
            gemini_prompt = _generate_gemini_prompt(entity, entity_type)

            task = {
                "label": label,
                "file": entity["file"],
                "entity_type": entity_type.rstrip("s"),
                "priority": priority,
                "missing_fields": missing_fields,
                "errors": errors,
                "warnings": warnings,
                "gemini_prompt": gemini_prompt,
                "current_data": entity["data"],
            }

            if entity_type not in completion_tasks:
                completion_tasks[entity_type] = []
            completion_tasks[entity_type].append(task)

            # Print summary
            print(f"  {label}")
            print(f"    Priority: {priority}")
            print(f"    Missing: {', '.join(missing_fields[:5])}")
            if len(missing_fields) > 5:
                print(f"             ... and {len(missing_fields) - 5} more")

    # Generate completion plan
    return {
        "refined_dir": str(refined_dir),
        "statistics": statistics,
        "completion_tasks": completion_tasks,
        "instructions": _generate_instructions(completion_tasks),
    }


def _generate_gemini_prompt(entity: dict, entity_type: str) -> str:
    """Generate Gemini prompt for completing entity.

    Args:
        entity: Entity information from incomplete_entities.json
        entity_type: Entity type (theorems, axioms, objects, etc.)

    Returns:
        Gemini prompt string
    """
    label = entity["label"]
    data = entity["data"]
    missing_fields = entity["missing_fields"]

    prompt_lines = [
        f"I need to complete the following {entity_type.rstrip('s')} entity:",
        "",
        f"**Label**: {label}",
    ]

    # Add existing content
    if "name" in data:
        prompt_lines.append(f"**Name**: {data['name']}")
    if "statement" in data:
        prompt_lines.append(f"**Statement**: {data['statement']}")
    elif "mathematical_expression" in data:
        prompt_lines.append(f"**Expression**: {data['mathematical_expression']}")
    if "content" in data:
        prompt_lines.append(f"**Content**: {data['content']}")

    prompt_lines.extend(["", "**Missing fields to fill:**"])

    # Describe what needs to be filled
    for field in missing_fields:
        if field == "statement":
            prompt_lines.append("- `statement`: Full mathematical statement/theorem")
        elif field == "name":
            prompt_lines.append("- `name`: Short descriptive name")
        elif field == "tags":
            prompt_lines.append("- `tags`: Descriptive tags for discoverability")
        elif field == "input_objects":
            prompt_lines.append("- `input_objects`: List of object labels this depends on")
        elif field == "input_axioms":
            prompt_lines.append("- `input_axioms`: List of axiom labels this requires")
        elif field == "input_parameters":
            prompt_lines.append("- `input_parameters`: List of parameter labels used")
        elif field == "output_type":
            prompt_lines.append(
                "- `output_type`: One of: property, bound, convergence, existence, uniqueness, equivalence, characterization"
            )
        elif field == "properties_required":
            prompt_lines.append(
                "- `properties_required`: Dict mapping object labels to required properties"
            )
        elif field == "foundational_framework":
            prompt_lines.append("- `foundational_framework`: Framework this axiom belongs to")
        elif field == "core_assumption":
            prompt_lines.append("- `core_assumption`: Core assumption this axiom makes")
        elif field == "object_type":
            prompt_lines.append(
                "- `object_type`: One of: SPACE, OPERATOR, MEASURE, FUNCTION, SET, METRIC, DISTRIBUTION, PROCESS, ALGORITHM, CONSTANT"
            )
        elif field == "current_attributes":
            prompt_lines.append(
                "- `current_attributes`: List of properties/attributes this object has"
            )
        else:
            prompt_lines.append(f"- `{field}`: (field description)")

    prompt_lines.extend([
        "",
        "Please provide the missing fields in JSON format:",
        "```json",
        "{",
        '  "field_name": "value",',
        "  ...",
        "}",
        "```",
    ])

    return "\n".join(prompt_lines)


def _generate_instructions(completion_tasks: dict[str, list]) -> str:
    """Generate instructions for Claude Code to execute completion.

    Args:
        completion_tasks: Completion tasks grouped by entity type

    Returns:
        Instructions string
    """
    total_tasks = sum(len(tasks) for tasks in completion_tasks.values())

    return f"""
# Completion Instructions for Claude Code

**Total entities to complete**: {total_tasks}

## Workflow

For each entity in the completion plan:

1. **Read the Gemini prompt** from `completion_plan.json`
2. **Call Gemini** using mcp__gemini-cli__ask-gemini:
   ```
   Model: gemini-2.5-pro
   Prompt: <gemini_prompt from plan>
   ```
3. **Parse Gemini response** to extract filled fields
4. **Update entity JSON file** with filled fields
5. **Validate updated entity** using validation module
6. **Log completion status** (success/failure/needs_review)

## Batch Processing

Process entities in priority order:
1. **Critical** (prevents loading)
2. **High** (missing required fields)
3. **Medium** (warnings only)

## Review Required

After Gemini fills fields, manually review:
- ✅ Correctness: Are filled values accurate?
- ✅ Consistency: Do they align with framework?
- ✅ Completeness: Are all required fields filled?

## Example Workflow (Single Entity)

```python
# 1. Get task
task = completion_plan["completion_tasks"]["theorems"][0]
label = task["label"]
file_path = refined_dir / task["file"]

# 2. Call Gemini
gemini_response = mcp__gemini_cli__ask_gemini(
    model="gemini-2.5-pro",
    prompt=task["gemini_prompt"]
)

# 3. Parse response
filled_fields = parse_json_from_gemini_response(gemini_response)

# 4. Update entity
with open(file_path) as f:
    entity_data = json.load(f)

entity_data.update(filled_fields)

# 5. Validate
validator = TheoremValidator()
result = validator.validate_entity(entity_data, file_path)

if result.is_valid:
    # Save updated entity
    with open(file_path, 'w') as f:
        json.dump(entity_data, f, indent=2)
    print(f"✅ {{label}}: Completed")
else:
    print(f"⚠️ {{label}}: Validation failed after completion")
    print(f"  Errors: {{len(result.errors)}}")
```

## Completion Report

After processing all entities, generate completion report:
- Total entities processed
- Successfully completed
- Failed validation
- Requires manual review

Save to: `completion_report.md`
"""


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate completion plan for incomplete entities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate completion plan
  python -m fragile.proofs.tools.complete_refinement \\
    --incomplete-file incomplete_entities.json \\
    --refined-dir docs/source/.../refined_data/

  # Specify output file
  python -m fragile.proofs.tools.complete_refinement \\
    --incomplete-file incomplete_entities.json \\
    --refined-dir docs/source/.../refined_data/ \\
    --output completion_plan.json

NOTE: This tool generates a completion plan. The actual completion
must be executed by Claude Code using the mcp__gemini-cli__ask-gemini tool.
        """,
    )

    parser.add_argument(
        "--incomplete-file",
        type=Path,
        default=Path("incomplete_entities.json"),
        help="Path to incomplete_entities.json from find_incomplete_entities",
    )

    parser.add_argument(
        "--refined-dir",
        type=Path,
        required=True,
        help="Path to refined_data directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("completion_plan.json"),
        help="Output completion plan JSON file (default: completion_plan.json)",
    )

    parser.add_argument(
        "--output-instructions",
        type=Path,
        default=Path("completion_instructions.md"),
        help="Output instructions markdown file (default: completion_instructions.md)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.incomplete_file.exists():
        print(f"Error: incomplete_file does not exist: {args.incomplete_file}", file=sys.stderr)
        print("Run find_incomplete_entities.py first to generate this file.", file=sys.stderr)
        sys.exit(1)

    if not args.refined_dir.exists():
        print(f"Error: refined_dir does not exist: {args.refined_dir}", file=sys.stderr)
        sys.exit(1)

    # Generate completion plan
    try:
        plan = generate_completion_plan(args.incomplete_file, args.refined_dir)

        # Save completion plan
        with open(args.output, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"\n✅ Completion plan saved to: {args.output}")

        # Save instructions
        with open(args.output_instructions, "w") as f:
            f.write(plan["instructions"])
        print(f"✅ Instructions saved to: {args.output_instructions}")

        print()
        print("=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("1. Review completion_plan.json to understand what will be filled")
        print("2. Use Claude Code to execute the completion workflow")
        print("3. Claude Code will call Gemini for each entity to fill missing fields")
        print("4. Review completed entities for accuracy")
        print("5. Re-run validation to verify all issues resolved")

        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
