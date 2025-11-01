#!/usr/bin/env python3
"""
Gemini Batch Processor for Cross-Reference Analysis.

Processes entities in batches, saves intermediate results.
Uses Gemini 2.5 Pro via MCP tool.

Usage:
    # NOTE: This script generates prompts and saves them.
    # The actual Gemini calls must be made interactively via Claude Code
    # since MCP tools are only available in that context.

    python gemini_batch_processor.py /path/to/raw_data --generate-batch 0
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class GeminiBatchProcessor:
    """Process entities in batches with Gemini 2.5 Pro."""

    def __init__(self, raw_data_dir: Path, batch_size: int = 10):
        self.raw_data_dir = Path(raw_data_dir)
        self.batch_size = batch_size
        self.context = self._build_context()

    def _build_context(self) -> dict[str, list[str]]:
        """Build entity context."""
        context = {"objects": [], "axioms": [], "parameters": []}

        for obj_file in (self.raw_data_dir / "objects").glob("*.json"):
            with open(obj_file) as f:
                data = json.load(f)
                label = data.get("label")
                name = data.get("name", "")
                if label:
                    context["objects"].append(f"{label}: {name}")

        for axiom_file in (self.raw_data_dir / "axioms").glob("*.json"):
            with open(axiom_file) as f:
                data = json.load(f)
                label = data.get("label")
                name = data.get("name", "")
                if label:
                    context["axioms"].append(f"{label}: {name}")

        for param_file in (self.raw_data_dir / "parameters").glob("*.json"):
            with open(param_file) as f:
                data = json.load(f)
                label = data.get("label")
                name = data.get("name", "")
                if label:
                    context["parameters"].append(f"{label}: {name}")

        return context

    def get_all_entities(self) -> list[tuple[Path, dict]]:
        """Get all theorem-like entities."""
        entities = []

        for subdir in ["theorems", "lemmas", "propositions", "corollaries"]:
            entity_dir = self.raw_data_dir / subdir
            if entity_dir.exists():
                for entity_file in entity_dir.glob("*.json"):
                    with open(entity_file) as f:
                        entity = json.load(f)
                    entities.append((entity_file, entity))

        return sorted(entities, key=lambda x: x[0].name)

    def create_prompt(self, entity: dict, statement: str) -> str:
        """Create Gemini prompt."""
        label = entity.get("label") or entity.get("label_text", "unknown")
        name = entity.get("name", "")

        return f"""Analyze this mathematical theorem/lemma to identify ALL dependencies.

LABEL: {label}
NAME: {name}

STATEMENT:
{statement}

AVAILABLE OBJECTS:
{chr(10).join(f"- {obj}" for obj in self.context["objects"])}

AVAILABLE AXIOMS:
{chr(10).join(f"- {axiom}" for axiom in self.context["axioms"])}

AVAILABLE PARAMETERS:
{chr(10).join(f"- {param}" for param in self.context["parameters"])}

Return JSON with:
- input_objects: List of obj-* labels this depends on
- input_axioms: List of axiom-* labels required
- input_parameters: List of param-* labels used
- output_type: One of: Bound|Property|Existence|Continuity|Lipschitz|Convergence|Equivalence|Other
- relations_established: List of specific relationships proven

Use ONLY labels from lists above. Return valid JSON only."""

    def generate_batch(self, batch_num: int, output_dir: Path):
        """Generate prompts for a batch."""
        entities = self.get_all_entities()
        start_idx = batch_num * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(entities))

        batch_entities = entities[start_idx:end_idx]

        output_dir.mkdir(parents=True, exist_ok=True)

        batch_info = {
            "batch_num": batch_num,
            "total_entities": len(entities),
            "batch_size": self.batch_size,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "entities": [],
        }

        for entity_file, entity in batch_entities:
            label = entity.get("label") or entity.get("label_text") or entity_file.stem

            # Extract statement
            statement = None
            for field in ["statement", "natural_language_statement", "full_statement_text"]:
                if entity.get(field):
                    statement = entity[field]
                    break

            if not statement:
                continue

            # Create prompt
            prompt = self.create_prompt(entity, statement)

            # Save prompt
            prompt_file = output_dir / f"{label}.txt"
            with open(prompt_file, "w") as f:
                f.write(prompt)

            batch_info["entities"].append({
                "label": label,
                "entity_file": str(entity_file),
                "prompt_file": str(prompt_file),
            })

        # Save batch info
        info_file = output_dir / "batch_info.json"
        with open(info_file, "w") as f:
            json.dump(batch_info, f, indent=2)

        return batch_info


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_dir", type=Path)
    parser.add_argument("--generate-batch", type=int, help="Generate batch N")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/gemini_batches"))

    args = parser.parse_args()

    processor = GeminiBatchProcessor(args.raw_data_dir, args.batch_size)

    if args.generate_batch is not None:
        batch_dir = args.output_dir / f"batch_{args.generate_batch:03d}"
        info = processor.generate_batch(args.generate_batch, batch_dir)

        print(f"Generated batch {info['batch_num']}")
        print(f"  Entities: {len(info['entities'])}")
        print(f"  Range: {info['start_idx']}-{info['end_idx']} of {info['total_entities']}")
        print(f"  Output: {batch_dir}")

    else:
        entities = processor.get_all_entities()
        total_batches = (len(entities) + args.batch_size - 1) // args.batch_size

        print(f"Total entities: {len(entities)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Total batches: {total_batches}")
        print()
        print("Run with --generate-batch N to generate batch N")


if __name__ == "__main__":
    main()
