#!/usr/bin/env python3
"""Process cross-reference batches with Gemini 2.5 Pro via subprocess."""

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict


def query_gemini(prompt: str) -> dict[str, Any]:
    """Query Gemini 2.5 Pro via Claude Code's MCP and return parsed JSON."""
    # Note: This must be called from within Claude Code environment
    # We'll use a marker format that can be parsed
    print(f"GEMINI_QUERY_START:{json.dumps({'prompt': prompt})}", flush=True)
    # Wait for response marker
    response = input()  # This would need to be coordinated with the caller
    return json.loads(response)


def process_batch_file(prompt_file: Path) -> dict[str, Any]:
    """Process a single prompt file."""
    label = prompt_file.stem
    print(f"  Processing {label}...", flush=True)

    with open(prompt_file) as f:
        prompt = f.read()

    # For now, return a marker that the caller can use to trigger Gemini
    return {"label": label, "prompt": prompt, "status": "pending"}


def process_batch_directory(batch_dir: Path) -> dict[str, Any]:
    """Process all prompts in a batch directory."""
    prompt_files = sorted(batch_dir.glob("*.txt"))

    batch_queries = []

    for prompt_file in prompt_files:
        result = process_batch_file(prompt_file)
        batch_queries.append(result)

    return {
        "batch_num": int(batch_dir.name.split("_")[1]),
        "queries": batch_queries,
        "count": len(batch_queries),
    }


def main():
    """Main processing function."""
    if len(sys.argv) < 2:
        print("Usage: process_gemini_batches.py <batches_dir> [batch_numbers...]")
        print("Example: process_gemini_batches.py /tmp/gemini_batches 003 004 005")
        sys.exit(1)

    batches_dir = Path(sys.argv[1])
    batch_nums = sys.argv[2:] if len(sys.argv) > 2 else []

    if not batch_nums:
        # Process all batches
        batch_dirs = sorted(batches_dir.glob("batch_*"))
    else:
        # Process specified batches
        batch_dirs = [batches_dir / f"batch_{num}" for num in batch_nums]

    all_queries = []

    for batch_dir in batch_dirs:
        if not batch_dir.exists():
            print(f"⚠️  Batch directory not found: {batch_dir}")
            continue

        print(f"\n{'=' * 80}")
        print(f"Batch {batch_dir.name}")
        print(f"{'=' * 80}")

        batch_data = process_batch_directory(batch_dir)
        all_queries.extend(batch_data["queries"])

        print(f"\n✓ Prepared {batch_data['count']} queries from {batch_dir.name}")

    # Output all queries as JSON for the caller to process
    output_file = batches_dir / "pending_queries.json"
    with open(output_file, "w") as f:
        json.dump(all_queries, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✅ Prepared {len(all_queries)} queries total")
    print(f"📝 Queries saved to: {output_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
