#!/usr/bin/env env python
"""Convert mermaid code blocks to Jupyter Book directives.

This script transforms GitHub-flavored markdown mermaid blocks (```mermaid)
into Jupyter Book MyST directive format (:::mermaid) for proper rendering.

Usage:
    python convert_mermaid_blocks.py <input_file> <output_file>
    python convert_mermaid_blocks.py docs/source --in-place  # Process directory
"""

import re
import sys
from pathlib import Path


def convert_mermaid_blocks(content: str) -> str:
    """Convert mermaid blocks to MyST mermaid directive format.

    Converts both ```mermaid and :::mermaid blocks to the correct
    ```{mermaid} MyST directive format for sphinxcontrib-mermaid.

    Args:
        content: Markdown file content

    Returns:
        Content with converted mermaid blocks
    """
    # Pattern 1: Match ```mermaid blocks
    pattern_backticks = r"```mermaid\n(.*?)```"

    # Pattern 2: Match :::mermaid blocks
    pattern_colons = r":::mermaid\n(.*?):::"

    def replace_block(match):
        mermaid_code = match.group(1).rstrip("\n")
        # Convert to MyST directive format for sphinxcontrib-mermaid
        # Using the ```{mermaid} directive syntax
        return f"```{{mermaid}}\n{mermaid_code}\n```"

    # Replace all occurrences of both patterns
    converted = re.sub(pattern_backticks, replace_block, content, flags=re.DOTALL)
    converted = re.sub(pattern_colons, replace_block, converted, flags=re.DOTALL)

    return converted


def process_file(input_path: Path, output_path: Path | None = None) -> None:
    """Process a single markdown file.

    Args:
        input_path: Path to input markdown file
        output_path: Path to output file (None for in-place editing)
    """
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Read input file
    content = input_path.read_text(encoding="utf-8")

    # Convert mermaid blocks
    converted = convert_mermaid_blocks(content)

    # Write output
    if output_path is None:
        output_path = input_path

    output_path.write_text(converted, encoding="utf-8")
    print(f"Converted: {input_path} -> {output_path}")


def process_directory(directory: Path, in_place: bool = True) -> None:
    """Process all markdown files in a directory recursively.

    Args:
        directory: Path to directory containing markdown files
        in_place: If True, modify files in place; if False, skip (would need output dir)
    """
    if not directory.exists() or not directory.is_dir():
        print(f"Error: Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)

    # Find all markdown files
    md_files = list(directory.rglob("*.md"))

    if not md_files:
        print(f"No markdown files found in {directory}")
        return

    print(f"Processing {len(md_files)} markdown files...")

    for md_file in md_files:
        try:
            process_file(md_file, output_path=None if in_place else md_file)
        except Exception as e:
            print(f"Error processing {md_file}: {e}", file=sys.stderr)


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_arg = Path(sys.argv[1])

    # Check if processing directory
    if input_arg.is_dir():
        in_place = "--in-place" in sys.argv
        if not in_place:
            print("Warning: Directory processing requires --in-place flag")
            print(__doc__)
            sys.exit(1)
        process_directory(input_arg, in_place=True)
    else:
        # Single file processing
        output_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        process_file(input_arg, output_arg)


if __name__ == "__main__":
    main()
