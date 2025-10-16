#!/usr/bin/env python3
"""
Fix LaTeX math formatting issues in markdown files.
Ensures:
1. Block equations use $$ on separate lines
2. Blank line before $$ blocks
3. Proper spacing around math
"""

from pathlib import Path
import re
import sys


def fix_single_line_display_math(content):
    """
    Convert single-line $$equation$$ to multi-line format.
    $$equation$$ -> \n$$\nequation\n$$
    """
    # Find all single-line display math (no newlines between $$)
    pattern = r"\$\$([^\n$]+)\$\$"

    def replace_func(match):
        equation = match.group(1).strip()
        return f"\n$$\n{equation}\n$$"

    return re.sub(pattern, replace_func, content)


def add_blank_line_before_display_math(content):
    """
    Ensure there's a blank line before $$ blocks.
    Handles cases like:
    - text\n$$ -> text\n\n$$
    But preserves:
    - \n\n$$ (already has blank line)
    - start of file
    - after block elements
    """
    lines = content.split("\n")
    result_lines = []

    for i, line in enumerate(lines):
        # Check if current line starts a display math block
        if line.strip() == "$$":
            # Check if we need to add blank line before
            if i > 0:
                prev_line = result_lines[-1] if result_lines else ""
                # Don't add blank line if:
                # - previous line is already empty
                # - previous line is a block element marker
                # - we're in a list or indented block
                if prev_line.strip() != "" and not line.startswith("   "):
                    # Add blank line
                    result_lines.append("")

        result_lines.append(line)

    return "\n".join(result_lines)


def process_file(input_path, output_path=None, dry_run=False):
    """
    Fix math formatting in markdown file.

    Args:
        input_path: Path to input markdown file
        output_path: Path to output file (default: overwrite input)
        dry_run: If True, print changes without modifying file

    Returns:
        Number of fixes made
    """
    with open(input_path, encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Apply fixes
    print("Fixing single-line display math...")
    content = fix_single_line_display_math(content)

    print("Adding blank lines before display math...")
    content = add_blank_line_before_display_math(content)

    # Count changes
    if content != original_content:
        # Count specific changes
        original_single_line = len(re.findall(r"\$\$[^\n]+\$\$", original_content))
        new_single_line = len(re.findall(r"\$\$[^\n]+\$\$", content))
        single_line_fixed = original_single_line - new_single_line

        original_lines = len(original_content.split("\n"))
        new_lines = len(content.split("\n"))
        lines_added = new_lines - original_lines

        print("\nChanges:")
        print(f"  - Fixed {single_line_fixed} single-line display math blocks")
        print(f"  - Added {lines_added} blank lines")

        if dry_run:
            print("\nDRY RUN - No changes written")
            # Show a sample
            print("\nSample of first difference:")
            orig_lines = original_content.split("\n")
            new_lines = content.split("\n")
            for i, (o, n) in enumerate(zip(orig_lines[:100], new_lines[:100])):
                if o != n:
                    print(f"Line {i + 1}:")
                    print(f"  Before: {o[:80]}")
                    print(f"  After:  {n[:80]}")
                    break
        else:
            output = output_path or input_path
            with open(output, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"\nWrote changes to {output}")

        return single_line_fixed + lines_added
    print("No changes needed")
    return 0


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python fix_math_formatting.py <input_file> [output_file] [--dry-run]")
        print("\nFixes LaTeX math formatting issues:")
        print("  - Converts single-line $$...$$ to multi-line format")
        print("  - Adds blank lines before $$ blocks")
        print("\nIf output_file is not specified, modifies input file in-place.")
        print("Use --dry-run to preview changes without modifying files.")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = None
    dry_run = False

    for arg in sys.argv[2:]:
        if arg == "--dry-run":
            dry_run = True
        else:
            output_file = arg

    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    process_file(input_file, output_file, dry_run)


if __name__ == "__main__":
    main()
