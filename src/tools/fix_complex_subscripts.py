#!/usr/bin/env python3
"""
Fix complex subscript notation in markdown files.
Converts patterns like V_Var,x and V_x_struct in backticks to proper LaTeX.
"""

from pathlib import Path
import re
import sys


# Common variable patterns that need proper LaTeX formatting
SUBSCRIPT_PATTERNS = [
    # Variance components
    (r"V_Var,x", r"V_{\text{Var},x}"),
    (r"V_Var,v", r"V_{\text{Var},v}"),
    (r"V_Var", r"V_{\text{Var}}"),
    # Structural components
    (r"V_x_struct", r"V_{x,\text{struct}}"),
    (r"V_W", r"V_W"),  # Keep as is - single letter subscript
    # Other common patterns
    (r"W_b", r"W_b"),  # Keep as is
    (r"k_{\text{alive}}", r"k_{\text{alive}}"),  # Already correct
    (r"c_V\*V_Var", r"c_V \cdot V_{\text{Var}}"),  # Fix multiplication
]


def fix_inline_code_subscripts(content):
    """
    Find inline code blocks with complex subscripts and convert them to math mode.
    Converts `V_Var,x` to $V_{\text{Var},x}$

    IMPORTANT: Must protect fenced code blocks first!
    """

    # First, protect ALL fenced code blocks
    protected_blocks = []

    def protect_fenced(match):
        protected_blocks.append(match.group(0))
        return f"__FENCED_{len(protected_blocks) - 1}__"

    content = re.sub(r"```.*?```", protect_fenced, content, flags=re.DOTALL)

    # Now process inline code (backticks) in the remaining content
    def replace_backtick(match):
        text = match.group(1)

        # Skip if it's a reference or code identifier
        if text.startswith(("def-", "eq-", "label:", "http", "www")):
            return match.group(0)

        # Skip simple code variables (no complex subscripts)
        if not any(pattern in text for pattern, _ in SUBSCRIPT_PATTERNS):
            return match.group(0)

        # Apply subscript fixes
        result = text
        for pattern, replacement in SUBSCRIPT_PATTERNS:
            result = result.replace(pattern, replacement)

        # Convert to math mode
        return f"${result}$"

    # Process inline code
    content = re.sub(r"`([^`]+)`", replace_backtick, content)

    # Restore protected fenced blocks
    for i, block in enumerate(protected_blocks):
        content = content.replace(f"__FENCED_{i}__", block)

    return content


def fix_text_subscripts(content):
    """
    Fix complex subscript patterns that appear in regular text (not in code or math).
    This is more conservative - only fixes clear mathematical variable references.
    """

    # Protect existing math and code blocks
    protected = []

    def protect(match):
        protected.append(match.group(0))
        return f"__PROTECTED_{len(protected) - 1}__"

    # Protect inline math
    content = re.sub(r"\$[^$]+\$", protect, content)

    # Protect display math
    content = re.sub(r"\$\$.*?\$\$", protect, content, flags=re.DOTALL)

    # Protect inline code
    content = re.sub(r"`[^`]+`", protect, content)

    # Protect fenced code blocks
    content = re.sub(r"```.*?```", protect, content, flags=re.DOTALL)

    # Now fix patterns in unprotected text
    # Be conservative - only fix obvious standalone variable references
    # Pattern: word boundary, variable, word boundary or punctuation
    for pattern, replacement in SUBSCRIPT_PATTERNS:
        # Match pattern when it appears as standalone or before punctuation/space
        content = re.sub(r"\b" + re.escape(pattern) + r"(?=[,\.\)\s:]|\b)", replacement, content)

    # Restore protected content
    for i, block in enumerate(protected):
        content = content.replace(f"__PROTECTED_{i}__", block)

    return content


def process_file(input_path, output_path=None, dry_run=False):
    """
    Process markdown file, fixing complex subscript notation.

    Args:
        input_path: Path to input markdown file
        output_path: Path to output file (default: overwrite input)
        dry_run: If True, print changes without modifying file

    Returns:
        Number of changes made
    """
    with open(input_path, encoding="utf-8") as f:
        content = f.read()

    original_content = content

    print("Fixing inline code subscripts...")
    content = fix_inline_code_subscripts(content)

    # Count changes
    changes = 0
    if content != original_content:
        # Count pattern occurrences
        for pattern, replacement in SUBSCRIPT_PATTERNS:
            orig_count = original_content.count(f"`{pattern}`")
            content.count(f"${replacement}$")
            if orig_count > 0:
                changes += orig_count
                print(f"  Converted {orig_count}× `{pattern}` → ${replacement}$")

    if dry_run:
        if changes > 0:
            print(f"\nWould make {changes} conversions")

            # Show first few examples
            print("\nFirst few examples:")
            lines_orig = original_content.split("\n")
            lines_new = content.split("\n")
            examples = 0
            for i, (orig, new) in enumerate(zip(lines_orig, lines_new), 1):
                if orig != new and examples < 5:
                    print(f"\nLine {i}:")
                    print(f"  Before: {orig[:100]}")
                    print(f"  After:  {new[:100]}")
                    examples += 1
        else:
            print("No changes needed")
    elif changes > 0:
        output = output_path or input_path
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\nWrote {changes} changes to {output}")
    else:
        print("No changes needed")

    return changes


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python fix_complex_subscripts.py <input_file> [output_file] [--dry-run]")
        print("\nFixes complex subscript notation in markdown files:")
        print("  `V_Var,x` → $V_{\\text{Var},x}$")
        print("  `V_Var,v` → $V_{\\text{Var},v}$")
        print("  `V_x_struct` → $V_{x,\\text{struct}}$")
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
