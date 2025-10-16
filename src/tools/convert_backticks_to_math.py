#!/usr/bin/env python3
"""
Convert inline backtick math expressions to dollar sign math.
Only converts expressions that contain LaTeX commands.
Preserves non-math code blocks and references.
"""

from pathlib import Path
import re
import sys


# LaTeX commands that indicate mathematical content
LATEX_COMMANDS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Upsilon",
    "Phi",
    "Psi",
    "Omega",
    "approx",
    "leq",
    "geq",
    "in",
    "notin",
    "cup",
    "cap",
    "times",
    "div",
    "sum",
    "prod",
    "int",
    "infty",
    "partial",
    "nabla",
    "pm",
    "neq",
    "equiv",
    "subseteq",
    "supseteq",
    "sqrt",
    "propto",
    "frac",
    "text",
}


def is_reference(text):
    """Check if text is a MyST reference (not math)."""
    return text.startswith(("def-", "eq-", "lem-", "thm-", "prf:", "ax:"))


def is_code_variable(text):
    """Check if text is a simple code variable or identifier."""
    # Simple identifiers without LaTeX
    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", text):
        return True
    # Simple comparisons/assignments without LaTeX
    if re.match(r"^[a-zA-Z_]\w*\s*[<>=!]+\s*[0-9a-zA-Z_]+$", text):
        if not any(f"\\{cmd}" in text for cmd in LATEX_COMMANDS):
            return True
    return False


def contains_latex(text):
    """Check if text contains LaTeX commands."""
    return any(f"\\{cmd}" in text for cmd in LATEX_COMMANDS)


def should_convert(text):
    """Determine if a backtick expression should be converted to math."""
    # Don't convert references
    if is_reference(text):
        return False

    # Don't convert simple code variables
    if is_code_variable(text):
        return False

    # Convert if it contains LaTeX commands
    if contains_latex(text):
        return True

    return False


def wrap_text_content(text):
    """Wrap plain text content (like 'Var') with \\text{}"""
    # Common function names that should be wrapped in \text
    text_patterns = [
        (r"\bVar\b", r"\\text{Var}"),
        (r"\bCov\b", r"\\text{Cov}"),
        (r"\bE\[", r"\\text{E}["),
        (r"\bchoose\b", r"\\text{choose}"),
    ]

    result = text
    for pattern, replacement in text_patterns:
        result = re.sub(pattern, replacement, result)

    return result


def process_line(line):
    """Process a single line, converting backtick math to dollar signs."""

    def replace_backtick(match):
        content = match.group(1)

        if should_convert(content):
            # Wrap text content appropriately
            converted = wrap_text_content(content)
            return f"${converted}$"
        # Keep as backticks
        return match.group(0)

    # Replace backtick expressions
    return re.sub(r"`([^`]+)`", replace_backtick, line)


def process_file(input_path, output_path=None, dry_run=False):
    """
    Process markdown file, converting backtick math to dollar signs.

    Args:
        input_path: Path to input markdown file
        output_path: Path to output file (default: overwrite input)
        dry_run: If True, print changes without modifying file

    Returns:
        Number of conversions made
    """
    with open(input_path, encoding="utf-8") as f:
        lines = f.readlines()

    converted_lines = []
    conversions = 0
    examples = []

    for line_num, line in enumerate(lines, 1):
        new_line = process_line(line)
        converted_lines.append(new_line)

        if new_line != line:
            conversions += 1
            if len(examples) < 10:
                # Extract the changed part for display
                examples.append((line_num, line.strip(), new_line.strip()))

    if dry_run:
        if conversions > 0:
            print(f"Would convert {conversions} lines")
            print("\nExample conversions:")
            for line_num, before, after in examples:
                print(f"\nLine {line_num}:")
                print(f"  Before: {before[:100]}")
                print(f"  After:  {after[:100]}")
        else:
            print("No conversions needed")
        return conversions

    if conversions > 0:
        output = output_path or input_path
        with open(output, "w", encoding="utf-8") as f:
            f.writelines(converted_lines)
        print(f"Converted {conversions} lines in {output}")
    else:
        print("No conversions needed")

    return conversions


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python convert_backticks_to_math.py <input_file> [output_file] [--dry-run]")
        print("\nConverts inline backtick math expressions to dollar signs.")
        print("Only converts expressions containing LaTeX commands.")
        print("If output_file is not specified, modifies input file in-place.")
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
