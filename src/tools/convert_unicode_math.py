#!/usr/bin/env python3
"""
Convert Unicode mathematical characters to native LaTeX in markdown files.
This script does FULL replacements of recurring expressions only.

Converts Unicode symbols like β, ε, ², ≈ to \beta, \varepsilon, ^{2}, \approx
while keeping the expressions in their original delimiters (` or $).
"""

from pathlib import Path
import sys


# Unicode to LaTeX mapping
UNICODE_TO_LATEX = {
    # Superscripts
    "²": r"^{2}",
    "³": r"^{3}",
    # Comparison operators
    "≈": r"\approx",
    "≤": r"\leq",
    "≥": r"\geq",
    "±": r"\pm",
    "≠": r"\neq",
    "≡": r"\equiv",
    # Set operations
    "∈": r"\in",
    "∉": r"\notin",
    "⊆": r"\subseteq",
    "⊇": r"\supseteq",
    "∪": r"\cup",
    "∩": r"\cap",
    # Arithmetic
    "×": r"\times",
    "÷": r"\div",
    # Special symbols
    "∞": r"\infty",
    "∂": r"\partial",
    "∇": r"\nabla",
    "∫": r"\int",
    "∑": r"\sum",
    "∏": r"\prod",
    "√": r"\sqrt",
    "∝": r"\propto",
    # Greek lowercase
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\varepsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    "φ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    # Greek uppercase
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Υ": r"\Upsilon",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
}


def convert_unicode_to_latex(text):
    """Convert Unicode math symbols to LaTeX equivalents."""
    result = text
    for unicode_char, latex in UNICODE_TO_LATEX.items():
        result = result.replace(unicode_char, latex)
    return result


def process_file(input_path, output_path=None, dry_run=False):
    """
    Process markdown file, converting Unicode math to LaTeX.

    Args:
        input_path: Path to input markdown file
        output_path: Path to output file (default: overwrite input)
        dry_run: If True, print changes without modifying file

    Returns:
        Number of substitutions made
    """
    with open(input_path, encoding="utf-8") as f:
        content = f.read()

    original_content = content
    converted_content = convert_unicode_to_latex(content)

    # Count changes
    changes = sum(1 for a, b in zip(original_content, converted_content) if a != b)

    if dry_run:
        if changes > 0:
            print(f"Would make {changes} character substitutions")
            # Show first few examples
            print("\nExample changes:")
            for unicode_char, latex in UNICODE_TO_LATEX.items():
                count = original_content.count(unicode_char)
                if count > 0:
                    print(f"  {unicode_char!r} → {latex!r} ({count} occurrences)")
        else:
            print("No changes needed")
        return changes

    if changes > 0:
        output = output_path or input_path
        with open(output, "w", encoding="utf-8") as f:
            f.write(converted_content)
        print(f"Made {changes} character substitutions in {output}")
    else:
        print("No changes needed")

    return changes


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python convert_unicode_math.py <input_file> [output_file] [--dry-run]")
        print("\nConverts Unicode mathematical characters to LaTeX equivalents.")
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
