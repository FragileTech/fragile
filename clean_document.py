#!/usr/bin/env python3
"""
Script to clean AI review meta-commentary from mathematical documents.
"""

from pathlib import Path
import re


def clean_document(input_path: Path) -> str:
    """Clean meta-commentary from document while preserving mathematical content."""

    with open(input_path, encoding="utf-8") as f:
        content = f.read()

    # Remove status line
    content = re.sub(
        r"\*\*Status\*\*:.*?(?:addressing|dual review|feedback).*?\n",
        "",
        content,
        flags=re.IGNORECASE,
    )

    # Remove "REVISED VERSION" markers
    content = re.sub(r"REVISED VERSION \d+.*?\n", "", content)

    # Remove entire admonitions about dual review/corrections
    # Pattern: :::{ type} ... content mentioning review/fix ... :::
    patterns_to_remove = [
        # Admonitions mentioning reviews, corrections, fixes
        r":::\{(important|note|warning)\}[^\n]*?(?:Revision|Dual Review|Correction|Fixed|Issue|Review|Gemini|Codex).*?\n(?:.*?\n)*?:::\n",
        # Specific "Revisions from Dual Review" block
        r":::\{important\} Revisions from Dual Review.*?\n(?:.*?\n)*?:::\n",
        # "Addressing Dual Review Concern" blocks
        r":::\{important\} Addressing Dual Review Concern.*?\n(?:.*?\n)*?:::\n",
    ]

    for pattern in patterns_to_remove:
        content = re.sub(pattern, "", content, flags=re.DOTALL | re.MULTILINE)

    # Remove inline mentions of reviews/corrections
    inline_removals = [
        # "(REVISED)" markers
        r"\(REVISED[^\)]*\)",
        # "Fixed:" mentions
        r"\*\*Fixed\*\*:.*?\n",
        # Checkmarks and issue tracking
        r"✅\s*\*\*Fixed\*\*:.*?(?:\n|$)",
        # "identified by dual review" phrases
        r"\(identified by (?:dual review|Gemini|Codex)[^\)]*\)",
        # "REVISED - " prefixes in headings
        r"\(REVISED - [^\)]+\)",
    ]

    for pattern in inline_removals:
        content = re.sub(pattern, "", content, flags=re.MULTILINE)

    # Clean up specific problem areas
    # Remove "Correction from Dual Review" warning boxes but keep mathematical content
    # This is tricky - we need to extract mathematical content from mixed admonitions

    # Pattern for warning boxes with "Correction from Dual Review"
    correction_pattern = (
        r":::\{warning\} Correction from Dual Review\n(.*?)\n\nThe correct mechanism is (.*?)\n:::"
    )

    def replace_correction(match):
        # Extract only the mathematical explanation
        explanation = match.group(2)
        return f"The correct mechanism is {explanation}"

    content = re.sub(correction_pattern, replace_correction, content, flags=re.DOTALL)

    # Remove "Circularity Note" that mentions dual review
    content = re.sub(
        r"\*\*Circularity Note\*\*:.*?identified by the dual review\.\n",
        "",
        content,
        flags=re.DOTALL,
    )

    # Clean up multiple blank lines
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Remove "(REVISED)" from section headings
    return re.sub(r"\(REVISED[^\)]*\)\s*", "", content)


if __name__ == "__main__":
    input_file = Path(
        "/home/guillem/fragile/docs/source/1_euclidean_gas/11_hk_convergence_bounded_density_rigorous_proof.md"
    )
    output = clean_document(input_file)
    print(output)
