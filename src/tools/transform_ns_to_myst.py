#!/usr/bin/env python3
"""
Transform NS draft to professional PDF-ready paper with MyST directives.

This script:
1. Adds YAML frontmatter
2. Moves abstract to frontmatter
3. Converts mathematical environments to MyST directives
4. Converts proofs
5. Removes § symbols from headings
6. Replaces references with citations
7. Adds section labels
"""

import re
from pathlib import Path
from typing import List, Tuple


def extract_abstract(lines: List[str]) -> Tuple[str, List[str]]:
    """Extract abstract from document body and return (abstract, remaining_lines)."""
    abstract_start = None
    abstract_end = None

    for i, line in enumerate(lines):
        if line.strip() == "**Abstract**":
            abstract_start = i
        elif abstract_start is not None and line.strip() == "---":
            abstract_end = i
            break

    if abstract_start is None:
        return "", lines

    # Extract abstract content (skip **Abstract** header and separator)
    abstract_lines = lines[abstract_start + 2:abstract_end]
    abstract = "\n".join(abstract_lines).strip()

    # Remove abstract and header from document
    remaining = lines[:abstract_start] + lines[abstract_end + 1:]

    return abstract, remaining


def create_yaml_frontmatter(abstract: str) -> str:
    """Create YAML frontmatter for pandoc."""
    # Indent abstract for YAML
    abstract_indented = "  " + abstract.replace("\n", "\n  ")

    return f"""---
title: "Global Regularity for the 3D Navier-Stokes Equations via Variational Stratification and Gevrey Structural Stability"
author:
  - name: "Guillem Duran-Ballester"
    affiliation: "Institution"
    email: "your.email@institution.edu"
date: "\\\\today"
abstract: |
{abstract_indented}

# Page geometry
geometry:
  - margin=1in
  - letterpaper

# Font settings
fontsize: 11pt
fontfamily: mathpazo

# Math packages
header-includes: |
  \\usepackage{{amsmath}}
  \\usepackage{{amssymb}}
  \\usepackage{{amsthm}}
  \\usepackage{{mathtools}}
  \\usepackage{{bbm}}
  \\usepackage{{bm}}
  \\usepackage{{mathrsfs}}
  \\usepackage{{thmtools}}

  % Theorem environments
  \\newtheorem{{theorem}}{{Theorem}}[section]
  \\newtheorem{{lemma}}[theorem]{{Lemma}}
  \\newtheorem{{proposition}}[theorem]{{Proposition}}
  \\newtheorem{{corollary}}[theorem]{{Corollary}}
  \\theoremstyle{{definition}}
  \\newtheorem{{definition}}[theorem]{{Definition}}
  \\newtheorem{{assumption}}[theorem]{{Assumption}}
  \\newtheorem{{hypothesis}}[theorem]{{Hypothesis}}
  \\theoremstyle{{remark}}
  \\newtheorem{{remark}}[theorem]{{Remark}}

  % Custom commands
  \\newcommand{{\\RR}}{{\\mathbb{{R}}}}
  \\newcommand{{\\NN}}{{\\mathbb{{N}}}}
  \\DeclareMathOperator{{\\supp}}{{supp}}
  \\DeclareMathOperator{{\\dist}}{{dist}}

  % Spacing
  \\setlength{{\\parskip}}{{0.5em}}
  \\setlength{{\\parindent}}{{0pt}}

# Bibliography
bibliography: references.bib
link-citations: true

# Document class options
documentclass: article
classoption:
  - 11pt
  - letterpaper
  - twoside

# Table of contents
toc: true
toc-depth: 2
number-sections: true
---

"""


def remove_section_symbols(line: str) -> str:
    """Remove § symbols from section headings."""
    # Match patterns like "## 1." or "## §1." or "### 1.1." or "### §1.1."
    return re.sub(r'(#{1,4})\s*§', r'\1 ', line)


def convert_math_environment(line: str) -> Tuple[str, str, str]:
    """
    Convert bold mathematical environment to MyST directive.
    Returns (directive_type, title, label) or (None, None, None) if not a math env.
    """
    # Pattern: **Type Number (Title).**
    pattern = r'\*\*([A-Z][a-z]+)\s+([\d.]+)\s+\(([^)]+)\)\.\*\*'
    match = re.match(pattern, line.strip())

    if match:
        env_type = match.group(1).lower()  # theorem, lemma, etc.
        number = match.group(2)
        title = match.group(3)

        # Create label from title
        label_base = title.lower()
        label_base = re.sub(r'[^\w\s-]', '', label_base)
        label_base = re.sub(r'\s+', '-', label_base)

        # Map to directive type
        type_map = {
            'theorem': 'theorem',
            'lemma': 'lemma',
            'definition': 'definition',
            'proposition': 'proposition',
            'corollary': 'corollary',
            'assumption': 'assumption',
            'hypothesis': 'hypothesis',
            'remark': 'remark',
        }

        if env_type in type_map:
            directive = type_map[env_type]
            label = f"{directive[:3]}-{label_base}"
            return directive, title, label

    # Pattern without number: **Type (Title).**
    pattern2 = r'\*\*([A-Z][a-z]+)\s+\(([^)]+)\)\.\*\*'
    match2 = re.match(pattern2, line.strip())

    if match2:
        env_type = match2.group(1).lower()
        title = match2.group(2)

        label_base = title.lower()
        label_base = re.sub(r'[^\w\s-]', '', label_base)
        label_base = re.sub(r'\s+', '-', label_base)

        type_map = {
            'theorem': 'theorem',
            'lemma': 'lemma',
            'definition': 'definition',
            'proposition': 'proposition',
            'corollary': 'corollary',
            'assumption': 'assumption',
            'hypothesis': 'hypothesis',
            'remark': 'remark',
        }

        if env_type in type_map:
            directive = type_map[env_type]
            label = f"{directive[:3]}-{label_base}"
            return directive, title, label

    return None, None, None


def is_proof_start(line: str) -> bool:
    """Check if line starts a proof."""
    stripped = line.strip()
    return stripped.startswith("**Proof.**") or stripped == "*Proof.*" or stripped.startswith("*Proof.*")


def is_proof_end(line: str) -> bool:
    """Check if line ends a proof."""
    return "$\\hfill \\square$" in line or "$\\square$" in line or "□" in line


def replace_citations(line: str) -> str:
    """Replace [1], [2], etc. with [@cite]."""
    citation_map = {
        '[1]': '[@beale1984]',
        '[2]': '[@constantin1993]',
        '[3]': '[@moffatt1992]',
        '[4]': '[@tao2016]',
        '[5]': '[@luo2014]',
        '[6]': '[@escauriaza2003]',
        '[7]': '[@benjamin1962]',
        '[8]': '[@caffarelli1982]',
        '[9]': '[@lin1998]',
        '[10]': '[@naber2017]',
        '[11]': '[@seregin2012]',
        '[12]': '[@bianchi1991]',
        '[13]': '[@dolbeault2024]',
    }

    for old, new in citation_map.items():
        line = line.replace(old, new)

    return line


def add_section_labels(line: str, section_counter: List[int]) -> str:
    """Add section labels before headings."""
    # Match heading levels
    heading_match = re.match(r'^(#{1,4})\s+(\d+\.?\d*\.?\d*\.?)\s+(.+)$', line)

    if heading_match:
        level = len(heading_match.group(1))
        number = heading_match.group(2).rstrip('.')
        title = heading_match.group(3)

        # Create label
        label_base = title.lower()
        label_base = re.sub(r'[^\w\s-]', '', label_base)
        label_base = re.sub(r'\s+', '-', label_base)
        label_base = label_base[:50]  # Truncate if too long

        label = f"(sec-{label_base})="

        return f"{label}\n{line}"

    return line


def transform_document(input_path: Path, output_path: Path):
    """Transform the NS draft document."""
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print("Extracting abstract...")
    abstract, lines = extract_abstract(lines)

    print("Creating YAML frontmatter...")
    frontmatter = create_yaml_frontmatter(abstract)

    print("Processing document body...")
    output_lines = []
    in_proof = False
    in_math_env = False
    math_env_content = []
    proof_content = []
    current_directive = None
    current_label = None
    section_counter = [0, 0, 0, 0]

    i = 0
    while i < len(lines):
        line = lines[i]
        original_line = line

        # Remove section symbols
        line = remove_section_symbols(line)

        # Replace citations
        line = replace_citations(line)

        # Add section labels
        line = add_section_labels(line, section_counter)

        # Check for mathematical environment start
        if not in_proof and not in_math_env:
            directive, title, label = convert_math_environment(line)
            if directive:
                in_math_env = True
                current_directive = directive
                current_label = label
                math_env_content = []
                i += 1
                continue

        # Check for proof start
        if not in_proof and is_proof_start(line):
            in_proof = True
            proof_content = []
            # Skip the **Proof.** line
            i += 1
            continue

        # Handle proof content
        if in_proof:
            if is_proof_end(line):
                # End proof (remove QED marker)
                line_without_qed = re.sub(r'\$\\hfill\s*\\square\$', '', line)
                line_without_qed = re.sub(r'\$\\square\$', '', line_without_qed)
                line_without_qed = line_without_qed.replace('□', '')
                if line_without_qed.strip():
                    proof_content.append(line_without_qed)

                # Output proof
                output_lines.append(":::{prf:proof}\n")
                output_lines.extend(proof_content)
                output_lines.append(":::\n")
                output_lines.append("\n")

                in_proof = False
                proof_content = []
                i += 1
                continue
            else:
                proof_content.append(line)
                i += 1
                continue

        # Handle math environment content
        if in_math_env:
            # Check if we've reached end of environment (blank line or next bold heading)
            if line.strip() == '' and len(math_env_content) > 0:
                # Check if next non-empty line is a new environment
                next_is_env = False
                for j in range(i + 1, min(i + 3, len(lines))):
                    if lines[j].strip():
                        if convert_math_environment(lines[j])[0] is not None:
                            next_is_env = True
                        break

                if next_is_env or i == len(lines) - 1:
                    # End of environment
                    output_lines.append(f":::{{prf:{current_directive}}} {title}\n")
                    output_lines.append(f":label: {current_label}\n")
                    output_lines.append("\n")
                    output_lines.extend(math_env_content)
                    output_lines.append(":::\n")
                    output_lines.append("\n")

                    in_math_env = False
                    math_env_content = []
                    current_directive = None
                    current_label = None
                    i += 1
                    continue
                else:
                    math_env_content.append(line)
            else:
                # Check if this is a new section or environment
                if line.startswith('#') or convert_math_environment(line)[0] is not None:
                    # End current environment
                    output_lines.append(f":::{{prf:{current_directive}}} {title}\n")
                    output_lines.append(f":label: {current_label}\n")
                    output_lines.append("\n")
                    output_lines.extend(math_env_content)
                    output_lines.append(":::\n")
                    output_lines.append("\n")

                    in_math_env = False
                    math_env_content = []
                    current_directive = None
                    current_label = None
                    # Don't increment i, reprocess this line
                    continue
                else:
                    math_env_content.append(line)

        # Regular content
        if not in_proof and not in_math_env:
            output_lines.append(line)

        i += 1

    print("Writing output...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(frontmatter)
        f.writelines(output_lines)

    print(f"✓ Transformation complete: {output_path}")


if __name__ == "__main__":
    input_file = Path("docs/source/navier_stokes/ns_draft.md")
    output_file = Path("docs/source/navier_stokes/ns_draft_myst.md")

    transform_document(input_file, output_file)
    print("\nNext steps:")
    print(f"1. Review {output_file}")
    print("2. Compile with: pandoc ns_draft_myst.md -o ns_draft.pdf --bibliography=references.bib --citeproc --pdf-engine=pdflatex --number-sections --toc")
