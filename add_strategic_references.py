#!/usr/bin/env python3
"""
Add strategic backward cross-references to improve document connectivity.

This script identifies opportunities for adding backward references and
applies them systematically based on connectivity patterns.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Document path
DOC_PATH = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

# Strategic reference additions based on connectivity analysis
# Format: (line_number_approx, search_pattern, replacement_pattern)
REFERENCE_ADDITIONS = [
    # Theorem referencing axioms it depends on
    (
        "Axiom of Boundary Regularity",
        r"(The stability of this process depends on how erratically the \"death probability\")",
        r"\1 (see {prf:ref}`axiom-boundary-regularity`)"
    ),
    (
        "Axiom of Guaranteed Revival in theorem statement",
        r"(Assume the global constraint.*from the Axiom of Guaranteed Revival)",
        r"\1 ({prf:ref}`axiom-guaranteed-revival`)"
    ),
    # Theorems referencing definitions
    (
        "N-Particle Displacement Metric in axiom",
        r"(where \$d_\{\\text\{Disp\},\\mathcal\{Y\}\}\$ is the N-Particle Displacement Metric)\.",
        r"\1 ({prf:ref}`def-n-particle-displacement-metric`)."
    ),
    # Lemmas referencing earlier lemmas
    (
        "Single-walker error lemmas in decomposition",
        r"(This lemma decomposes the stable walker error)",
        r"\1 using single-walker error bounds from {prf:ref}`lem-single-walker-positional-error` and {prf:ref}`lem-single-walker-structural-error`"
    ),
]

# Read the document
print(f"Reading document: {DOC_PATH}")
with open(DOC_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

original_content = content
modifications = 0

# Apply strategic reference additions
for description, pattern, replacement in REFERENCE_ADDITIONS:
    matches = list(re.finditer(pattern, content))
    if matches:
        print(f"Found {len(matches)} matches for: {description}")
        content = re.sub(pattern, replacement, content)
        modifications += len(matches)
    else:
        print(f"No matches for: {description}")

# Additional targeted fixes based on connectivity report
# Add references to foundational axioms in key theorems

# Add reference to axiom-non-degenerate-noise in perturbation theorems
content = re.sub(
    r"(:::\{prf:theorem\} Probabilistic Continuity of the Perturbation Operator\n:label: thm-perturbation-operator-continuity-reproof\n)",
    r"\1\nThis theorem relies on the Axiom of Bounded Second Moment of Perturbation ({prf:ref}`axiom-non-degenerate-noise`) to bound perturbation-induced displacement.\n",
    content
)

# Add reference to def-valid-state-space in boundary axioms
content = re.sub(
    r"(The boundary of the valid domain, \$\\partial \\mathcal\{X\}_\{\\mathrm\{valid\}\}\$)",
    r"\1 ({prf:ref}`def-valid-state-space`)",
    content
)

# Add reference to def-companion-selection-measure in distance measurement
content = re.sub(
    r"(is sampled from the \*\*Companion Selection Measure\*\* \$\\mathbb\{C\}_i\$\.)",
    r"is sampled from the **Companion Selection Measure** $\\mathbb{C}_i$ ({prf:ref}`def-companion-selection-measure`).",
    content
)

if content != original_content:
    print(f"\n Total modifications: {modifications + 3}")
    print(f"Writing updated document to: {DOC_PATH}")
    with open(DOC_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Done!")
else:
    print("\nNo modifications were made.")
