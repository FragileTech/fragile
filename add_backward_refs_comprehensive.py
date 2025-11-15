#!/usr/bin/env python3
"""
Comprehensively add backward cross-references to improve connectivity.

Focus areas based on connectivity report:
1. Convert source-only entities (147) to bidirectional by adding incoming refs
2. Strengthen axiom → theorem connections
3. Add definition → lemma/theorem connections
4. Increase bidirectional entities from 47 to 100+
"""

import re
from pathlib import Path

DOC_PATH = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

print(f"Reading: {DOC_PATH}")
with open(DOC_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()

modifications = []

# Track changes
def add_ref_after_pattern(line_idx, pattern, ref_text, description):
    """Add a reference after finding a pattern in a line"""
    global modifications
    line = lines[line_idx]
    if re.search(pattern, line):
        # Check if reference already exists
        if ref_text not in line:
            # Add the reference
            new_line = re.sub(pattern, lambda m: m.group(0) + f" ({ref_text})", line, count=1)
            if new_line != line:
                lines[line_idx] = new_line
                modifications.append((line_idx+1, description))
                return True
    return False

# STRATEGY 1: Add axiom references in theorem statements that rely on them

# Axiom of Boundary Regularity
for i, line in enumerate(lines):
    if "death probability" in line and "continuous function" in line:
        if add_ref_after_pattern(i, r"continuous function", "{prf:ref}`axiom-boundary-regularity`",
                                  "Axiom of Boundary Regularity ref in continuity discussion"):
            break

# Axiom of Guaranteed Revival
for i, line in enumerate(lines):
    if "revival mechanism" in line.lower() and "preventing" in line:
        if add_ref_after_pattern(i, r"revival mechanism", "{prf:ref}`axiom-guaranteed-revival`",
                                  "Axiom of Guaranteed Revival ref in revival discussion"):
            break

# Axiom of Bounded Measurement Variance
for i, line in enumerate(lines):
    if "variance of the raw value measurement" in line.lower():
        if add_ref_after_pattern(i, r"measurement process", "{prf:ref}`axiom-bounded-measurement-variance`",
                                  "Axiom of Bounded Measurement Variance ref"):
            break

# STRATEGY 2: Add definition references in theorems/lemmas that use them

# def-walker references
for i, line in enumerate(lines):
    if re.search(r"walker.*position.*status", line, re.IGNORECASE) and "walker" not in line.lower()+"({prf:ref}":
        # Skip if already has reference
        if "{prf:ref}`def-walker`" not in line:
            # Add reference to first occurrence of "walker" in definitions
            new_line = re.sub(r"\bwalker\b", "walker ({prf:ref}`def-walker`)", line, count=1)
            if new_line != line and i < 300:  # Only in early sections
                lines[i] = new_line
                modifications.append((i+1, "Added def-walker ref in foundational section"))

# def-alive-dead-sets references
for i, line in enumerate(lines):
    if "alive set" in line.lower() and "mathcal{A}" in line:
        if "{prf:ref}`def-alive-dead-sets`" not in line and "{prf:ref}`def-alive-dead-sets`" not in lines[max(0,i-2):i+3]:
            # Add reference
            new_line = re.sub(r"alive set(?!\s*\()", "alive set ({prf:ref}`def-alive-dead-sets`)", line, count=1)
            if new_line != line:
                lines[i] = new_line
                modifications.append((i+1, "Added def-alive-dead-sets ref"))
                if len(modifications) > 50:  # Limit to avoid over-referencing
                    break

# def-swarm-and-state-space references
for i, line in enumerate(lines):
    if re.search(r"swarm\s+state.*\$\\mathcal\{S\}", line, re.IGNORECASE):
        if "{prf:ref}`def-swarm-and-state-space`" not in line and i > 200:  # After definition
            new_line = re.sub(r"swarm\s+state(?!\s*\()", "swarm state ({prf:ref}`def-swarm-and-state-space`)", line, count=1)
            if new_line != line:
                lines[i] = new_line
                modifications.append((i+1, "Added def-swarm-and-state-space ref"))
                if len(modifications) > 70:
                    break

# STRATEGY 3: Add axiom references in axiom descriptions

# Link axioms to each other when they're related
axiom_relationships = [
    (r"Axiom of Boundary Regularity", r"valid domain", "{prf:ref}`def-valid-state-space`"),
    (r"Axiom of Non-Degenerate Noise", r"perturbation", "{prf:ref}`def-perturbation-measure`"),
    (r"Axiom of Bounded Algorithmic Diameter", r"algorithmic space", "{prf:ref}`def-algorithmic-space-generic`"),
]

for axiom_name, context_pattern, ref_to_add in axiom_relationships:
    in_axiom = False
    for i, line in enumerate(lines):
        if axiom_name in line:
            in_axiom = True
        if in_axiom and context_pattern in line.lower():
            if ref_to_add not in line:
                new_line = re.sub(context_pattern, lambda m: m.group(0) + f" ({ref_to_add})", line, count=1, flags=re.IGNORECASE)
                if new_line != line:
                    lines[i] = new_line
                    modifications.append((i+1, f"Added cross-reference in {axiom_name}"))
                    break
        if in_axiom and ":::" in line and "prf:" in line and axiom_name not in line:
            in_axiom = False

# STRATEGY 4: Add lemma → lemma references in proofs

# Single-walker error lemmas
lemma_dependencies = [
    ("lem-total-squared-error-stable", ["lem-single-walker-positional-error", "lem-single-walker-structural-error"]),
    ("thm-distance-operator-mean-square-continuity", ["lem-total-squared-error-stable", "lem-total-squared-error-unstable"]),
    ("thm-deterministic-potential-continuity", ["lem-component-potential-lipschitz"]),
]

for theorem_label, dependency_labels in lemma_dependencies:
    in_theorem = False
    for i, line in enumerate(lines):
        if f"label: {theorem_label}" in line:
            in_theorem = True
        if in_theorem and i < len(lines) - 1:
            # Check if we're in the statement (before proof)
            if ":::{prf:proof}" in line:
                break
            # Add references to dependencies if not present
            for dep_label in dependency_labels:
                if dep_label.replace("-", " ") in line.lower() and f"{{prf:ref}}`{dep_label}`" not in line:
                    # Try to add reference
                    pattern = dep_label.replace("-", "[-\\s]")
                    new_line = re.sub(pattern, f"{{prf:ref}}`{dep_label}`", line, count=1, flags=re.IGNORECASE)
                    if new_line != line:
                        lines[i] = new_line
                        modifications.append((i+1, f"Added {dep_label} ref in {theorem_label}"))

# STRATEGY 5: Add def-algorithmic-space-generic references
for i, line in enumerate(lines):
    if "algorithmic space" in line.lower() and "$D_{\\mathcal{Y}}$" in line:
        if "{prf:ref}`def-algorithmic-space-generic`" not in line and i > 250:
            new_line = re.sub(r"algorithmic space(?!\s*\()", "algorithmic space ({prf:ref}`def-algorithmic-space-generic`)", line, count=1)
            if new_line != line and len(modifications) < 100:
                lines[i] = new_line
                modifications.append((i+1, "Added def-algorithmic-space-generic ref"))

print(f"\nTotal modifications: {len(modifications)}")
print("\nFirst 20 modifications:")
for line_no, desc in modifications[:20]:
    print(f"  Line {line_no}: {desc}")

if modifications:
    print(f"\nWriting changes to {DOC_PATH}...")
    with open(DOC_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Done!")
    print(f"\nSummary: Added {len(modifications)} backward cross-references")
else:
    print("\nNo modifications made.")
