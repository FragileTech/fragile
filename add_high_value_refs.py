#!/usr/bin/env python3
"""
Add high-value backward cross-references targeting source-only entities.

Strategy: Add incoming references TO source-only entities from earlier parts of the document.
This converts them from source-only to bidirectional, improving overall connectivity.
"""

import re
from pathlib import Path

DOC_PATH = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

# High-value targets: source-only axioms and definitions that should receive incoming refs
HIGH_VALUE_TARGETS = {
    # Axioms that are frequently used but not referenced enough
    "axiom-boundary-smoothness": ["boundary", "smooth", "C^1"],
    "axiom-reward-regularity": ["reward", "Lipschitz", "L_R"],
    "axiom-environmental-richness": ["richness", "environment"],
    "axiom-sufficient-amplification": ["amplification"],
    "axiom-geometric-consistency": ["geometric consistency"],

    # Definitions heavily used in later theorems
    "def-raw-value-operator": ["raw value", "measurement operator"],
    "def-cloning-probability-function": ["cloning probability", "clone"],
    "def-perturbation-fluctuation-bounds-reproof": ["perturbation bound", "fluctuation"],
    "def-expected-cloning-action": ["expected cloning"],
    "def-displacement-components": ["displacement component"],
}

print(f"Reading: {DOC_PATH}")
with open(DOC_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

original = content
mods = 0

# For each high-value target, find natural places to reference it
for target_label, keywords in HIGH_VALUE_TARGETS.items():
    # Find where the target is defined
    target_match = re.search(rf":::\{{prf:\w+\}}.*?\n:label:\s*{target_label}", content)
    if not target_match:
        print(f"Warning: Could not find definition for {target_label}")
        continue

    target_pos = target_match.start()

    # Find mentions of keywords AFTER the definition that don't already have a reference
    for keyword in keywords:
        # Look for keyword mentions after the target definition
        pattern = rf"(?<![`{{])({re.escape(keyword)})(?![`}}])"

        for match in re.finditer(pattern, content[target_pos:], re.IGNORECASE):
            actual_pos = target_pos + match.start()

            # Check if this location already has a reference to our target
            context_start = max(0, actual_pos - 100)
            context_end = min(len(content), actual_pos + 100)
            context = content[context_start:context_end]

            if target_label in context:
                continue  # Already referenced

            # Check if we're in a theorem/lemma statement (not in a proof)
            # Find the nearest directive before this point
            before_text = content[max(0, actual_pos - 500):actual_pos]
            if ":::{prf:theorem}" in before_text or ":::{prf:lemma}" in before_text:
                # Check we're not in a proof
                if ":::{prf:proof}" not in before_text.split(":::{prf:")[-1]:
                    # Good candidate! Add reference
                    old_text = match.group(1)
                    new_text = f"{old_text} ({{prf:ref}}`{target_label}`)"

                    # Replace only this specific occurrence
                    before = content[:actual_pos]
                    at_match = content[actual_pos:actual_pos + len(old_text)]
                    after = content[actual_pos + len(old_text):]

                    content = before + new_text + after[len(old_text):]
                    mods += 1
                    print(f"Added ref to {target_label} (keyword: '{keyword}')")

                    if mods >= 40:  # Limit to avoid over-referencing
                        break

        if mods >= 40:
            break

    if mods >= 40:
        break

# Also add some strategic references for commonly used theorems/lemmas
STRATEGIC_REFS = [
    # In McDiarmid's inequality applications, reference the theorem
    (r"(bounded differences)", "{prf:ref}`thm-mcdiarmids-inequality`",
     "McDiarmid application"),

    # In standardization discussions, reference the operator
    (r"(standardization operator)(?!\s*\({{prf:ref}})", "{prf:ref}`def-standardization-operator-n-dimensional`",
     "Standardization operator"),

    # In status update discussions
    (r"(status update operator)(?!\s*\({{prf:ref}})", "{prf:ref}`def-status-update-operator`",
     "Status update operator"),

    # In perturbation discussions
    (r"(perturbation operator)(?!\s*\({{prf:ref}})", "{prf:ref}`def-perturbation-operator`",
     "Perturbation operator"),
]

for pattern, ref, description in STRATEGIC_REFS:
    matches = list(re.finditer(pattern, content, re.IGNORECASE))
    count = 0
    for match in matches:
        # Check if reference not already present nearby
        start = max(0, match.start() - 50)
        end = min(len(content), match.end() + 50)
        if ref not in content[start:end]:
            content = content[:match.end()] + f" ({ref})" + content[match.end():]
            count += 1
            mods += 1
            if count >= 3:  # Limit per pattern
                break

    if count > 0:
        print(f"Added {count} refs for {description}")

print(f"\nTotal modifications: {mods}")

if content != original:
    print(f"Writing to {DOC_PATH}...")
    with open(DOC_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Done!")
else:
    print("No changes made.")
