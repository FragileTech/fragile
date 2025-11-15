#!/usr/bin/env python3
import re
from collections import Counter

# Read 02_euclidean_gas.md
with open("docs/source/1_euclidean_gas/02_euclidean_gas.md") as f:
    content = f.read()

# Extract existing references
existing_refs = set(re.findall(r'\{prf:ref\}`([^`]+)`', content))

print("Existing framework references in 02_euclidean_gas.md:")
framework_refs = [ref for ref in existing_refs if ref.startswith(('def-', 'axiom-', 'thm-', 'lem-'))]
for ref in sorted(framework_refs)[:30]:
    print(f"  - {ref}")

print(f"\nTotal framework references: {len(framework_refs)}")

# Count mentions of key framework concepts WITHOUT references
concepts_to_check = [
    ("walker", r'\bwalker\b'),
    ("swarm", r'\bswarm\b(?! state)'),  # Exclude "swarm state"
    ("alive set", r'\balive set\b'),
    ("cemetery", r'\bcemetery\b'),
    ("axiom", r'\baxiom\b'),
    ("guaranteed revival", r'\bguaranteed revival\b'),
    ("boundary regularity", r'\bboundary regularity\b'),
]

print("\nConcept mentions (excluding those already with {prf:ref}):")
for concept, pattern in concepts_to_check:
    # Find lines with concept but WITHOUT {prf:ref}
    lines = content.split('\n')
    count = 0
    for line in lines:
        if re.search(pattern, line, re.IGNORECASE) and '{prf:ref}' not in line:
            count += 1
    print(f"  {concept}: {count} unlinked mentions")

