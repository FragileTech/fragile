#!/usr/bin/env python3
"""
Strategic backward cross-reference enrichment for 02_euclidean_gas.md

Adds cross-references at key locations where concepts from 01_fragile_gas_framework.md
are first introduced or significantly used.
"""

import re
from pathlib import Path

def enrich_02_euclidean_gas():
    """Apply strategic enrichments to 02_euclidean_gas.md"""

    input_path = Path("docs/source/1_euclidean_gas/02_euclidean_gas.md")
    output_path = input_path  # Overwrite in place

    print(f"Reading {input_path}...")
    content = input_path.read_text()
    lines = content.split('\n')

    modifications = []

    # Key enrichment patterns:
    # Format: (line_pattern, replacement_pattern, description)

    enrichments = [
        # Section 0 TLDR - First mentions of framework concepts
        (
            r'(all framework axioms) \(\{prf:ref\}`def-fragile-gas-algorithm`\)',
            r'\1 ({prf:ref}`def-fragile-gas-algorithm`, {prf:ref}`def-fragile-swarm-instantiation`)',
            "Add instantiation reference in TLDR"
        ),

        # Section 1.1 - Framework references
        (
            r'walker \$w_i = \(x_i \(\{prf:ref\}`def-walker`\), v_i, s_i\)',
            r'walker $w_i = (x_i, v_i, s_i)$ ({prf:ref}`def-walker`)',
            "Move walker reference after full tuple"
        ),

        # Section 1.2 - Physical motivation
        (
            r'(The BAOAB integrator) \(\{prf:ref\}`alg-euclidean-gas`\)',
            r'\1 ({prf:ref}`alg-euclidean-gas`)',
            "Keep existing reference"
        ),

        # Section 3.1 - Algorithm specification
        (
            r'(\*\*Cemetery check\.\*\*) (If all walkers are dead)',
            r'\1 ({prf:ref}`def-cemetery-state-measure`) \2',
            "Add cemetery state reference"
        ),

        (
            r'(compute raw reward) \$r_i:=R\(x_i,v_i\)\$ (and algorithmic distance)',
            r'\1 $r_i:=R(x_i,v_i)$ ({prf:ref}`def-reward-measurement`) \2',
            "Add reward measurement reference"
        ),

        (
            r'(apply the regularized standard deviation from) \{prf:ref\}`def-statistical-properties-measurement`',
            r'\1 {prf:ref}`def-statistical-properties-measurement`',
            "Keep existing reference"
        ),

        # Section 3.3 - Position-velocity foundations
        (
            r'(\*\*Algorithmic space and Sasaki metric\*\*) \$\(\\\mathcal Y,d_\{\\\mathcal Y\}\^\{\\\mathrm\{Sasaki\}\}\)\$ (where the algorithmic space is the) \(\{prf:ref\}`def-algorithmic-space-generic`\)',
            r'\1 $(\mathcal Y,d_{\mathcal Y}^{\mathrm{Sasaki}})$ \2 ({prf:ref}`def-algorithmic-space-generic`)',
            "Keep existing algorithmic space reference"
        ),

        # Section 3.4 - Swarm distance
        (
            r'(aggregation pipeline:\n\n- \*\*Dispersion distance\.\*\*)',
            r'\1',
            "Dispersal distance section"
        ),

        (
            r'(The canon ical measurement and potential pipeline now operate)',
            r'\1',
            "Framework adaptation note"
        ),

        # Section 4.1 - Viability axioms
        (
            r'(\*\*Guaranteed Revival\.\*\*) (.*?)(and Stage 3 draws independent thresholds)',
            r'\1 ({prf:ref}`axiom-guaranteed-revival`) \2\3',
            "Add axiom reference to guaranteed revival"
        ),

        (
            r'(so each dead walker survives.*?satisfies the axiom) \(Theorem \*Almost-sure revival\* \(\{prf:ref\}`thm-revival-guarantee`,',
            r'\1 ({prf:ref}`thm-revival-guarantee`,',
            "Keep existing revival theorem reference"
        ),

        # Section 4.2 - Environmental axioms
        (
            r'(\*\*Valid noise measure \(kinetic perturbation\)\.\*\*) (Lemma)',
            r'\1 ({prf:ref}`def-valid-noise-measure`) \2',
            "Add valid noise measure reference"
        ),

        # Section 4.3 - Algorithmic axioms
        (
            r'(The Sasaki distance satisfies) \$d_\{\\\mathcal Y\}\^\{\\\mathrm\{Sasaki\}\}',
            r'\1 $d_{\mathcal Y}^{\mathrm{Sasaki}}',
            "Sasaki distance continuity"
        ),
    ]

    # Apply enrichments line by line
    for i, line in enumerate(lines):
        for pattern, replacement, description in enrichments:
            if re.search(pattern, line):
                new_line = re.sub(pattern, replacement, line)
                if new_line != line:
                    lines[i] = new_line
                    modifications.append(f"Line {i+1}: {description}")
                    print(f"  ✓ Line {i+1}: {description}")

    # Special section-level enrichments
    # Add cross-references in key theorem/definition statements

    print(f"\nApplied {len(modifications)} modifications")

    # Write output
    enriched_content = '\n'.join(lines)
    output_path.write_text(enriched_content)
    print(f"Wrote enriched content to {output_path}")

    return len(modifications)

if __name__ == "__main__":
    count = enrich_02_euclidean_gas()
    print(f"\nTotal enrichments: {count}")
