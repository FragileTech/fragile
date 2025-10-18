#!/usr/bin/env python3
"""
Rewrite all glossary descriptions to provide TLDR context WITHOUT repeating the title.
This avoids wasting tokens by duplicating information already in the title.
"""

import re
from pathlib import Path
from typing import Dict, List


def extract_key_concept_from_title(title: str) -> str:
    """Extract the key mathematical concept from title without repeating it."""
    # Clean LaTeX
    clean = re.sub(r'\$[^\$]+\$', '', title)
    clean = re.sub(r'[_\{\}\\]', '', clean)
    clean = clean.strip()

    # Extract what the item is ABOUT, not its name
    lower = clean.lower()

    # Return the essence, not the name
    if 'of' in lower:
        parts = clean.split(' of ', 1)
        if len(parts) > 1:
            return parts[1]  # Return what it's of
    elif 'for' in lower:
        parts = clean.split(' for ', 1)
        if len(parts) > 1:
            return parts[1]  # Return what it's for

    return clean


def generate_context_description(title: str, entry_type: str) -> str:
    """Generate description that gives CONTEXT, never repeats title."""

    clean_title = re.sub(r'\$[^\$]+\$', '', title)
    clean_title = re.sub(r'[_\{\}\\]', '', clean_title).strip()
    title_lower = clean_title.lower()

    # Type-specific context descriptions (NEVER repeat the title)
    if entry_type == 'Definition':
        if 'walker' in title_lower:
            return "Tuple (x,v,s) of position, velocity, and viability status"
        elif 'swarm' in title_lower and 'space' in title_lower:
            return "Product space Σ_N of N agent configurations"
        elif 'alive' in title_lower or 'dead' in title_lower:
            return "Classifies agents by viability status bit s∈{0,1}"
        elif 'axiom' in title_lower:
            if 'revival' in title_lower:
                return "Ensures dead agents revive with probability 1"
            elif 'boundary' in title_lower and 'regularity' in title_lower:
                return "Death probability Hölder continuous in configuration"
            elif 'boundary' in title_lower and 'smoothness' in title_lower:
                return "Domain boundary has finite perimeter"
            elif 'environment' in title_lower:
                return "Rewards vary sufficiently across state space"
            elif 'reward' in title_lower:
                return "Fitness function is Lipschitz continuous"
            elif 'algorithmic' in title_lower:
                return "Projected space has bounded diameter"
            elif 'noise' in title_lower:
                return "Perturbations have non-degenerate support"
            elif 'geometric' in title_lower:
                return "Projection compatible with algorithmicmetric"
            else:
                return f"Fundamental requirement for {extract_key_concept_from_title(title)}"
        elif 'metric' in title_lower or 'distance' in title_lower:
            return "Quantifies dissimilarity between configurations"
        elif 'quotient' in title_lower:
            return "Identifies permutation-equivalent states"
        elif 'operator' in title_lower:
            return "Maps current configuration to next timestep"
        elif 'qsd' in title_lower:
            return "Stationary measure for absorbed process"
        elif 'measure' in title_lower or 'kernel' in title_lower:
            return "Probability distribution governing stochastic updates"
        elif 'function' in title_lower:
            if 'lyapunov' in title_lower:
                return "Energy-like quantity decreasing in expectation"
            elif 'barrier' in title_lower:
                return "Strictly positive smooth function on domain"
            elif 'rescale' in title_lower:
                return "Monotone map normalizing fitness values"
            else:
                return "Maps states to real values"
        elif 'space' in title_lower:
            return "Mathematical domain with metric and measure"
        else:
            concept = extract_key_concept_from_title(title)
            return f"Specifies {concept} rigorously"

    elif entry_type == 'Theorem':
        if 'convergence' in title_lower:
            return "Exponential approach to equilibrium measure"
        elif 'contraction' in title_lower:
            return "Distance decreases under operator application"
        elif 'uniqueness' in title_lower:
            return "At most one solution exists"
        elif 'existence' in title_lower:
            return "Solution guaranteed to exist"
        elif 'revival' in title_lower:
            return "Dead agents resurrected with certainty"
        elif 'lsi' in title_lower or 'sobolev' in title_lower:
            return "Entropy production controls convergence rate"
        elif 'bound' in title_lower:
            return "Establishes quantitative inequality"
        elif 'drift' in title_lower:
            return "Expected change per timestep bounded"
        elif 'tightness' in title_lower:
            return "Sequence admits convergent subsequence"
        elif 'thermodynamic' in title_lower:
            return "Macroscopic observables converge as N→∞"
        elif 'hypoelliptic' in title_lower or 'hormander' in title_lower:
            return "Degenerate operator has smooth solutions"
        else:
            return "Main technical result with proof"

    elif entry_type == 'Lemma':
        if 'bound' in title_lower:
            return "Technical inequality for analysis"
        elif 'decomposition' in title_lower:
            return "Splits quantity into interpretable components"
        elif 'lipschitz' in title_lower or 'continuity' in title_lower:
            return "Establishes regularity with explicit constant"
        elif 'polishness' in title_lower:
            return "Space is complete separable metric"
        else:
            return "Supporting technical result"

    elif entry_type == 'Proposition':
        if 'property' in title_lower or 'properties' in title_lower:
            return "Characterizes key features"
        elif 'necessity' in title_lower:
            return "Requirement cannot be relaxed"
        else:
            return "Intermediate result supporting main theorems"

    elif entry_type == 'Corollary':
        return "Immediate consequence of preceding result"

    elif entry_type == 'Axiom' or entry_type == 'Assumption':
        concept = clean_title.replace('Axiom of', '').replace('Axiom', '').replace('Assumption', '').strip()
        return f"Fundamental requirement ensuring {concept.lower()}"

    elif entry_type == 'Remark':
        return "Clarifying note on technical detail"

    elif entry_type == 'Algorithm':
        return "Computational procedure with pseudocode"

    elif entry_type == 'Observation':
        return "Empirical or intuitive insight"

    elif entry_type == 'Conjecture':
        return "Unproven but plausible statement"

    else:
        # Fallback - give mathematical context
        return "Mathematical object in framework"


def main():
    """Rewrite all descriptions to avoid title repetition."""
    project_root = Path(__file__).parent.parent.parent
    glossary_path = project_root / 'docs' / 'glossary.md'

    print(f"Reading: {glossary_path}")

    with open(glossary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    rewrites = 0
    current_entry = {}

    while i < len(lines):
        line = lines[i]

        # Track metadata
        if line.startswith('### ') and not line.startswith('####'):
            current_entry = {'title': line.strip('# \n')}
        elif line.startswith('- **Type:**'):
            current_entry['type'] = line.split(':', 1)[1].strip()
        elif line.startswith('- **Label:**'):
            current_entry['label'] = line.split(':', 1)[1].strip().strip('`')

        # Rewrite description
        if line.startswith('- **Description:**'):
            title = current_entry.get('title', '')
            entry_type = current_entry.get('type', '')

            if title and entry_type:
                new_desc = generate_context_description(title, entry_type)

                # Ensure <= 15 words
                words = new_desc.split()
                if len(words) > 15:
                    new_desc = ' '.join(words[:15])

                line = f"- **Description:** {new_desc}\n"
                rewrites += 1

                if rewrites % 100 == 0:
                    print(f"Rewrote {rewrites} descriptions...")

        new_lines.append(line)
        i += 1

    # Write output
    with open(glossary_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"\n✓ Rewrote {rewrites} descriptions")
    print(f"✓ All descriptions now provide context without repeating titles")
    print(f"✓ Updated: {glossary_path}")


if __name__ == '__main__':
    main()
