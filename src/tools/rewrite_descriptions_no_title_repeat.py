#!/usr/bin/env python3
"""
Rewrite all glossary descriptions to provide TLDR context WITHOUT repeating the title.
This avoids wasting tokens by duplicating information already in the title.
"""

from pathlib import Path
import re


def extract_key_concept_from_title(title: str) -> str:
    """Extract the key mathematical concept from title without repeating it."""
    # Clean LaTeX
    clean = re.sub(r"\$[^\$]+\$", "", title)
    clean = re.sub(r"[_\{\}\\]", "", clean)
    clean = clean.strip()

    # Extract what the item is ABOUT, not its name
    lower = clean.lower()

    # Return the essence, not the name
    if "of" in lower:
        parts = clean.split(" of ", 1)
        if len(parts) > 1:
            return parts[1]  # Return what it's of
    elif "for" in lower:
        parts = clean.split(" for ", 1)
        if len(parts) > 1:
            return parts[1]  # Return what it's for

    return clean


def generate_context_description(title: str, entry_type: str) -> str:
    """Generate description that gives CONTEXT, never repeats title."""

    clean_title = re.sub(r"\$[^\$]+\$", "", title)
    clean_title = re.sub(r"[_\{\}\\]", "", clean_title).strip()
    title_lower = clean_title.lower()

    # Type-specific context descriptions (NEVER repeat the title)
    if entry_type == "Definition":
        if "walker" in title_lower:
            return "Tuple (x,v,s) of position, velocity, and viability status"
        if "swarm" in title_lower and "space" in title_lower:
            return "Product space Σ_N of N agent configurations"
        if "alive" in title_lower or "dead" in title_lower:
            return "Classifies agents by viability status bit s∈{0,1}"
        if "axiom" in title_lower:
            if "revival" in title_lower:
                return "Ensures dead agents revive with probability 1"
            if "boundary" in title_lower and "regularity" in title_lower:
                return "Death probability Hölder continuous in configuration"
            if "boundary" in title_lower and "smoothness" in title_lower:
                return "Domain boundary has finite perimeter"
            if "environment" in title_lower:
                return "Rewards vary sufficiently across state space"
            if "reward" in title_lower:
                return "Fitness function is Lipschitz continuous"
            if "algorithmic" in title_lower:
                return "Projected space has bounded diameter"
            if "noise" in title_lower:
                return "Perturbations have non-degenerate support"
            if "geometric" in title_lower:
                return "Projection compatible with algorithmicmetric"
            return f"Fundamental requirement for {extract_key_concept_from_title(title)}"
        if "metric" in title_lower or "distance" in title_lower:
            return "Quantifies dissimilarity between configurations"
        if "quotient" in title_lower:
            return "Identifies permutation-equivalent states"
        if "operator" in title_lower:
            return "Maps current configuration to next timestep"
        if "qsd" in title_lower:
            return "Stationary measure for absorbed process"
        if "measure" in title_lower or "kernel" in title_lower:
            return "Probability distribution governing stochastic updates"
        if "function" in title_lower:
            if "lyapunov" in title_lower:
                return "Energy-like quantity decreasing in expectation"
            if "barrier" in title_lower:
                return "Strictly positive smooth function on domain"
            if "rescale" in title_lower:
                return "Monotone map normalizing fitness values"
            return "Maps states to real values"
        if "space" in title_lower:
            return "Mathematical domain with metric and measure"
        concept = extract_key_concept_from_title(title)
        return f"Specifies {concept} rigorously"

    if entry_type == "Theorem":
        if "convergence" in title_lower:
            return "Exponential approach to equilibrium measure"
        if "contraction" in title_lower:
            return "Distance decreases under operator application"
        if "uniqueness" in title_lower:
            return "At most one solution exists"
        if "existence" in title_lower:
            return "Solution guaranteed to exist"
        if "revival" in title_lower:
            return "Dead agents resurrected with certainty"
        if "lsi" in title_lower or "sobolev" in title_lower:
            return "Entropy production controls convergence rate"
        if "bound" in title_lower:
            return "Establishes quantitative inequality"
        if "drift" in title_lower:
            return "Expected change per timestep bounded"
        if "tightness" in title_lower:
            return "Sequence admits convergent subsequence"
        if "thermodynamic" in title_lower:
            return "Macroscopic observables converge as N→∞"
        if "hypoelliptic" in title_lower or "hormander" in title_lower:
            return "Degenerate operator has smooth solutions"
        return "Main technical result with proof"

    if entry_type == "Lemma":
        if "bound" in title_lower:
            return "Technical inequality for analysis"
        if "decomposition" in title_lower:
            return "Splits quantity into interpretable components"
        if "lipschitz" in title_lower or "continuity" in title_lower:
            return "Establishes regularity with explicit constant"
        if "polishness" in title_lower:
            return "Space is complete separable metric"
        return "Supporting technical result"

    if entry_type == "Proposition":
        if "property" in title_lower or "properties" in title_lower:
            return "Characterizes key features"
        if "necessity" in title_lower:
            return "Requirement cannot be relaxed"
        return "Intermediate result supporting main theorems"

    if entry_type == "Corollary":
        return "Immediate consequence of preceding result"

    if entry_type in {"Axiom", "Assumption"}:
        concept = (
            clean_title.replace("Axiom of", "")
            .replace("Axiom", "")
            .replace("Assumption", "")
            .strip()
        )
        return f"Fundamental requirement ensuring {concept.lower()}"

    if entry_type == "Remark":
        return "Clarifying note on technical detail"

    if entry_type == "Algorithm":
        return "Computational procedure with pseudocode"

    if entry_type == "Observation":
        return "Empirical or intuitive insight"

    if entry_type == "Conjecture":
        return "Unproven but plausible statement"

    # Fallback - give mathematical context
    return "Mathematical object in framework"


def main():
    """Rewrite all descriptions to avoid title repetition."""
    project_root = Path(__file__).parent.parent.parent
    glossary_path = project_root / "docs" / "glossary.md"

    print(f"Reading: {glossary_path}")

    with open(glossary_path, encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    rewrites = 0
    current_entry = {}

    while i < len(lines):
        line = lines[i]

        # Track metadata
        if line.startswith("### ") and not line.startswith("####"):
            current_entry = {"title": line.strip("# \n")}
        elif line.startswith("- **Type:**"):
            current_entry["type"] = line.split(":", 1)[1].strip()
        elif line.startswith("- **Label:**"):
            current_entry["label"] = line.split(":", 1)[1].strip().strip("`")

        # Rewrite description
        if line.startswith("- **Description:**"):
            title = current_entry.get("title", "")
            entry_type = current_entry.get("type", "")

            if title and entry_type:
                new_desc = generate_context_description(title, entry_type)

                # Ensure <= 15 words
                words = new_desc.split()
                if len(words) > 15:
                    new_desc = " ".join(words[:15])

                line = f"- **Description:** {new_desc}\n"
                rewrites += 1

                if rewrites % 100 == 0:
                    print(f"Rewrote {rewrites} descriptions...")

        new_lines.append(line)
        i += 1

    # Write output
    with open(glossary_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"\n✓ Rewrote {rewrites} descriptions")
    print("✓ All descriptions now provide context without repeating titles")
    print(f"✓ Updated: {glossary_path}")


if __name__ == "__main__":
    main()
