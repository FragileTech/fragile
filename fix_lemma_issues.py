#!/usr/bin/env python3
"""
Fix specific validation issues in lemma files.
"""

import json
from pathlib import Path


LEMMAS_DIR = Path(
    "/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/lemmas"
)


def load_json(filepath):
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Fix 1: Standardize output types
OUTPUT_TYPE_MAPPING = {
    "Lipschitz Bound": "Lipschitz Continuity",
    "Continuity": "Lipschitz Continuity",
}


def fix_output_types():
    """Fix non-standard output types."""
    print("Fixing output types...")
    files_to_fix = [
        "lem-cloning-probability-lipschitz.json",
        "lem-component-potential-lipschitz.json",
        "lem-empirical-moments-lipschitz.json",
        "lem-lipschitz-variance-functional.json",
        "lem-stats-structural-continuity.json",
        "lem-stats-value-continuity.json",
        "lem-validation-uniform-ball.json",
        "sub-lem-potential-stable-error-mean-square.json",
    ]

    for filename in files_to_fix:
        filepath = LEMMAS_DIR / filename
        if not filepath.exists():
            continue

        data = load_json(filepath)
        old_type = data["output_type"]

        if old_type in OUTPUT_TYPE_MAPPING:
            data["output_type"] = OUTPUT_TYPE_MAPPING[old_type]
            save_json(filepath, data)
            print(f"  {filename}: '{old_type}' -> '{data['output_type']}'")


# Fix 2: Rename unlabeled-lemma-72
def fix_unlabeled_lemma():
    """Rename unlabeled lemma with proper label."""
    print("\nFixing unlabeled lemma...")
    old_path = LEMMAS_DIR / "unlabeled-lemma-72.json"
    new_path = LEMMAS_DIR / "lem-empirical-moments-lipschitz-dup.json"

    if old_path.exists():
        data = load_json(old_path)

        # Update label
        data["label"] = "lem-empirical-moments-lipschitz-dup"
        data["name"] = "Empirical moments are Lipschitz in L2"

        # Add missing fields
        data["input_objects"] = [
            "obj-swarm-aggregation-operator-axiomatic",
            "obj-aggregator-lipschitz-constants",
        ]
        data["relations_established"] = [
            "Lipschitz constants for empirical mean: L_μ,M = 1/√k",
            "Lipschitz constant for second moment: L_m2,M = 2V_max/√k",
        ]

        # Save with new name
        save_json(new_path, data)
        print("  Renamed: unlabeled-lemma-72.json -> lem-empirical-moments-lipschitz-dup.json")
        print(f"  Label: {data['label']}")

        # Remove old file
        old_path.unlink()
        print("  Removed: unlabeled-lemma-72.json")


# Fix 3: Add missing statements from source document
MISSING_STATEMENTS = {
    "lem-final-positional-displacement-bound.json": {
        "natural_language_statement": "The total squared positional displacement from all walkers is bounded by a quadratic function of the initial displacement metric.",
        "input_objects": ["obj-n-particle-displacement-metric", "obj-swarm-and-state-space"],
        "relations_established": [
            "Final positional displacement bounded by initial displacement squared"
        ],
    },
    "lem-final-status-change-bound.json": {
        "natural_language_statement": "The expected total squared status change from all walkers is bounded by a quadratic function of the initial displacement metric.",
        "input_objects": ["obj-n-particle-displacement-metric", "obj-walker"],
        "relations_established": ["Final status change bounded by initial displacement squared"],
    },
    "lem-inequality-toolbox.json": {
        "natural_language_statement": "Collection of standard inequalities: Triangle inequality, Cauchy-Schwarz, Hölder, Jensen, subadditivity of fractional powers.",
        "input_objects": [],
        "relations_established": ["Standard inequalities for bounding composite expressions"],
    },
    "lem-subadditivity-power.json": {
        "natural_language_statement": "For α ∈ (0, 1], the function t ↦ t^α is subadditive: (a + b)^α ≤ a^α + b^α for all a, b ≥ 0.",
        "input_objects": [],
        "relations_established": ["Subadditivity property for fractional powers with 0 < α ≤ 1"],
    },
    "sub-lem-perturbation-positional-bound-reproof.json": {
        "natural_language_statement": "The squared positional displacement induced by the perturbation operator is bounded in expectation by a function of the initial displacement and perturbation constants.",
        "input_objects": ["obj-perturbation-operator", "obj-perturbation-constants"],
        "relations_established": ["Perturbation-induced positional displacement is bounded"],
    },
    "sub-lem-probabilistic-bound-perturbation-displacement-reproof.json": {
        "natural_language_statement": "With high probability (under concentration inequalities), the total perturbation-induced displacement across all walkers is bounded by a function of the number of walkers and perturbation scale.",
        "input_objects": ["obj-perturbation-operator", "obj-perturbation-measure"],
        "relations_established": ["High-probability bound on perturbation displacement"],
    },
    "sub-lem-unify-holder-terms.json": {
        "natural_language_statement": "Multiple Hölder-type terms (√V, V^{1/3}, etc.) arising from different components can be unified into a single dominant term using subadditivity and case analysis.",
        "input_objects": [],
        "relations_established": ["Unified Hölder bound for multiple terms"],
    },
}


def fix_missing_statements():
    """Add missing statements and metadata."""
    print("\nFixing missing statements...")

    for filename, updates in MISSING_STATEMENTS.items():
        filepath = LEMMAS_DIR / filename
        if not filepath.exists():
            continue

        data = load_json(filepath)

        # Update fields
        if not data.get("natural_language_statement"):
            data["natural_language_statement"] = updates["natural_language_statement"]
            print(f"  {filename}: Added statement")

        if not data.get("input_objects"):
            data["input_objects"] = updates["input_objects"]
            print(f"  {filename}: Added input_objects")

        if not data.get("relations_established"):
            data["relations_established"] = updates["relations_established"]
            print(f"  {filename}: Added relations_established")

        save_json(filepath, data)


# Fix 4: Add missing relations for lemmas that have statements but no relations
def fix_missing_relations():
    """Add relations_established for lemmas that have content but no relations."""
    print("\nFixing missing relations...")

    fixes = {
        "lem-perturbation-positional-bound.json": [
            "Perturbation-induced positional error bounded by displacement and perturbation constants"
        ],
        "lem-probabilistic-bound-perturbation-displacement.json": [
            "High-probability bound on total perturbation-induced displacement across swarm"
        ],
    }

    for filename, relations in fixes.items():
        filepath = LEMMAS_DIR / filename
        if not filepath.exists():
            continue

        data = load_json(filepath)

        if not data.get("relations_established"):
            data["relations_established"] = relations
            save_json(filepath, data)
            print(f"  {filename}: Added relations")


if __name__ == "__main__":
    print("=" * 80)
    print("FIXING LEMMA VALIDATION ISSUES")
    print("=" * 80)

    fix_output_types()
    fix_unlabeled_lemma()
    fix_missing_statements()
    fix_missing_relations()

    print("\n" + "=" * 80)
    print("FIXES COMPLETE")
    print("=" * 80)
    print("\nRe-run refine_lemmas.py to verify all issues are resolved.")
