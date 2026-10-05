"""Exact source inventories for entropy, mass/transport and exchangeability."""

import json
from pathlib import Path

from generate_inventory import build


CHAPTERS = {
    10: "10_kl_hypocoercive.md",
    11: "11_hk_convergence.md",
    12: "12_qsd_exchangeability_theory.md",
}


def main():
    destination = Path(__file__).resolve().parent
    for chapter, filename in CHAPTERS.items():
        inventory = build(chapter, filename)
        scope = (
            f"Chapter {chapter}: preserve the exact reference law and generator/kernel, "
            "normalization, kinetic units, entropy conventions, alive mass, component-shared "
            "innovations and permutation-invariant swarm observables. Continuous densities, "
            "atomic empirical measures and their declared coarse-grained pushforwards are "
            "distinct measures. Native stationary QSD/LSI/attraction hypotheses require "
            "separate analytic justification; finite runs cannot establish their quantifiers."
        )
        inventory["scope_definitions"] = {"chapter_default": scope}
        for claim in inventory["formal_items"]:
            claim["scope"] = scope
        for expression in inventory["quantitative_expressions"]:
            expression["scope_ref"] = "chapter_default"
        inventory["scope"] = (
            "Every exact expression retains its source formula and hypotheses. Numerical "
            "credit requires matching operands under the stated law; definitions, analytic "
            "conditions and infinite-limit clauses are individually dispositioned."
        )
        output = destination / f"chapter{chapter:02d}_inventory.json"
        output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"chapter": chapter, "counts": inventory["counts"]}))


if __name__ == "__main__":
    main()
