"""Inventory Chapters 7–9 while preserving exact formulas and hypotheses."""

import json
from pathlib import Path

from generate_inventory import build


CHAPTERS = {
    7: "07_discrete_qsd.md",
    8: "08_mean_field.md",
    9: "09_propagation_chaos.md",
}


def main():
    destination = Path(__file__).resolve().parent
    for chapter, filename in CHAPTERS.items():
        inventory = build(chapter, filename)
        scope = (
            f"Chapter {chapter}: equilibrium references, the actual full marked native update, "
            "and finite-population/mean-field convergence concern distinct laws. Preserve "
            "complete local hypotheses, self-exclusion, sampled fitness marks, connected "
            "collision randomness, probability normalization and survival conditioning. "
            "Finite native observations do not certify stationarity, QSD concentration or "
            "universal attraction. Array indices only address data in an exchangeable swarm."
        )
        inventory["scope_definitions"] = {"chapter_default": scope}
        for claim in inventory["formal_items"]:
            claim["scope"] = scope
        for expression in inventory["quantitative_expressions"]:
            expression["scope_ref"] = "chapter_default"
        inventory["scope"] = (
            "Complete source inventory. Numerical credit requires matching exact expressions, "
            "actual operands, recorded hypotheses and immutable execution evidence."
        )
        output = destination / f"chapter{chapter:02d}_inventory.json"
        output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"chapter": chapter, "counts": inventory["counts"]}))


if __name__ == "__main__":
    main()
