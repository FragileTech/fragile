"""Inventory chapters 4–6 without granting credit for unexecuted expressions."""

import json
from pathlib import Path

from generate_inventory import build


CHAPTERS = {
    4: "04_wasserstein_contraction.md",
    5: "05_kinetic_contraction.md",
    6: "06_convergence.md",
}


def main():
    destination = Path(__file__).resolve().parent
    for chapter, filename in CHAPTERS.items():
        inventory = build(chapter, filename)
        scope = (
            f"Chapter {chapter} in the finite-particle convergence sequence. Each expression "
            "retains its local hypotheses and transition law. Stored canonical native runs, "
            "linear controls, conservative laws and survival-conditioned killed laws have "
            "separate applicability; none is substituted for another."
        )
        inventory["scope_definitions"] = {"chapter_default": scope}
        for claim in inventory["formal_items"]:
            claim["scope"] = scope
        for expression in inventory["quantitative_expressions"]:
            expression["scope_ref"] = "chapter_default"
        inventory["scope"] = (
            "Complete source inventory for analysis of existing simulations. Inventory "
            "membership supplies no numerical evidence or global theorem certificate."
        )
        output = destination / f"chapter{chapter:02d}_inventory.json"
        output.write_text(json.dumps(inventory, indent=2) + "\n")
        print(json.dumps({"chapter": chapter, "counts": inventory["counts"]}))


if __name__ == "__main__":
    main()
