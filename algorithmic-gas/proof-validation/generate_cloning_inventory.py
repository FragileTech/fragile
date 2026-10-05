"""Generate the complete, compact source inventory for convergence chapter 03."""

import bisect
import hashlib
import json
from operator import itemgetter
from pathlib import Path
import re

from estimate_catalog import (
    annotate_expressions,
    inline_math_matches,
    validate_display_delimiters,
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = Path("docs/source/2_fractal_gas/convergence_program/03_cloning.md")
    raw = (root / source).read_text()
    lines = raw.splitlines()
    starts = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line) + 1

    def lineof(offset):
        return bisect.bisect_right(starts, offset)

    stack = []
    blocks = []
    for i, line in enumerate(lines, 1):
        m = re.match(r"^(\s*)(:{3,})\{([^}]+)\}(.*)$", line)
        if m:
            stack.append({
                "start": i,
                "fence": len(m[2]),
                "directive": m[3],
                "title": m[4].strip(),
            })
        elif re.match(r"^\s*:{3,}\s*$", line):
            fence = len(line.strip())
            if stack:
                match = next(
                    (j for j in range(len(stack) - 1, -1, -1) if stack[j]["fence"] == fence), None
                )
                if match is not None:
                    while len(stack) > match:
                        b = stack.pop()
                        b["end"] = i
                        if b["directive"].startswith("prf:"):
                            blocks.append(b)
    for b in stack:
        if b["directive"].startswith("prf:"):
            b["end"] = len(lines)
            blocks.append(b)
    blocks.sort(key=itemgetter("start"))
    implemented = {
        "def-patched-std-dev-function": "Recompute the real Rust Global standardizer scale from alive population variance and its configured quadratic floor.",
        "lem-patching-properties": "Check configured global scale lower and Popoviciu upper bounds on each recorded alive measurement vector.",
        "def-standardization-operator": "Recompute real Rust standardization and compare every recorded score with the actual alive mask.",
        "lem-compact-support-z-scores": "Use the observed input range divided by the configured positive floor to check all alive scores; do not infer a global bound on an unbounded reward.",
        "def-max-patched-std": "Check the regularized scale against sqrt(observed range squared/4 + sigma_min squared).",
        "def-fitness-potential-operator": "Recompute the actual Rust fitness-channel product using recorded standardized scores.",
        "lem-potential-bounds": "Compute bounded logistic channel endpoints with the actual floors, amplitudes and exponents; omit legacy unbounded-map upper claims.",
        "def-cloning-score": "Compare the real Rust CloneDecision accepted probability with its exact clipped relative-fitness expression.",
        "def-cloning-probability": "Integrate independent current donor probabilities using the actual configured geometry/kernel and gate; revival remains probability one.",
        "lem-dead-walker-clone-prob": "Check actual dead-row decisions revive with probability one and current entering live donors.",
        "def-cloning-frozen-positional-moments": "Integrate row means and covariance traces exactly under retained sampled fitness, independent donors/gates and isotropic centered jitter.",
        "lem-cloning-individual-centered-displacement": "Check the row-centered drift identity including the moving population mean, and the conditional barycenter variance bound.",
        "lem-variance-change-decomposition": "Integrate the conditional position law; preserve the distinction between conditional row independence and measurement-averaged dependence.",
        "lem-keystone-contraction-alive": "Check the complete conditional collective flux identity with incoming donor flux, deterministic mean shift, copy covariance and jitter.",
        "thm-positional-variance-contraction": "Check the reset bound using the actual eligible physical diameter; increasing realized or expected variance is allowed below the reset constant.",
        "prop-cloning-component-conservation": "Check actual full-slot component momentum and energy restitution at the post_transform stage.",
        "prop-bounded-velocity-expansion": "Check full-slot kinetic-energy nonincrease while preserving alive-only revival terms.",
        "thm-cloning-canonical-barycenter-concentration": "Compute D_m, epsilon_s,kappa_C,C,F_*,F^*,L_0,L_a,B_0 exactly for canonical parameters; integrate conditional barycenter variance; exact small-fixture measurement averaging adds the separate fitness variance.",
        "ex-cloning-position-spreading": "Enumerate all eight distinct measurement patterns for x=[0,0,0,.1], retain all outcomes and integrate actual donor/gate/jitter moments; compare with independent real-engine ensembles.",
        "lem-keystone-complete-coverage-constants": "Compute canonical D_0,B_f,m_x,m_z,kappa_D,kappa_C,D_m,s_*,Z_*,L_A,v_0,h_f,rho_f,Delta_f,t_f,omega_f,r,gamma_0,a_0,C_0 from supplied analytic reward Lipschitz bound and W_0. Retain logarithms for constants outside representable f64 range.",
        "thm-keystone-complete-error-coverage": "Compute M_r,chi_0 and the finite-population range in logs; checking the lower pressure bound requires measurement-law averaging and the actual entering error weights.",
        "thm-keystone-discharged-averaged-pressure": "Compute chi_*,B_*,N_0 and affine/surviving-label corrections without replacing tiny positive rates by zero; the full two-swarm pressure inequality remains conditional on its entering geometry.",
        "cor-keystone-canonical-balanced-structural": "Compute the canonical balanced two-cluster rate (4/9) A_0(.5); scope is even N>=4, zero velocities, equal two-site populations, radii [.5,2), quadratic rewards.",
        "prop-cloning-two-cluster-noise-balance": "Compute A_0(a) and retain the explicit positive-jitter drift lower bound; averaged cloning variance may increase.",
    }
    exact = set(implemented) - {
        "lem-keystone-complete-coverage-constants",
        "thm-keystone-complete-error-coverage",
        "thm-keystone-discharged-averaged-pressure",
        "prop-cloning-two-cluster-noise-balance",
        "cor-keystone-canonical-balanced-structural",
        "thm-cloning-canonical-barycenter-concentration",
    }

    def classification(kind, label, text):
        if kind in {"axiom", "remark"} and label not in implemented:
            return "not_numerically_testable"
        if label in exact:
            return "exact"
        if kind == "definition" and (
            not re.search(
                r"probability|expectation|coupled|coupling|infimum|Wasserstein",
                text,
                re.IGNORECASE,
            )
        ):
            return "exact"
        return "conditional"

    def scope(label, text):
        base = "Finite-particle specified cloning proposal, retaining actual input alive/dead marks, sampled fitness, donor law and stage order. Complete source statement is authoritative; no pathwise monotonicity or continuum convergence inferred."
        if any(t in text.lower() for t in ["greedy", "idealized", "uniform companion"]):
            base += " Matching/idealized/uniform law is a separate branch from canonical independent weighted companions."
        if any(t in text.lower() for t in ["boundary", "barrier"]):
            base += " Boundary claims require the stated occupancy, favorable-companion, jitter-integrability and killed/survival conventions; finite samples cannot certify these uniform hypotheses."
        if "canonical" in text.lower():
            base += " Canonical constants apply only when all specified parameters and entering-region hypotheses match."
        if label in {
            "prop-cloning-macroscopic-structural-expansion",
            "prop-cloning-no-global-affine-structural-contraction",
            "prop-canonical-fullstep-structural-expansion",
        }:
            base += " This is a counterexample/noncontraction statement; positive drift validates its stated scope."
        return base

    validate_display_delimiters(raw)
    quant = []
    for m in re.finditer(r"\$\$([\s\S]*?)\$\$", raw):
        quant.append({
            "source_line": lineof(m.start()),
            "source_end_line": lineof(m.end() - 1),
            "kind": "display_math",
            "formula": m[1].strip(),
        })
    for m in inline_math_matches(raw):
        quant.append({
            "source_line": lineof(m.start()),
            "source_end_line": lineof(m.end() - 1),
            "kind": "inline_math",
            "formula": raw[m.start() + 1 : m.end() - 1],
        })
    quant.sort(key=itemgetter("source_line", "source_end_line", "kind"))
    items = []
    unlabeled = []
    for b in blocks:
        full = "\n".join(lines[b["start"] - 1 : b["end"]])
        lm = re.search(r"^:label:\s*(.+)$", full, re.MULTILINE)
        label = lm[1].strip() if lm else None
        kind = b["directive"].split(":", 1)[1]
        body = "\n".join(lines[b["start"] : b["end"] - 1])
        body = re.sub(r"^:(?:label|class|name):.*\n?", "", body, flags=re.MULTILINE).strip()
        cut = re.search(
            r"(?m)^\*?\*?Proof\.?\*?\*?|^\*\*Proof\.[^\n]*|^\*Proof\.\*|^\*\*1\. Bound",
            body,
        )
        statement = body[: cut.start()].rstrip() if cut else body
        hypothesis_lines = [
            line
            for line in statement.splitlines()
            if re.search(
                r"\b(assume|suppose|let|fix|under|provided|if |require|conditional|bounded|independent|positive|nonempty|nonextinct|canonical|for every|for any|use the|with |where )",
                line,
                re.IGNORECASE,
            )
        ]
        formulas = [
            dict(q)
            for q in quant
            if b["start"] <= q["source_line"] and q["source_end_line"] <= b["end"]
        ]
        status = classification(kind, label or "", statement)
        diagnostic = implemented.get(
            label,
            "Track the exact source formulas and stated hypotheses. Add an explicit targeted fixture or an analytic hypothesis check before marking this item validated; finite simulation alone cannot prove this uniform statement.",
        )
        item = {
            "id": label or f"chapter03-{kind}-line-{b['start']}",
            "source_label": label,
            "label": label,
            "kind": kind,
            "title": b["title"],
            "source_line": b["start"],
            "source_end_line": b["end"],
            "source_end": b["end"],
            "source_excerpt": full,
            "statement": statement,
            "hypotheses_text": statement,
            "hypotheses": hypothesis_lines,
            "formulas": formulas,
            "computability_kind": status,
            "diagnostic": diagnostic,
            "proposed_diagnostic": diagnostic,
            "scope": scope(label or "", statement),
            "validation_status": "inventoried_unvalidated",
            "curated_initial_diagnostic_hint": label in implemented,
        }
        (items if label else unlabeled).append(item)
    for q in quant:
        containers = [
            b
            for b in blocks
            if b["start"] <= q["source_line"] and q["source_end_line"] <= b["end"]
        ]
        if containers:
            b = max(containers, key=itemgetter("start"))
            item = next((i for i in items + unlabeled if i["source_line"] == b["start"]), None)
            if item and item["source_label"] is None:
                preceding = [i for i in items if i["source_line"] < b["start"]]
                owner = preceding[-1] if preceding else item
            else:
                owner = item
        else:
            preceding = [i for i in items if i["source_line"] <= q["source_line"]]
            owner = preceding[-1] if preceding else None
        q["source_label"] = owner["source_label"] if owner else None
        q["formal_item_id"] = owner["id"] if owner else None
        q["source_context"] = "\n".join(
            lines[max(0, q["source_line"] - 3) : min(len(lines), q["source_end_line"] + 2)]
        )
        q["scope"] = (
            owner["scope"]
            if owner
            else "Unattached mathematical expression; source context is authoritative."
        )
        q["computability_kind"] = owner["computability_kind"] if owner else "conditional"
        q["diagnostic"] = (
            owner["diagnostic"]
            if owner
            else "Review the source-context expression before constructing its diagnostic."
        )
        q["latex"] = q["formula"]
        q["source_end"] = q["source_end_line"]
    annotate_expressions(quant)
    labels = re.findall(r"^:label:\s*(.+)$", raw, re.MULTILINE)
    assert sorted(labels) == sorted(i["source_label"] for i in items)
    out = {
        "schema_version": 1,
        "chapter": 3,
        "source_path": str(source),
        "source": str(source),
        "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "source_line_count": len(lines),
        "scope": "Exhaustive retained-source inventory. Every labeled formal item, unlabeled proof and mathematical expression is recorded with exact source lines. Inventoried is distinct from validated; finite samples do not establish universal hypotheses or uniform rates. Source was already modified in the workspace and is preserved verbatim.",
        "formal_items": items,
        "unlabeled_formal_blocks": unlabeled,
        "quantitative_expressions": quant,
        "coverage": {
            "labeled_formal_items": len(items),
            "unlabeled_formal_blocks": len(unlabeled),
            "display_math_expressions": sum(q["kind"] == "display_math" for q in quant),
            "inline_math_expressions": sum(q["kind"] == "inline_math" for q in quant),
            "curated_initial_diagnostic_hints": sum(
                i["curated_initial_diagnostic_hint"] for i in items
            ),
        },
    }
    scopes = {}
    diagnostics = {}

    def intern(registry, text, prefix):
        for key, value in registry.items():
            if value == text:
                return key
        key = f"{prefix}-{len(registry) + 1:03d}"
        registry[key] = text
        return key

    for number, expression in enumerate(out["quantitative_expressions"], 1):
        expression["id"] = f"chapter03-expression-{number:04d}"
        expression["scope_ref"] = intern(scopes, expression.pop("scope"), "scope")
        expression["diagnostic_ref"] = intern(
            diagnostics, expression.pop("diagnostic"), "diagnostic"
        )
        expression["source_context"] = expression["source_context"][:80]
        expression.pop("latex", None)
        expression.pop("source_end", None)
    expression_ids = {
        (q["source_line"], q["source_end_line"], q["kind"], q["formula"]): q["id"]
        for q in out["quantitative_expressions"]
    }
    for item in out["formal_items"]:
        item["formula_expression_ids"] = [
            expression_ids[q["source_line"], q["source_end_line"], q["kind"], q["formula"]]
            for q in item.pop("formulas")
        ]
        item["hypotheses_ref"] = "statement"
        item["scope_ref"] = intern(scopes, item.pop("scope"), "scope")
        item["diagnostic_ref"] = intern(diagnostics, item.pop("diagnostic"), "diagnostic")
        for key in [
            "source_excerpt",
            "hypotheses_text",
            "hypotheses",
            "proposed_diagnostic",
            "source_end",
        ]:
            item.pop(key, None)
    for block in out["unlabeled_formal_blocks"]:
        keep = {"id", "source_line", "source_end_line", "kind", "title"}
        for key in list(block):
            if key not in keep:
                block.pop(key)
    out["scopes"] = scopes
    out["diagnostics"] = diagnostics
    out["inventory_method"] = (
        "Retain each exact formal statement once; hypotheses_ref points to that statement. Record every display and inline expression once, including proof expressions; formal items list their expression IDs. Unlabeled proofs have source ranges and do not increase formal claim counts. A SHA256 ties all ranges to the retained source; rerun this generator after source edits. Curated hints cover an initial subset; the runtime report records actual diagnostic source references and scopes. Neither hint availability nor a source reference establishes a complete theorem."
    )
    path = root / "algorithmic-gas/proof-validation/chapter03_inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps({"path": str(path), "coverage": out["coverage"], "bytes": path.stat().st_size})
    )


if __name__ == "__main__":
    main()
