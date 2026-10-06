"""Generate source-complete convergence inventories for framework chapters 1 and 2."""

import bisect
import collections
import hashlib
import json
from pathlib import Path
import re

from estimate_catalog import (
    annotate_expressions,
    inline_math_matches,
    validate_display_delimiters,
)


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"definition", "assumption", "axiom", "lemma", "theorem", "corollary", "proposition"}
OPEN = re.compile(r"^\s*(:{3,}|`{3,})\{prf:(\w+)\}\s*(.*)$")
ANY_OPEN = re.compile(r"^\s*(:{3,})\s*\{[^}]+\}")
CLOSE = re.compile(r"^\s*(:{3,})\s*$")
LABEL = re.compile(r"^\s*:label:\s*(\S+)")
REF = re.compile(r"\{prf:ref\}`([^`]+)`")
CONSTANT = re.compile(
    r"(?:[CLKMDEFBAVR]\s*_|\\(?:kappa|sigma|varepsilon|epsilon|lambda|gamma|eta)\b|p_\{|p_[a-z]|\\alpha_[BH]|p_[-+])"
)

DIAGNOSTICS = {
    "topology": "Retain this as an analytic obligation. Check concrete state shapes, finite coordinates and discrete marks, and exercise finite input sequences as a smoke test; finite simulations cannot certify Polishness, measurability, Feller continuity, or existence of a reference measure.",
    "geometry": "Compute physical and squashed coordinates for randomized and adversarial pairs, including zero vectors and large radii. Evaluate every displayed distance, Jacobian norm and diameter inequality; report maximum signed lhs-minus-rhs and the scope of the physical compact set.",
    "moments": "Use the actual Rust measurement output on bounded scalar arrays and alive masks. Recompute population mean, second moment and variance; compare value perturbations and support changes with each displayed coefficient, including singleton, constant-array and near-zero-variance cases.",
    "standardization": "Freeze each realized raw array and alive mask, recompute population variance and the configured quadratic denominator, and evaluate the displayed decomposition and value/structural bounds. For stochastic channels average after the nonlinear pipeline, using independent replicates or the stated coupling; retain each raw-array bound and floor.",
    "rescale": "Evaluate the specified rescale and its analytic derivative over a dense/adversarial score grid and saturation endpoints. Check range, monotonicity, patch continuity and the displayed Lipschitz constant. The cubic-patch formulas apply only to that extension; canonical Rust logistic maps use derivative A/4.",
    "companion": "Enumerate the finite donor support and normalize its exact weights. For uniform-companion claims use the explicit uniform extension, compute categorical expectations and total variation exactly, and evaluate support-change bounds. Gaussian-weight changes require their additional weight term.",
    "reward": "Record the landscape and physical region, analytically bound reward/force gradients where possible, and sample point/segment/local-region pairs to evaluate the displayed modulus or variance lower bound. A sampled minimum is an estimate, never a certified global infimum.",
    "kinetic": "Freeze a post-collision Rust input and repeat the actual kinetic stage over independent seeds. Compare sample position means, covariance, physical squared increments and terminal survival probabilities with the displayed formulas, including Gaussian sampling uncertainty and the gamma=0 limit.",
    "cloning": "Freeze actual sampled fitness, current donor law and alive mask. Compute clipped acceptance probabilities and their exact finite categorical expectations, compare observed gate frequencies over independent seeds, and verify source-coordinate copying, jitter moments, singleton behavior and stated floor constraints.",
    "collision": "For every accepted connected component, use all frozen velocities including dead slots. Check momentum and restitution identities per realization; then fix the graph and repeat Haar rotations to compare conditional means and all cross-covariances with alpha^2(u_i dot u_j)I/d.",
    "revival": "Test every alive count from zero to N, retaining dead coordinates. Check all-dead absorption and all-slot post-cloning revival. Distinguish the abstract threshold-score revival inequality from canonical Rust deterministic dead-row acceptance; terminal deaths are a later observable.",
    "boundary": "For the actual Gaussian position law compute box survival probability as the product of normal-CDF interval probabilities. Compare empirical deaths with that probability, and compare paired-input probability changes with the displayed physical modulus. For alternative perimeter bounds retain unspecified dimension constants and domain hypotheses.",
    "concentration": "Repeat complete independently seeded experiments with the independence structure required in the statement. Evaluate its threshold, expected value, empirical exceedance frequency and confidence interval; component rotations are shared within each component and cannot be treated as independent row noise.",
    "composite": "Trace every stage using the stated coupling, normalization and raw-array convention. Assemble each displayed coefficient only when all dependency constants are available and their hypotheses hold; measure output displacement over replicates. Preserve independent-output stochastic offsets and fractional powers. Law distances require an actual coupling or exact finite mixture, not independent-output distance.",
    "rates": "Sweep alive counts and regularization floors while holding the stated inputs fixed, exclude transitions violating the relative-collapse constraint, and fit log-log slopes with uncertainty. Report finite-range consistency only; Big-O statements and unspecified prefactors do not supply numerical certificates.",
    "algebra": "Evaluate the exact displayed expressions on valid randomized and adversarial inputs and report signed residuals with floating-point tolerances. Preserve zero, singleton and endpoint conventions, units, and every local hypothesis; definitions are implementation contracts rather than stochastic convergence guarantees.",
}

OVERRIDES = {
    "lem-sasaki-total-squared-error-stable": {
        "computability_kind": "exact",
        "diagnostic": "Evaluate every clause of the repaired positional-plus-support bound under the declared uniform companion law. Include unchanged positions with changing support, independent permutations, singleton branches, and the native intermediate revival stage. Recompute the structural support term from current donors.",
        "limitation": "The measurement-support term applies to the current conditional law. Revived coordinates are copied from live donors; discarded dead coordinates are not a permanent convergence error.",
    },
    "axiom-margin-stability": {
        "computability_kind": "conditional",
        "diagnostic": "Attempt a boundary-straddling pair at arbitrarily small physical separation and compare terminal marks. Hard absorbing boxes have no uniform positive positional margin over all states; apply this assumption only to a restricted domain separated from the boundary.",
        "limitation": "The canonical hard terminal boundary does not satisfy a global positive deterministic margin. Gaussian smoothing supplies probability regularity, not deterministic margin invariance.",
    },
    "thm-forced-activity": {
        "computability_kind": "conditional",
        "diagnostic": "Compute the receiver fraction, qualifying donor probability mass and retained realized fitness gap. Compare the exact clipped average acceptance with the stated population-independent lower bound. Include equal-fitness arrays where the positive-gap hypotheses fail and the exact native live-cloning probability is zero.",
        "limitation": "The lower bound uses the explicit receiver, donor-mass and fitness-gap hypotheses; environmental richness alone does not supply them for every realized fitness array.",
    },
    "lem-eg-scheduled-revival": {
        "computability_kind": "exact",
        "diagnostic": "For M=0 require exact input-state absorption. For M>0 require all N intermediate marks equal one and every initially dead row accepted, even if eta^(alpha+beta)/(epsilon_clone*p_max)<=1. Distinguish terminal killing afterward.",
        "limitation": "Canonical deterministic dead-row acceptance has no fitness-ratio revival requirement.",
    },
    "lem-euclidean-geometric-consistency": {
        "computability_kind": "empirical",
        "diagnostic": "Repeat the canonical kinetic stage on a frozen input and compare mean position b(v+hF(x)/2), covariance s_h^2 I and full joint covariance eigenvalues. Check h^2 L_F/4<1 before invoking phase-space nondegeneracy. For F(x)=-x, run h=2 as an exact degeneracy control: v3=-x1 is deterministic conditional on input.",
        "limitation": "Positional isotropy does not imply phase-space isotropy; local joint condition-number bounds have unspecified compact-set constants.",
    },
    "axiom-non-deceptive": {
        "computability_kind": "conditional",
        "diagnostic": "Numerically integrate squared reward-gradient norm along many segments of length at least L_grad and record the sampled minimum. For a strongly convex quadratic U=x^T H x/2, lambda_min(H)=m>0, the segment average is at least m^2 L_grad^2/12 by minimizing over midpoint and orientation; use this analytic specialization as a certificate and contrast flat landscapes.",
        "limitation": "The general kappa_grad and L_grad are existence/hypothesis constants; sampled minima alone do not certify every segment.",
    },
    "lem-euclidean-richness": {
        "computability_kind": "conditional",
        "diagnostic": "Specify the reference probability pi_B and measurable subsets B1,B2, compute or bound p1,p2 and the reward gap Delta, then compare Var_pi_B R with p1*p2*Delta^2. Include a constant objective and zero kinetic penalty with variance exactly zero.",
        "limitation": "Distinct reward values without quantitative probability mass factors do not establish uniform richness.",
    },
    "prop-w2-bound-no-offset": {
        "computability_kind": "conditional",
        "diagnostic": "For a finite exactly represented transition mixture or explicitly constructed coupling, compute TV and Wasserstein bounds using diameter D_o or fourth moments M4. On a fixed status stratum estimate local regularity; record C_N as unspecified until a full finite-history Lipschitz bound is derived. Independent output samples are not a Wasserstein coupling certificate.",
        "limitation": "C_N is only locally asserted and can depend on N. Fourth-moment and TV inputs must be established independently.",
    },
    "def-cloning-operator-continuity-coeffs-recorrected": {
        "computability_kind": "conditional",
        "diagnostic": "Use the cloning proof formulas C_clone,L=3+3D_Y^2 C_P/N, C_clone,H=3D_Y^2 H_P/N and K_clone=3D_Y^2 K_P/N. Supply the probability-bound coefficients explicitly, or derive them from certified A1..A4 potential-error moduli. The coefficient module evaluates this conditional algebra; fitted trajectory ratios are not theorem constants.",
        "limitation": "The upstream stochastic potential-error moduli are not universal numerical constants. CompositeInputs preserves them as supplied analytic hypotheses.",
    },
}


def block_end(lines, start, marker):
    if marker.startswith("`"):
        for j in range(start + 1, len(lines)):
            if lines[j].strip() == marker:
                return j
        return len(lines) - 1
    stack = [len(marker)]
    for j in range(start + 1, len(lines)):
        opening = ANY_OPEN.match(lines[j])
        if opening:
            stack.append(len(opening.group(1)))
            continue
        closing = CLOSE.match(lines[j])
        if closing and len(closing.group(1)) >= stack[-1]:
            stack.pop()
            if not stack:
                return j
    return len(lines) - 1


def math_expressions(text):
    validate_display_delimiters(text)
    offsets = [0]
    offsets.extend(m.end() for m in re.finditer(r"\n", text))
    display = list(re.finditer(r"(?<!\\)\$\$(.*?)\$\$", text, re.DOTALL))
    out = []
    masked = list(text)
    for match in display:
        line = bisect.bisect_right(offsets, match.start())
        end = bisect.bisect_right(offsets, match.end())
        out.append((line, end, "display_equation", match.group(1).strip()))
        for i in range(match.start(), match.end()):
            if masked[i] != "\n":
                masked[i] = " "
    remainder = "".join(masked)
    for match in inline_math_matches(remainder):
        formula = match.group(1).strip()
        if re.search(r"=|<|>|\\(?:leq?|geq?|in|approx|propto|sim)\b", formula) or CONSTANT.search(
            formula
        ):
            line = bisect.bisect_right(offsets, match.start())
            end = bisect.bisect_right(offsets, match.end())
            out.append((line, end, "inline_quantity_or_constraint", formula))
    for line, raw in enumerate(text.splitlines(), 1):
        # A line inside a display is already represented by that complete
        # expression. Counting it again as unfenced math obscures actual gaps.
        if any(start <= line <= end for start, end, kind, _ in out if kind == "display_equation"):
            continue
        if (
            raw.lstrip().startswith("|")
            and "$" in raw
            and not re.match(r"^\s*\|[ :|\-]+\|\s*$", raw)
        ):
            out.append((line, line, "quantitative_table_row", raw.strip()))
        elif "$" not in raw and CONSTANT.search(raw) and re.search(r":=|=|\\(?:leq?|geq?)", raw):
            out.append((line, line, "unfenced_quantitative_expression", raw.strip()))
    return sorted(out)


def classify(label, title, kind, statement):
    title_clean = REF.sub("", title).lower()
    low = REF.sub("", title + " " + statement[:1400]).lower()
    if any(
        x in title_clean
        for x in (
            "polish",
            "borel",
            "feller",
            "markov kernel",
            "kernel structure",
            "reference measure",
            "metric quotient",
            "valid state space",
        )
    ):
        return "not_numerically_testable", "topology"
    if any(x in low for x in ("asymptotic", "growth exponent", "growth rate", "big-o", "scaling")):
        return "conditional", "rates"
    if any(x in low for x in ("revival", "single survivor", "single-survivor", "alive and dead")):
        return "exact" if kind == "definition" else "conditional", "revival"
    if any(
        x in low
        for x in ("shared covariance", "restitution", "component cloning", "frozen component")
    ):
        return "empirical" if "covariance" in low else "exact", "collision"
    if any(
        x in low
        for x in (
            "non-deceptive",
            "richness",
            "reward regularity",
            "reward measurement",
            "reward function",
        )
    ):
        return "conditional", "reward"
    if any(x in low for x in ("death probability", "boundary", "status update", "final status")):
        return "empirical" if "probability" in low else "conditional", "boundary"
    if any(
        x in low
        for x in (
            "mcdiarmid",
            "probabilistic bound",
            "fluctuation",
            "bounded differences",
            "conditional product structure",
        )
    ):
        return "empirical", "concentration"
    if any(
        x in low
        for x in (
            "baoab",
            "kinetic",
            "perturbation",
            "drift",
            "non-degenerate noise",
            "nondegeneracy",
            "anisotropy",
        )
    ):
        return "empirical" if kind != "definition" else "exact", "kinetic"
    if any(
        x in low
        for x in (
            "composite displacement",
            "swarm update",
            "coupling bound",
            "composite coefficient",
        )
    ):
        return "conditional", "composite"
    if any(
        x in low
        for x in (
            "cloning",
            "fitness potential",
            "potential operator",
            "cloning action",
            "potential assembly",
        )
    ):
        return "exact" if kind == "definition" else "conditional", "cloning"
    if any(x in low for x in ("rescale", "polynomial patch", "cubic patch", "logistic")):
        return "exact", "rescale"
    if any(
        x in low
        for x in (
            "standardiz",
            "value error",
            "structural error",
            "denominator shift",
            "direct shift",
            "mean shift",
            "regularized standard deviation",
            "sigma",
        )
    ):
        return "exact", "standardization"
    if any(
        x in low
        for x in (
            "empirical moment",
            "empirical measure",
            "aggregat",
            "variance functional",
            "statistical properties",
            "variance production",
            "relative collapse",
        )
    ):
        return "exact" if kind != "axiom" else "conditional", "moments"
    if any(
        x in low
        for x in (
            "companion",
            "normalization",
            "set change",
            "positional error",
            "distance measurement",
            "raw distance",
            "raw value",
            "structural change",
        )
    ):
        return "exact" if kind != "axiom" else "conditional", "companion"
    if any(
        x in low
        for x in (
            "sasaki",
            "squashing",
            "projection",
            "displacement",
            "algorithmic space",
            "distance",
            "metric",
            "geometric",
        )
    ):
        return "exact", "geometry"
    return ("conditional" if kind in {"axiom", "assumption"} else "exact"), "algebra"


def scope_for(chapter, line, auxiliary_end=1575):
    if chapter == 1:
        return "Abstract fragile-gas framework. Uniform-companion and bounded-domain formulas apply only to their stated specialization; independent-output bounds retain stochastic offsets. This is not automatically the full canonical Rust component-collision kernel."
    if 458 <= line < auxiliary_end:
        return "Auxiliary finite-swarm estimates. Expected raw distances use unregularized distance and uniform companions; scalar-array estimates require the displayed V_max and compact physical reward region. Canonical quadratic floor uses kappa_var,min=0 and epsilon_std=sigma_min. Average only after sampled nonlinear fitness."
    return "Canonical marked Euclidean Gas at fixed step size, Gaussian distance-weighted current-frame companions, shared component Haar rotations, BAOAB plus independent final position diffusion, smooth radial cap, and terminal-only classification, unless an explicit extension is stated."


def build(chapter, filename):
    relative = Path("docs/source/2_fractal_gas/convergence_program") / filename
    source = ROOT / relative
    text = source.read_text()
    lines = text.splitlines()
    auxiliary_end = next(
        (i + 1 for i, line in enumerate(lines) if line.startswith("### 4.4 ")), 1575
    )
    sections = []
    anchors = []
    for i, line in enumerate(lines):
        if re.match(r"^#{1,4}\s", line):
            sections.append((i + 1, line.lstrip("#").strip()))
        anchor = re.match(r"^\(([^)]+)\)=$", line)
        if anchor:
            anchors.append((i + 1, anchor.group(1)))
    all_expressions = math_expressions(text)
    formal = []
    for i, line in enumerate(lines):
        m = OPEN.match(line)
        if not m or m.group(2) not in REQUIRED | {"algorithm", "remark"}:
            continue
        end = block_end(lines, i, m.group(1))
        body_lines = lines[i + 1 : end]
        # Options belong to this directive; do not steal labels from nested proof blocks.
        source_label = None
        for candidate in body_lines[:8]:
            lm = LABEL.match(candidate)
            if lm:
                source_label = lm.group(1)
                break
            if candidate.strip() and not candidate.lstrip().startswith(":"):
                break
        content = "\n".join(body_lines)
        cuts = [
            match.start()
            for match in re.finditer(
                r"(?m)^\s*(?:`{3,}\{dropdown\}\s*Proof|:{3,}\{prf:proof\}|\*{1,2}Proof\.?\*{1,2}|\*{1,2}Proof\.\*{0,2}|#{1,4}\s.*Proof)",
                content,
            )
        ]
        statement = content[: min(cuts)] if cuts else content
        statement = re.sub(r"(?m)^\s*:(?:label|class):.*\n?", "", statement).strip()
        label = source_label or f"chapter{chapter:02d}-unlabeled-line-{i + 1}"
        kind = m.group(2)
        computability, category = classify(label, m.group(3), kind, statement)
        item = {
            "id": label,
            "kind": kind,
            "title": m.group(3).strip(),
            "source_label": source_label,
            "label": source_label,
            "source_path": str(relative),
            "source_line": i + 1,
            "source_end_line": end + 1,
            "source_label_line": next(
                (j + 1 for j in range(i + 1, min(end, i + 9)) if LABEL.match(lines[j])), None
            )
            if source_label
            else None,
            "statement": statement,
            "source_excerpt": "\n".join(lines[i : end + 1]),
            "hypotheses_text": statement,
            "hypotheses": [
                p.strip()
                for p in re.split(r"\n\s*\n", statement)
                if re.search(
                    r"(?i)\b(?:assume|suppose|provided|condition|under|where|for any|for each|let|if|fixed|positive|finite|bounded|nonempty)\b",
                    p,
                )
            ],
            "scope": scope_for(chapter, i + 1, auxiliary_end),
            "dependencies": sorted(set(REF.findall(statement))),
            "formulas": [
                {"source_line": a, "source_end_line": b, "kind": k, "formula": f}
                for a, b, k, f in all_expressions
                if i + 1 <= a <= end + 1
            ],
            "computability_kind": computability,
            "diagnostic_category": category,
            "diagnostic": DIAGNOSTICS[category],
            "proposed_diagnostic": DIAGNOSTICS[category],
            "validation_status": "inventoried_unvalidated",
            "simulation_can_certify_global_claim": False,
            "analytic_only": computability == "not_numerically_testable",
        }
        if source_label in OVERRIDES:
            item.update(OVERRIDES[source_label])
        item["proposed_diagnostic"] = item["diagnostic"]
        item["source_end"] = item["source_end_line"]
        for formula in item["formulas"]:
            formula["latex"] = formula["formula"]
            formula["source_end"] = formula["source_end_line"]
        formal.append(item)
    expressions = []
    for j, (a, b, kind, formula) in enumerate(all_expressions, 1):
        containing = [f for f in formal if f["source_line"] <= a <= f["source_end_line"]]
        owner = (
            min(containing, key=lambda f: f["source_end_line"] - f["source_line"])
            if containing
            else None
        )
        section = next((t for p, t in reversed(sections) if p <= a), "")
        anchor = next((l for p, l in reversed(anchors) if p <= a), None)
        # Table anchors own their actual rows, including the inline equations
        # inside those rows. They must not become the owner of every unlabeled
        # proof below the table until another anchor happens to appear.
        if anchor and anchor.startswith("tab-") and not lines[a - 1].lstrip().startswith("|"):
            anchor = None
        cname = (
            owner["computability_kind"]
            if owner
            else ("conditional" if re.search(r"\\(?:sup|inf)|\\lesssim|O\(", formula) else "exact")
        )
        expr = {
            "id": f"chapter{chapter:02d}-expression-{j:04d}",
            "source_path": str(relative),
            "source_line": a,
            "source_end_line": b,
            "source_label": owner["source_label"] if owner else anchor,
            "formal_item_id": owner["id"] if owner else None,
            "section": section,
            "kind": kind,
            "formula": formula,
            "source_context": "\n".join(lines[max(0, a - 3) : min(len(lines), b + 2)]),
            "scope": scope_for(chapter, a, auxiliary_end),
            "computability_kind": cname,
            "diagnostic": owner["diagnostic"]
            if owner
            else "Evaluate this exact source expression under the accompanying paragraph/table hypotheses and the active transition configuration. Record dependency quantities and distinguish exact constants from sampled extrema or unspecified analytic inputs.",
            "validation_status": "inventoried_unvalidated",
        }
        if owner and owner["validation_status"] == "analytic_counterexample_identified":
            expr["validation_status"] = "analytic_counterexample_identified"
        if re.search(r"\bC_d\b|C_d\x27|C_\{(?:N|\\text\{perim\})\}|C_N", formula):
            expr["unspecified_quantities"] = (
                "Dimension/local constants are not numerically specified by this expression; preserve them as analytic inputs."
            )
            if cname != "not_numerically_testable":
                expr["computability_kind"] = "conditional"
        expr["latex"] = expr["formula"]
        expr["source_end"] = expr["source_end_line"]
        expr["proposed_diagnostic"] = expr["diagnostic"]
        expressions.append(expr)
    annotate_expressions(expressions, chapter=chapter)
    by_kind = dict(sorted(collections.Counter(f["kind"] for f in formal).items()))
    result = {
        "schema_version": 1,
        "chapter": chapter,
        "title": lines[0].lstrip("#").strip(),
        "source_path": str(relative),
        "source": str(relative),
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_line_count": len(lines),
        "line_count": len(lines),
        "scope": "Source-grounded inventory and proposed diagnostics. Unrun diagnostics are explicitly unvalidated; this file supplies no empirical theorem certificate.",
        "inventory_method": "Enumerate every prf definition/assumption/axiom/lemma/theorem/corollary/proposition plus algorithm and remark, with unlabeled items identified by source line. Preserve every display equation, every inline quantity/constraint and quantitative table row, including proof equations. The complete statement is retained once and hypotheses_ref points to it so extraction cannot silently discard a condition. Formulas are stored once in quantitative_expressions and linked by formula_expression_ids; diagnostic_ref and scope_ref resolve inside this file.",
        "computability_kinds": {
            "exact": "An algebraic identity, configured parameter, explicit formula, or finite categorical sum evaluable once its stated inputs are supplied. Exact refers to the observable/formula, not a proof by simulation.",
            "empirical": "An expectation, covariance, probability or finite-range behavior estimable with the actual transition and independent replicates; sampling uncertainty is required.",
            "conditional": "A result depending on unspecified constants, global infima/suprema, analytic landscape hypotheses, a particular specialization/coupling, or asymptotic prefactors. Missing dependencies must remain missing.",
            "not_numerically_testable": "A topological, measure-theoretic, global existence or continuity assertion that finite numerical simulations cannot certify.",
        },
        "counts": {
            "formal_items": len(formal),
            "required_formal_items": sum(f["kind"] in REQUIRED for f in formal),
            "labeled_required_formal_items": sum(
                f["kind"] in REQUIRED and f["source_label"] is not None for f in formal
            ),
            "formal_items_by_kind": by_kind,
            "quantitative_expressions": len(expressions),
            "formal_items_by_computability": dict(
                collections.Counter(f["computability_kind"] for f in formal)
            ),
        },
        "formal_items": formal,
        "quantitative_expressions": expressions,
    }
    # Keep each statement and formula once; all references resolve inside this file.
    result["diagnostic_registry"] = DIAGNOSTICS
    result["scope_definitions"] = {
        "chapter_default": scope_for(chapter, 1),
        "auxiliary_finite_swarm": scope_for(chapter, 458, auxiliary_end),
    }
    for expression in result["quantitative_expressions"]:
        expression.pop("latex", None)
        expression.pop("source_end", None)
        expression.pop("proposed_diagnostic", None)
        expression.pop("diagnostic", None)
        expression.pop("scope", None)
        expression.pop("source_path", None)
        expression["scope_ref"] = (
            "auxiliary_finite_swarm"
            if chapter == 2 and 458 <= expression["source_line"] < auxiliary_end
            else "chapter_default"
        )
        expression["source_context"] = expression["source_context"][:32]
        expression.pop("section", None)
        expression.pop("validation_status", None)
        expression["diagnostic_ref"] = (
            {"formal_item_id": expression["formal_item_id"]}
            if expression["formal_item_id"]
            else {"registry_key": "algebra"}
        )
    result["expression_validation_default"] = "inventoried_unvalidated"
    result["expression_diagnostic_rule"] = (
        "Resolve formal_item_id to its formal_items diagnostic; expressions outside a formal item use diagnostic_registry.algebra. Exact formula and line references remain on each expression."
    )
    for expression in result["quantitative_expressions"]:
        expression.pop("diagnostic_ref", None)
    for claim in result["formal_items"]:
        claim["formula_expression_ids"] = [
            expression["id"]
            for expression in result["quantitative_expressions"]
            if claim["source_line"] <= expression["source_line"] <= claim["source_end_line"]
        ]
        claim.pop("formulas", None)
        claim.pop("source_excerpt", None)
        claim.pop("hypotheses_text", None)
        claim.pop("hypotheses", None)
        claim.pop("proposed_diagnostic", None)
        claim["hypotheses_ref"] = "statement"
    return result


def item(result, label):
    return next(f for f in result["formal_items"] if f["id"] == label)


def add_quantities(result, label, quantities):
    source = item(result, label)
    for symbol, formula, inputs, kind, diagnostic, limitation in quantities:
        result.setdefault("named_constants_and_rates", []).append({
            "symbol": symbol,
            "formula": formula,
            "inputs": inputs,
            "computability_kind": kind,
            "source_label": source["source_label"],
            "source_path": result["source_path"],
            "source_line": source["source_line"],
            "source_end_line": source["source_end_line"],
            "source_formula_lines": [
                f["source_line"]
                for f in result["quantitative_expressions"]
                if f["id"] in source["formula_expression_ids"]
                and symbol.replace("\\", "") in f["formula"].replace("\\", "")
            ],
            "hypotheses_ref": source["id"] + ".statement",
            "scope_ref": source["id"] + ".scope",
            "diagnostic": diagnostic,
            "limitation": limitation,
            "validation_status": "inventoried_unvalidated",
        })


def attach_implementation_diagnostics(result):
    """Link implemented checks without declaring their outcomes validated."""
    labels = {claim["source_label"] for claim in result["formal_items"]}
    checks = []
    framework = ROOT / "crates/benchmarks/src/convergence_framework.rs"
    if framework.exists():
        content = framework.read_text()
        pattern = r'BoundCheck::upper\s*\(\s*"([^"]+)"\s*,\s*&\[(.*?)\]'
        for match in re.finditer(pattern, content, re.DOTALL):
            source_labels = re.findall(r'"([^"]+)"', match.group(2))
            relevant = [label for label in source_labels if label in labels]
            if relevant:
                checks.append({
                    "id": match.group(1),
                    "source_labels": relevant,
                    "implementation_path": str(framework.relative_to(ROOT)),
                    "implementation_line": content.count("\n", 0, match.start()) + 1,
                    "status": "implemented_check_outcome_is_in_run_report",
                })
    coefficients = ROOT / "crates/benchmarks/src/convergence_coefficients.rs"
    if coefficients.exists():
        content = coefficients.read_text()
        pattern = r'check\s*\(\s*report,\s*"([^"]+)"\s*,\s*&\[(.*?)\]'
        for match in re.finditer(pattern, content, re.DOTALL):
            source_labels = re.findall(r'"([^"]+)"', match.group(2))
            relevant = [label for label in source_labels if label in labels]
            if relevant:
                checks.append({
                    "id": match.group(1),
                    "source_labels": relevant,
                    "implementation_path": str(coefficients.relative_to(ROOT)),
                    "implementation_line": content.count("\n", 0, match.start()) + 1,
                    "status": "implemented_check_outcome_is_in_run_report",
                })
    if result["chapter"] == 2:
        kinetic = ROOT / "crates/benchmarks/src/convergence_kinetic.rs"
        content = kinetic.read_text() if kinetic.exists() else ""
        specifications = [
            ("thermostat_innovation_mean", "def-eg-baoab-canonical"),
            ("thermostat_innovation_covariance", "def-eg-baoab-canonical"),
            ("native_b2_to_final_cap_identity", "def-eg-baoab-canonical"),
            ("position_drift", "lem-euclidean-geometric-consistency"),
            ("position_covariance", "lem-euclidean-geometric-consistency"),
            ("physical_squared_increment_bound", "lem-euclidean-perturb-moment"),
            ("synchronous_position_lipschitz", "lem-sasaki-kinetic-lipschitz"),
            ("radial_velocity_cap", "def-eg-baoab-canonical"),
            ("terminal_death_probability", "lem-euclidean-boundary-holder"),
            ("terminal_death_probability_lipschitz", "lem-euclidean-boundary-holder"),
            ("full_phase_space_covariance_positivity", "lem-euclidean-geometric-consistency"),
        ]
        for name, label in specifications:
            match = re.search('"' + re.escape(name), content)
            if match:
                checks.append({
                    "id": name,
                    "source_labels": [label],
                    "implementation_path": str(kinetic.relative_to(ROOT)),
                    "implementation_line": content.count("\n", 0, match.start()) + 1,
                    "status": "implemented_check_outcome_is_in_run_report",
                })
    result["implementation_diagnostics"] = checks
    for claim in result["formal_items"]:
        claim["implementation_diagnostic_ids"] = [
            check["id"] for check in checks if claim["source_label"] in check["source_labels"]
        ]


def main():
    ch1 = build(1, "01_fragile_gas_framework.md")
    ch2 = build(2, "02_euclidean_gas.md")

    ch1["scope_obligations"] = [
        "N>=2 is the abstract framework convention; canonical Euclidean Gas explicitly permits N=1.",
        "Uniform companion support-change constants do not cover finite-width Gaussian weight changes without extra terms.",
        "Bounded algorithmic features do not imply bounded physical coordinates or confinement on an unbounded domain.",
        "All-slot shared component rotations correlate collision outputs, so abstract product-output hypotheses must be checked before transfer.",
        "Independent-output displacement retains additive noise even at identical inputs. This observable is different from a distance between transition laws.",
        "Global positional status margins fail for unrestricted states with a hard absorbing boundary; probability smoothing is a separate statement.",
        "Dimension/perimeter/local constants C_d, C_d_prime, C_perim and C_N remain unspecified where the source only supplies existence or asymptotic bounds.",
    ]
    ch2["scope_obligations"] = [
        "Canonical scheduled revival uses deterministic acceptance of initially dead rows when M>0 and does not need abstract revival-score inequalities.",
        "The canonical standardization denominator is sqrt(population variance + sigma_min^2); auxiliary notation specializes with kappa_var,min=0 and epsilon_std=sigma_min.",
        "Auxiliary expected-distance inequalities use the uniform companion extension and unregularized comparison distance, not finite-width canonical Gaussian companions or pre-averaged sampled fitness.",
        "Physical phase-space displacement and the compact squashed Sasaki distance have distinct global bounds. Transfer requires a compact physical set and an inverse-projection modulus.",
        "Collision identities use all slots including retained dead velocities; alive-only momentum changes on revival and terminal killing.",
        "Positive positional covariance does not imply isotropic or globally uniformly nondegenerate full phase-space covariance.",
        "Non-deception and quantitative richness are extra landscape hypotheses; constant objectives remain well-defined despite failing positive richness.",
        "Finite empirical verification of Feller continuity or global existence is impossible; numerical exercises only test consistency of the transition contract.",
    ]

    add_quantities(
        ch1,
        "axiom-guaranteed-revival",
        [
            (
                "kappa_revival",
                "eta^(alpha+beta)/(epsilon_clone*p_max)",
                ["eta", "alpha", "beta", "epsilon_clone", "p_max"],
                "exact",
                "Compute ratio, require >1 for abstract threshold-score revival; compare dead-slot acceptance frequencies.",
                "Canonical Rust scheduled revival does not require this condition.",
            ),
        ],
    )
    add_quantities(
        ch1,
        "axiom-sufficient-amplification",
        [
            (
                "kappa_amplification",
                "alpha+beta",
                ["alpha", "beta"],
                "exact",
                "Check exponents nonnegative and sum positive; zero-sum is a negative control.",
                "Positive amplification alone does not give a quantitative activity lower bound.",
            ),
        ],
    )
    add_quantities(
        ch1,
        "lem-empirical-aggregator-properties",
        [
            ("L_mu_M", "k^(-1/2)", ["k"], "exact", DIAGNOSTICS["moments"], None),
            ("L_m2_M", "2*V_max/sqrt(k)", ["k", "V_max"], "exact", DIAGNOSTICS["moments"], None),
            ("L_var_M", "4*V_max/sqrt(k)", ["k", "V_max"], "exact", DIAGNOSTICS["moments"], None),
            (
                "L_mu_S",
                "2*V_max/k2",
                ["k2", "V_max"],
                "exact",
                DIAGNOSTICS["moments"],
                "Both alive sets must be nonempty.",
            ),
            (
                "L_m2_S",
                "2*V_max^2/k2",
                ["k2", "V_max"],
                "exact",
                DIAGNOSTICS["moments"],
                "Both alive sets must be nonempty.",
            ),
            (
                "kappa_var",
                "1",
                ["empirical population variance"],
                "exact",
                "Compare sum squared deviations with k times population variance.",
                None,
            ),
            ("kappa_range", "1", ["V_max"], "exact", "Check population variance<=V_max^2.", None),
            (
                "p_mu_S,p_m2_S,p_worst_case",
                "-1",
                ["k2>=c_min*k1"],
                "conditional",
                DIAGNOSTICS["rates"],
                "Asymptotic scaling excludes relative-collapse violations.",
            ),
        ],
    )
    add_quantities(
        ch1,
        "lem-cubic-patch-derivative-bounds",
        [
            (
                "L_P",
                "1+(3*ln(2)-2)^2/(3*(2*ln(2)-1))",
                ["z_max>1"],
                "exact",
                DIAGNOSTICS["rescale"],
                "Applies to the cubic patch, not canonical logistic rescaling.",
            ),
        ],
    )
    add_quantities(
        ch1,
        "lem-sigma-reg-derivative-bounds",
        [
            (
                "L_sigma_reg",
                "1/(2*a)",
                ["a>0"],
                "exact",
                "Evaluate derivative at variance zero and along a nonnegative variance grid.",
                None,
            ),
            (
                "sup_abs_second_derivative",
                "1/(4*a^3)",
                ["a>0"],
                "exact",
                "Evaluate analytic second derivative over nonnegative variance.",
                None,
            ),
            (
                "sup_abs_nth_derivative",
                "(2*n-3)!!/(2^n*a^(2*n-1))",
                ["a>0", "integer n>=1"],
                "exact",
                "Use (-1)!!=1 and evaluate finite derivative orders.",
                None,
            ),
        ],
    )
    add_quantities(
        ch1,
        "lem-cloning-probability-lipschitz",
        [
            (
                "L_pi_c",
                "1/(p_max*epsilon_clone)",
                ["p_max>0", "epsilon_clone>0"],
                "exact",
                DIAGNOSTICS["cloning"],
                None,
            ),
            (
                "L_pi_i",
                "(V_pot_max+epsilon_clone)/(p_max*epsilon_clone^2)",
                ["V_pot_max", "p_max>0", "epsilon_clone>0"],
                "exact",
                DIAGNOSTICS["cloning"],
                None,
            ),
        ],
    )
    add_quantities(
        ch1,
        "def-final-status-change-coeffs",
        [
            (
                "C_status_H",
                "L_death^2*N^(1-alpha_B)",
                ["L_death", "N", "0<alpha_B<=1"],
                "conditional",
                DIAGNOSTICS["boundary"],
                "Requires a valid modulus in the correct input metric.",
            ),
            (
                "K_status_var",
                "N/2",
                ["N"],
                "exact",
                "Compare total independent Bernoulli disagreement variance with N/2.",
                "Use the independence structure of the source proof.",
            ),
        ],
    )
    add_quantities(
        ch1,
        "def-composite-continuity-coeffs-recorrected",
        [
            (
                "C_Psi_L",
                "3*a",
                ["a=C_clone_L"],
                "conditional",
                DIAGNOSTICS["composite"],
                "Upstream cloning coefficient can be unspecified.",
            ),
            (
                "C_Psi_H",
                "3*b+A*a^alpha+A*b^alpha",
                ["a", "b", "A", "alpha=alpha_B"],
                "conditional",
                DIAGNOSTICS["composite"],
                "Upstream dependency constants and normalization must be established.",
            ),
            (
                "K_Psi",
                "3*c+A*c^alpha+K0",
                ["c=K_clone", "A", "alpha=alpha_B", "K0"],
                "conditional",
                DIAGNOSTICS["composite"],
                "Nonzero independent-output offset is essential.",
            ),
            (
                "p_minus",
                "alpha_B/2",
                ["0<alpha_B<=1"],
                "exact",
                "Check retained small-displacement power for 0<=V<=1.",
                None,
            ),
            (
                "p_plus",
                "max(1/2,alpha_B)",
                ["0<alpha_B<=1"],
                "exact",
                "Check retained large-displacement power for V>1.",
                None,
            ),
        ],
    )

    add_quantities(
        ch2,
        "lem-sasaki-kinetic-lipschitz",
        [
            ("b", "h*(1+c)/2", ["h>0", "c=exp(-gamma*h)"], "exact", DIAGNOSTICS["kinetic"], None),
            (
                "s_h_squared",
                "h^2*q^2/4+h*sigma_x^2",
                ["h", "q", "sigma_x"],
                "exact",
                DIAGNOSTICS["kinetic"],
                "This is positional covariance, not joint phase-space covariance.",
            ),
            (
                "L_flow",
                "1+b*h*L_F/2+b/sqrt(lambda_v)",
                ["h", "b", "L_F", "lambda_v>0"],
                "exact",
                "Compare common-innovation paired kinetic position maps in physical phase-space metric.",
                "Do not apply this globally in squashed metric.",
            ),
        ],
    )
    add_quantities(
        ch2,
        "def-eg-baoab-canonical",
        [
            ("c", "exp(-gamma*h)", ["gamma>=0", "h>0"], "exact", DIAGNOSTICS["kinetic"], None),
            (
                "q_squared",
                "sigma_v^2*((1-exp(-2*gamma*h))/(2*gamma) if gamma>0 else h)",
                ["sigma_v", "gamma>=0", "h>0"],
                "exact",
                DIAGNOSTICS["kinetic"],
                "Use a stable expm1 form near zero gamma.",
            ),
            (
                "velocity_cap",
                "V_alg*v3/(V_alg+norm(v3))",
                ["V_alg>0", "finite v3"],
                "exact",
                "Check every output velocity norm<V_alg, including zero and very large input vectors.",
                "The cap compresses all nonzero vectors, even below the radius.",
            ),
        ],
    )
    add_quantities(
        ch2,
        "lem-euclidean-boundary-holder",
        [
            (
                "L_death_physical",
                "L_flow/(sqrt(2*pi)*s_h)",
                ["L_flow", "s_h>0"],
                "exact",
                DIAGNOSTICS["boundary"],
                "Input distance is physical phase-space; conversion to squashed distance is local.",
            ),
            (
                "equal_covariance_Gaussian_TV",
                "2*Phi(r/(2*s_h))-1",
                ["r>=0", "s_h>0"],
                "exact",
                "Evaluate exact shifted-Gaussian TV against its linear upper bound.",
                None,
            ),
        ],
    )
    add_quantities(
        ch2,
        "lem-euclidean-perturb-moment",
        [
            (
                "C_x",
                "3*b^2*h^2*L_F^2/4",
                ["b", "h", "L_F"],
                "exact",
                DIAGNOSTICS["kinetic"],
                "Requires global force growth.",
            ),
            ("C_v", "3*b^2+2*lambda_v", ["b", "lambda_v"], "exact", DIAGNOSTICS["kinetic"], None),
            (
                "C_0",
                "3*b^2*h^2*B_F^2/4+d*s_h^2+2*lambda_v*V_alg^2",
                ["b", "h", "B_F", "d", "s_h", "lambda_v", "V_alg"],
                "exact",
                DIAGNOSTICS["kinetic"],
                None,
            ),
        ],
    )
    add_quantities(
        ch2,
        "lem-euclidean-geometric-consistency",
        [
            (
                "position_mean_increment",
                "b*(v+h*F(x)/2)",
                ["b", "v", "h", "F(x)"],
                "empirical",
                DIAGNOSTICS["kinetic"],
                None,
            ),
            (
                "position_covariance",
                "s_h^2*I_d",
                ["s_h", "d"],
                "empirical",
                DIAGNOSTICS["kinetic"],
                None,
            ),
            (
                "position_covariance_condition_number",
                "1",
                ["s_h>0"],
                "exact",
                "Compare sampled covariance eigenvalue ratio with 1 using covariance uncertainty.",
                "Joint condition number is separate.",
            ),
            (
                "phase_space_nondegeneracy_condition",
                "q>0 and sigma_x>0 and h^2*L_F/4<1",
                ["q", "sigma_x", "h", "L_F"],
                "exact",
                "Sweep h through the threshold; unit quadratic h=2 is an exact degeneracy control.",
                "No explicit uniform global joint condition-number constant is supplied.",
            ),
        ],
    )
    add_quantities(
        ch2,
        "thm-eg-component-balances",
        [
            (
                "component_momentum_residual",
                "sum(v_tilde_i)-sum(v_i)=0",
                ["component", "frozen all-slot velocities"],
                "exact",
                DIAGNOSTICS["collision"],
                None,
            ),
            (
                "relative_energy_ratio",
                "alpha_restitution^2",
                ["orthogonal R_C", "component"],
                "exact",
                DIAGNOSTICS["collision"],
                None,
            ),
            (
                "component_conditional_covariance",
                "alpha_restitution^2*(u_i dot u_j)/d*I_d",
                ["fixed graph", "fixed frozen velocities", "Haar O(d)"],
                "empirical",
                DIAGNOSTICS["collision"],
                "Output rows are correlated by their shared rotation.",
            ),
        ],
    )
    add_quantities(
        ch2,
        "lem-squashing-properties-generic",
        [
            ("L_psi_C", "1", ["C>0"], "exact", DIAGNOSTICS["geometry"], None),
            (
                "D_psi_tangential",
                "C/(C+norm(z))",
                ["C>0", "z!=0"],
                "exact",
                "Compare analytic squash Jacobian singular values with tangential and radial factors.",
                "At zero all factors are 1.",
            ),
            (
                "D_psi_radial",
                "C^2/(C+norm(z))^2",
                ["C>0", "z!=0"],
                "exact",
                "Compare analytic squash Jacobian singular values with radial factor.",
                "Inverse-projection factors diverge near feature boundary.",
            ),
        ],
    )
    add_quantities(
        ch2,
        "lem-sasaki-aggregator-lipschitz",
        [
            ("L_mu_M", "1/sqrt(k)", ["k>=1"], "exact", DIAGNOSTICS["moments"], None),
            (
                "L_m2_M",
                "2*V_max/sqrt(k)",
                ["k>=1", "abs(values)<=V_max"],
                "exact",
                DIAGNOSTICS["moments"],
                None,
            ),
            (
                "L_mu_S",
                "3*V_max/k_min",
                ["k_min=max(1,min(k1,k2))"],
                "exact",
                DIAGNOSTICS["moments"],
                None,
            ),
            (
                "L_m2_S",
                "3*V_max^2/k_min",
                ["k_min=max(1,min(k1,k2))"],
                "exact",
                DIAGNOSTICS["moments"],
                None,
            ),
            (
                "p_mu_S,p_m2_S,p_worst_case",
                "-1",
                ["relative-collapse bound"],
                "conditional",
                DIAGNOSTICS["rates"],
                None,
            ),
        ],
    )

    # Configuration facts are explicit in this compact source definition, including numbers in prose.
    canonical = item(ch2, "def-eg-canonical-rust")
    ch2["canonical_parameters"] = [
        {
            "name": name,
            "default": value,
            "constraint": constraint,
            "source_label": canonical["source_label"],
            "source_line": canonical["source_line"],
            "diagnostic": diagnostic,
            "validation_status": "inventoried_unvalidated",
        }
        for name, value, constraint, diagnostic in [
            (
                "dimension",
                None,
                "integer 1<=d<=256",
                "Sweep representative dimensions and test dimension validation.",
            ),
            (
                "h",
                None,
                "finite h>0",
                "Record actual step size; check nondegeneracy h^2 L_F/4<1 where invoked.",
            ),
            (
                "epsilon_D",
                2,
                "positive finite width; infinity is a separate uniform extension",
                "Recompute measurement categorical probabilities and empirical frequencies.",
            ),
            (
                "epsilon_C",
                2,
                "positive finite width; infinity is a separate uniform extension",
                "Recompute clone categorical probabilities and empirical frequencies.",
            ),
            ("R_x", 2, "positive", "Check squashed feature norms and diameter."),
            ("R_v", 2, "positive", "Check velocity feature norms and smooth cap."),
            ("lambda_v", 1, "positive", "Check physical/squashed weighted distances."),
            ("sigma_min_reward", 0.1, "positive", "Check sqrt(population variance+floor^2)."),
            ("sigma_min_diversity", 0.1, "positive", "Check sqrt(population variance+floor^2)."),
            (
                "A_reward",
                2,
                "positive",
                "Check logistic derivative bound A/4 and range (eta,A+eta).",
            ),
            (
                "A_diversity",
                2,
                "positive",
                "Check logistic derivative bound A/4 and range (eta,A+eta).",
            ),
            ("eta_reward", 0.1, "positive", "Check strictly positive reward factor."),
            ("eta_diversity", 0.1, "positive", "Check strictly positive diversity factor."),
            (
                "alpha",
                1,
                "nonnegative, alpha+beta>0",
                "Sweep selection exponents and compute minimum/maximum fitness bounds.",
            ),
            (
                "beta",
                1,
                "nonnegative, alpha+beta>0",
                "Sweep diversity exponent and compute minimum/maximum fitness bounds.",
            ),
            (
                "delta_D",
                0.001,
                "positive",
                "Check sampled separation sqrt(distance^2+delta_D^2), singleton delta_D.",
            ),
            ("p_max", 1, "positive", "Compute clipped clone acceptance probability."),
            (
                "epsilon_clone",
                0.000001,
                "positive",
                "Check denominator floor and clone probability derivatives.",
            ),
            (
                "alpha_restitution",
                0.5,
                "canonical collision restitution",
                "Check component relative-energy ratio alpha_restitution^2.",
            ),
            (
                "sigma_clone",
                0.1,
                "Gaussian clone jitter",
                "Estimate accepted-recipient jitter covariance sigma_clone^2 I.",
            ),
            (
                "mass",
                1,
                "acceleration incorporates nonunit mass",
                "Ensure force is acceleration in kinetic formulas.",
            ),
            ("gamma", 1, "nonnegative", "Check c and q, including gamma=0 limiting formula."),
            (
                "sigma_v",
                1,
                "canonical velocity diffusion factor",
                "Compare O-stage Gaussian covariance q^2 I.",
            ),
            (
                "sigma_x",
                0.1,
                "positive for stated Feller/nondegeneracy assertion",
                "Compare final position diffusion covariance h sigma_x^2 I.",
            ),
            ("V_alg", 2, "positive", "Check smooth cap strict radius and moment constant."),
            (
                "absorbing_box_half_width",
                2,
                "terminal-only classification on [-2,2]^d",
                "Compare terminal marks with coordinates; retain intermediate and terminal dead coordinates.",
            ),
            (
                "lambda_vel",
                None,
                "additional optional kinetic objective choice; zero is permitted",
                "Check reward is negative minimized objective minus kinetic penalty if configured.",
            ),
        ]
    ]
    ch2["analytic_specializations"] = [
        {
            "name": "Sasaki feature diameter",
            "formula": "D_Y=2*sqrt(R_x^2+lambda_v*V_alg^2)",
            "source_line": 109,
            "source_label": "lem-projection-lipschitz",
            "status": "derived_exact",
            "diagnostic": "Compare all squashed pair distances with the product-ball diameter.",
        },
        {
            "name": "Local inverse squash modulus",
            "formula": "L_inverse(C,r_phys)=(1+r_phys/C)^2",
            "source_line": 155,
            "source_label": "lem-squashing-properties-generic",
            "status": "derived_exact",
            "diagnostic": "Use only on norm(z)<=r_phys; compose reward and kinetic physical Lipschitz bounds with this factor.",
        },
        {
            "name": "Quadratic non-deception certificate",
            "formula": "kappa_grad=m^2*L_grad^2/12",
            "source_line": 1592,
            "source_label": "axiom-non-deceptive",
            "status": "derived_exact_for_SPD_quadratic",
            "hypotheses": [
                "U(x)=x^T H x/2",
                "H symmetric positive definite",
                "lambda_min(H)=m>0",
                "segments of length>=L_grad within convex valid domain",
            ],
            "diagnostic": "Compare numerical line integrals with certificate; flat and nonconvex landscapes need separate hypotheses.",
        },
        {
            "name": "Canonical fitness range",
            "formula": "eta_r^alpha*eta_d^beta <= V_fit <= (A_r+eta_r)^alpha*(A_d+eta_d)^beta",
            "source_line": 1640,
            "source_label": "def-eg-frozen-measurements",
            "status": "derived_exact",
            "diagnostic": "Compare each frozen sampled fitness with these positive range bounds.",
        },
        {
            "name": "Canonical logistic Lipschitz",
            "formula": "L_g=A/4",
            "source_line": 1640,
            "source_label": "def-eg-frozen-measurements",
            "status": "derived_exact",
            "diagnostic": "Compare derivative A*sigmoid(z)*(1-sigmoid(z)) with A/4.",
        },
    ]

    for specialization in ch2["analytic_specializations"]:
        specialization["source_line"] = item(ch2, specialization["source_label"])["source_line"]

    for result in (ch1, ch2):
        attach_implementation_diagnostics(result)
        out = (
            ROOT
            / "proof-validation"
            / f"chapter{result['chapter']:02d}_inventory.json"
        )
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(out.relative_to(ROOT), result["counts"])


if __name__ == "__main__":
    main()
