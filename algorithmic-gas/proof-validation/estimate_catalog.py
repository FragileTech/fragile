"""Identify expression roles without promoting source membership to evidence."""

import re


RELATION = re.compile(r"(?<!\\)[=<>]|\\(?:leq?|geq?|lesssim|gtrsim|sim|asymp|approx|propto)\b")
DOMAIN = re.compile(r"\\(?:in|subseteq?|to|mapsto)\b")
SIMPLE_CONDITION = re.compile(
    r"^(?:[A-Za-z0-9_{}.,'\\\-+()\[\]/ ]|\\(?:alpha|beta|gamma|eta|epsilon|"
    r"varepsilon|sigma|lambda|kappa|rho|in|leq?|geq?|text|mathrm|mathbb))*[<>=].*$"
)
SPACE_MEMBERSHIP = re.compile(
    r"\\in\s*(?:C\^|\\mathcal|\\Sigma|\\overline|\\(?:mathbb|mathbf)\{R\}\^|"
    r"[^;=<>]*\\times)"
)


def inline_math_matches(text):
    """Retain exact offsets for single-line and multiline inline formulas."""
    masked = re.sub(
        r"(?<!\\)\$\$[\s\S]*?\$\$",
        lambda match: "".join("\n" if char == "\n" else " " for char in match[0]),
        text,
    )
    return re.finditer(r"(?<![\\$])\$(?!\$)(.+?)(?<!\\)\$(?!\$)", masked, re.DOTALL)


def validate_display_delimiters(text):
    """Reject missing math delimiters before a catalog silently loses formulas."""
    for line in text.splitlines():
        count = len(re.findall(r"(?<!\\)\$\$", line))
        if count == 1 and line.strip() != "$$":
            message = "A block-math delimiter must occupy its own line"
            raise ValueError(message)
    delimiters = list(re.finditer(r"(?<!\\)\$\$", text))
    if len(delimiters) % 2:
        message = "Unpaired display-math delimiter; repair the source before indexing estimates"
        raise ValueError(message)
    for opening, closing in zip(delimiters[::2], delimiters[1::2]):
        formula = text[opening.end() : closing.start()]
        if re.search(r"(?m)^\s*(?:[:]{3,}\{|`{3,}|#{1,6}\s)", formula):
            message = "Display math crosses a Markdown block; repair its source delimiters"
            raise ValueError(message)


def expression_role(formula, kind):
    """Retain estimates, definitions and hypotheses as separate obligations.

    The role is an indexing aid. Only an exact source-bound numerical comparison
    supplies numerical evidence; this classification never gives a passed status.
    """
    if kind == "quantitative_table_row":
        return "quantitative_summary_row"
    # A space declaration or quotient definition has no numerical lhs/rhs.
    # Keep any accompanying inequality as its own quantitative obligation.
    if not re.search(r"\\(?:leq?|geq?)\b|(?<!\\)[<>]", formula) and (
        SPACE_MEMBERSHIP.search(formula)
        or re.search(r"(?:/|\\big/|\\bigl/|\\bigm/)\s*\\(?:mathfrak|mathcal)\s*\{?S", formula)
        or re.match(r"^\s*(?:w_?\{?i\}?|\\mathcal\s*S)\s*(?::=|=)\s*\(", formula)
    ):
        return "domain_or_operator_contract"
    if re.search(r"\\(?:lesssim|gtrsim|asymp|propto)\b|(?:\\in\s*)?O\s*\(", formula):
        return "asymptotic_or_unspecified_bound"
    if re.search(r"\\(?:leq?|geq?)\b|(?<!\\)[<>]", formula):
        # A displayed chain remains one target. Its evidence must check every
        # relation; splitting it heuristically can silently drop a proof step.
        if kind.startswith("inline") and len(formula) < 100 and SIMPLE_CONDITION.match(formula):
            return "parameter_or_domain_condition"
        return "quantitative_bound"
    if RELATION.search(formula):
        return "identity_constant_or_distribution"
    if DOMAIN.search(formula):
        if re.match(r"^\s*[ijrs](?:_\{?[^}]+\}?)?\s*\\in\b", formula):
            return "domain_or_operator_contract"
        if r"\in" in formula and len(formula) < 100:
            return "parameter_or_domain_condition"
        return "domain_or_operator_contract"
    return "symbol_or_expression_reference"


def annotate_expressions(expressions, chapter=None):
    """Classify every expression; all evidence still comes from executed checks."""
    for expression in expressions:
        for key in (
            "evidence_status",
            "referenced_chapters",
            "external_reference_rationale",
            "external_reference_source_context",
        ):
            expression.pop(key, None)
        expression["expression_role"] = expression_role(expression["formula"], expression["kind"])
        expression["requires_expression_evidence"] = expression["expression_role"] not in {
            "symbol_or_expression_reference",
            "domain_or_operator_contract",
        }
        # This one TLDR equation explicitly cites the later mean-field chapter.
        # Retain its exact source and external scope, without pretending that
        # a native finite-population experiment verifies an independent law.
        tokens = re.sub(r"\s+", "", expression["formula"])
        context = expression.get("source_context", "")
        if (
            chapter == 2
            and expression.get("source_label") == "sec-eg-tldr"
            and expression.get("formal_item_id") is None
            and tokens == r"\mu_{n+1}=\mathcalF_h(\mu_n)"
            and all(
                term in context for term in ("{doc}`08_mean_field`", "mean-field", "hypotheses")
            )
        ):
            expression.update(
                expression_role="external_chapter_result_reference",
                requires_expression_evidence=False,
                evidence_status="not_evaluated_external_scope",
                referenced_chapters=[8],
                external_reference_source_context=context,
                external_reference_rationale=(
                    "This TLDR explicitly assigns the deterministic mean-field recurrence to "
                    "chapter 8 under that chapter's hypotheses. Chapter 2 establishes the full "
                    "finite-population Markov kernel; its empirical measure is random."
                ),
            )
