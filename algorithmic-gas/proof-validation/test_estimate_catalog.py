"""Regression checks for numerical obligations versus state-space contracts."""

import unittest

from estimate_catalog import (
    annotate_expressions,
    expression_role,
    inline_math_matches,
    validate_display_delimiters,
)
from generate_inventory import build


class EstimateRoles(unittest.TestCase):
    def test_multiline_inline_formulas_keep_exact_offsets_and_full_text(self):
        raw = "Escaped \\$ stays text. $a\n \\le b$ and $c=1$.\n$$\nx=2\n$$\n"
        matches = list(inline_math_matches(raw))
        self.assertEqual([match[1] for match in matches], ["a\n \\le b", "c=1"])
        for match in matches:
            self.assertEqual(raw[match.start() + 1 : match.end() - 1], match[1])

    def test_table_anchor_cannot_steal_later_unlabeled_proof(self):
        catalog = build(1, "01_fragile_gas_framework.md")
        expressions = catalog["quantitative_expressions"]
        proof = next(
            expression
            for expression in expressions
            if expression["formula"].startswith(r"\sum_k A_k V^{p_k} \le \sum_k A_k")
        )
        self.assertIsNone(proof["source_label"])
        rows = [
            expression
            for expression in expressions
            if expression["source_label"] == "tab-framework-theorem-summary"
        ]
        self.assertTrue(rows)
        self.assertTrue(
            all(
                expression["kind"] in {"quantitative_table_row", "inline_quantity_or_constraint"}
                for expression in rows
            )
        )

    def test_only_explicit_external_tldr_reference_has_external_scope(self):
        expression = {
            "formula": r"\mu_{n+1}=\mathcal F_h(\mu_n)",
            "kind": "inline_quantity_or_constraint",
            "source_label": "sec-eg-tldr",
            "formal_item_id": None,
            "source_context": "Under the mean-field hypotheses of {doc}`08_mean_field`.",
        }
        annotate_expressions([expression], chapter=2)
        self.assertFalse(expression["requires_expression_evidence"])
        self.assertEqual(expression["evidence_status"], "not_evaluated_external_scope")
        for change in (
            {"formal_item_id": "thm-local-population"},
            {"source_label": "thm-local-population"},
            {"source_context": "A native empirical population evolves by this map."},
        ):
            local = {**expression, **change}
            annotate_expressions([local], chapter=2)
            self.assertTrue(local["requires_expression_evidence"])
            self.assertNotIn("evidence_status", local)

    def test_missing_delimiter_cannot_silently_drop_source_estimates(self):
        validate_display_delimiters("$$\na \\le b\n$$\n")
        with self.assertRaises(ValueError):
            validate_display_delimiters("$$\na \\le b\n")
        with self.assertRaises(ValueError):
            validate_display_delimiters("$$\na \\le b\n:::{prf:lemma}\n$$\n")
        with self.assertRaises(ValueError):
            validate_display_delimiters("$$\na \\le b\n$$ To see this, consider...\n")

    def test_domain_contracts_do_not_become_numerical_estimates(self):
        for formula in (
            r"\psi_C\in C^\infty(\mathbb R^d\setminus\{0\})",
            r"w_i\in\mathcal X\times\mathbb R^d\times\{0,1\}",
            r"\Sigma_N=\widetilde\Sigma_N/\mathfrak S_N",
            r"i\in I_{11}",
        ):
            self.assertEqual(
                expression_role(formula, "inline_quantity_or_constraint"),
                "domain_or_operator_contract",
                formula,
            )

    def test_real_estimates_and_parameter_conditions_remain_required(self):
        formulas = (
            r"W_2^2(\mu,\nu)\le D^2\varepsilon",
            r"M_4(\mu)=\int d(o,x)^4\mu(dx)",
            r"\alpha\in(0,1]",
            r"\psi_C\in C^\infty,\quad |\nabla\psi_C|\le L",
            r"N\ge N_0",
        )
        expressions = [{"formula": formula, "kind": "display_equation"} for formula in formulas]
        annotate_expressions(expressions)
        self.assertTrue(
            all(expression["requires_expression_evidence"] for expression in expressions)
        )


if __name__ == "__main__":
    unittest.main()
