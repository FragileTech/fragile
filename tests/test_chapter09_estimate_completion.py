"""Adversarial regression for the survival-aware Chapter 9 validation evidence."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "algorithmic-gas/proof-validation/complete_chapter09_estimates.py"


@pytest.fixture(scope="module")
def evidence():
    spec = importlib.util.spec_from_file_location("chapter09_completion_checks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(SCRIPT.parent))
        spec.loader.exec_module(module)
    inventory = json.loads(SCRIPT.with_name("chapter09_inventory.json").read_text())
    return module.algebra(inventory) + module.residual_checks(inventory), inventory


def test_stationary_variance_uses_survival_reweighted_inputs(evidence):
    checks, _ = evidence
    record = next(c for c in checks if c["id"] == "qsd-total-variance")
    q = np.array(record["inputs"]["Q"])
    nu = np.array(record["inputs"]["nu"])
    h = np.array(record["inputs"]["H"])
    survival = q.sum(axis=1)
    means = (q @ h) / survival
    variances = (q @ h**2) / survival - means**2
    wrong_budget = float(nu @ variances + nu @ ((means - nu @ means) ** 2))
    assert abs(wrong_budget - record["observed"]) > 1e-4
    assert record["observed"] == pytest.approx(record["bound"], abs=1e-12)
    assert record["passed"]


def test_zero_alive_ratio_retains_an_auxiliary_probability(evidence):
    checks, _ = evidence
    records = [c for c in checks if c["id"].startswith("alive-ratio-") and c["inputs"]["v"] == 0]
    assert records
    for record in records:
        assert record["inputs"]["U"] == 0
        assert record["bound"] >= 1
        assert record["observed"] <= record["bound"] + 1e-12
    # A zero denominator is not counted as an observed normalized alive law.
    assert all(c["inputs"]["zero_alive_auxiliary"] == "worst bounded test" for c in records)


def test_exact_source_bindings_do_not_credit_unmeasured_theorem_formulas(evidence):
    checks, inventory = evidence
    formulas = {e["id"]: e["formula"] for e in inventory["quantitative_expressions"]}
    for check in checks:
        for source in check["source_expressions"]:
            assert source["formula"] == formulas[source["id"]]
    native_stationary_attraction = next(
        e
        for e in inventory["quantitative_expressions"]
        if e["source_label"] == "thm-uniqueness-of-qsd" and "\\mathcal F_h^n" in e["formula"]
    )
    credited = {e["id"] for check in checks for e in check["source_expressions"]}
    assert native_stationary_attraction["id"] not in credited
