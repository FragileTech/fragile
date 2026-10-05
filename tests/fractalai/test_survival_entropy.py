"""Independently check survivor entropy using input/output/flag enumeration."""

import math

import numpy as np
import pytest
from scipy.linalg import eig
from scipy.special import rel_entr

from fragile.fractalai.theory.survival_entropy import (
    finite_survivor_entropy_balance,
    UniformSurvivalEntropyBudget,
)


@pytest.mark.parametrize("p", [[0.6, 0.3, 0.1], [0.01, 0.04, 0.95], [0.0, 1.0, 0.0]])
def test_exact_rejected_and_surviving_entropy_against_joint_enumeration(p):
    q = np.array([[0.75, 0.15, 0.02], [0.05, 0.6, 0.1], [0.01, 0.04, 0.4]])
    values, vectors = eig(q.T)
    index = int(np.argmax(values.real))
    nu = vectors[:, index].real
    nu = nu / nu.sum()
    p = np.array(p)
    result = finite_survivor_entropy_balance(q, p, nu)
    # Retain the input on rejection and each real output on acceptance.
    # Both experiments have the same conditional law given the input.
    channels = np.column_stack([1 - q.sum(1), q])
    joint_p, joint_nu = p[:, None] * channels, nu[:, None] * channels
    assert float(rel_entr(joint_p, joint_nu).sum()) == pytest.approx(result.input_entropy)
    assert result.reconstructed_input_entropy == pytest.approx(result.input_entropy, abs=2e-14)
    assert result.backward_information >= -2e-14
    assert result.output_entropy <= result.input_entropy / q.sum(1).min() + 2e-14
    # Compute backward information from normalized posteriors independently.
    output = p @ q
    expected = 0.0
    for j in range(3):
        if output[j] > 0:
            pp = p * q[:, j] / output[j]
            rr = nu * q[:, j] / (nu @ q)[j]
            expected += output[j] / output.sum() * float(rel_entr(pp, rr).sum())
    assert result.backward_information == pytest.approx(expected, abs=2e-14)


@pytest.mark.parametrize("exponent", [2e-6, 7e-11, 0.2])
def test_uniform_survival_prefactor_is_stable_and_monotone(exponent):
    budget = UniformSurvivalEntropyBudget(exponent)
    assert budget.uniform_prefactor == pytest.approx(1 / -math.expm1(-exponent))
    values = [budget.log_prefactor(n) for n in [1, 2, 100, 1000000]]
    assert values == sorted(values, reverse=True)
    assert budget.log_prefactor(100, steps=3) == pytest.approx(3 * budget.log_prefactor(100))
    assert budget.log_prefactor(1000000, steps=0) == pytest.approx(0)


def test_qsd_identity_is_required_and_not_replaced_by_a_reference_density():
    with pytest.raises(ValueError, match="QSD"):
        finite_survivor_entropy_balance(
            np.diag([0.9, 0.5]), np.array([0.5, 0.5]), np.array([0.5, 0.5])
        )


@pytest.mark.parametrize("alive_floor", [2e-6, 7e-11])
def test_actual_laplace_floor_retains_its_stronger_survival_bound(alive_floor):
    budget = UniformSurvivalEntropyBudget.from_alive_floor(alive_floor)
    assert budget.uniform_prefactor == pytest.approx(1 / alive_floor)
    assert budget.log_prefactor(31) == pytest.approx(
        -math.log(-math.expm1(31 * math.log1p(-alive_floor)))
    )
