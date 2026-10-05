"""Primitive applicability and native conditional survival regression tests."""

import math
import unittest

from complete_chapter04_qsd_primitives import (
    analytic_primitive,
    conditional_survival,
    viscous_position_jacobian,
)
import numpy as np
from test_complete_chapter04_estimates import configuration


def reference_configuration(row=False):
    config = configuration()
    noise = {
        "innovation": "gaussian",
        "geometry": {"kind": "isotropic", "scale": {"kind": "constant", "values": [1.0]}},
    }
    config["kinetic"].update({
        "integrator": {"kind": "baoab", "dt": 0.04, "friction": 1.0},
        "boundary_schedule": "end_of_step",
        "noise": noise,
        "position_diffusion": 0.1,
    })
    config["clone_transform"]["jitter"] = noise
    config["clone_transform"].update({"restitution": 0.5, "collision_rotation": "haar"})
    config["clone_decision"]["revival_from_companion"] = True
    config["qft"] = {
        "viscosity": {"coefficient": 0.3, "bandwidth": 1.0, "row_normalized": row},
        "curl": None,
        "graph_viscosity": None,
        "innovation_shifts": [],
    }
    return config


class QSDPrimitiveTests(unittest.TestCase):
    def test_dense_jacobian_agrees_with_independent_directional_difference(self):
        x = np.array([[-0.3, 0.1], [0.7, -0.2], [0.0, 0.9]])
        v = np.array([[0.1, 0.2], [-0.2, 0.3], [0.4, -0.3]])
        direction = np.array([[0.4, -0.1], [0.0, 0.5], [-0.3, 0.2]])
        for row in (False, True):

            def force(z):
                w = np.exp(-np.sum((z[:, None] - z[None, :]) ** 2, axis=2) / 2)
                np.fill_diagonal(w, 0)
                w /= np.sum(w, axis=1, keepdims=True) if row else len(z)
                return 0.3 * np.sum(w[:, :, None] * (v[None, :] - v[:, None]), axis=1)

            epsilon = 1e-6
            finite_difference = (
                force(x + epsilon * direction) - force(x - epsilon * direction)
            ) / (2 * epsilon)
            action = viscous_position_jacobian(x, v, 0.3, 1.0, row) @ direction.ravel()
            np.testing.assert_allclose(action.reshape(x.shape), finite_difference, atol=1e-10)

    def test_row_pair_ratio_has_zero_position_derivative(self):
        jacobian = viscous_position_jacobian([[-1.0], [2.0]], [[0.4], [-0.5]], 0.3, 1.0, True)
        np.testing.assert_array_equal(jacobian, np.zeros((2, 2)))

    def test_logarithms_preserve_positive_minorization_below_float_range(self):
        p = analytic_primitive(reference_configuration(), 200, 1, False)
        self.assertTrue(math.isfinite(p["log_doob_minorization_lower_bound"]))
        self.assertLess(p["log_doob_minorization_lower_bound"], -100000)
        self.assertEqual(math.exp(p["log_doob_minorization_lower_bound"]), 0.0)
        self.assertGreater(p["coercivity_margin"], 0)

    def test_global_coercivity_failure_rejected(self):
        config = reference_configuration()
        config["qft"]["viscosity"]["coefficient"] = 100.0
        with self.assertRaisesRegex(ValueError, "coercivity"):
            analytic_primitive(config, 4, 1, False)

    def test_extra_force_provider_excluded(self):
        config = reference_configuration()
        config["qft"]["curl"] = {"coefficient": 0.1}
        with self.assertRaisesRegex(ValueError, "Additional providers"):
            analytic_primitive(config, 4, 1, False)

    def test_realized_jitter_event_and_noise_violation_are_separate(self):
        p = analytic_primitive(reference_configuration(), 4, 1, False)
        step = {
            "stages": [
                {
                    "stage": name,
                    "fields": {"positions": {"values": x}, "velocities": {"values": v}},
                }
                for name, x, v in (
                    ("literal_clone", [0.0, 0.0], [0.0, 0.0]),
                    ("B1_input", [0.0, 0.2], [0.0, 0.0]),
                    ("B1", [0.0, 0.2], [0.0, 0.0]),
                    ("terminal", [0.001, 0.2], [0.0, 0.0]),
                )
            ],
            "field_evaluations": [
                {"stage": name, "field": "executed_noise", "values": [0.0, 0.0]}
                for name in ("O", "position_diffusion")
            ],
        }
        rows = conditional_survival(step, p, 2, 1)
        self.assertTrue(rows[0]["eligible_jitter_event"])
        self.assertFalse(rows[1]["eligible_jitter_event"])
        self.assertIsNone(rows[1]["survival_floor_residual"])
        self.assertGreater(rows[0]["position_noise_identity_absolute_residual"], 0.0009)
        # The native executed_noise field stores eta, before the integrator's
        # q/s amplitudes. A nonzero probe prevents treating it as scaled noise.
        step["field_evaluations"][0]["values"] = [1.0, 0.0]
        step["field_evaluations"][1]["values"] = [2.0, 0.0]
        step["stages"][-1]["fields"]["positions"]["values"][0] = p["t"] * p["q"] + 2 * p["s"]
        corrected = conditional_survival(step, p, 2, 1)
        self.assertLess(corrected[0]["position_noise_identity_absolute_residual"], 1e-14)


if __name__ == "__main__":
    unittest.main()
