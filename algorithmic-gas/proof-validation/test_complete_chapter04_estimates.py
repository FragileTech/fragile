"""Applicability and exact conditional-moment regressions for the Ch4 audit."""

import copy
import unittest

from complete_chapter04_estimates import (
    comparison,
    conditional_copying,
    integrate_measurement,
    keystone_constants,
    reward_profile,
)


def configuration():
    distance = {
        "kind": "squashed_phase_space",
        "position_radius": 2.0,
        "velocity_radius": 2.0,
        "lambda": 1.0,
    }
    donor = {
        "distance": distance,
        "law": "independent",
        "kernel": {"kind": "gaussian", "width": 2.0},
        "count": 1,
        "history_window": 0,
        "allow_self": False,
    }
    return {
        "distance_donors": copy.deepcopy(donor),
        "cloning_donors": copy.deepcopy(donor),
        "fitness": {
            "reward_standardizer": {"kind": "global", "sigma_min": 0.1},
            "diversity_standardizer": {"kind": "global", "sigma_min": 0.1},
            "reward_map": {"kind": "logistic", "amplitude": 2.0, "floor": 0.1},
            "diversity_map": {"kind": "logistic", "amplitude": 2.0, "floor": 0.1},
            "reward_exponent": 1.0,
            "diversity_exponent": 1.0,
            "distance_floor": 0.001,
        },
        "boundary": {
            "kind": "absorbing_box",
            "domain": {"lower": [-2.0], "upper": [2.0]},
        },
        "kinetic": {"velocity_cap": 2.0},
        "clone_decision": {"every": 1, "epsilon": 1e-6, "saturation": 1.0},
        "clone_transform": {
            "jitter_amplitude": 0.1,
            "jitter": {"innovation": "gaussian"},
        },
    }


class CompletionAuditTests(unittest.TestCase):
    def test_forced_copy_two_point_variance_retains_jitter_and_normalization(self):
        # Low fitness recipient certainly copies the high-fitness donor;
        # both output means coincide. Independent single-recipient Gaussian
        # jitter contributes sigma²/4 to population variance, not sigma².
        p, v = conditional_copying([[-1.0], [1.0]], [0.1, 2.0], [[0, 1], [1, 0]], configuration())
        self.assertEqual(p, [1, 0])
        self.assertAlmostEqual(v, 0.1**2 / 4)

    def test_complete_product_law_and_permutation_invariance(self):
        x, v, r = [[-1.0], [0.0], [0.5], [1.0]], [[0.0]] * 4, [-0.5, 0, -0.125, -0.5]
        result = integrate_measurement(x, v, r, configuration())
        self.assertEqual(len(result["patterns"]), 81)
        self.assertAlmostEqual(result["probability_mass"], 1)
        permutation = [2, 0, 3, 1]
        moved = integrate_measurement(
            [x[i] for i in permutation],
            [v[i] for i in permutation],
            [r[i] for i in permutation],
            configuration(),
        )
        self.assertAlmostEqual(result["expected_variance"], moved["expected_variance"])
        for i, j in enumerate(permutation):
            self.assertAlmostEqual(result["mean_row_pressure"][j], moved["mean_row_pressure"][i])

    def test_mutual_donors_rejected_before_product_integration(self):
        config = configuration()
        config["distance_donors"]["law"] = "mutual"
        with self.assertRaisesRegex(ValueError, "independence"):
            integrate_measurement([[-1.0], [1.0]], [[0.0]] * 2, [-1, -1], config)

    def test_real_violation_never_hidden_by_standard_error(self):
        bad = comparison("bad", ["reference"], 1.01, 1.0, standard_error=100.0)
        self.assertFalse(bad["passed"])

    def test_disabled_diversity_cannot_supply_keystone_constant(self):
        config = configuration()
        config["fitness"]["diversity_exponent"] = 0
        with self.assertRaisesRegex(ValueError, "diversity exponent"):
            keystone_constants(config, "quadratic", 1)

    def test_powered_map_has_positive_specialized_constant_and_exact_scope(self):
        config = configuration()
        config["fitness"]["diversity_exponent"] = 0.5
        constants = keystone_constants(config, "quadratic", 1)
        self.assertGreater(constants["log_chi_star"], -1000)
        self.assertIn("derivative-floor", constants["omega_scope"])
        self.assertTrue(constants["N_independent"])

    def test_nonquadratic_reward_bound_has_explicit_dimension(self):
        one, _ = reward_profile("rastrigin", 1)
        four, osc = reward_profile("rastrigin", 4)
        self.assertAlmostEqual(four, 2 * one)
        self.assertEqual(osc, 96)
        self.assertGreater(one, 60)


if __name__ == "__main__":
    unittest.main()
