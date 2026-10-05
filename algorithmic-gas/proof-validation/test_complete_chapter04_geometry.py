"""Optimal normalized transport and regional applicability regressions."""

import unittest

from complete_chapter04_geometry import assignment_transport, regional_checks


class GeometryCompletionTests(unittest.TestCase):
    def test_unlabeled_permutations_have_zero_transport(self):
        self.assertEqual(assignment_transport([[-1.0], [2.0]], [[2.0], [-1.0]]), 0.0)

    def test_transport_is_probability_normalized(self):
        self.assertEqual(assignment_transport([[0.0]], [[1.0]]), 1.0)
        self.assertEqual(assignment_transport([[0.0]] * 4, [[1.0]] * 4), 1.0)

    def test_no_eligible_region_point_is_not_a_vacuous_bound_check(self):
        comparisons, count = regional_checks(
            [[2.0]], "constant", {"region_lower": [-0.1], "region_upper": [0.1]}, "outside"
        )
        self.assertEqual(comparisons, [])
        self.assertEqual(count, 0)

    def test_wrong_curvature_certificate_causes_actual_failure(self):
        certificate = {
            "region_lower": [-0.1],
            "region_upper": [0.1],
            "force_sup_upper": 1.0,
            "curvature_lower": 2.0,
            "curvature_upper": 2.0,
            "reward_quadratic_growth": 0.5,
            "reward_constant_growth": 0,
            "restoring_k": 0.5,
            "radial_defect_upper": 0,
        }
        comparisons, count = regional_checks([[0.0]], "quadratic", certificate, "wrong")
        self.assertEqual(count, 1)
        self.assertTrue(any(not c["passed"] for c in comparisons))


if __name__ == "__main__":
    unittest.main()
