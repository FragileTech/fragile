"""Applicability and individual-violation regressions for local native audit."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from audit_local_structural_kinetics import (
    AllStepsReader,
    live_mask,
    load_skipper,
    pathwise_result,
    same_well_membership,
    SKIPPER,
)
from complete_local_structural_kinetics import add_cached_records


class LocalKineticAuditTests(unittest.TestCase):
    def test_second_query_must_remain_in_same_well(self):
        center, first, second = same_well_membership(([1.001], [1.002]), ([2.001], [2.002]), 0.01)
        self.assertEqual(center, [1])
        self.assertTrue(first)
        self.assertFalse(second)

    def test_individual_numerical_violation_is_rejected(self):
        self.assertFalse(pathwise_result(1.0, 0.99, 0.96, True)["passed"])
        self.assertTrue(pathwise_result(1.0, 0.95, 0.96, True)["passed"])
        self.assertIsNone(pathwise_result(1.0, 0.99, 0.96, False)["passed"])

    def test_dead_storage_is_excluded_from_applicability(self):
        step = {
            "stages": [
                {
                    "stage": "B1_input",
                    "validity": [
                        {"invalid": False, "terminated": False},
                        {"invalid": False, "terminated": True},
                    ],
                }
            ]
        }
        self.assertEqual(live_mask(step, "B1_input", 2), [True, False])

    def test_decoder_keeps_all_steps_and_validates_remainder(self):
        # {"steps": [1, 2]} -- the existing first-step decoder keeps only 1.
        raw = b"\xa1\x65steps\x82\x01\x02"
        reader = AllStepsReader(raw)
        self.assertEqual(reader.value(), {"steps": [1, 2]})
        self.assertEqual(reader.position, len(raw))

    def test_cached_ledger_recomputes_every_individual_predicate(self):
        walker = pathwise_result(1.0, 0.95, 0.96, True)
        walker.update({"query1_inside": True, "query2_inside": True, "input_cost": 1.0})
        payload = {
            "records": [
                {
                    "shared_noise": True,
                    "potential_force_identity": True,
                    "live_masks": {"B1_input": [[True], [True]]},
                    "radii": {
                        "0.01": {
                            "walkers": [walker],
                            "normalized_swarm": pathwise_result(1.0, 0.95, 0.96, True),
                        }
                    },
                }
            ]
        }

        def stats():
            return {
                "0.01": {
                    "walker_candidates": 0,
                    "query1_inside": 0,
                    "query2_inside": 0,
                    "qualified_checks": 0,
                    "failed_checks": 0,
                    "full_swarm_steps": 0,
                    "full_swarm_failed": 0,
                    "maximum_signed_residual": None,
                    "maximum_observed_ratio": None,
                }
            }

        result = stats()
        self.assertEqual(add_cached_records(result, payload, {0.01: {"rho_upper": 0.96}}), (0, 0))
        self.assertEqual(result["0.01"]["qualified_checks"], 1)
        walker["observed"] = 0.99
        with self.assertRaises(ValueError):
            add_cached_records(stats(), payload, {0.01: {"rho_upper": 0.96}})


@unittest.skipUnless(shutil.which("cc"), "Optional native skip reader needs a system C compiler")
class NativeSubtreeSkipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        path = Path(cls.temp.name) / "skip.so"
        subprocess.run(
            [
                "cc",
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-shared",
                "-fPIC",
                str(Path(__file__).with_name("skip_native_cbor.c")),
                "-o",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        load_skipper(path)

    @classmethod
    def tearDownClass(cls):
        SKIPPER.clear()
        cls.temp.cleanup()

    def test_discarded_map_arrays_match_definite_decoder(self):
        raw = b"\xa1\x61a\x84\x01\x18\xff\xfb\x3f\xf0\x00\x00\x00\x00\x00\x00\xf6"
        reader = AllStepsReader(raw)
        self.assertIsNone(reader.value(keep=False))
        self.assertEqual(reader.position, len(raw))

    def test_truncated_indefinite_and_excess_depth_rejected(self):
        for raw in (
            b"\x9f",
            b"\xa1\x61a",
            b"\x5b\xff\xff\xff\xff\xff\xff\xff\xff",
            b"\x81" * 66 + b"\x01",
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                AllStepsReader(raw).value(keep=False)


if __name__ == "__main__":
    unittest.main()
