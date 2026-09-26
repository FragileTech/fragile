"""Native benchmark accounting checks; run with Python's unittest runner."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "million_benchmark", ROOT / "tools/benchmark-million.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class BenchmarkTargetTest(unittest.TestCase):
    def test_initialization_can_reach_target(self):
        row = RUNNER.BENCH.run_case(
            ("quadratic", 20, "Gaussian", 0, 1000), target_error=1e20
        )
        self.assertTrue(row["target_reached"])
        self.assertEqual(row["iterations"], 0)
        self.assertEqual(row["evaluations"], 8)
        self.assertEqual(row["stop_reason"], "target_reached")

    def test_unreached_target_preserves_trajectory(self):
        case = ("quadratic", 20, "Gaussian", 0, 1000)
        baseline = RUNNER.BENCH.run_case(case)
        targeted = RUNNER.BENCH.run_case(case, target_error=0)
        self.assertFalse(targeted["target_reached"])
        for key in ("best", "evaluations", "iterations", "final_status", "error"):
            self.assertEqual(baseline[key], targeted[key], key)
        self.assertLessEqual(targeted["evaluations"], 1000)

    def test_cma_stops_at_target(self):
        row = RUNNER.run(("quadratic", "cma", 0, 0, 1_000_000, 1e-5))
        self.assertTrue(row["target_reached"])
        self.assertLessEqual(abs(row["best"] - row["reference_minimum"]), 1e-5)
        self.assertLess(row["evaluations"], 1_000_000)
        self.assertEqual(row["precision"]["cma_coordinates_bits"], 64)

    def test_fp64_request_rejects_mixed_engine(self):
        if RUNNER.precision_info()["swarm_coordinates_bits"] == 64:
            self.skipTest("This build supports FP64 swarm coordinates")
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/benchmark-million.py"),
             "--dry-run", "--output", "/tmp/unused-fp64-benchmark"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Requested FP64 is unsupported", result.stderr)


if __name__ == "__main__":
    unittest.main()
