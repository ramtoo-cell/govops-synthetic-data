
"""Tests for the synthetic data validator."""
import unittest

from src.synthetic_data import SyntheticDataValidator

REAL = [{"age": 30, "zip": "10001", "score": 0.5},
        {"age": 40, "zip": "10002", "score": 0.6},
        {"age": 50, "zip": "10003", "score": 0.7}]


class TestSyntheticData(unittest.TestCase):
    def setUp(self):
        self.v = SyntheticDataValidator(REAL, quasi_identifiers=("age", "zip"),
                                        ranges={"age": (0, 120), "score": (0.0, 1.0)})

    def test_overlap_detected(self):
        synth = [{"age": 30, "zip": "10001", "score": 0.55}]
        result = self.v.check_overlap(synth)
        self.assertFalse(result.passed)
        self.assertEqual(result.value, 1.0)

    def test_no_overlap_passes(self):
        synth = [{"age": 31, "zip": "10001", "score": 0.55}]
        self.assertTrue(self.v.check_overlap(synth).passed)

    def test_range_violation(self):
        synth = [{"age": 200, "zip": "99999", "score": 0.5}]
        result = self.v.check_ranges(synth, ["age", "score"])
        self.assertFalse(result.passed)
        self.assertEqual(result.value, 1.0)

    def test_range_ok(self):
        synth = [{"age": 35, "zip": "10005", "score": 0.55}]
        self.assertTrue(self.v.check_ranges(synth, ["age", "score"]).passed)

    def test_drift_detected(self):
        synth = [{"age": 300, "score": 9.0} for _ in range(3)]
        result = self.v.check_drift(synth, ["age", "score"])
        self.assertFalse(result.passed)

    def test_drift_ok(self):
        synth = [{"age": 32, "score": 0.52},
                 {"age": 42, "score": 0.62},
                 {"age": 52, "score": 0.72}]
        self.assertTrue(self.v.check_drift(synth, ["age", "score"]).passed)

    def test_validate_all_pass(self):
        synth = [{"age": 32, "zip": "10005", "score": 0.52},
                 {"age": 42, "zip": "10006", "score": 0.62},
                 {"age": 52, "zip": "10007", "score": 0.72}]
        report = self.v.validate(synth, ["age", "score"])
        self.assertTrue(report["passed"])
        self.assertEqual(set(report["checks"]), {"overlap", "ranges", "drift"})

    def test_validate_fails(self):
        synth = [{"age": 30, "zip": "10001", "score": 0.5}]
        report = self.v.validate(synth, ["age", "score"])
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["overlap"]["passed"])

    def test_no_qids_overlap_vacuous(self):
        v = SyntheticDataValidator(REAL)
        self.assertTrue(v.check_overlap([{"age": 30}]).passed)

    def test_bad_tolerance(self):
        with self.assertRaises(ValueError):
            SyntheticDataValidator(REAL, mean_tolerance=-1)


if __name__ == "__main__":
    unittest.main()
