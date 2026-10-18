import unittest
import numpy as np
from cellnura_release.evaluation.statistics import paired_bootstrap, validate_reference
from cellnura_release.evaluation.artifacts import ArtifactError
from cellnura_release.evaluation.unified_matcher import Instance, EvalInput, evaluate


class StatisticsProtocolTests(unittest.TestCase):
    def test_one_draw_reference_difference_is_recorded_not_hidden(self):
        catalog = {"bootstrap_reference_iterations": 10000, "bootstrap_comparisons": [
            {"expected": {"point_delta": 0.0062, "ci_low": -0.0653, "ci_high": 0.0483,
                          "bootstrap_fraction_positive": 0.4848}}]}
        row = {"dataset": "CoNSeP", "opponent": "DIST", "point_delta": 0.0062,
               "ci_low": -0.0653, "ci_high": 0.0483, "bootstrap_fraction_positive": 0.4847}
        result = validate_reference(catalog, [row], 10000)
        self.assertEqual(result["reference_mismatches"], [])
        self.assertFalse(result["reference_all_values_identical_4dp"])
        self.assertEqual(len(result["reference_fraction_differences"]), 1)
        row["bootstrap_fraction_positive"] = 0.4846
        self.assertEqual(len(validate_reference(catalog, [row], 10000)["reference_mismatches"]), 1)

    def test_fold_mean_does_not_pool_nuclei(self):
        b1 = np.array([[[1, 1, 1]]]); o1 = np.array([[[0, 1, 1]]])
        b2 = np.tile([[[0, 1, 1]]], (9, 1, 1)); o2 = np.tile([[[1, 1, 1]]], (9, 1, 1))
        r = paired_bootstrap([b1, b2], [o1, o2], iterations=100, seed=42)
        self.assertEqual(r["point_delta"], 0)
        self.assertEqual(r["ci_low"], 0)
        self.assertEqual(r["ci_high"], 0)
        # The pooled comparison would be 0.1 - 0.9 = -0.8.

    def test_pairing_equal_models_yields_zero_interval(self):
        x = np.array([[[1, 2, 3]], [[0, 1, 2]], [[1, 1, 1]]])
        r = paired_bootstrap([x], [x.copy()], iterations=100)
        self.assertEqual(r["ci_low"], 0)
        self.assertEqual(r["ci_high"], 0)

    def test_different_per_image_gt_cannot_be_paired(self):
        x = np.array([[[1, 1, 1]]]); y = np.array([[[1, 2, 1]]])
        with self.assertRaisesRegex(ArtifactError, "identical per-image GT"):
            paired_bootstrap([x], [y], 10)

    def test_maximum_cardinality_precedes_distance(self):
        # g0 can match p0/p1; g1 only p0. Nearest-first would lose a match.
        ev = EvalInput("demo", 1.1,
                       [Instance("a", "g0", 0, 0, 0), Instance("a", "g1", 2, 0, 0)],
                       [Instance("a", "p0", 1, 0, 0), Instance("a", "p1", -1, 0, 0)])
        self.assertEqual(evaluate(ev).n_matched, 2)

    def test_legacy_duplicate_counter_matches_claims(self):
        ev = EvalInput("demo", 1,
                       [Instance("a", "g", 0, 0, 0)],
                       [Instance("a", "p0", 0, 0, 0), Instance("a", "p1", 0.1, 0, 0)])
        r = evaluate(ev, "legacy_r1")
        self.assertEqual(r.n_duplicate_claims, 1)
        self.assertEqual(r.n_gt, r.n_matched + r.n_missed - r.n_duplicate_claims)
