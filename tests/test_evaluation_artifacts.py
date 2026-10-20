"""Refuse missing/corrupt data, preserve zero-detection/zero-GT images."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from cellnura_release.evaluation.artifacts import (ArtifactError, checked_file, load_catalog,
                                                  load_input, sha256, verify_catalog_inputs)
from cellnura_release.evaluation.unified_matcher import evaluate, write_outputs


class FrozenArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.job = {"job_id": "demo", "dataset": "demo", "tau": 1,
                    "class_ids": [-1, 0, 1], "real_class_ids": [0]}
        self.job["images"] = self.write("images.csv", ["image_id"], [["a"], ["empty"], ["zero_detection"]])
        schema = ["image_id", "instance_id", "cx", "cy", "class_id"]
        self.job["ground_truth"] = self.write("gt.csv", schema, [["a", "g0", 0, 0, 0], ["zero_detection", "g1", 5, 5, 0]])
        self.job["prediction"] = self.write("pred.csv", schema, [["a", "p0", 0, 0, 0]])

    def write(self, name, fields, rows):
        path = self.root / name
        with path.open("w", newline="") as h:
            w = csv.writer(h); w.writerow(fields); w.writerows(rows)
        return {"path": name, "sha256": sha256(path), "rows": len(rows)}

    def replace_prediction(self, rows):
        self.job["prediction"] = self.write("pred.csv", ["image_id", "instance_id", "cx", "cy", "class_id"], rows)

    def test_zero_detection_and_empty_images_are_counted(self):
        r = evaluate(load_input(self.root, self.job))
        self.assertEqual(r.n_gt, 2)
        self.assertEqual(r.n_missed, 1)
        self.assertEqual([x["image_id"] for x in r.per_image], ["a", "empty", "zero_detection"])
        self.assertAlmostEqual(r.macro("unified"), 2 / 3)

    def test_missing_gt_file_is_not_a_zero_gt_image(self):
        (self.root / "gt.csv").unlink()
        with self.assertRaisesRegex(ArtifactError, "Missing artifact"):
            load_input(self.root, self.job)

    def test_changed_file_refused_before_evaluation(self):
        with (self.root / "pred.csv").open("a") as h:
            h.write("a,p1,0,0,0\n")
        with self.assertRaisesRegex(ArtifactError, "SHA-256 mismatch"):
            load_input(self.root, self.job)

    def test_missing_gt_row_refused_even_with_new_hash(self):
        replacement = self.write("gt.csv", ["image_id", "instance_id", "cx", "cy", "class_id"], [["a", "g0", 0, 0, 0]])
        self.job["ground_truth"]["sha256"] = replacement["sha256"]
        with self.assertRaisesRegex(ArtifactError, "Instance count mismatch"):
            load_input(self.root, self.job)

    def test_unknown_image_refused(self):
        self.replace_prediction([["not_in_manifest", "p", 0, 0, 0]])
        with self.assertRaisesRegex(ArtifactError, "Unknown image"):
            load_input(self.root, self.job)

    def test_undeclared_class_refused(self):
        self.replace_prediction([["a", "p", 0, 0, 99]])
        with self.assertRaisesRegex(ArtifactError, "Undeclared prediction class"):
            load_input(self.root, self.job)

    def test_declared_unknown_kept_and_penalizes_true_class(self):
        self.replace_prediction([["a", "p", 0, 0, -1]])
        r = evaluate(load_input(self.root, self.job))
        self.assertEqual(r.n_pred, 1)
        self.assertEqual(r.n_matched, 1)
        self.assertEqual(r.classes[0]["counts"]["FN_mis"], 1)
        self.assertEqual(r.macro("unified"), 0)

    def test_duplicate_instance_identifier_refused(self):
        self.replace_prediction([["a", "p", 0, 0, 0], ["a", "p", 1, 0, 0]])
        with self.assertRaisesRegex(ArtifactError, "Duplicate instance"):
            load_input(self.root, self.job)

    def test_nonfinite_coordinate_refused(self):
        self.replace_prediction([["a", "p", "nan", 0, 0]])
        with self.assertRaisesRegex(ArtifactError, "Nonfinite"):
            load_input(self.root, self.job)

    def test_path_traversal_refused(self):
        with self.assertRaisesRegex(ArtifactError, "escapes"):
            checked_file(self.root, {"path": "../outside.csv", "sha256": "0" * 64})

    def test_absolute_path_refused(self):
        with self.assertRaisesRegex(ArtifactError, "must be relative"):
            checked_file(self.root, {"path": str(self.root / "gt.csv"), "sha256": "0" * 64})

    def test_macro_real_csv_columns_match_header(self):
        r = evaluate(load_input(self.root, self.job))
        write_outputs(r, str(self.root / "eval"))
        with (self.root / "eval_per_class.csv").open() as h:
            rows = list(csv.reader(h))
        macro = next(x for x in rows if x[0] == "MACRO_REAL")
        self.assertEqual(len(macro), len(rows[0]))
        self.assertAlmostEqual(float(macro[rows[0].index("uni_F1")]), 2 / 3, places=4)

    def test_catalog_manifest_detects_changed_registry(self):
        catalog = {"schema_version": 1, "jobs": [self.job]}
        (self.root / "jobs.json").write_text(json.dumps(catalog))
        entry = {"path": "jobs.json", "sha256": sha256(self.root / "jobs.json")}
        (self.root / "artifact_manifest.json").write_text(json.dumps({"files": [entry]}))
        load_catalog(self.root)
        (self.root / "jobs.json").write_text(json.dumps({**catalog, "jobs": []}))
        with self.assertRaisesRegex(ArtifactError, "SHA-256 mismatch"):
            load_catalog(self.root)

    def test_catalog_alone_does_not_prove_job_inputs_are_present(self):
        missing = dict(self.job)
        missing["prediction"] = {"path": "missing_prediction.csv", "sha256": "0" * 64, "rows": 0}
        catalog = {"schema_version": 1, "jobs": [missing]}
        (self.root / "jobs.json").write_text(json.dumps(catalog))
        entry = {"path": "jobs.json", "sha256": sha256(self.root / "jobs.json")}
        (self.root / "artifact_manifest.json").write_text(json.dumps({"files": [entry]}))
        loaded = load_catalog(self.root)
        with self.assertRaisesRegex(ArtifactError, "Missing artifact"):
            verify_catalog_inputs(self.root, loaded)

    def test_all_job_inputs_can_be_verified_explicitly(self):
        catalog = {"schema_version": 1, "jobs": [self.job]}
        self.assertEqual(verify_catalog_inputs(self.root, catalog), 1)
