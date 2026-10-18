"""Code/data separation must preserve immutable data and avoid partial conflict writes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from cellnura_release.supplementary import attach_artifacts


class SupplementaryDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.destination = self.root / "source", self.root / "destination"
        self.source.mkdir()
        entries = []
        for name, value in [("a.csv", b"frozen a"), ("b.csv", b"frozen b")]:
            (self.source / name).write_bytes(value)
            entries.append({"path": name, "sha256": hashlib.sha256(value).hexdigest()})
        (self.source / "artifact_manifest.json").write_text(json.dumps({"files": entries}))
        self.pinned = self.root / "pinned.json"
        self.pinned.write_bytes((self.source / "artifact_manifest.json").read_bytes())

    def test_check_only_and_idempotent_attachment(self):
        check = attach_artifacts(self.source, self.destination, self.pinned, check_only=True)
        self.assertEqual(check["copied_files"], 0)
        self.assertFalse(self.destination.exists())
        first = attach_artifacts(self.source, self.destination, self.pinned)
        self.assertEqual(first["copied_files"], 3)
        second = attach_artifacts(self.source, self.destination, self.pinned)
        self.assertEqual(second["copied_files"], 0)
        self.assertEqual(second["existing_identical_files"], 3)

    def test_corrupt_source_is_refused_before_copy(self):
        (self.source / "b.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Source SHA256 mismatch"):
            attach_artifacts(self.source, self.destination, self.pinned)
        self.assertFalse(self.destination.exists())

    def test_destination_conflict_preserves_existing_and_copies_nothing(self):
        self.destination.mkdir()
        (self.destination / "b.csv").write_bytes(b"user's data")
        with self.assertRaisesRegex(ValueError, "no files replaced"):
            attach_artifacts(self.source, self.destination, self.pinned)
        self.assertEqual((self.destination / "b.csv").read_bytes(), b"user's data")
        self.assertFalse((self.destination / "a.csv").exists())

    def test_changed_manifest_and_path_traversal_are_refused(self):
        changed = {"files": [{"path": "../outside.csv", "sha256": "0" * 64}]}
        (self.source / "artifact_manifest.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            attach_artifacts(self.source, self.destination, self.pinned)
        self.pinned.write_bytes((self.source / "artifact_manifest.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "canonical relative path"):
            attach_artifacts(self.source, self.destination, self.pinned)
        self.assertFalse(self.destination.exists())
