import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from cellnura_release.features import read_image, validate_image_references
from cellnura_release.paths import project_root
from cellnura_release.training import split_roles
from cellnura_release.verification import verify_files


class ReleaseGuardrails(unittest.TestCase):
    def test_failed_source_check_reaches_process_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            config=json.loads((project_root()/"configs/release.json").read_text())
            config["path_base"]=str(project_root())
            config["baseline_paths"]["tsfd_source"]=str(Path(directory)/"missing_tsfd")
            path=Path(directory)/"config.json";path.write_text(json.dumps(config))
            result=subprocess.run([sys.executable,"-m","cellnura_release","baseline","check","tsfd",
                                   "--sources-only","--config",str(path)],
                                  cwd=project_root(),capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn("TSFD upstream not fetched",result.stdout)

    def test_array_positions_are_not_silently_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"images.npy"
            np.save(path,np.zeros((2,4,4,3),np.uint8))
            for bad in [-1,0.5,np.inf]:
                row=pd.Series({"path":"images.npy","array_index":bad})
                with self.assertRaises(ValueError):read_image(row,Path(directory),{})

    def test_alias_image_ids_cannot_hide_same_tile(self):
        frame=pd.DataFrame([{"image_id":"train_a","path":"images.npy","array_index":0},
                            {"image_id":"val_b","path":"images.npy","array_index":0}]).set_index("image_id")
        with self.assertRaisesRegex(ValueError,"same physical tile"):
            validate_image_references(frame,Path("."),{"train_a","val_b"})

    def test_external_training_manifest_materializes_existing_validation(self):
        expected={"CoNSeP":{"train":22,"val":5,"test":14},"CRCHisto":{"train":68,"val":12,"test":20}}
        for ds,counts in expected.items():
            roles=split_roles(project_root()/"artifacts/training_split_manifest.csv",ds)
            self.assertEqual({k:len(v) for k,v in roles.items()},counts)
            self.assertFalse(roles["train"]&roles["val"])
            self.assertFalse((roles["train"]|roles["val"])&roles["test"])

    def test_release_identity_rejects_tamper_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file=root/"fixture.txt";file.write_bytes(b"frozen")
            manifest={"files":[{"path":"fixture.txt","bytes":6,"sha256":hashlib.sha256(b"frozen").hexdigest()}]}
            self.assertEqual(verify_files(root,manifest),1)
            file.write_bytes(b"edited")
            with self.assertRaisesRegex(ValueError,"Identity changed"):verify_files(root,manifest)
            manifest["files"][0]["path"]="../outside.txt"
            with self.assertRaises(ValueError):verify_files(root,manifest)


if __name__=="__main__":
    unittest.main()

