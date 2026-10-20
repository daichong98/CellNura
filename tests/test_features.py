"""Small image tests; optional strict Swin load uses an explicitly supplied file."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from cellnura_release import features
from cellnura_release.paths import load_config, get_path


class FeatureTests(unittest.TestCase):
    def test_crop_and_png_channel_roundtrip(self):
        image=np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3)
        nucleus={"centroid":[32,30],"contour":[[25,22],[42,22],[42,39],[25,39]]}
        crop=features.crop_nucleus(image,nucleus)
        x,y,w,h=cv2.boundingRect(np.asarray(nucleus["contour"],np.int32))
        np.testing.assert_array_equal(crop,image[max(0,y-10):min(96,y+h+10),max(0,x-10):min(96,x+w+10)])
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"crop.png";cv2.imwrite(str(path),crop)
            canonical=Image.open(path).convert("RGB").resize((64,64),Image.Resampling.BILINEAR)
            np.testing.assert_array_equal(np.asarray(features.mobilevit_crop(crop)),np.asarray(canonical))

    def test_morph_and_ring_dimensions(self):
        image=np.full((48,48,3),230,np.uint8)
        cv2.circle(image,(24,24),12,(20,40,70),-1)
        morphology=features.morph_feats(image);rings=features.ring_feats(image)
        self.assertEqual(list(morphology),features.MORPH_COLS)
        self.assertEqual(list(rings),features.RING_COLS)
        self.assertEqual(len(rings),36)
        self.assertTrue(np.isfinite(list(morphology.values())).all())

    def test_config_paths_relocate_with_config(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/"config").mkdir();(root/"data").mkdir()
            (root/"data/file.csv").write_text("example")
            path=root/"config/run.json"
            path.write_text(json.dumps({"path_base":"..","paths":{"resource":"data/file.csv"}}))
            config=load_config(path)
            self.assertEqual(get_path("resource",config,required=True),root/"data/file.csv")

    def test_train_val_labels_zero_image_coverage_and_frozen_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            names=["train_a","val_a","test_a","test_zero"]
            image=np.full((64,64,3),230,np.uint8);cv2.circle(image,(32,32),9,(20,40,60),-1)
            for name in names:
                Image.fromarray(image).save(root/f"{name}.png")
                nucleus={} if name=="test_zero" else {"1":{"centroid":[32,32],"contour":[[24,24],[40,24],[40,40],[24,40]]}}
                (root/f"{name}.json").write_text(json.dumps({"tiles":[{"nuc":nucleus}]}))
            pd.DataFrame({"image_id":names,"path":[f"{n}.png" for n in names]}).to_csv(root/"images.csv",index=False)
            pd.DataFrame({"image_id":names,"path":[f"{n}.json" for n in names]}).to_csv(root/"segments.csv",index=False)
            pd.DataFrame({"dataset":["CoNSeP"]*4,"image_id":names,"split":["train","val","test","test"]}).to_csv(root/"split.csv",index=False)
            # Invalid test GT is removed before validation or label assignment.
            pd.DataFrame({"image_id":names,"cx":[32,32,np.nan,np.nan],"cy":[32,32,np.nan,np.nan],"class_id":[1,2,-999,-999]}).to_csv(root/"gt.csv",index=False)
            config={"paths":{},"backbones":{"swin":{"sha256":"mock"},"mobilevit":{"file_sha256":{}}}}
            (root/"config.json").write_text(json.dumps(config))
            argv=["--config",str(root/"config.json"),"--dataset","CoNSeP","--image-index",str(root/"images.csv"),
                  "--segmentation-index",str(root/"segments.csv"),"--split-manifest",str(root/"split.csv"),
                  "--training-ground-truth",str(root/"gt.csv"),"--swin-checkpoint",str(root/"swin.pth"),
                  "--mobilevit-checkpoint",str(root/"mobilevit"),"--out",str(root/"features.csv")]
            with patch.object(features,"load_swin",return_value=(None,{})),patch.object(features,"load_mobilevit",return_value=(None,None,{})),\
                 patch.object(features,"global_vector",return_value=np.zeros(1024,np.float32)),\
                 patch.object(features,"local_vectors",side_effect=lambda patches,*args:np.zeros((len(patches),96),np.float32)):
                features.main(argv)
            frame=pd.read_csv(root/"features.csv")
            self.assertEqual(frame.set_index("original_image").label.to_dict(),{"test_a":-1,"train_a":1,"val_a":2})
            coverage=pd.read_csv(root/"features.coverage.csv").set_index("image_id")
            self.assertEqual(set(coverage.index),set(names))
            self.assertEqual(coverage.loc["test_zero","feature_instances"],0)
            self.assertEqual(json.loads((root/"features.manifest.json").read_text())["test_label_assignment_count"],0)

    def test_config_training_ground_truth_is_used_without_cli_path(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            names=["train_a","val_a"]
            image=np.full((64,64,3),230,np.uint8);cv2.circle(image,(32,32),9,(20,40,60),-1)
            for name in names:
                Image.fromarray(image).save(root/f"{name}.png")
                nucleus={"1":{"centroid":[32,32],"contour":[[24,24],[40,24],[40,40],[24,40]]}}
                (root/f"{name}.json").write_text(json.dumps({"tiles":[{"nuc":nucleus}]}))
            pd.DataFrame({"image_id":names,"path":[f"{n}.png" for n in names]}).to_csv(root/"images.csv",index=False)
            pd.DataFrame({"image_id":names,"path":[f"{n}.json" for n in names]}).to_csv(root/"segments.csv",index=False)
            pd.DataFrame({"dataset":["CoNSeP"]*2,"image_id":names,"split":["train","val"]}).to_csv(root/"split.csv",index=False)
            pd.DataFrame({"image_id":names,"cx":[32,32],"cy":[32,32],"class_id":[1,2]}).to_csv(root/"gt.csv",index=False)
            config={"path_base":".","paths":{"training_ground_truth":"gt.csv"},
                    "backbones":{"swin":{"sha256":"mock"},"mobilevit":{"file_sha256":{}}}}
            (root/"config.json").write_text(json.dumps(config))
            argv=["--config",str(root/"config.json"),"--dataset","CoNSeP","--image-index",str(root/"images.csv"),
                  "--segmentation-index",str(root/"segments.csv"),"--split-manifest",str(root/"split.csv"),
                  "--swin-checkpoint",str(root/"swin.pth"),"--mobilevit-checkpoint",str(root/"mobilevit"),
                  "--out",str(root/"features.csv")]
            with patch.object(features,"load_swin",return_value=(None,{})),patch.object(features,"load_mobilevit",return_value=(None,None,{})),\
                 patch.object(features,"global_vector",return_value=np.zeros(1024,np.float32)),\
                 patch.object(features,"local_vectors",side_effect=lambda patches,*args:np.zeros((len(patches),96),np.float32)):
                features.main(argv)
            frame=pd.read_csv(root/"features.csv").set_index("original_image")
            self.assertEqual(frame.label.to_dict(),{"train_a":1,"val_a":2})

    @unittest.skipUnless(os.environ.get("CELLNURA_TEST_SWIN"),"optional audited Swin checkpoint not supplied")
    def test_real_swin_loads_all_325_tensors(self):
        model,info=features.load_swin(os.environ["CELLNURA_TEST_SWIN"],"cpu","70812ab6b0a7a38712409d13976df9431632466eaacf991d5e90d9a1e91f3ab1")
        self.assertEqual(info["loaded_tensors"],325)
        self.assertEqual(info["target_tensors"],325)
        self.assertFalse(model.training)


if __name__=="__main__":
    unittest.main()
