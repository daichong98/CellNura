"""Meaningful CPU tests for split independence, complete states and inference."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from cellnura_release.models import MORPH_COLS
from cellnura_release.training import main as train_main,split_roles
from cellnura_release.inference import (load_checkpoint,predict_frame,feature_columns,validate_features,
                                      build_graphs,collate_graphs,main as inference_main,
                                      mlp_class_ids,evaluation_table,read_detections)


def synthetic_inputs(folder):
    generator=np.random.default_rng(2026)
    image_names=["train_a","train_b","val_a","test_a"]
    feature_names=([f"mobilevit_feature_{i}" for i in range(96)]+[f"swin_global_{i}" for i in range(1024)]+
                   MORPH_COLS+["FR1_2"]+[f"FR{i}" for i in range(3,38)])
    rows=[]
    for image in image_names:
        global_=generator.normal(size=1024).astype(np.float32)
        for nucleus in range(6):
            values=generator.normal(size=len(feature_names)).astype(np.float32)
            values[96:1120]=global_
            row=dict(zip(feature_names,values))
            row.update(original_image=image,nucleus_id=nucleus,cx=float(nucleus*3),cy=1.0,
                       label=99999 if image.startswith("test") else nucleus%4)
            rows.append(row)
    frame=pd.DataFrame(rows)
    # Shift test features far away; train-side scaler expectations remain exact.
    frame.loc[frame.original_image=="test_a",feature_names]+=10000
    feature_path=Path(folder)/"features.csv";frame.to_csv(feature_path,index=False)
    split_path=Path(folder)/"split.csv"
    pd.DataFrame({"dataset":["CoNSeP"]*4,"image_id":image_names,
                  "split":["train","train","val","test"]}).to_csv(split_path,index=False)
    return feature_path,split_path,frame


class CellNuraTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.feature,cls.split,cls.frame=synthetic_inputs(cls.root)
        cls.out=cls.root/"run"
        cls.argv=["--features",str(cls.feature),"--split-manifest",str(cls.split),
                  "--dataset","CoNSeP","--out",str(cls.out),"--device","cpu","--threads","2",
                  "--fusion-epochs","1","--gat-epochs","1","--mlp-epochs","1"]
        train_main(cls.argv)
        cls.state,cls.fusion,cls.gat,cls.mlp=load_checkpoint(cls.out/"checkpoint.pth")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_states_scalers_fit_train_only(self):
        state=self.state
        self.assertTrue({"fusion","gat","mlp","sc_l","sc_g","scaler","columns","config"}<=set(state))
        self.assertEqual(state["config"]["n_classes"],4)
        self.assertEqual(state["config"]["mlp_dropout"],0.3)
        self.assertEqual(state["config"]["gat_dropout"],0.4)
        self.assertEqual(state["sc_l"].n_samples_seen_,12)
        self.assertEqual(state["sc_g"].n_samples_seen_,12)
        self.assertEqual(state["scaler"].n_samples_seen_,12)
        train=self.frame[self.frame.original_image.str.startswith("train")]
        np.testing.assert_allclose(state["sc_l"].mean_,train[state["columns"]["mv_cols"]].to_numpy(np.float32).mean(axis=0,dtype=np.float64),atol=1e-7)

    def test_inference_ignores_gt_and_does_not_fit(self):
        test=self.frame[self.frame.original_image=="test_a"].copy()
        original=predict_frame(test,self.state,self.fusion,self.gat,self.mlp)
        test["label"]=np.nan
        no_gt=predict_frame(test.drop(columns="label"),self.state,self.fusion,self.gat,self.mlp)
        pd.testing.assert_frame_equal(original,no_gt)
        self.assertEqual(self.state["scaler"].n_samples_seen_,12)
        pd.testing.assert_frame_equal(original,pd.read_csv(self.out/"test_predictions.csv"),check_dtype=False)

    def test_empty_inference_and_key_validation(self):
        result=predict_frame(self.frame.iloc[:0],self.state,self.fusion,self.gat,self.mlp)
        self.assertEqual(list(result),["original_image","nucleus_id","cx","cy","pred"])
        self.assertEqual(len(result),0)
        with self.assertRaisesRegex(ValueError,"Duplicate image/nucleus"):
            validate_features(pd.concat([self.frame,self.frame.iloc[:1]]),self.state["columns"])
        bad=self.frame.copy();bad["swin_global_0_swin"]=123
        with self.assertRaisesRegex(ValueError,"Conflicting"):
            feature_columns(bad)

    def test_resume_retains_manifest_and_refuses_identity_change(self):
        before=(self.out/"run_manifest.json").read_bytes()
        train_main(self.argv+["--resume"])
        self.assertEqual((self.out/"run_manifest.json").read_bytes(),before)
        with self.assertRaisesRegex(ValueError,"identity/checkpoint mismatch"):
            train_main(self.argv+["--resume","--seed","43"])
        self.assertEqual((self.out/"run_manifest.json").read_bytes(),before)

    def test_duplicate_split_roles_rejected(self):
        duplicate=pd.read_csv(self.split)
        duplicate=pd.concat([duplicate,duplicate.iloc[[0]].assign(split="test")])
        path=self.root/"bad_split.csv";duplicate.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,"duplicated image roles"):
            split_roles(path,"CoNSeP")

    def test_block_graphs_have_no_cross_image_edges(self):
        frame=self.frame.iloc[:12].reset_index(drop=True)
        graphs=build_graphs(frame,np.zeros((len(frame),1024),np.float32))
        _,adjacency,_=collate_graphs(graphs,"cpu")
        self.assertEqual(float(adjacency[0,:6,6:].sum()),0.0)
        self.assertEqual(float(adjacency[0,6:,:6].sum()),0.0)
        self.assertEqual(float(adjacency[0].diag().sum()),12.0)

    def test_frozen_inference_refuses_relabeling_trained_images_as_test(self):
        manifest=pd.read_csv(self.split)
        manifest.loc[manifest.image_id=="train_a","split"]="test"
        changed=self.root/"changed_manifest.csv";manifest.to_csv(changed,index=False)
        argv=["--features",str(self.feature),"--checkpoint",str(self.out/"checkpoint.pth"),
              "--out",str(self.root/"relabel_pred.csv"),"--split","test","--split-manifest",str(changed)]
        with self.assertRaisesRegex(ValueError,"roles differ from checkpoint"):
            inference_main(argv)
        self.assertFalse((self.root/"relabel_pred.csv").exists())


class UnknownClassRuleTests(unittest.TestCase):
    def test_mlp_argmax_stays_inside_real_classes(self):
        logits = torch.tensor([[0.1, 3.0, 0.2], [4.0, 0.0, 0.1]])
        self.assertEqual(mlp_class_ids(logits, 3), [1, 0])

    def test_detection_without_mlp_row_is_unknown(self):
        classified = pd.DataFrame({"original_image": ["a"], "nucleus_id": [1], "cx": [1.0], "cy": [2.0], "pred": [4]})
        detections = pd.DataFrame({
            "image_id": ["a", "a"], "instance_id": ["d0", "d1"],
            "cx": [9.0, 1.0], "cy": [8.0, 2.0],
        })
        result = evaluation_table(classified, detections)
        self.assertEqual(result.pred.tolist(), [-1, 4])
        self.assertEqual(result.nucleus_id.tolist(), [0, 1])
        self.assertEqual(sorted(set(result.class_id.tolist())), [-1, 4])

    def test_detection_list_requires_instance_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "detections.csv"
            pd.DataFrame({"cx": [1.0], "cy": [2.0]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Detection list lacks"):
                read_detections(path)


if __name__=="__main__":
    unittest.main()
