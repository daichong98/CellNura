#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib, json, os, sys

import pandas as pd
import torch

from cellnura_release.baseline_runtime import resource, frozen_roles
BASE = resource('project_root')
HERE = resource('scripts')
sys.path.insert(0, HERE)
import micronet_standalone
import t08_micronet_external as ext
from unified_matcher import EvalInput, Instance, evaluate, write_outputs

# Released t08_micronet_external.infer_image already uses the native probability.
# Do not reapply softmax before overlap averaging.

RUNS = resource('runs')
SEG_DIR = f"{RUNS}/micronet_retrain_crchisto"
FINAL = resource('evaluation')
DS = "crchisto"
TAU = 10.0


def sha256(path, bs=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    seg_ckpt = f"{SEG_DIR}/best_model.pth"
    assert os.path.exists(seg_ckpt), f"Missing segmentation weights {seg_ckpt} (first run t06_train_micronet_crchisto.py)"
    seg_manifest = json.load(open(f"{SEG_DIR}/run_manifest.json"))

    model = micronet_standalone.MicroNet(num_input_channels=3, num_output_channels=2)
    state = torch.load(seg_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state["desc"] if "desc" in state else state, strict=True)
    model = model.cuda().eval()

    gt = ext.load_gt(DS)
    tr_names, te_names = ext.image_lists(DS)
    assert set(gt) >= set(tr_names) | set(te_names), "GT is missing test or training images"

    det_tr = ext.detect_split(model, DS, tr_names, f"{SEG_DIR}/det_train")
    det_te = ext.detect_split(model, DS, te_names, f"{SEG_DIR}/det_test")
    del model
    torch.cuda.empty_cache()

    val_names = set(frozen_roles('crchisto')['val'])
    items, ledger = ext.build_train_items(DS, det_tr, gt, TAU)
    ledger.to_csv(f"{SEG_DIR}/train_crop_ledger.csv", index=False)

    cls_out = f"{SEG_DIR}/classifier"
    os.makedirs(cls_out, exist_ok=True)
    best_acc, epochs_run, counts = ext.train_classifier(DS, items, val_names, cls_out)

    from torchvision import models as tvm
    clf = tvm.resnet18()
    clf.fc = torch.nn.Linear(clf.fc.in_features, ext.DS[DS]["n_cls"])
    clf.load_state_dict(torch.load(f"{cls_out}/best_model.pth", weights_only=True))
    cls_test = ext.classify_dets(DS, clf, det_te)

    # ---- Unified evaluation (τ=10, macro averaging over real classes, all 20 test images) ----
    gt_list, pr_list = [], []
    for name in te_names:
        pts, cls = gt[name]
        for j, (c, k) in enumerate(zip(pts, cls)):
            gt_list.append(Instance(name, f"g{j}", float(c[0]), float(c[1]), int(k)))
        for k, n in enumerate(det_te[name], 1):
            pr_list.append(Instance(name, f"d{k}", float(n["centroid"][0]), float(n["centroid"][1]),
                                    int(cls_test[name][k - 1])))
    ev = EvalInput(dataset="CRCHisto", tau=TAU, gt=gt_list, pred=pr_list,
                   image_ids=tuple(te_names), class_ids=tuple(range(4)), real_class_ids=(0, 1, 2, 3))
    res = evaluate(ev, mode="one_to_one")
    rc = (0, 1, 2, 3)
    outdir = f"{FINAL}/micronet_retrain_crchisto"
    os.makedirs(outdir, exist_ok=True)
    write_outputs(res, f"{outdir}/eval")

    pred_rows = [(p.image_id, p.x, p.y, p.cls) for p in pr_list]
    pd.DataFrame(pred_rows, columns=["image_id", "cx", "cy", "pred"]).to_csv(
        f"{SEG_DIR}/pred_all.csv", index=False)

    per_class = {c: {"uni_P": round(res.classes[c]["unified"]["P"], 4),
                     "uni_R": round(res.classes[c]["unified"]["R"], 4),
                     "uni_F1": round(res.classes[c]["unified"]["F1"], 4),
                     "TP": res.classes[c]["counts"]["TP"],
                     "FP_mis": res.classes[c]["counts"]["FP_mis"],
                     "FP_spur": res.classes[c]["counts"]["FP_spur"],
                     "FN_mis": res.classes[c]["counts"]["FN_mis"],
                     "FN_miss": res.classes[c]["counts"]["FN_miss"]}
                 for c in rc}
    print(json.dumps({"N_gt": res.n_gt, "N_pred": res.n_pred, "M": res.n_matched,
                      "missed": res.n_missed, "spurious": res.n_spurious,
                      "cls": round(res.macro("classification", real_class_ids=rc), 4),
                      "plusFN": round(res.macro("plus_missed_FN", real_class_ids=rc), 4),
                      "uni": round(res.macro("unified", real_class_ids=rc), 4),
                      "per_class": per_class}, ensure_ascii=False, indent=1), flush=True)

    manifest = {
        "run_id": "micronet_retrain_crchisto",
        "method": "Micro-Net within-fold retraining + shared ResNet-18 classifier",
        "tier": "A (within-fold retraining: segmentation trained from scratch using the CRCHisto training set of 80 images with a 68/12 image-level train/validation split; radius-8 synthetic disk GT disclosed)",
        "dataset": DS, "tau_px": TAU,
        "segmentation_checkpoint": seg_ckpt, "segmentation_sha256": sha256(seg_ckpt),
        "segmentation_training": {k: seg_manifest[k] for k in
                                  ("split", "epochs", "batch", "patches_per_img", "seed", "lr",
                                   "best_val_dice", "loss", "input", "train_images", "val_images")},
        "inference": "native-scale tile 256 stride 128, prob-mean stitching, 0.5 threshold, min_size 50; native output probability (no second softmax)",
        "softmax_correction": {
            "reason": "MicroNet output head already softmax-normalized; removed second softmax before tile averaging",
            "changed_inference_line": "prob = out[0, 1]",
            "implementation": "t08_micronet_external.py::infer_image",
            "implementation_sha256": sha256(os.path.join(os.path.dirname(__file__), "t08_micronet_external.py")),
        },
        "classifier": "resnet18-imagenet, 64x64 bbox+8px crops, Adam 1e-4, sqrt-class-weight, 15ep early-stop pat4",
        "classifier_split": "image-level 68/12 (frozen external_validation_splits.json)",
        "classifier_checkpoint": f"{cls_out}/best_model.pth",
        "classifier_sha256": sha256(f"{cls_out}/best_model.pth"),
        "best_val_acc": round(best_acc, 4), "epochs_run": epochs_run,
        "n_train_crops": len(items), "class_counts_train": counts,
        "prediction_export": f"{SEG_DIR}/pred_all.csv",
        "prediction_sha256": sha256(f"{SEG_DIR}/pred_all.csv"),
        "eval": {"N_gt": res.n_gt, "N_pred": res.n_pred, "M": res.n_matched,
                 "missed": res.n_missed, "spurious": res.n_spurious,
                 "cls_macroF1_real": round(res.macro("classification", real_class_ids=rc), 4),
                 "plusFN_macroF1_real": round(res.macro("plus_missed_FN", real_class_ids=rc), 4),
                 "uni_macroF1_real": round(res.macro("unified", real_class_ids=rc), 4)},
        "per_class": per_class,
        "gt_source": "step0_preprocessed classification_data centroids, CMAP{inf:0,epi:1,fib:2,others:3}; train target = r=8 disk GT (disclosed)",
        "code_sha256": sha256(__file__),
        "evaluator_sha256": sha256(f"{HERE}/unified_matcher.py"),
    }
    json.dump(manifest, open(f"{SEG_DIR}/run_eval_manifest.json", "w"), indent=2, ensure_ascii=False)
    print(f"[retrain-eval] manifest → {SEG_DIR}/run_eval_manifest.json", flush=True)
    print(f"[retrain-eval] eval outputs → {outdir}/eval_*.csv", flush=True)


if __name__ == "__main__":
    main()
