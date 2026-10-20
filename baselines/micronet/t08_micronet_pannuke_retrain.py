#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t08_micronet_pannuke_retrain.py — Micro-Net PanNuke within-fold retraining evaluation (Group A).

The protocol uses segmentation weights trained from scratch within each fold:
  * Segmentation: runs/micronet_retrain_pannuke_fold{f}/best_model.pth, trained
    from scratch with t06_train_micronet_pannuke.py.
  * Inference: full 256-pixel image → resize to 252 → preproc → forward pass
    → softmax argmax → connected components with min_size 50 → fill_holes
    → scale centroids by 256/252 back to the original image coordinates.
  * Classification: shared ResNet-18 v3 with GT-centered 64×64 training crops
    and identical windows centered on test detections; no detections discarded
    (F1/F4 audit fixes). Uses the same within-fold classifier weights as the
    Micro-Net/StarDist/DIST PanNuke rows.
  * Evaluation: one-to-one unified_matcher with τ=12 px, macro averaging over
    real classes 0–4, and all test images.
Outputs: final_tables/micronet_retrain_pannuke/eval_fold{f}_*;
         runs/micronet_retrain_pannuke_fold{f}/pred_all.csv
Usage: python t08_micronet_pannuke_retrain.py [--folds 1 2 3]
"""
import argparse, hashlib, json, os, sys, time

import cv2
import numpy as np
import pandas as pd
import torch

from cellnura_release.baseline_runtime import resource, frozen_roles
BASE = resource('project_root')
HERE = resource('scripts')
sys.path.insert(0, HERE)
import micronet_standalone
import t08_classifier_v3 as cv3
from unified_matcher import write_outputs
from scipy import ndimage
from skimage import morphology

RUNS = resource('runs')
FINAL = resource('evaluation')
OUTD = f"{FINAL}/micronet_retrain_pannuke"


def sha256(path, bs=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def detect_test_images(model, imgs_npy, te_ids, fold):
    """Whole-image inference with within-fold retrained weights. Return rows [{image_key, nucleus_id, cx, cy}] and detection counts per image."""
    from PIL import Image  # noqa: F401  Keep imports consistent with Group B
    rows = []
    per_img = {}
    t0 = time.time()
    for n, img_id in enumerate(sorted(te_ids), 1):
        f, i, _, _ = cv3.img_arrays_static[img_id]
        img = np.clip(np.asarray(imgs_npy[f][i]), 0, 255).astype(np.uint8)
        resized = cv2.resize(img, (252, 252), interpolation=cv2.INTER_LINEAR)
        value = micronet_standalone.MicroNet.preproc(resized).astype(np.float32)
        x = torch.from_numpy(value.transpose(2, 0, 1)[None]).cuda()
        with torch.inference_mode():
            output = model(x)[0]
        probs = output[0].permute(1, 2, 0).cpu().numpy()
        inst = ndimage.label(probs.argmax(2))[0]
        inst = morphology.remove_small_objects(inst, min_size=50)
        canvas = np.zeros(inst.shape, dtype=np.int32)
        for iid in np.unique(inst):
            if iid:
                canvas[ndimage.binary_fill_holes(inst == iid)] = iid
        inst = canvas
        k = 0
        for iid in np.unique(inst):
            if iid == 0:
                continue
            y, x = np.nonzero(inst == iid)
            ox = (x.mean() + 0.5) * 256 / 252 - 0.5
            oy = (y.mean() + 0.5) * 256 / 252 - 0.5
            rows.append({"image_key": f"fold{fold}_image_{i}_segmentation",
                         "nucleus_id": k, "cx": float(ox), "cy": float(oy)})
            k += 1
        per_img[img_id] = k
        if n % 500 == 0:
            print(f"  detect fold{fold}: {n}/{len(te_ids)} ({n/(time.time()-t0):.1f}/s)", flush=True)
    print(f"  detect fold{fold} DONE: {len(rows)} detections ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows, columns=["image_key", "nucleus_id", "cx", "cy"]), per_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    cv3._IMGS = cv3.img_arrays()
    cv3.img_arrays_static = cv3._IMGS
    imgs_npy = {f: np.load(f"{resource('pannuke')}/Fold {f}/images/fold{f}/images.npy", mmap_mode="r")
                for f in (1, 2, 3)}
    os.makedirs(OUTD, exist_ok=True)
    summary = []
    for tf in a.folds:
        seg_dir = f"{RUNS}/micronet_retrain_pannuke_fold{tf}"
        seg_ckpt = f"{seg_dir}/best_model.pth"
        assert os.path.exists(seg_ckpt), f"Missing {seg_ckpt} (first run t06_train_micronet_pannuke.py --folds {tf})"
        seg_manifest = json.load(open(f"{seg_dir}/run_manifest.json"))

        model = micronet_standalone.MicroNet(num_input_channels=3, num_output_channels=2)
        state = torch.load(seg_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state["desc"] if "desc" in state else state, strict=True)
        model = model.cuda().eval()

        role, tr_ids, va_ids, te_ids = cv3.split_for(tf)
        det_df, per_img = detect_test_images(model, imgs_npy, te_ids, tf)
        det_df.to_csv(f"{seg_dir}/det_test.csv", index=False)
        del model
        torch.cuda.empty_cache()

        # Shared classifier v3 (same weights as the Micro-Net/StarDist/DIST PanNuke rows)
        clf = cv3.models.resnet18()
        clf.fc = cv3.nn.Linear(clf.fc.in_features, 5)
        clf_path = f"{cv3.RUNS}/resnet18_v3_classifier_testfold{tf}/best_model.pth"
        clf.load_state_dict(torch.load(clf_path, map_location="cpu", weights_only=True))
        clf = clf.cuda().eval()

        res, cls_m, pfn_m, uni_m = cv3.eval_method(str(seg_dir), f"{seg_dir}/det_test.csv", tf,
                                                   str(OUTD), clf)
        write_outputs(res, f"{OUTD}/eval_fold{tf}")
        pred_rows = [(p.image_id, p.x, p.y, p.cls) for p in res.pred_instances] \
            if hasattr(res, "pred_instances") else None
        # EvalResult does not store the instance list; reconstruct it from detections and classification results
        det = pd.read_csv(f"{seg_dir}/det_test.csv")
        det["img_id"] = det.image_key.str.replace(r"_segmentation$", "", regex=True)
        items, keys = [], []
        for r in det.itertuples():
            items.append((r.img_id, float(r.cx), float(r.cy), 0))
            keys.append(r.img_id)
        xs = []
        B = 512
        for s in range(0, len(items), B):
            chunk = items[s:s + B]
            pil_list = []
            for img_id, cx, cy, _ in chunk:
                f, i, imgs, _ = cv3.img_arrays_static[img_id]
                img = np.clip(np.asarray(imgs[i]), 0, 255).astype(np.uint8)
                pil_list.append(cv3.crop64(img, cx, cy))
            batch = torch.stack([cv3._tf(p) for p in pil_list]).cuda()
            with torch.inference_mode():
                xs.extend(clf(batch).argmax(1).cpu().tolist())
        pd.DataFrame([(img_id, cx, cy, int(c)) for (img_id, cx, cy, _), c in zip(items, xs)],
                     columns=["image_id", "cx", "cy", "pred"]).to_csv(f"{seg_dir}/pred_all.csv", index=False)

        from unified_matcher import Instance, evaluate, EvalInput  # noqa: E402
        summary.append({"fold": tf, "N_gt": res.n_gt, "N_pred": res.n_pred, "M": res.n_matched,
                        "missed": res.n_missed, "spurious": res.n_spurious,
                        "cls": round(cls_m, 4), "plusFN": round(pfn_m, 4), "uni": round(uni_m, 4),
                        "best_val_dice": seg_manifest["best_val_dice"]})
        print(f"[fold{tf}] N_gt={res.n_gt} N_pred={res.n_pred} M={res.n_matched} "
              f"missed={res.n_missed} spur={res.n_spurious} cls={cls_m:.4f} +FN={pfn_m:.4f} uni={uni_m:.4f}",
              flush=True)

    import numpy as np2
    unis = [s["uni"] for s in summary]
    mean, sd = float(np2.mean(unis)), float(np2.std(unis, ddof=1))
    manifest = {
        "run_id": "micronet_retrain_pannuke",
        "method": "Micro-Net within-fold retraining (per outer fold) + shared ResNet-18 v3 classifier",
        "tier": "A (within-fold retraining: each segmentation model is trained from scratch on its fold training images, with fixed image-level splits)",
        "dataset": "PanNuke", "tau_px": 12.0,
        "segmentation_checkpoints": {f"fold{tf}": f"{RUNS}/micronet_retrain_pannuke_fold{tf}/best_model.pth"
                                     for tf in a.folds},
        "segmentation_sha256": {f"fold{tf}": sha256(f"{RUNS}/micronet_retrain_pannuke_fold{tf}/best_model.pth")
                                for tf in a.folds},
        "inference": "whole 256 patch -> resize 252 -> preproc -> argmax CC min_size 50 + fill_holes -> centroid x256/252",
        "classifier": "shared ResNet-18 v3 (GT-anchored 64x64 center window, zero-dropout, per-fold)",
        "classifier_checkpoints": {f"fold{tf}": f"{cv3.RUNS}/resnet18_v3_classifier_testfold{tf}/best_model.pth"
                                   for tf in a.folds},
        "folds": summary,
        "uni_macroF1_mean": round(mean, 4), "uni_macroF1_sd": round(sd, 4),
        "prediction_exports": {f"fold{tf}": f"{RUNS}/micronet_retrain_pannuke_fold{tf}/pred_all.csv"
                               for tf in a.folds},
        "prediction_sha256": {f"fold{tf}": sha256(f"{RUNS}/micronet_retrain_pannuke_fold{tf}/pred_all.csv")
                              for tf in a.folds},
        "code_sha256": sha256(__file__),
        "evaluator_sha256": sha256(f"{HERE}/unified_matcher.py"),
    }
    json.dump(manifest, open(f"{RUNS}/micronet_retrain_pannuke_run_manifest.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"[pannuke-retrain] mean uni = {mean:.4f} ± {sd:.4f} over {len(summary)} folds", flush=True)
    print(f"[pannuke-retrain] manifest → {RUNS}/micronet_retrain_pannuke_run_manifest.json", flush=True)


if __name__ == "__main__":
    main()
