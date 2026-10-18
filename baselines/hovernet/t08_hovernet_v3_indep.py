"""t08_hovernet_v3_indep.py — HoVer-Net v3.1 independent baseline: native type branch + unified evaluation.

Method: use v3.1 independent within-fold segmentation weights and padded 2×2-window inference to extract the type-branch output at each detected centroid
(the same correct method as t08_hovernet_tphead_eval.py: per-window lookup, without accumulating argmax outputs).
GPU memory usage <2GB (batch=8 fp16); can run concurrently with TSFD.
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import glob, json, os, re, sys, time
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
BASE = resource('project_root')
PKG = resource('data_package')
RUNS = f"{resource('runs')}"
STEP1_V3 = f"{resource('hovernet_segmentation')}"
OUT = f"{resource('evaluation')}/hovernet_v3_indep"
sys.path.insert(0, f"{resource('vendor_hovernet_original')}")
sys.path.insert(0, f"{resource('vendor_hovernet')}")
sys.path.insert(0, f"{resource('vendor_hovernet')}/models")
from unified_matcher import EvalInput, Instance, evaluate, write_outputs
from models.hovernet.net_desc import create_model
PAD, IN, WOUT = (46, 256, 164)
OFFS = [0, 92]
OFF = 46
CLASS_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']

@torch.no_grad()
def fold_type_maps(model, fold, bs=8):
    """Inference over the full fold with 2×2 padded windows → per-window type map (argmax, 0-5).
    Correct method: take argmax independently in each window, then select the window covering the centroid for lookup."""
    imgs = np.load(f"{resource('pannuke')}/Fold {fold}/images/fold{fold}/images.npy", mmap_mode='r')
    n = imgs.shape[0]
    tms = np.zeros((n, 4, WOUT, WOUT), dtype=np.uint8)
    t0 = time.time()
    for s in range(0, n, bs):
        batch = np.asarray(imgs[s:s + bs]).astype(np.float32)
        padded = np.stack([cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101) for im in batch])
        for oy in OFFS:
            for ox in OFFS:
                crops = padded[:, oy:oy + IN, ox:ox + IN]
                x = torch.from_numpy(crops.transpose(0, 3, 1, 2)).cuda()
                with torch.autocast('cuda', dtype=torch.float16):
                    tp = model(x)['tp'].float()
                k = (1 if oy else 0) * 2 + (1 if ox else 0)
                tms[s:s + len(batch), k] = tp.argmax(dim=1).cpu().numpy().astype(np.uint8)
        if s // bs % 50 == 0:
            print(f'  fold{fold} {s + len(batch)}/{n} ({(s + len(batch)) / (time.time() - t0):.1f}/s)', flush=True)
    return tms

def lookup_type(tms_i, cx, cy):
    """Find the window covering the centroid among the 4 windows and return its argmax class (0-5)."""
    px, py = (int(round(cx)) + PAD, int(round(cy)) + PAD)
    for oy in OFFS:
        for ox in OFFS:
            lx, ly = (px - (ox + OFF), py - (oy + OFF))
            if 0 <= lx < WOUT and 0 <= ly < WOUT:
                k = (1 if oy else 0) * 2 + (1 if ox else 0)
                return int(tms_i[k, ly, lx])
    return -1

def load_gt(fold):
    masks = np.load(f"{resource('pannuke')}/Fold {fold}/masks/fold{fold}/masks.npy", mmap_mode='r')
    gt = {}
    for i in range(masks.shape[0]):
        m = np.asarray(masks[i])
        pts, cls = ([], [])
        for c in range(5):
            ch = m[..., c]
            for iid in np.unique(ch):
                if iid == 0:
                    continue
                ys, xs = np.nonzero(ch == iid)
                pts.append((float(xs.mean()), float(ys.mean())))
                cls.append(c)
        gt[f'fold{fold}_image_{i}'] = (np.array(pts) if pts else np.zeros((0, 2)), np.array(cls, dtype=int))
    return gt

def main():
    os.makedirs(OUT, exist_ok=True)
    summaries, per_class_dfs = ([], [])
    for fold in (int(os.environ['CELLNURA_BASELINE_FOLD']),) if os.environ.get('CELLNURA_BASELINE_FOLD') else (1, 2, 3):
        print(f'\n===== HoVer-Net v3.1 independent baseline fold{fold} =====', flush=True)
        model = create_model(mode='fast', nr_types=6)
        ck = torch.load(f'{RUNS}/hovernet_v3_fast_pannuke_testfold{fold}/best_model.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['desc'] if 'desc' in ck else ck, strict=True)
        model = model.cuda().eval()
        t0 = time.time()
        tms = fold_type_maps(model, fold)
        print(f'  type maps done ({time.time() - t0:.0f}s), GPU mem: {torch.cuda.max_memory_allocated() / 1000000000.0:.1f}GB', flush=True)
        del model
        torch.cuda.reset_peak_memory_stats()
        gt = load_gt(fold)
        gt_list, pr_list = ([], [])
        n_unk = 0
        for i in range(tms.shape[0]):
            img_id = f'fold{fold}_image_{i}'
            jp = f'{STEP1_V3}/outer_fold{fold}/step1/source_fold{fold}/{img_id}_segmentation.json'
            pts, cls = gt[img_id]
            for j in range(len(pts)):
                gt_list.append(Instance(img_id, f'g{j}', pts[j][0], pts[j][1], int(cls[j])))
            if not os.path.exists(jp):
                raise FileNotFoundError(f'Missing segmentation for test image {img_id}: {jp}')
            d = json.load(open(jp))
            nuc = d['tiles'][0]['nuc']
            for nid, v in nuc.items():
                cx, cy = (float(v['centroid'][0]), float(v['centroid'][1]))
                tp = lookup_type(tms[i], cx, cy)
                if tp <= 0:
                    n_unk += 1
                    tp = 0
                pr_list.append(Instance(img_id, f'd{nid}', cx, cy, tp - 1))
        ev = EvalInput(dataset='PanNuke', tau=12.0, gt=gt_list, pred=pr_list, image_ids=tuple([f'fold{fold}_image_{i}' for i in range(tms.shape[0])]), real_class_ids=(0, 1, 2, 3, 4))
        res = evaluate(ev, mode='one_to_one')
        write_outputs(res, f'{OUT}/eval_fold{fold}')
        row = {'fold': fold, 'N_GT': res.n_gt, 'N_pred': len(pr_list), 'M': res.n_matched, 'missed': res.n_missed, 'spurious': res.n_spurious, 'cls_macroF1': round(res.macro('classification'), 4), 'plusFN_macroF1': round(res.macro('plus_missed_FN'), 4), 'uni_macroF1': round(res.macro('unified'), 4)}
        summaries.append(row)
        print(f'  {row} | unknown/skipped={n_unk}', flush=True)
        for ci, cname in enumerate(CLASS_NAMES):
            d = res.classes.get(ci)
            if not d:
                continue
            per_class_dfs.append({'fold': fold, 'class': cname, 'cls_F1': round(d['classification']['F1'], 4), 'plusFN_F1': round(d['plus_missed_FN']['F1'], 4), 'uni_F1': round(d['unified']['F1'], 4)})
        del tms
    pf = pd.DataFrame(summaries)
    pf.to_csv(f'{OUT}/per_fold_summary.csv', index=False)
    pc = pd.DataFrame(per_class_dfs)
    pc.to_csv(f'{OUT}/per_class_per_fold.csv', index=False)
    print(f'\n=== HoVer-Net v3.1 Independent Baseline ===')
    print(pf.to_string(index=False))
    m, s = (pf.cls_macroF1.mean(), pf.cls_macroF1.std())
    mu, su = (pf.uni_macroF1.mean(), pf.uni_macroF1.std())
    print(f'cls {m:.4f}±{s:.3f} | uni {mu:.4f}±{su:.3f}')
    print('\nPer class:')
    print(pc.groupby('class')[['cls_F1', 'uni_F1']].mean().round(4).loc[CLASS_NAMES].to_string())
if __name__ == '__main__':
    main()
