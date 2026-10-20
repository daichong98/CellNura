"""t08_hovernet_tphead_eval.py — T08: HoVer-Net within-fold tp-head baseline (PanNuke, unified protocol).

Pipeline = backbone with official HoVer-Net fast weights (frozen, exactly the same segmentation as CellNura)
         + within-fold tp-head fine-tuning (only the two training folds, no test-fold information) → type predictions for all detections
Evaluation = one-to-one unified_matcher (τ=12px), with all GT instances and all detections included in the statistics.
Output: final_tables/hovernet_tphead_baseline/{per_fold_summary,per_class_per_fold}.csv
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import glob
import json
import os
import re
import sys
import time
import cv2
import numpy as np
import pandas as pd
import torch
BASE = resource('project_root')
PKG = resource('data_package')
TPDIR = f"{resource('tphead_runs')}"
STEP1 = f"{resource('official_hovernet_detections')}"
OUT = f"{resource('evaluation')}/hovernet_tphead_baseline"
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
def fold_type_maps(model, fold, bs=32):
    """Inference over the full fold with 2×2 padded windows → type_maps (n,4,164,164) uint8 (1-5, 0=outside window/background)."""
    img = np.load(f"{resource('pannuke')}/Fold {fold}/images/fold{fold}/images.npy", mmap_mode='r')
    n = img.shape[0]
    tms = np.zeros((n, 4, WOUT, WOUT), dtype=np.uint8)
    t0 = time.time()
    for s in range(0, n, bs):
        batch = np.asarray(img[s:s + bs]).astype(np.uint8)
        padded = np.stack([cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101) for im in batch])
        for oy in OFFS:
            for ox in OFFS:
                crops = padded[:, oy:oy + IN, ox:ox + IN]
                x = torch.from_numpy(crops.astype(np.float32).transpose(0, 3, 1, 2)).cuda()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    tp = model(x)['tp'].float()
                k = (1 if oy else 0) * 2 + (1 if ox else 0)
                tms[s:s + len(batch), k] = tp.argmax(dim=1).cpu().numpy().astype(np.uint8)
        if s // bs % 20 == 0:
            print(f'  fold{fold} imgs {s + len(batch)}/{n} ({(s + len(batch)) / (time.time() - t0):.1f}/s)', flush=True)
    return tms

def lookup(tms, fold):
    """Look up types for all standard step1 detections → DataFrame(image_key, nucleus_id, cx, cy, pred)."""
    rows = []
    files = sorted((p for p in glob.glob(f'{STEP1}/fold{fold}_image_*_segmentation.json') if os.path.isfile(p)), key=lambda p: int(re.search('image_(\\d+)', p).group(1)))
    for jp in files:
        base = os.path.basename(jp).replace('_segmentation.json', '')
        _m = re.search('image_(\\d+)', base)
        if not _m:
            print(f'  [skip] unexpected filename: {base}', flush=True)
            continue
        img_i = int(_m.group(1))
        d = json.load(open(jp))
        nuc = {}
        for t in d.get('tiles', [d]):
            nuc.update(t.get('nuc', {}) or {})
        tms_i = tms[img_i]
        for nid, v in nuc.items():
            px = int(round(float(v['centroid'][0]))) + PAD
            py = int(round(float(v['centroid'][1]))) + PAD
            pred = -1
            for oy in OFFS:
                for ox in OFFS:
                    lx, ly = (px - (ox + OFF), py - (oy + OFF))
                    if 0 <= lx < WOUT and 0 <= ly < WOUT:
                        vv = tms_i[(1 if oy else 0) * 2 + (1 if ox else 0), ly, lx]
                        if vv > 0:
                            pred = int(vv) - 1
                        break
                else:
                    continue
                break
            rows.append({'image_key': base + '_segmentation', 'nucleus_id': int(nid), 'cx': float(v['centroid'][0]), 'cy': float(v['centroid'][1]), 'pred': pred})
    return pd.DataFrame(rows, columns=['image_key', 'nucleus_id', 'cx', 'cy', 'pred'])

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
    summary, per_class = ([], [])
    for fold in (int(os.environ['CELLNURA_BASELINE_FOLD']),) if os.environ.get('CELLNURA_BASELINE_FOLD') else (1,2,3):
        print(f'===== HoVer-Net tp-head baseline fold{fold} =====', flush=True)
        model = create_model(mode='fast', nr_types=6)
        ck = torch.load(f'{TPDIR}/cv{fold}_tp_head.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['full'], strict=True)
        model = model.cuda().eval()
        tms = fold_type_maps(model, fold)
        del model
        torch.cuda.empty_cache()
        preds = lookup(tms, fold)
        preds.to_csv(f'{OUT}/pred_all_fold{fold}.csv', index=False)
        print(f'  detections with type: {len(preds)} (untyped {int((preds.pred < 0).sum())})', flush=True)
        gt = load_gt(fold)
        gt_list, pr_list = ([], [])
        for img, (pts, cls) in gt.items():
            for j in range(len(pts)):
                gt_list.append(Instance(img, f'g{j}', pts[j][0], pts[j][1], int(cls[j])))
        for r in preds.itertuples():
            pr_list.append(Instance(r.image_key.replace('_segmentation', ''), f'd{r.nucleus_id}', r.cx, r.cy, int(r.pred)))
        ev = EvalInput(dataset='PanNuke', tau=12.0, gt=gt_list, pred=pr_list, real_class_ids=real_classes('PanNuke'), image_ids=test_image_ids('pannuke',fold))
        res = evaluate(ev, mode='one_to_one')
        write_outputs(res, f'{OUT}/eval_fold{fold}')
        summary.append({'fold': fold, 'N_GT': res.n_gt, 'N_pred': res.n_pred, 'M': res.n_matched, 'missed': res.n_missed, 'spurious': res.n_spurious, 'cls_macroF1': round(res.macro('classification'), 4), 'plusFN_macroF1': round(res.macro('plus_missed_FN'), 4), 'uni_macroF1': round(res.macro('unified'), 4)})
        print('  ' + str(summary[-1]), flush=True)
        for ci, cname in enumerate(CLASS_NAMES):
            d = res.classes.get(ci)
            if not d:
                continue
            per_class.append({'fold': fold, 'class': cname, 'cls_F1': round(d['classification']['F1'], 4), 'plusFN_F1': round(d['plus_missed_FN']['F1'], 4), 'uni_F1': round(d['unified']['F1'], 4)})
        del tms
    pd.DataFrame(summary).to_csv(f'{OUT}/per_fold_summary.csv', index=False)
    pc = pd.DataFrame(per_class)
    pc.to_csv(f'{OUT}/per_class_per_fold.csv', index=False)
    print('\n=== HoVer-Net tp-head baseline (unified protocol, PanNuke) ===')
    print(pd.DataFrame(summary).to_string(index=False))
    m = pc.groupby('class')[['cls_F1', 'plusFN_F1', 'uni_F1']].mean().round(4)
    print('\nPer-class means over three folds:')
    print(m.loc[[c for c in CLASS_NAMES if c in m.index]].to_string())
if __name__ == '__main__':
    main()
