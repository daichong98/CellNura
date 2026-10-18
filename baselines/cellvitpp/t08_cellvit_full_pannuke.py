"""t08_cellvit_full_pannuke.py — Full CellViT-SAM pipeline inference on the three PanNuke test folds.

Native protocol: official CellViT-SAM-H x40 weights, direct inference on 256×256
RGB tiles with RGB/255 → (x-.5)/.5, the model's instance segmentation
(calculate_instance_map, magnification=40), and type mapping (type-1 → 0-4).
Unified evaluation: one-to-one matching with unified_matcher (τ=12 px) and
per-class results at all three evaluation levels.
Disclosure: the official x40 weights were trained on 90% of all PanNuke folds,
including overlap with test folds; see the official CellViT documentation.
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE = Path(resource('scripts'))
BASE = Path(resource('project_root'))
BLD = Path(resource('baselines'))
sys.path.insert(0, str(Path(resource('vendor_cellvit'))))
sys.path.insert(0, str(HERE))
from unified_matcher import EvalInput, Instance, evaluate, write_outputs
from models.segmentation.cell_segmentation.cellvit import CellViTSAM
PKG = Path(resource('data_package'))
CKPT = Path(resource('cellvit_sam_weight'))
OUT = Path(resource('evaluation') + '/cellvit_sam_full_pannuke')
MANIFEST = Path(resource('split_manifest'))
TAU = 12.0
CLASS_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']

def load_gt(fold):
    masks = np.load(Path(resource('pannuke')) / f'Fold {fold}/masks/fold{fold}/masks.npy', mmap_mode='r')
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

def main(score_thr=0.5):
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    torch.set_num_threads(8)
    ck = torch.load(CKPT, map_location='cpu', weights_only=True)
    model = CellViTSAM(model_path=None, num_nuclei_classes=6, num_tissue_classes=19, vit_structure='SAM-H')
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model = model.cuda().eval()
    print('CellViT-SAM-H loaded (strict)', flush=True)
    man = pd.read_csv(MANIFEST)
    pan = man[man.dataset == 'PanNuke']
    out_rows, per_class = ([], [])
    for fold in (int(os.environ['CELLNURA_BASELINE_FOLD']),) if os.environ.get('CELLNURA_BASELINE_FOLD') else (1,2,3):
        te = set(pan[(pan.outer_fold == f'fold{fold}') & (pan.split == 'test')].image_id)
        imgs = np.load(Path(resource('pannuke')) / f'Fold {fold}/images/fold{fold}/images.npy', mmap_mode='r')
        gt = load_gt(fold)
        gt_list, pr_list = ([], [])
        n_done = 0
        n_unk = 0
        t0 = time.time()
        for i in range(imgs.shape[0]):
            image_id = f'fold{fold}_image_{i}'
            if image_id not in te:
                continue
            img = np.clip(np.asarray(imgs[i]), 0, 255).astype(np.uint8)
            tensor = torch.from_numpy(((img.astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)[None]).cuda()
            with torch.inference_mode():
                out = model(tensor)
                for key in ('nuclei_binary_map', 'nuclei_type_map'):
                    out[key] = torch.softmax(out[key], 1)
                _, infos = model.calculate_instance_map(out, magnification=40)
            pts, cls = gt[image_id]
            for j in range(len(pts)):
                gt_list.append(Instance(image_id, f'g{j}', pts[j][0], pts[j][1], int(cls[j])))
            n_unk_i = 0
            for iid, cell in infos[0].items():
                t = int(cell['type']) - 1
                x, y = cell['centroid']
                pr_list.append(Instance(image_id, f'd{int(iid)}', float(x), float(y), t))
                if t < 0:
                    n_unk_i += 1
            n_unk += n_unk_i
            n_done += 1
            if n_done % 500 == 0:
                print(f'  fold{fold}: {n_done}/{len(te)} imgs ({n_done / (time.time() - t0):.1f}/s)', flush=True)
        ev = EvalInput(dataset='PanNuke', tau=TAU, gt=gt_list, pred=pr_list, real_class_ids=real_classes('PanNuke'))
        res = evaluate(ev, mode='one_to_one')
        write_outputs(res, str(OUT / f'eval_fold{fold}'))
        row = {'fold': fold, 'N_GT': res.n_gt, 'N_pred': res.n_pred, 'M': res.n_matched, 'missed': res.n_missed, 'spurious': res.n_spurious, 'cls_macroF1': round(res.macro('classification'), 4), 'plusFN_macroF1': round(res.macro('plus_missed_FN'), 4), 'uni_macroF1': round(res.macro('unified'), 4)}
        out_rows.append(row)
        print(str(row), flush=True)
        for ci, cname in enumerate(CLASS_NAMES):
            d = res.classes.get(ci)
            if not d:
                continue
            per_class.append({'fold': fold, 'class': cname, 'cls_F1': round(d['classification']['F1'], 4), 'plusFN_F1': round(d['plus_missed_FN']['F1'], 4), 'uni_F1': round(d['unified']['F1'], 4)})
    df = pd.DataFrame(out_rows)
    df.to_csv(OUT / 'per_fold_summary.csv', index=False)
    pc = pd.DataFrame(per_class)
    pc.to_csv(OUT / 'per_class_per_fold.csv', index=False)
    meta = {'method': 'CellViT-SAM-H x40 (official weights, native inference)', 'contamination_disclosure': 'official checkpoint trained on 90% of all PanNuke folds (per CellViT official documentation); PanNuke row is a reference, not a held-out test estimate', 'protocol': 'native 256x256 inference; unified one-to-one matching tau=12px', 'rows': out_rows}
    json.dump(meta, open(OUT / 'run_manifest.json', 'w'), indent=2, ensure_ascii=False)
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--score-thr', type=float, default=0.5)
    a = ap.parse_args()
    main(a.score_thr)
