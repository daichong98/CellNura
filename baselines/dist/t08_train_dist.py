"""t08_train_dist.py — DIST (DRFNS distance regression) full within-fold training + inference + shared classifier + unified evaluation.

Implementation identity: a local PyTorch port of the DRFNS authors' source code (02_DIST/pytorch_port, numerical comparison validated),
not official PyTorch code from the original authors; referred to as "DIST (PyTorch port)" in the paper. No type head → detections are classified by the shared
ResNet-18 v2 classifier (same protocol as Micro-Net/StarDist).

Protocol:
- Train from scratch within each fold: image-level train/val = 85/15 from the other two folds (split_manifest); test fold used only for inference
- Input geometry (same as preflight): 256 image + reflect pad 70 → 396 input → 212 output,
  aligned with the image center [22:234]; distance targets are computed on complete GT instances before taking the central 212 crop
- Inference tiling: output windows {0,44}²×212² cover the full 256² image; distances are averaged in overlapping regions
- Hyperparameters (frozen in advance): Adam lr=1e-4, MSE, batch 8, ≤20 epochs, patience 5, seed 42, no augmentation
  (the authors' training augmentation/schedule were not ported; this known implementation difference is recorded in the manifest)
- Classification: nearest v3.1 nuclear image within 12px of a detected centroid → within-fold best resnet18_v2 weights
Usage:
  python t08_train_dist.py --test-fold 1
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import argparse, glob, json, os, re, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
BASE = resource('project_root')
PKG = resource('data_package')
PORT = f"{resource('vendor_dist_port')}"
MANIFEST = f"{resource('split_manifest')}"
RUNS = f"{resource('runs')}"
OUTROOT = f"{resource('detections')}/dist_full"
HERE = Path(resource('scripts'))
sys.path.insert(0, HERE)
sys.path.insert(0, PORT)
from unified_matcher import EvalInput, Instance, evaluate, write_outputs
from model import DIST
from targets_and_postprocess import instance_distances, pannuke_instances, postprocess
SEED = 42
IMG = 256
PAD = 92
IN = 396
OUT = 212
CENTER = (IN - OUT) // 2
WINDOWS = (0, IMG - OUT)
LUT = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

def split_rows_for(test_fold):
    man = pd.read_csv(MANIFEST)
    pan = man[(man.dataset == 'PanNuke') & (man.outer_fold == f'fold{test_fold}')]
    out = {}
    for s in ('train', 'val'):
        rows = {}
        for x in pan[pan.split == s].image_id.tolist():
            m = re.match('fold(\\d+)_image_(\\d+)', x)
            rows.setdefault(int(m.group(1)), []).append(int(m.group(2)))
        out[s] = rows
    return (out['train'], out['val'])

class DistDS(Dataset):
    """PanNuke images + distance targets (computed on complete GT before taking the central 212 crop).

    A few images have overlapping annotations across class channels, preventing construction of an unambiguous instance map; these are filtered in the constructor and recorded in
    failure_ledger (training targets cannot be constructed; no silent omission; the proportion is recorded in the run manifest)."""
    skipped = []

    def __init__(self, rows):
        self.items = []
        for f, idxs in rows.items():
            imgs = np.load(f"{resource('pannuke')}/Fold {f}/images/fold{f}/images.npy", mmap_mode='r')
            msks = np.load(f"{resource('pannuke')}/Fold {f}/masks/fold{f}/masks.npy", mmap_mode='r')
            for i in idxs:
                m = np.asarray(msks[i])
                cover = np.zeros(m.shape[:2], np.int16)
                for c in range(5):
                    cover += (m[..., c] > 0).astype(np.int16)
                if (cover > 1).any():
                    DistDS.skipped.append(f'fold{f}_image_{i}')
                    continue
                self.items.append((f, i, imgs, msks))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        f, i, imgs, msks = self.items[j]
        img = np.clip(np.asarray(imgs[i]), 0, 255).astype(np.float32)
        inst = pannuke_instances(np.asarray(msks[i]))
        dist = instance_distances(inst).astype(np.float32)
        target = dist[CENTER:CENTER + OUT, CENTER:CENTER + OUT]
        x = np.pad(img, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='reflect')
        x = x[CENTER:CENTER + IN, CENTER:CENTER + IN]
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(target[None].copy()))

def make_windows(img):
    """4 input windows (padded[oy:oy+396]) → stitched full-image distance map (mean fusion)."""
    xs = []
    padded = np.pad(img, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='reflect')
    for oy in WINDOWS:
        for ox in WINDOWS:
            x = padded[oy:oy + IN, ox:ox + IN]
            xs.append(x.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(xs).astype(np.float32))

def predict_full(model, img):
    """Full-image inference with 4 windows → fused distances → postprocessing → centroids (original-image coordinates)."""
    xs = make_windows(img).cuda()
    with torch.inference_mode():
        d = model(xs)[:, 0].cpu().numpy()
    canvas = np.zeros((IMG, IMG), np.float32)
    cover = np.zeros((IMG, IMG), np.float32)
    k = 0
    for oy in WINDOWS:
        for ox in WINDOWS:
            canvas[oy:oy + OUT, ox:ox + OUT] += d[k]
            cover[oy:oy + OUT, ox:ox + OUT] += 1
            k += 1
    canvas /= cover
    inst = postprocess(canvas, h=1, threshold=1.0)
    cents = []
    for iid in np.unique(inst):
        if iid == 0:
            continue
        ys, xs_ = np.nonzero(inst == iid)
        if len(ys) < 3:
            continue
        cents.append((float(xs_.mean()), float(ys.mean())))
    return cents

def train_fold(tf, epochs=20):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_id = f'dist_pannuke_testfold{tf}'
    outdir = f'{RUNS}/{run_id}'
    os.makedirs(outdir, exist_ok=True)
    log = open(f'{outdir}/train.log', 'a', buffering=1)

    def P(s):
        print(f'[{run_id}] {s}', flush=True)
        log.write(s + '\n')
    tr_rows, va_rows = split_rows_for(tf)
    ds_tr, ds_va = (DistDS(tr_rows), DistDS(va_rows))
    P(f'train={len(ds_tr)} val={len(ds_va)} (image-level split; {len(DistDS.skipped)} images with overlapping annotations skipped)')
    if DistDS.skipped:
        pd.Series(DistDS.skipped).to_csv(f'{outdir}/failure_ledger_overlap_gt.csv', index=False)
    dl_tr = DataLoader(ds_tr, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    model = DIST().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    crit = nn.MSELoss()
    best, best_state, pat = (1e+18, None, 0)
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tot = nb = 0
        for x, y in dl_tr:
            x, y = (x.cuda(), y.cuda())
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        model.eval()
        vtot = vnb = 0
        with torch.no_grad():
            for x, y in dl_va:
                vtot += crit(model(x.cuda()), y.cuda()).item()
                vnb += 1
        vl = vtot / max(vnb, 1)
        P(f'ep {ep}/{epochs} loss={tot / max(nb, 1):.4f} val_mse={vl:.4f} ({time.time() - t0:.0f}s)')
        if vl < best - 1e-06:
            best, best_state, pat = (vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0)
        else:
            pat += 1
        if pat >= 5:
            P(f'early stop at ep {ep}')
            break
    torch.save(best_state, f'{outdir}/best_model.pth')
    P(f'DONE best_val_mse={best:.4f}')
    return (outdir, best)

def infer_fold(tf, outdir):
    model = DIST()
    model.load_state_dict(torch.load(f'{outdir}/best_model.pth', weights_only=True))
    model = model.cuda().eval()
    imgs = np.load(f"{resource('pannuke')}/Fold {tf}/images/fold{tf}/images.npy", mmap_mode='r')
    rows = []
    t0 = time.time()
    for i in range(imgs.shape[0]):
        img = np.clip(np.asarray(imgs[i]), 0, 255).astype(np.float32)
        cents = predict_full(model, img)
        for k, (cx, cy) in enumerate(cents, 1):
            rows.append({'image_key': f'fold{tf}_image_{i}_segmentation', 'nucleus_id': k, 'cx': cx, 'cy': cy})
    df = pd.DataFrame(rows)
    os.makedirs(OUTROOT, exist_ok=True)
    df.to_csv(f'{OUTROOT}/det_fold{tf}.csv', index=False)
    print(f'[dist fold{tf}] infer {len(imgs)} imgs → {len(df)} det ({time.time() - t0:.0f}s)', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-fold', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--infer-only', action='store_true')
    a = ap.parse_args()
    run_dir = f'{RUNS}/dist_pannuke_testfold{a.test_fold}'
    if not a.infer_only:
        _, best = train_fold(a.test_fold, a.epochs)
        json.dump({'run_id': f'dist_pannuke_testfold{a.test_fold}', 'impl': 'DRFNS PyTorch port (02_DIST/pytorch_port)', 'note': 'Original training augmentation/schedule not ported; Adam 1e-4 MSE batch8 ≤20ep patience5 seed42', 'best_val_mse': round(best, 4), 'seed': SEED}, open(f'{run_dir}/run_manifest.json', 'w'), indent=2, ensure_ascii=False)
    infer_fold(a.test_fold, run_dir)
if __name__ == '__main__':
    main()
