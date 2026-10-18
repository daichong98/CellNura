"""t06_train_hovernet_crchisto.py — CellNura v3.3 within-fold CRCHisto segmenter (fast, nr_types=5).

GT limitation (important disclosure): CRCHisto provides nuclear centroids only (detection.mat / processed.json), without instance masks.
This script synthesizes instance masks as r=8px disks centered on the centroids (overlapping pixels belong to the first drawn instance), enabling HoVer-Net
training; this synthesis is a known protocol deviation recorded in the run manifest and response letter. Evaluation uses only centroids (unified matching),
so the evaluation annotations are unaffected by mask synthesis.

Protocol aligned with t06_train_hovernet_consep.py: train from scratch on the 80 training images; image-level train/validation split 68/12 (same as the shared classifier);
class mapping {inflammatory:0, epithelial:1, fibroblast:2, others:3}; tp channel 0=background.
Usage: python t06_train_hovernet_crchisto.py [--smoke]
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
BASE = resource('project_root')
CRC_DIR = f"{resource('crchisto')}"
sys.path.insert(0, f"{resource('vendor_hovernet_original')}")
sys.path.insert(0, f"{resource('vendor_hovernet')}")
sys.path.insert(0, f"{resource('vendor_hovernet')}/models")
from models.hovernet.net_desc import create_model
from models.hovernet.targets import gen_instance_hv_map
OUTROOT = f"{resource('runs')}"
SEED = 42
CROP, INP = (164, 256)
NR_TYPES = 5
CMAP = {'inflammatory': 0, 'epithelial': 1, 'fibroblast': 2, 'others': 3}
DISK_R = 8
PATCHES_PER_IMG = 48
EPOCHS = 60

def _c(x, shape):
    hs = (x.shape[0] - shape[0]) // 2
    ws = (x.shape[1] - shape[1]) // 2
    return x[hs:hs + shape[0], ws:ws + shape[1]]

def build_maps(name):
    """Centroid GT → (ann, typ): instances synthesized as r=8 disks (disclosed limitation)."""
    d = json.load(open(f'{CRC_DIR}/crchisto_features_output/step0_preprocessed/{name}_processed.json'))
    from PIL import Image
    img = np.array(Image.open(f'{CRC_DIR}/CRCHisto_database/Detection/{name}/{name}.bmp').convert('RGB'))
    H, W = img.shape[:2]
    ann = np.zeros((H, W), dtype=np.int32)
    typ = np.zeros((H, W), dtype=np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    disk = (xx - 0) ** 2 + (yy - 0) ** 2
    import math
    r2 = DISK_R ** 2
    nxt = 1
    for cl, coords in d.get('classification_data', {}).items():
        if cl not in CMAP or not isinstance(coords, list):
            continue
        for c in coords:
            if not (isinstance(c, list) and len(c) >= 2):
                continue
            cx, cy = (float(c[0]), float(c[1]))
            x0, x1 = (max(0, int(cx) - DISK_R - 1), min(W, int(cx) + DISK_R + 2))
            y0, y1 = (max(0, int(cy) - DISK_R - 1), min(H, int(cy) + DISK_R + 2))
            sel = (xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2 <= r2
            sel &= ann[y0:y1, x0:x1] == 0
            patch_ann = ann[y0:y1, x0:x1]
            patch_typ = typ[y0:y1, x0:x1]
            patch_ann[sel] = nxt
            patch_typ[sel] = CMAP[cl] + 1
            nxt += 1
    return (img, ann, typ)

class CRCImageCache:

    def __init__(self, names):
        self.data = {}
        for n in names:
            self.data[n] = build_maps(n)

    def __getitem__(self, n):
        return self.data[n]

class CRCPatch(Dataset):

    def __init__(self, cache, names, train, ppi=PATCHES_PER_IMG, seed=SEED):
        self.cache, self.names, self.train, self.ppi = (cache, names, train, ppi)
        self.rng = np.random.RandomState(seed if train else seed + 1)

    def __len__(self):
        return len(self.names) * self.ppi

    def __getitem__(self, k):
        n = self.names[k % len(self.names)]
        img, ann, typ = self.cache[n]
        H, W = ann.shape
        if self.train:
            oy = self.rng.randint(0, H - INP)
            ox = self.rng.randint(0, W - INP)
        else:
            r = np.random.RandomState(k)
            oy, ox = (r.randint(0, H - INP), r.randint(0, W - INP))
        img = img[oy:oy + INP, ox:ox + INP]
        ann = ann[oy:oy + INP, ox:ox + INP].copy()
        typ = typ[oy:oy + INP, ox:ox + INP].copy()
        if self.train:
            rot = self.rng.randint(0, 4)
            if rot:
                img, ann, typ = (np.rot90(img, rot), np.rot90(ann, rot), np.rot90(typ, rot))
            if self.rng.rand() < 0.5:
                img, ann, typ = (img[:, ::-1], ann[:, ::-1], typ[:, ::-1])
            if self.rng.rand() < 0.5:
                img, ann, typ = (img[::-1], ann[::-1], typ[::-1])
        img = np.ascontiguousarray(img)
        ann = np.ascontiguousarray(ann)
        typ = np.ascontiguousarray(typ)
        hv = gen_instance_hv_map(ann.astype(np.int32), [CROP, CROP]).astype(np.float32)
        np_map = (ann > 0).astype(np.float32)
        return (torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)), torch.from_numpy(_c(np_map, [CROP, CROP]))[None], torch.from_numpy(_c(hv, [CROP, CROP]).transpose(2, 0, 1)), torch.from_numpy(_c(typ.astype(np.int64), [CROP, CROP])))

def dice_loss(prob, true, eps=1.0):
    num = 2 * (prob * true).sum(dim=(1, 2, 3)) + eps
    den = (prob * prob).sum(dim=(1, 2, 3)) + (true * true).sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()

def loss_fn(out, np_t, hv_t, tp_t):
    np_prob = torch.softmax(out['np'], dim=1)[:, 1:2]
    l_bce = F.binary_cross_entropy(np_prob.clamp(1e-06, 1 - 1e-06), np_t)
    dl = dice_loss(np_prob, np_t)
    l_np = 0.5 * l_bce + 0.5 * dl
    fg = (np_t[:, 0] > 0).float()
    n_fg = fg.sum().clamp(min=1)
    hv_err = ((out['hv'] - hv_t) ** 2).mean(dim=1)
    l_hv = (hv_err * fg).sum() / n_fg
    l_tp = F.cross_entropy(out['tp'], tp_t)
    return (l_np + l_hv + l_tp, {'np': l_np.item(), 'hv': l_hv.item(), 'tp': l_tp.item()})

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    inter = card_p = card_t = 0.0
    tp_hit = tp_tot = 0
    for img, np_t, hv_t, tp_t in loader:
        img = img.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(img)
        pred = (torch.softmax(out['np'].float(), dim=1)[:, 1] > 0.5).float()
        true = np_t[:, 0].to(device)
        inter += (pred * true).sum().item()
        card_p += pred.sum().item()
        card_t += true.sum().item()
        tp_hit += (out['tp'].argmax(1).to(device) == tp_t.to(device))[true > 0].sum().item()
        tp_tot += (true > 0).sum().item()
    return (2 * inter / max(card_p + card_t, 1), tp_hit / max(tp_tot, 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_id = 'hovernet_fast_crchisto_v33'
    outdir = f'{OUTROOT}/{run_id}'
    os.makedirs(outdir, exist_ok=True)
    log = open(f'{outdir}/train.log', 'a', buffering=1)

    def P(s):
        print(f'[{run_id}] {s}', flush=True)
        log.write(s + '\n')
    roles = frozen_roles('crchisto')
    sp = {'train_images': roles['train'], 'val_images':roles['val']}
    tr_names = sorted(sp['train_images'])
    va_names = sorted(sp['val_images'])
    epochs = EPOCHS
    if a.smoke:
        tr_names, va_names, epochs = (tr_names[:3], va_names[:1], 2)
    P(f'train={len(tr_names)} val={len(va_names)} imgs (image-level 68/12 split, same as the shared classifier; GT synthesized as r={DISK_R} disks)')
    cache = CRCImageCache(tr_names + va_names)
    ds_tr = CRCPatch(cache, tr_names, True)
    ds_va = CRCPatch(cache, va_names, False, ppi=16)
    dl_tr = DataLoader(ds_tr, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=8, num_workers=4, pin_memory=True, persistent_workers=True)
    model = create_model(mode='fast', nr_types=NR_TYPES).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=5e-05)
    best = 0.0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep == 48:
            for g in opt.param_groups:
                g['lr'] = 1e-05
        model.train()
        tot = nb = 0
        parts = {'np': 0, 'hv': 0, 'tp': 0}
        for img, np_t, hv_t, tp_t in dl_tr:
            img, np_t, hv_t, tp_t = (img.cuda(non_blocking=True), np_t.cuda(non_blocking=True), hv_t.cuda(non_blocking=True), tp_t.cuda(non_blocking=True))
            opt.zero_grad(set_to_none=True)
            out = model(img)
            out = {k: v.float() for k, v in out.items()}
            loss, parts = loss_fn(out, np_t, hv_t, tp_t)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        dice, tp_acc = validate(model, dl_va, 'cuda')
        P(f"ep {ep}/{epochs} loss={tot / max(nb, 1):.4f} (np={parts['np']:.3f} hv={parts['hv']:.3f} tp={parts['tp']:.3f}) val_dice={dice:.4f} val_tp_acc={tp_acc:.4f} ({time.time() - t0:.0f}s)")
        if dice > best:
            best = dice
            torch.save({'desc': model.state_dict()}, f'{outdir}/best_model.pth')
    json.dump({'run_id': run_id, 'impl': 'HoVer-Net fast nr_types=5, from scratch on CRCHisto 80 train', 'gt_note': 'CRCHisto has no instance-mask GT; r=8 disks synthesized from centroids (overlaps assigned to the first drawn instance); protocol deviation disclosed', 'split': 'image-level 68/12 (same as shared classifier)', 'epochs': epochs, 'batch': 8, 'lr': 5e-05, 'grad_clip': 1.0, 'amp': False, 'note': '1e-4+fp16 failed twice (rising loss/NaN assertion); retrained with fp32+5e-5', 'seed': SEED, 'best_val_dice': round(best, 4), 'train_images': tr_names, 'val_images': va_names}, open(f'{outdir}/run_manifest.json', 'w'), indent=2, ensure_ascii=False)
    P(f'DONE best_val_dice={best:.4f}')
if __name__ == '__main__':
    main()
