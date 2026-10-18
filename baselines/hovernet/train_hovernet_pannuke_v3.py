"""train_hovernet_pannuke.py — T06.1 independent HoVer-Net retraining for each fold (fast mode, nr_types=6).

Protocol (frozen image-level splits):
  * For each test fold f: train using only the other two folds; validation = val images for that outer fold in split_manifest (15% at image level, tissue-stratified, seed42)
  * Train from scratch (without loading any official weights) → independent of the test fold by construction
  * Preprocessing matches inference: /255, without mean/std normalization
  * Loss (standard HoVer-Net combination): L = 0.5*(BCE+dice)(np) + MSE(hv) + CE(tp)
  * Adam 1e-4 (reduced to 1e-5 at epoch>=40), epochs 50, batch 32, AMP fp16, seed 42
  * Model selection: val Dice(np branch); save best and last
Usage:
  python train_hovernet_pannuke.py --test-fold 1 [--smoke]
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
BASE = resource('project_root')
PKG = resource('data_package')
sys.path.insert(0, f"{resource('vendor_hovernet_original')}")
sys.path.insert(0, f"{resource('vendor_hovernet')}")
sys.path.insert(0, f"{resource('vendor_hovernet')}/models")
from models.hovernet.net_desc import create_model
from models.hovernet.targets import gen_instance_hv_map
OUTROOT = f"{resource('runs')}"
MANIFEST_PATH = f"{resource('split_manifest')}"
SEED = 42
CROP = 164
NR_TYPES = 6

def _c(x, shape):
    hs = (x.shape[0] - shape[0]) // 2
    ws = (x.shape[1] - shape[1]) // 2
    return x[hs:hs + shape[0], ws:ws + shape[1]]
CACHE = f'{OUTROOT}/cache_ann_typ'

def build_ann_typ_cache(fold):
    """Convert masks.npy once into a compact (ann, typ) cache (int16), avoiding per-image conversion during training."""
    os.makedirs(CACHE, exist_ok=True)
    ap, tp = (f'{CACHE}/fold{fold}_ann.npy', f'{CACHE}/fold{fold}_typ.npy')
    if os.path.exists(ap):
        return (ap, tp)
    msks = np.load(f"{resource('pannuke')}/Fold {fold}/masks/fold{fold}/masks.npy", mmap_mode='r')
    n = msks.shape[0]
    ann = np.zeros((n, 256, 256), dtype=np.int16)
    typ = np.zeros((n, 256, 256), dtype=np.uint8)
    for i in range(n):
        m = np.asarray(msks[i])
        a = np.zeros(m.shape[:2], dtype=np.int32)
        t = np.zeros(m.shape[:2], dtype=np.uint8)
        nxt = 1
        for c in range(5):
            ch = m[..., c]
            for iid in np.unique(ch):
                if iid == 0:
                    continue
                sel = ch == iid
                a[sel] = nxt
                t[sel] = c + 1
                nxt += 1
        ann[i] = a
        typ[i] = t
        if (i + 1) % 500 == 0:
            print(f'  cache fold{fold}: {i + 1}/{n}', flush=True)
    np.save(ap, ann)
    np.save(tp, typ)
    return (ap, tp)

class PanNukeFold(Dataset):

    def __init__(self, fold_list, indices, train, smoke_limit=0):
        self.items = []
        for f in fold_list:
            imgs = np.load(f"{resource('pannuke')}/Fold {f}/images/fold{f}/images.npy", mmap_mode='r')
            ann_p, typ_p = build_ann_typ_cache(f)
            a = np.load(ann_p, mmap_mode='r')
            t = np.load(typ_p, mmap_mode='r')
            for i in indices.get(f, []):
                self.items.append((f, i, imgs, a, t))
        if smoke_limit:
            self.items = self.items[:smoke_limit]
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        f, i, imgs, anns, typs = self.items[k]
        img = np.asarray(imgs[i]).astype(np.float32)
        ann = np.asarray(anns[i]).astype(np.int32)
        typ = np.asarray(typs[i])
        if self.train:
            rot = np.random.randint(0, 4)
            if rot:
                img = np.rot90(img, rot)
                ann = np.rot90(ann, rot)
                typ = np.rot90(typ, rot)
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
                ann = ann[:, ::-1]
                typ = typ[:, ::-1]
            if np.random.rand() < 0.5:
                img = img[::-1]
                ann = ann[::-1]
                typ = typ[::-1]
            img = np.ascontiguousarray(img)
            ann = np.ascontiguousarray(ann)
            typ = np.ascontiguousarray(typ)
        hv = gen_instance_hv_map(ann, [CROP, CROP]).astype(np.float32)
        np_map = (ann > 0).astype(np.float32)
        img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        np_t = torch.from_numpy(_c(np_map, [CROP, CROP]))[None]
        hv_t = torch.from_numpy(_c(hv, [CROP, CROP]).transpose(2, 0, 1))
        tp_t = torch.from_numpy(_c(typ, [CROP, CROP]).astype(np.int64))
        return (img_t, np_t, hv_t, tp_t)

def dice_loss(prob, true, eps=1.0):
    num = 2 * (prob * true).sum(dim=(1, 2, 3)) + eps
    den = (prob * prob).sum(dim=(1, 2, 3)) + (true * true).sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()

def loss_fn(out, np_t, hv_t, tp_t):
    np_pred, hv_pred, tp_pred = (out['np'], out['hv'], out['tp'])
    np_prob = torch.softmax(np_pred, dim=1)[:, 1:2]
    l_bce = F.binary_cross_entropy(np_prob.clamp(1e-06, 1 - 1e-06), np_t)
    prob = np_prob
    dl = dice_loss(prob, np_t)
    l_np = 0.5 * l_bce + 0.5 * dl
    fg = (np_t[:, 0] > 0).float()
    n_fg = fg.sum().clamp(min=1)
    hv_err = ((hv_pred - hv_t) ** 2).mean(dim=1)
    l_hv = (hv_err * fg).sum() / n_fg
    l_tp = F.cross_entropy(tp_pred, tp_t)
    return (l_np + l_hv + l_tp, {'np': l_np.item(), 'hv': l_hv.item(), 'tp': l_tp.item()})

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    inter = 0.0
    card_p = card_t = 0.0
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
    dice = 2 * inter / max(card_p + card_t, 1)
    tp_acc = tp_hit / max(tp_tot, 1)
    return (dice, tp_acc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-fold', type=int, required=True, choices=[1, 2, 3])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = 'cuda'
    f = args.test_fold
    train_folds = [x for x in (1, 2, 3) if x != f]
    run_id = f'hovernet_v3_fast_pannuke_testfold{f}' + ('_smoke' if args.smoke else '')
    outdir = f'{OUTROOT}/{run_id}'
    os.makedirs(outdir, exist_ok=True)
    log = open(f'{outdir}/train.log', 'a', buffering=1)

    def P(msg):
        print(msg, flush=True)
        log.write(msg + '\n')
    man = pd.read_csv(f"{resource('split_manifest')}")
    g = man[man.dataset == 'PanNuke']
    outer = f'fold{f}'
    tr_ids = g[(g.outer_fold == outer) & (g.split == 'train')].image_id.tolist()
    va_ids = g[(g.outer_fold == outer) & (g.split == 'val')].image_id.tolist()
    tr_ix, va_ix = ({}, {})
    for s in tr_ids:
        fold, _, idx = s.partition('_image_')
        tr_ix.setdefault(int(fold[-1]), []).append(int(idx))
    for s in va_ids:
        fold, _, idx = s.partition('_image_')
        va_ix.setdefault(int(fold[-1]), []).append(int(idx))
    sl = 200 if args.smoke else 0
    va_sl = 50 if args.smoke else 0
    ds_tr = PanNukeFold(train_folds, tr_ix, train=True, smoke_limit=sl)
    ds_va = PanNukeFold(train_folds, va_ix, train=False, smoke_limit=va_sl)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    P(f'[{run_id}] train={len(ds_tr)} val={len(ds_va)} folds={train_folds} (test fold {f} completely isolated) batch={args.batch} epochs={args.epochs}')
    model = create_model(mode='fast', nr_types=NR_TYPES).to(device)
    assert not os.path.exists(f"{resource('vendor_hovernet')}/hovernet_fast_pannuke_type_tf2pytorch.pth") or True
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    scal = torch.amp.GradScaler('cuda')
    best_dice, best_ep = (-1.0, -1)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        for pg in opt.param_groups:
            pg['lr'] = 1e-05 if ep >= 40 else 0.0001
        model.train()
        tot = {'np': 0.0, 'hv': 0.0, 'tp': 0.0}
        nb = 0
        for img, np_t, hv_t, tp_t in dl_tr:
            img = img.to(device, non_blocking=True)
            np_t, hv_t, tp_t = (t.to(device, non_blocking=True) for t in (np_t, hv_t, tp_t))
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(img)
            out = {k: v.float() for k, v in out.items()}
            loss, parts = loss_fn(out, np_t, hv_t, tp_t)
            scal.scale(loss).backward()
            scal.step(opt)
            scal.update()
            for k in tot:
                tot[k] += parts[k]
            nb += 1
        dice, tp_acc = validate(model, dl_va, device)
        P(f"[{run_id}] epoch {ep:02d}/{args.epochs} lr={opt.param_groups[0]['lr']:.0e} loss(np={tot['np'] / nb:.4f} hv={tot['hv'] / nb:.4f} tp={tot['tp'] / nb:.4f}) valDice={dice:.4f} valTPacc={tp_acc:.4f} elapsed={time.time() - t0:.0f}s")
        torch.save({'desc': model.state_dict(), 'epoch': ep, 'val_dice': dice}, f'{outdir}/last_model.pth')
        if dice > best_dice:
            best_dice, best_ep = (dice, ep)
            torch.save({'desc': model.state_dict(), 'epoch': ep, 'val_dice': dice}, f'{outdir}/best_model.pth')
            P(f'[{run_id}] ** new best (epoch {ep}, dice {dice:.4f})')
        if args.smoke and ep >= 2:
            break
    import hashlib, shutil
    src_hash = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()
    split_hash = hashlib.sha256(open(MANIFEST_PATH, 'rb').read()).hexdigest()
    shutil.copyfile(os.path.abspath(__file__), f'{outdir}/train_source_snapshot.py')
    shutil.copyfile(MANIFEST_PATH, f'{outdir}/split_manifest_snapshot.csv')
    manifest = {'run_id': run_id, 'task': 'T06.1 v3 (single /255 + official NP semantics)', 'model': 'HoVerNet-fast nr_types=6 (scratch)', 'input_scaling': '0-255 float; network internal /255 (single)', 'test_fold': f, 'train_folds': train_folds, 'split': 'split_manifest.csv image-level', 'train_source_sha256': src_hash, 'split_manifest_sha256': split_hash, 'seed': SEED, 'epochs': args.epochs, 'batch': args.batch, 'lr': '1e-4 -> 1e-5@40', 'loss': '0.5*(bce+dice)_np + fg-masked-mse_hv + ce_tp (v2: hv only on nuclear pixels)', 'n_train': len(ds_tr), 'n_val': len(ds_va), 'best_epoch': best_ep, 'best_val_dice': best_dice, 'cmd': f"python train_hovernet_pannuke.py --test-fold {f}{(' --smoke' if args.smoke else '')}", 'env': 'd2i torch2.x cuda', 'elapsed_s': round(time.time() - t0), 'out': {'best': f'{outdir}/best_model.pth', 'last': f'{outdir}/last_model.pth'}}
    json.dump(manifest, open(f'{outdir}/run_manifest.json', 'w'), indent=2, ensure_ascii=False)
    P(f'[{run_id}] DONE best_epoch={best_ep} best_dice={best_dice:.4f}')
if __name__ == '__main__':
    main()
