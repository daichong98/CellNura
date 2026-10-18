"""t06_train_hovernet_consep.py — CellNura v3.3 independent within-fold CoNSeP segmenter (fast mode, nr_types=5).

Protocol (aligned with the frozen image-level split in train_hovernet_pannuke_v3.py):
  * Train from scratch using only the 27 training images; image-level train/validation split = 22/5 (reuse micronet_external/consep/classifier_split.json,
    the same split as the shared classifier; all trainable modules share this internal image-level split)
  * Class mapping: 7 official CoNSeP classes → 4 classes ({1:3, 2:0, 3:1, 4:1, 5:2, 6:2, 7:2}); tp channel 0=background
  * Random 256 crops from 1000×1000 images (same geometric augmentation as PanNuke v3), with a central 164 output crop
  * Loss/optimization: same as v3 (0.5BCE+0.5dice / nuclear-pixel MSE / CE; Adam 1e-4, reduced to 1e-5 at ep≥48; 60ep; batch 32; AMP)
  * Model selection: val Dice(np)
Usage: python t06_train_hovernet_consep.py [--smoke]
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
BASE = resource('project_root')
CONSEP = f"{resource('consep')}"
sys.path.insert(0, f"{resource('vendor_hovernet_original')}")
sys.path.insert(0, f"{resource('vendor_hovernet')}")
sys.path.insert(0, f"{resource('vendor_hovernet')}/models")
from models.hovernet.net_desc import create_model
from models.hovernet.targets import gen_instance_hv_map
OUTROOT = f"{resource('runs')}"
SEED = 42
CROP = 164
INP = 256
NR_TYPES = 5
CMAP = {1: 3, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}
PATCHES_PER_IMG = 64
EPOCHS = 60

def _c(x, shape):
    hs = (x.shape[0] - shape[0]) // 2
    ws = (x.shape[1] - shape[1]) // 2
    return x[hs:hs + shape[0], ws:ws + shape[1]]

def load_split():
    roles=frozen_roles('consep')
    return roles['train'], roles['val']

def build_maps(name):
    """inst_map + inst_type → (ann int32, typ uint8)."""
    import scipy.io as sio
    m = sio.loadmat(f'{CONSEP}/label/{name}.mat')
    inst = (m['inst_map'] if 'inst_map' in m else m['inst_instancemap']).astype(np.int32)
    types = m['inst_type'].ravel().astype(int)
    typ = np.zeros(inst.shape, dtype=np.uint8)
    nxt = 1
    ann = np.zeros(inst.shape, dtype=np.int32)
    for iid in np.unique(inst):
        if iid == 0:
            continue
        if iid - 1 >= len(types) or types[iid - 1] == 0 or types[iid - 1] not in CMAP:
            continue
        sel = inst == iid
        ann[sel] = nxt
        nxt += 1
        typ[sel] = CMAP[int(types[iid - 1])] + 1
    return (ann, typ)

class ImageCache:
    """Load the 27 training images + (ann, typ) into memory once (1000²×27 ≈ 0.8GB int16/uint8)."""

    def __init__(self, names):
        self.data = {}
        for n in names:
            from PIL import Image
            img = np.array(Image.open(f'{CONSEP}/image/{n}.png').convert('RGB'))
            ann, typ = build_maps(n)
            self.data[n] = (img.astype(np.uint8), ann.astype(np.int16), typ)

    def __getitem__(self, n):
        return self.data[n]

class ConSePPatch(Dataset):

    def __init__(self, cache, names, train, patches_per_img=PATCHES_PER_IMG, seed=SEED):
        self.cache, self.names, self.train = (cache, names, train)
        self.ppi = patches_per_img
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
        ann = ann[oy:oy + INP, ox:ox + INP]
        typ = typ[oy:oy + INP, ox:ox + INP]
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
    run_id = 'hovernet_fast_consep_v33'
    outdir = f'{OUTROOT}/{run_id}'
    os.makedirs(outdir, exist_ok=True)
    log = open(f'{outdir}/train.log', 'a', buffering=1)

    def P(s):
        print(f'[{run_id}] {s}', flush=True)
        log.write(s + '\n')
    tr_names, va_names = load_split()
    epochs = EPOCHS
    if a.smoke:
        tr_names, va_names = (tr_names[:3], va_names[:1])
        epochs = 2
    P(f'train={len(tr_names)} imgs val={len(va_names)} imgs (image-level 22/5 split, same as the shared classifier)')
    cache = ImageCache(tr_names + va_names)
    ds_tr = ConSePPatch(cache, tr_names, True)
    ds_va = ConSePPatch(cache, va_names, False, patches_per_img=16)
    dl_tr = DataLoader(ds_tr, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=16, num_workers=4, pin_memory=True, persistent_workers=True)
    model = create_model(mode='fast', nr_types=NR_TYPES).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    best = 0.0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep == 48:
            for g in opt.param_groups:
                g['lr'] = 1e-05
        model.train()
        tot = nb = 0
        for img, np_t, hv_t, tp_t in dl_tr:
            img, np_t, hv_t, tp_t = (img.cuda(non_blocking=True), np_t.cuda(non_blocking=True), hv_t.cuda(non_blocking=True), tp_t.cuda(non_blocking=True))
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(img)
            out = {k: v.float() for k, v in out.items()}
            loss, parts = loss_fn(out, np_t, hv_t, tp_t)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        dice, tp_acc = validate(model, dl_va, 'cuda')
        P(f"ep {ep}/{epochs} loss={tot / max(nb, 1):.4f} (np={parts['np']:.3f} hv={parts['hv']:.3f} tp={parts['tp']:.3f}) val_dice={dice:.4f} val_tp_acc={tp_acc:.4f} ({time.time() - t0:.0f}s)")
        if dice > best:
            best = dice
            torch.save({'desc': model.state_dict()}, f'{outdir}/best_model.pth')
    json.dump({'run_id': run_id, 'impl': 'HoVer-Net fast nr_types=5, from scratch on CoNSeP 27 train', 'split': 'image-level 22/5 (same as shared classifier)', 'epochs': epochs, 'batch': 16, 'seed': SEED, 'best_val_dice': round(best, 4), 'train_images': tr_names, 'val_images': va_names}, open(f'{outdir}/run_manifest.json', 'w'), indent=2, ensure_ascii=False)
    P(f'DONE best_val_dice={best:.4f}')
if __name__ == '__main__':
    main()
