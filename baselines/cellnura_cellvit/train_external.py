from pathlib import Path
from cellnura_release.baseline_runtime import resource, frozen_roles
import argparse, glob, hashlib, json, os, re, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
BASE = resource('project_root')
SRC = f"{resource('vendor_cellvit')}"
HERE = resource('scripts')
RUNS = f"{resource('runs')}"
MANIFEST = f"{resource('split_manifest')}"
sys.path.insert(0, SRC)
from models.segmentation.cell_segmentation.cellvit import CellViT256
SEED = 42
BATCH = 16
DISK_R = 8

def log(s):
    print(s, flush=True)

def _fix_mirror_padding(ann):
    cur = np.amax(ann)
    for inst_id in list(np.unique(ann)):
        if inst_id == 0:
            continue
        m = np.array(ann == inst_id, np.uint8)
        r = _label(m)[0]
        r[r > 1] += cur
        ann[r > 1] = r[r > 1]
        cur = np.amax(ann)
    return ann
from scipy.ndimage import measurements as _labeler
_label = _labeler.label

def _bbox(img):
    rows, cols = (np.any(img, 1), np.any(img, 0))
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [rmin, rmax + 1, cmin, cmax + 1]

def gen_hv(inst):
    """Official HoVer-Net HV target-generation algorithm (vendored implementation)."""
    fixed = _fix_mirror_padding(inst.copy())
    x_map = np.zeros(inst.shape[:2], np.float32)
    y_map = np.zeros(inst.shape[:2], np.float32)
    for iid in np.unique(fixed):
        if iid == 0:
            continue
        m = np.array(fixed == iid, np.uint8)
        b = _bbox(m)
        b[0] -= 2
        b[2] -= 2
        b[1] += 2
        b[3] += 2
        b[0] = max(0, b[0])
        b[2] = max(0, b[2])
        m = m[b[0]:b[1], b[2]:b[3]]
        if m.shape[0] < 2 or m.shape[1] < 2:
            continue
        com = list(_labeler.center_of_mass(m))
        com = [int(com[0] + 0.5), int(com[1] + 0.5)]
        xr = np.arange(1, m.shape[1] + 1) - com[1]
        yr = np.arange(1, m.shape[0] + 1) - com[0]
        gx, gy = np.meshgrid(xr, yr)
        gx[m == 0] = 0
        gy[m == 0] = 0
        gx, gy = (gx.astype(np.float32), gy.astype(np.float32))
        if np.min(gx) < 0:
            gx[gx < 0] /= -np.amin(gx[gx < 0])
        if np.min(gy) < 0:
            gy[gy < 0] /= -np.amin(gy[gy < 0])
        if np.max(gx) > 0:
            gx[gx > 0] /= np.amax(gx[gx > 0])
        if np.max(gy) > 0:
            gy[gy > 0] /= np.amax(gy[gy > 0])
        x_map[b[0]:b[1], b[2]:b[3]][m > 0] = gx[m > 0]
        y_map[b[0]:b[1], b[2]:b[3]][m > 0] = gy[m > 0]
    return np.dstack([x_map, y_map])
CFG = {'consep': dict(tau=5.0, n_types=5, classes=['Inflammatory', 'Epithelial', 'Spindle', 'Misc'], img_dir=f"{resource('consep')}/image", lbl_dir=f"{resource('consep')}/label", split_json=f'{RUNS}/micronet_external/consep/classifier_split.json', crops_per_img=8, val_crops=4, epochs=400), 'crchisto': dict(tau=10.0, n_types=5, classes=['inflammatory', 'epithelial', 'fibroblast', 'others'], img_dir=f"{resource('crchisto')}/CRCHisto_database/Detection", split_json=f'{RUNS}/micronet_external/crchisto/classifier_split.json', crops_per_img=4, val_crops=2, epochs=400)}
CMAP_CONSEP = {1: 3, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}
CMAP_CRC = {'inflammatory': 0, 'epithelial': 1, 'fibroblast': 2, 'others': 3}

def load_image(ds, name):
    if ds == 'consep':
        from PIL import Image
        return np.array(Image.open(f"{CFG[ds]['img_dir']}/{name}.png").convert('RGB'))
    from PIL import Image
    return np.array(Image.open(f"{CFG[ds]['img_dir']}/{name}/{name}.bmp").convert('RGB'))

def build_ann(ds, name, img):
    """Return (inst int32, type_map uint8 1-4): instance masks for consep, synthetic disks for crchisto."""
    H, W = img.shape[:2]
    inst = np.zeros((H, W), np.int32)
    typ = np.zeros((H, W), np.uint8)
    if ds == 'consep':
        import scipy.io as sio
        m = sio.loadmat(f"{CFG[ds]['lbl_dir']}/{name}.mat")
        instm = (m['inst_map'] if 'inst_map' in m else m['inst_instancemap']).astype(np.int32)
        types = m['inst_type'].ravel().astype(int)
        nxt = 1
        for iid in np.unique(instm):
            if iid == 0 or iid - 1 >= len(types) or types[iid - 1] == 0 or (types[iid - 1] not in CMAP_CONSEP):
                continue
            sel = instm == iid
            inst[sel] = nxt
            typ[sel] = CMAP_CONSEP[int(types[iid - 1])] + 1
            nxt += 1
    else:
        d = json.load(open(f"{resource('crchisto')}/crchisto_features_output/step0_preprocessed/{name}_processed.json"))
        yy, xx = np.mgrid[0:H, 0:W]
        nxt = 1
        for cl, coords in d.get('classification_data', {}).items():
            if cl not in CMAP_CRC or not isinstance(coords, list):
                continue
            for c in coords:
                if not (isinstance(c, list) and len(c) >= 2):
                    continue
                cx, cy = (float(c[0]), float(c[1]))
                x0, x1 = (max(0, int(cx) - DISK_R - 1), min(W, int(cx) + DISK_R + 2))
                y0, y1 = (max(0, int(cy) - DISK_R - 1), min(H, int(cy) + DISK_R + 2))
                sel = (xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2 <= DISK_R ** 2
                sel &= inst[y0:y1, x0:x1] == 0
                inst[y0:y1, x0:x1][sel] = nxt
                typ[y0:y1, x0:x1][sel] = CMAP_CRC[cl] + 1
                nxt += 1
    return (inst, typ)

class ExtDS(Dataset):

    def __init__(self, ds, names, train, crops_per_img, seed=SEED):
        self.ds, self.train, self.cpi = (ds, train, crops_per_img)
        self.rng = np.random.RandomState(seed if train else seed + 1)
        self.data = {}
        for n in names:
            img = load_image(ds, n)
            inst, typ = build_ann(ds, n, img)
            self.data[n] = (img, inst, typ)
        self.names = []
        for n in names:
            for _ in range(crops_per_img):
                self.names.append(n)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, k):
        n = self.names[k % len(self.names)]
        img, inst, typ = self.data[n]
        H, W = inst.shape
        C = 256
        if self.train:
            oy, ox = (self.rng.randint(0, H - C + 1), self.rng.randint(0, W - C + 1))
        else:
            r = np.random.RandomState(k)
            oy, ox = (r.randint(0, H - C + 1), r.randint(0, W - C + 1))
        img = img[oy:oy + C, ox:ox + C]
        inst = inst[oy:oy + C, ox:ox + C].copy()
        typ = typ[oy:oy + C, ox:ox + C].copy()
        if self.train:
            rot = self.rng.randint(0, 4)
            if rot:
                img, inst, typ = (np.rot90(img, rot), np.rot90(inst, rot), np.rot90(typ, rot))
            if self.rng.rand() < 0.5:
                img, inst, typ = (img[:, ::-1], inst[:, ::-1], typ[:, ::-1])
            if self.rng.rand() < 0.5:
                img, inst, typ = (img[::-1], inst[::-1], typ[::-1])
            if self.rng.rand() < 0.5:
                a = 1.0 + self.rng.uniform(-0.2, 0.2)
                b = self.rng.uniform(-0.1, 0.1) * 255
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        img_f = (np.ascontiguousarray(img).astype(np.float32) / 255 - 0.5) / 0.5
        binary = (np.ascontiguousarray(inst) > 0).astype(np.int64)
        hv = gen_hv(np.ascontiguousarray(inst)).astype(np.float32)
        return (torch.from_numpy(img_f.transpose(2, 0, 1).copy()), torch.from_numpy(binary), torch.from_numpy(np.ascontiguousarray(typ).astype(np.int64)), torch.from_numpy(hv.transpose(2, 0, 1).copy()))

def focal_loss(p, target, alpha=0.25, gamma=2.0, eps=1e-07):
    pt = p.gather(1, target.unsqueeze(1)).squeeze(1).clamp(eps, 1 - eps)
    return (-(alpha * (1 - pt) ** gamma * pt.log())).mean()

def dice_loss(p, target, eps=1.0):
    C = p.shape[1]
    t = F.one_hot(target, C).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (p * t).sum(dims)
    card = p.sum(dims) + t.sum(dims)
    return (1 - (2 * inter + eps) / (card + eps)).mean()

def compute_loss(out, binary_t, type_t, hv_t):
    pb = torch.softmax(out['nuclei_binary_map'].float(), 1)
    ptm = torch.softmax(out['nuclei_type_map'].float(), 1)
    loss = focal_loss(pb, binary_t) + dice_loss(pb, binary_t) + focal_loss(ptm, type_t) + dice_loss(ptm, type_t)
    fg = (binary_t > 0).float()
    n_fg = fg.sum().clamp(min=1)
    err = ((out['hv_map'].float() - hv_t) ** 2).mean(1)
    return loss + (err * fg).sum() / n_fg

@torch.no_grad()
def validate(model, dl):
    model.eval()
    tot = nb = 0
    for img, binary_t, type_t, hv_t in dl:
        out = model(img.cuda(non_blocking=True))
        tot += compute_loss(out, binary_t.cuda(), type_t.cuda(), hv_t.cuda()).item()
        nb += 1
    return tot / max(nb, 1)

def train(ds, smoke=False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = CFG[ds]
    run_id = f'cellnura_cellvit_classifier_{ds}' + ('_smoke' if smoke else '')
    outdir = f'{RUNS}/{run_id}'
    os.makedirs(outdir, exist_ok=True)
    sp = {'train_images': frozen_roles(ds)['train'], 'val_images': frozen_roles(ds)['val']}
    tr_names, va_names = (sorted(sp['train_images']), sorted(sp['val_images']))
    if ds == 'consep':
        te_names = sorted((os.path.basename(p)[:-4] for p in glob.glob(f"{cfg['img_dir']}/test_*.png")))
    else:
        man = pd.read_csv(MANIFEST)
        te_names = sorted(set(man[(man.dataset == 'CRCHisto') & (man.split == 'test')].image_id))
    if smoke:
        tr_names, va_names = (tr_names[:4], va_names[:2])
    n_types = cfg['n_types']
    log(f'[{run_id}] train={len(tr_names)} val={len(va_names)} test={len(te_names)} images (image-level split)')
    ds_tr = ExtDS(ds, tr_names, True, cfg['crops_per_img'])
    ds_va = ExtDS(ds, va_names, False, cfg['val_crops'])
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH, num_workers=4, pin_memory=True)
    model = CellViT256(model256_path=None, num_nuclei_classes=n_types, num_tissue_classes=4).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0003, betas=(0.85, 0.95), weight_decay=0.0001)
    epochs = 2 if smoke else cfg['epochs']
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, ep = (1e+18, 0)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = nb = 0
        for img, binary_t, type_t, hv_t in dl_tr:
            out = model(img.cuda(non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss = compute_loss(out, binary_t.cuda(), type_t.cuda(), hv_t.cuda())
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        vl = validate(model, dl_va)
        if ep % 20 == 0 or ep == 1 or smoke:
            log(f'[{run_id}] ep {ep}/{epochs} loss={tot / max(nb, 1):.4f} val_loss={vl:.4f} ({time.time() - t0:.0f}s)')
        if vl < best:
            best = vl
            torch.save({'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()}, 'epoch': ep, 'best_val_loss': round(best, 4)}, f'{outdir}/best_model.pth')
    log(f'[{run_id}] DONE best_val_loss={best:.4f}')
    json.dump({'run_id': run_id, 'impl': 'CellViT256 arch, random init, adapted trainer (disclosed)', 'dataset': ds, 'splits': {'train': tr_names, 'val': va_names, 'test': te_names}, 'epochs': epochs, 'best_val_loss': round(best, 4), 'seed': SEED, 'gt_note': 'crchisto: disk r=8 synthesized instances (disclosed)' if ds == 'crchisto' else 'consep: official inst masks', 'code_sha256': hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()}, open(f'{outdir}/run_manifest.json', 'w'), indent=2, ensure_ascii=False)
    return outdir

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=['consep', 'crchisto'])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(8)
    train(a.dataset, a.smoke)


if __name__ == '__main__':
    main()
