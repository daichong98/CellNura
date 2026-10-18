"""t08_classifier_v3.py — Shared classifier v3 with the F1/F4 audit fixes.

Background (independent review in RESULT_AUDIT_20260928):
  F1 (P0): The v2 classifier did not filter the role mapping by outer_fold, so
           all three runs used fold3 roles. Test images and labels from folds
           1 and 2 entered training (44,121 and 42,186 test nuclei, respectively).
  F4 (P1): At test time, v2 classified a detection only if it matched a v3.1
           HoVer-Net nucleus within 12 px. This discarded 22–25% of Micro-Net,
           9–14% of DIST, and 5–7% of StarDist detections, introducing a filter
           based on a second segmentation model.

v3 protocol (a method-neutral scheme independent of the crop source):
  * Split: filter by `(dataset==PanNuke)&(outer_fold==fold{tf})`, assert pairwise
    disjoint image roles, and save the image IDs.
  * Training/validation crops: fixed 64×64 windows centered on GT centroids,
    using only train/val images and GT class labels. This crop rule does not
    depend on any segmentation model and implements one option from the audit.
  * Test crops: use the same 64×64 window at each baseline detection (cx, cy)
    in the original image. Classify every detection without discarding any.
  * Model/hyperparameters: ResNet-18 with ImageNet initialization (as in v2),
    Adam at 1e-4, at most 15 epochs with patience 4, square-root class weighting,
    64×64 inputs, rot90/flip augmentation, and seed 42.
Usage:
  python t08_classifier_v3.py --test-fold 1          # Train one of the three folds
  python t08_classifier_v3.py --test-fold 1 --eval-all # Then evaluate Micro-Net/StarDist/DIST
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import argparse, hashlib, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
BASE = resource('project_root')
PKG = resource('data_package')
MANIFEST = f"{resource('split_manifest')}"
RUNS = f"{resource('runs')}"
DET = f"{resource('detections')}"
FINAL = f"{resource('evaluation')}"
HERE = Path(resource('scripts'))
sys.path.insert(0, HERE)
from unified_matcher import EvalInput, Instance, evaluate, write_outputs
SEED = 42
WIN = 64
_tf = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def log(s):
    print(s, flush=True)

def split_for(tf):
    """F1 audit fix: filter by the current outer fold and check assertions. Return {img_id: role} and the image index {img_id: (fold_n, idx)}."""
    man = pd.read_csv(MANIFEST)
    g = man[(man.dataset == 'PanNuke') & (man.outer_fold == f'fold{tf}')]
    assert len(g) == g.image_id.nunique() == 7901
    role = dict(zip(g.image_id, g.split))
    tr = {x for x, r in role.items() if r == 'train'}
    va = {x for x, r in role.items() if r == 'val'}
    te = {x for x, r in role.items() if r == 'test'}
    assert not tr & va and (not tr & te) and (not va & te), 'Image roles are not pairwise disjoint'
    exp = {1: (4458, 787, 2656), 2: (4571, 807, 2523), 3: (4402, 777, 2722)}[tf]
    assert (len(tr), len(va), len(te)) == exp, f'Unexpected split counts: {len(tr)},{len(va)},{len(te)}'
    return (role, tr, va, te)

def img_arrays():
    """{img_id: (fold_n, images npy mmap, masks npy mmap)}."""
    out = {}
    for f in (1, 2, 3):
        imgs = np.load(f"{resource('pannuke')}/Fold {f}/images/fold{f}/images.npy", mmap_mode='r')
        msks = np.load(f"{resource('pannuke')}/Fold {f}/masks/fold{f}/masks.npy", mmap_mode='r')
        for i in range(imgs.shape[0]):
            out[f'fold{f}_image_{i}'] = (f, i, imgs, msks)
    return out

def gt_centroids(m):
    """Return per-instance (x, y, class), using ch == iid to select each instance."""
    pts = []
    for c in range(5):
        ch = m[..., c]
        for iid in np.unique(ch):
            if iid == 0:
                continue
            ys, xs = np.nonzero(ch == iid)
            pts.append((float(xs.mean()), float(ys.mean()), c))
    return pts

def crop64(img, cx, cy):
    """Return a fixed center window as a PIL image, clamped to the image bounds. F4 audit fix: independent of the segmentation model."""
    H, W = img.shape[:2]
    h0 = int(round(cy - WIN / 2))
    w0 = int(round(cx - WIN / 2))
    h0 = max(0, min(h0, H - WIN))
    w0 = max(0, min(w0, W - WIN))
    return Image.fromarray(np.ascontiguousarray(img[h0:h0 + WIN, w0:w0 + WIN]))

class GTCropDS(Dataset):
    """GT-centered 64×64 windows from train/val images, labelled with the GT class."""

    def __init__(self, items, augment):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        img_id, cx, cy, cls = self.items[k]
        f, i, imgs, _ = _IMGS[img_id]
        img = np.clip(np.asarray(imgs[i]), 0, 255).astype(np.uint8)
        pil = crop64(img, cx, cy)
        x = _tf(pil)
        if self.augment:
            r = k % 4
            if r:
                x = torch.rot90(x, r, dims=(1, 2))
            if k % 2:
                x = torch.flip(x, dims=(2,))
        return (x, cls)
_IMGS = None

def build_items(img_ids):
    """GT-centered crop records for the specified image set only."""
    items = []
    for img_id in sorted(img_ids):
        f, i, _, msks = _IMGS[img_id]
        m = np.asarray(msks[i])
        for cx, cy, c in gt_centroids(m):
            items.append((img_id, cx, cy, c))
    return items

def train_classifier(tf, tr_items, va_items, outdir):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dl_tr = DataLoader(GTCropDS(tr_items, True), batch_size=256, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    dl_va = DataLoader(GTCropDS(va_items, False), batch_size=512, num_workers=8, pin_memory=True)
    model = models.resnet18(weights=None)
    model.load_state_dict(torch.load(resource('resnet18_imagenet_weight', required=True), map_location='cpu', weights_only=True), strict=True)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    counts = np.bincount([c for _, _, _, c in tr_items], minlength=5).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w = (w / w.sum() * 5).astype(np.float32)
    crit = nn.CrossEntropyLoss(weight=torch.FloatTensor(w).cuda())
    best, best_state, pat, ep = (0.0, None, 0, 0)
    t0 = time.time()
    for ep in range(1, 16):
        model.train()
        for x, y in dl_tr:
            x, y = (x.cuda(), y.cuda())
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dl_va:
                p = model(x.cuda()).argmax(1).cpu()
                correct += (p == y).sum().item()
                total += len(y)
        acc = correct / max(total, 1)
        log(f'  [cls-v3 f{tf}] ep{ep} val_acc={acc:.4f} ({time.time() - t0:.0f}s)')
        if acc > best:
            best, best_state, pat = (acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0)
        else:
            pat += 1
        if pat >= 4:
            break
    torch.save(best_state, f'{outdir}/best_model.pth')
    return (best, ep)

def load_gt_test(te):
    gt = {}
    for img_id in sorted(te):
        f, i, _, msks = _IMGS[img_id]
        pts = gt_centroids(np.asarray(msks[i]))
        gt[img_id] = pts
    return gt

def eval_method(method_dir, det_fmt, tf, outdir, model):
    """Evaluation with the F4 audit fix: all detections → center-window classification → unified matching; no detections discarded."""
    det = pd.read_csv(det_fmt.format(f=tf))
    det['img_id'] = det.image_key.str.replace('_segmentation$', '', regex=True)
    items, keys = ([], [])
    for r in det.itertuples():
        items.append((r.img_id, float(r.cx), float(r.cy), 0))
        keys.append(r.img_id)
    xs = []
    B = 512
    for s in range(0, len(items), B):
        chunk = items[s:s + B]
        pil_list = []
        for img_id, cx, cy, _ in chunk:
            f, i, imgs, _ = _IMGS[img_id]
            img = np.clip(np.asarray(imgs[i]), 0, 255).astype(np.uint8)
            pil_list.append(crop64(img, cx, cy))
        batch = torch.stack([_tf(p) for p in pil_list]).cuda()
        with torch.inference_mode():
            xs.extend(model(batch).argmax(1).cpu().tolist())
    man = pd.read_csv(MANIFEST)
    te = sorted(set(man[(man.dataset == 'PanNuke') & (man.outer_fold == f'fold{tf}') & (man.split == 'test')].image_id))
    gt = load_gt_test(te)
    gt_list, pr_list = ([], [])
    for img_id in te:
        for j, (cx, cy, c) in enumerate(gt[img_id]):
            gt_list.append(Instance(img_id, f'g{j}', cx, cy, c))
    k = 0
    for (img_id, cx, cy, _), pred in zip(items, xs):
        pr_list.append(Instance(img_id, f'd{k}', cx, cy, int(pred)))
        k += 1
    ev = EvalInput(dataset='PanNuke', tau=12.0, gt=gt_list, pred=pr_list, image_ids=tuple(te), real_class_ids=real_classes('PanNuke'))
    res = evaluate(ev, mode='one_to_one')
    n_real = 5
    uni = float(np.mean([res.classes[c]['unified']['F1'] for c in range(n_real) if c in res.classes]))
    cls = float(np.mean([res.classes[c]['classification']['F1'] for c in range(n_real) if c in res.classes]))
    pfn = float(np.mean([res.classes[c]['plus_missed_FN']['F1'] for c in range(n_real) if c in res.classes]))
    return (res, cls, pfn, uni)

def main():
    global _IMGS
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-fold', type=int, required=True)
    ap.add_argument('--eval-all', action='store_true')
    ap.add_argument('--method', choices=['Micro-Net', 'StarDist', 'DIST'])
    ap.add_argument('--inference-only', action='store_true')
    a = ap.parse_args()
    tf = a.test_fold
    torch.set_num_threads(8)
    outdir = f'{RUNS}/resnet18_v3_classifier_testfold{tf}'
    os.makedirs(outdir, exist_ok=True)
    log(f'[cls-v3 f{tf}] Loading image index...')
    _IMGS = img_arrays()
    role, tr, va, te = split_for(tf)
    tr_items = build_items(tr)
    va_items = build_items(va)
    log(f'[cls-v3 f{tf}] train={len(tr_items)} val={len(va_items)} GT crops (image-level split: {len(tr)}/{len(va)})')
    import argparse as _ap
    resume = os.path.exists(f'{outdir}/best_model.pth') and os.path.exists(f'{outdir}/run_manifest.json')
    if resume:
        prev = json.load(open(f'{outdir}/run_manifest.json'))
        _cur_split_sha = hashlib.sha256(open(MANIFEST, 'rb').read()).hexdigest()
        assert prev.get('split_manifest_sha256') == _cur_split_sha, '[cls-v3 resume aborted] The split-manifest hash differs from the previous run. Do not overwrite historical training records; use a new run_id directory or restore the original inputs.'
        assert str(prev.get('protocol', '')).startswith('F1+F4'), '[cls-v3 resume aborted] The previous manifest uses a different protocol version; refusing to resume from incompatible artifacts.'
        if prev.get('train_images') != sorted(tr) or prev.get('val_images') != sorted(va) or prev.get('run_id') != f'resnet18_v3_classifier_testfold{tf}':
            raise ValueError('Shared classifier v3 outer-fold training identity mismatch; do not restore a different fold checkpoint into this run')
        best, ep = (prev['best_val_acc'], prev['epochs_run'])
        log(f'[cls-v3 f{tf}] Resuming from trained weights; val_acc={best}')
    else:
        if a.inference_only:
            raise FileNotFoundError('Shared classifier v3 checkpoint+manifest required; run train first')
        best, ep = train_classifier(tf, tr_items, va_items, outdir)
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(f'{outdir}/best_model.pth', weights_only=True))
    model = model.cuda().eval()
    json.dump({'run_id': f'resnet18_v3_classifier_testfold{tf}', 'protocol': 'F1+F4 fix: outer_fold-filtered roles + GT-anchored 64x64 center crops; test = every detection center-cropped (zero drop)', 'crop_rule': f'fixed center window {WIN}x{WIN} at centroid, clamped to image', 'train_images': sorted(tr), 'val_images': sorted(va), 'n_train_crops': len(tr_items), 'n_val_crops': len(va_items), 'best_val_acc': round(best, 4), 'epochs_run': ep, 'seed': SEED, 'split_manifest_sha256': hashlib.sha256(open(MANIFEST, 'rb').read()).hexdigest(), 'code_sha256': hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()}, open(f'{outdir}/' + ('evaluation_manifest.json' if resume else 'run_manifest.json'), 'w'), indent=2, ensure_ascii=False)
    if a.eval_all:
        methods = [('Micro-Net', f'{DET}/micronet_full/det_fold{{f}}.csv', f'{FINAL}/micronet_v3_pannuke'), ('StarDist', f'{DET}/stardist_full/det_fold{{f}}.csv', f'{FINAL}/stardist_v3_pannuke'), ('DIST', f'{DET}/dist_full/det_fold{{f}}.csv', f'{FINAL}/dist_v3_pannuke')]
        for name, fmt, out in methods:
            if a.method and name != a.method:
                continue
            res, cls, pfn, uni = eval_method(name, fmt, tf, outdir, model)
            os.makedirs(out, exist_ok=True)
            write_outputs(res, f'{out}/eval_fold{tf}')
            row = {'method': name, 'fold': tf, 'N_GT': res.n_gt, 'N_pred': res.n_pred, 'M': res.n_matched, 'missed': res.n_missed, 'spurious': res.n_spurious, 'cls_macroF1': round(cls, 4), 'plusFN_macroF1': round(pfn, 4), 'uni_macroF1': round(uni, 4), 'tier': 'real-class macro (v3 image-level, GT anchored)'}
            json.dump(row, open(f'{out}/fold{tf}_result.json', 'w'), indent=2)
            log(f'  [{name} f{tf}] N_gt={res.n_gt} det={res.n_pred} M={res.n_matched} uni={uni:.4f}')
    log(f'[cls-v3 f{tf}] DONE')
if __name__ == '__main__':
    main()
