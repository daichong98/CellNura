"""t08_train_dist_external.py — DIST (PyTorch port) within-fold CoNSeP/CRCHisto training + shared classifier + unified evaluation.

Protocol (aligned with the PanNuke DIST results and the external Micro-Net results):
  * Within-fold distance-regression training from scratch: random 256 crops from 22 consep training images (distances precomputed on full-image instances before cropping, preserving correct distances for boundary instances);
    68 crchisto training images (disk-synthesized instances disclosed, as in the other CRCHisto baselines); image-level train/val (22/5, 68/12, same frozen manifests)
  * Input geometry matches the PanNuke DIST results: crop 256 → reflect 92 → 440 → [22:418] 396 input → central 212 output
  * Inference: pad full images to multiples of 256, non-overlapping 256 tiles × 4-window fusion → postprocess → instance bbox+centroids
  * Classification: shared ResNet-18 (same protocol and code as the external Micro-Net results: one-to-one matching of DIST detections on its own training images to GT → bbox
    crops for training; early stopping on image-level val), reusing t08_micronet_external components
  * Evaluation: one-to-one unified_matcher at τ=5/10px, including all image_ids
Usage: python t08_train_dist_external.py --dataset consep|crchisto [--smoke]
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import argparse, glob, hashlib, json, os, sys, time
import numpy as np
import pandas as pd
import torch
BASE = resource('project_root')
HERE = resource('scripts')
RUNS = f"{resource('runs')}"
PORT = f"{resource('vendor_dist_port')}"
FINAL = f"{resource('evaluation')}"
sys.path.insert(0, HERE)
sys.path.insert(0, PORT)
from unified_matcher import EvalInput, Instance, evaluate, write_outputs
from model import DIST
from targets_and_postprocess import postprocess
from t08_micronet_external import load_gt, image_lists, CropDS, build_train_items, train_classifier, classify_dets, DS, SEED, sha256
from torch.utils.data import Dataset, DataLoader
SEED_D = SEED

def log(s):
    print(s, flush=True)

def image_and_inst(ds, name):
    """→ (img uint8, inst int32). Instance masks for consep; synthesized disks for crchisto."""
    if ds == 'consep':
        import scipy.io as sio
        from PIL import Image
        img = np.array(Image.open(f"{resource('consep')}/image/{name}.png").convert('RGB'))
        m = sio.loadmat(f"{resource('consep')}/label/{name}.mat")
        instm = (m['inst_map'] if 'inst_map' in m else m['inst_instancemap']).astype(np.int32)
        types = m['inst_type'].ravel().astype(int)
        CMAP = {1: 3, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}
        inst = np.zeros(img.shape[:2], np.int32)
        nxt = 1
        for iid in np.unique(instm):
            if iid == 0 or iid - 1 >= len(types) or types[iid - 1] == 0 or (types[iid - 1] not in CMAP):
                continue
            inst[instm == iid] = nxt
            nxt += 1
        return (img, inst)
    from PIL import Image
    img = np.array(Image.open(f"{resource('crchisto')}/CRCHisto_database/Detection/{name}/{name}.bmp").convert('RGB'))
    H, W = img.shape[:2]
    d = json.load(open(f"{resource('crchisto')}/crchisto_features_output/step0_preprocessed/{name}_processed.json"))
    yy, xx = np.mgrid[0:H, 0:W]
    inst = np.zeros((H, W), np.int32)
    nxt = 1
    CMAP = {'inflammatory': 0, 'epithelial': 1, 'fibroblast': 2, 'others': 3}
    for cl, coords in d.get('classification_data', {}).items():
        if cl not in CMAP or not isinstance(coords, list):
            continue
        for c in coords:
            if not (isinstance(c, list) and len(c) >= 2):
                continue
            cx, cy = (float(c[0]), float(c[1]))
            x0, x1 = (max(0, int(cx) - 9), min(W, int(cx) + 10))
            y0, y1 = (max(0, int(cy) - 9), min(H, int(cy) + 10))
            sel = (xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2 <= 64
            sel &= inst[y0:y1, x0:x1] == 0
            inst[y0:y1, x0:x1][sel] = nxt
            nxt += 1
    return (img, inst)

class DistDSExt(Dataset):
    """Random/deterministic 256 crops; distances precomputed (cached) on full-image instances, then cropped to the central 212 region."""

    def __init__(self, ds, names, train, crops_per_img, seed=SEED_D):
        from targets_and_postprocess import instance_distances as _idist
        self.ds, self.train, self.cpi = (ds, train, crops_per_img)
        self.rng = np.random.RandomState(seed if train else seed + 1)
        self.items = []
        for n in names:
            img, inst = image_and_inst(ds, n)
            dist = _idist(inst).astype(np.float32)
            for _ in range(crops_per_img):
                self.items.append((n, img, dist))
        self.n = len(self.items)

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        n, img, dist = self.items[k % self.n]
        H, W = dist.shape
        C = 256
        if self.train:
            oy, ox = (self.rng.randint(0, H - C + 1), self.rng.randint(0, W - C + 1))
        else:
            r = np.random.RandomState(k)
            oy, ox = (r.randint(0, H - C + 1), r.randint(0, W - C + 1))
        crop = img[oy:oy + C, ox:ox + C].astype(np.float32)
        padded = np.pad(crop, ((92, 92), (92, 92), (0, 0)), mode='reflect')
        x = padded[92:92 + 348, 92:92 + 348]
        target = dist[oy + 92:oy + 92 + 164, ox + 92:ox + 92 + 164]
        if self.train:
            rot = self.rng.randint(0, 4)
            if rot:
                x = np.rot90(x, rot)
                target = np.rot90(target, rot)
            if self.rng.rand() < 0.5:
                x, target = (x[:, ::-1], target[:, ::-1])
            if self.rng.rand() < 0.5:
                x, target = (x[::-1], target[::-1])
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)).astype(np.float32)), torch.from_numpy(np.ascontiguousarray(target[None].copy())))

def train_dist(ds, smoke=False):
    torch.manual_seed(SEED_D)
    np.random.seed(SEED_D)
    run_id = f'dist_scratch_{ds}' + ('_smoke' if smoke else '')
    outdir = f'{RUNS}/{run_id}'
    os.makedirs(outdir, exist_ok=True)
    roles = frozen_roles(ds)
    tr_names, va_names = (roles['train'], roles['val'])
    if smoke:
        tr_names, va_names = (tr_names[:3], va_names[:1])
    cpi = 8 if ds == 'consep' else 4
    ds_tr = DistDSExt(ds, tr_names, True, cpi)
    ds_va = DistDSExt(ds, va_names, False, 2)
    dl_tr = DataLoader(ds_tr, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=8, num_workers=4, pin_memory=True)
    log(f'[{run_id}] train={len(ds_tr)} tiles val={len(ds_va)} tiles')
    model = DIST().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    crit = torch.nn.MSELoss()
    epochs = 2 if smoke else 300
    best, ep = (1e+18, 0)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = nb = 0
        for x, y in dl_tr:
            x, y = (x.cuda(), y.cuda())
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        if ep % 25 == 0 or ep == 1 or smoke:
            log(f'[{run_id}] ep {ep}/{epochs} loss={tot / max(nb, 1):.4f} val_mse={vl:.4f} ({time.time() - t0:.0f}s)')
        if vl < best:
            best = vl
            torch.save(model.state_dict(), f'{outdir}/best_model.pth')
    log(f'[{run_id}] DONE best_val_mse={best:.4f}')
    json.dump({'run_id': run_id, 'impl': 'DRFNS PyTorch port distance regression, from scratch (fold-internal)', 'dataset': ds, 'epochs': epochs, 'best_val_mse': round(best, 4), 'seed': SEED_D, 'gt_note': 'crchisto: disk r=8 synthesized instances (disclosed)' if ds == 'crchisto' else 'consep: official inst masks', 'code_sha256': sha256(__file__)}, open(f'{outdir}/run_manifest.json', 'w'), indent=2, ensure_ascii=False)
    return outdir

def predict_full(model, img, dist_scale=1.0):
    """Full-image padding → non-overlapping 256 tiles × 4-window fusion → distance map → postprocessing → instances."""
    H, W = img.shape[:2]
    ph = (256 - H % 256) % 256
    pw = (256 - W % 256) % 256
    imgp = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='reflect').astype(np.float32)
    Hp, Wp = imgp.shape[:2]
    canvas = np.zeros((Hp, Wp), np.float32)
    for oy in range(0, Hp, 256):
        for ox in range(0, Wp, 256):
            crop = imgp[oy:oy + 256, ox:ox + 256]
            padded = np.pad(crop, ((92, 92), (92, 92), (0, 0)), mode='reflect')
            xs = []
            for wy in (0, 44):
                for wx in (0, 44):
                    xs.append(padded[wy:wy + 396, wx:wx + 396])
            xb = torch.from_numpy(np.stack([t.transpose(2, 0, 1) for t in xs]).astype(np.float32)).cuda()
            with torch.inference_mode():
                d = model(xb)[:, 0].cpu().numpy()
            cv = np.zeros((256, 256), np.float32)
            cc = np.zeros((256, 256), np.float32)
            k = 0
            for wy in (0, 44):
                for wx in (0, 44):
                    cv[wy:wy + 212, wx:wx + 212] += d[k]
                    cc[wy:wy + 212, wx:wx + 212] += 1
                    k += 1
            canvas[oy:oy + 256, ox:ox + 256] += cv / cc
    inst = postprocess(canvas[:H, :W], h=1, threshold=1.0)
    out = []
    for iid in np.unique(inst):
        if iid == 0:
            continue
        ys, xs_ = np.nonzero(inst == iid)
        if len(ys) < 3:
            continue
        out.append({'centroid': [float(xs_.mean()), float(ys.mean())], 'bbox': [int(xs_.min()), int(ys.min()), int(xs_.max()) + 1, int(ys.max()) + 1]})
    return out

def infer_all(ds, outdir, smoke=False):
    model = DIST()
    model.load_state_dict(torch.load(f'{outdir}/best_model.pth', map_location='cpu', weights_only=True))
    model = model.cuda().eval()
    tr_names, te_names = image_lists(ds)
    names = tr_names + te_names
    if smoke:
        names = names[:2]
    savedir = f'{outdir}/det'
    os.makedirs(savedir, exist_ok=True)
    identity = {'distance_checkpoint_sha256': sha256(f'{outdir}/best_model.pth'), 'dataset': ds, 'code_sha256': sha256(__file__)}
    identity_path=f'{savedir}/DETECTION_IDENTITY.json'
    if os.path.exists(identity_path):
        if json.load(open(identity_path)) != identity: raise ValueError('DIST distance cache identity mismatch; use a fresh output tree')
    elif glob.glob(f'{savedir}/*.json'):raise ValueError('DIST legacy cache has no checkpoint identity; use a fresh output tree')
    else:json.dump(identity,open(identity_path,'w'),indent=2)
    dets = {}
    t0 = time.time()
    for i, name in enumerate(names, 1):
        cache = f'{savedir}/{name}.json'
        if os.path.exists(cache):
            dets[name] = json.load(open(cache))['nuclei']
            continue
        img, _ = image_and_inst(ds, name)
        nuc = predict_full(model, img)
        dets[name] = nuc
        json.dump({'image_id': name, 'nuclei': nuc}, open(cache, 'w'))
        if i % 10 == 0:
            log(f'  infer {i}/{len(names)} ({time.time() - t0:.0f}s)')
    log(f'[{ds}] infer DONE {len(names)} imgs ({time.time() - t0:.0f}s)')
    return dets

def classify_and_eval(ds, outdir, dets, smoke=False):
    tau = DS[ds]['tau']
    gt = load_gt(ds)
    tr_names, te_names = image_lists(ds)
    val_names = set(frozen_roles(ds)['val'])
    items, ledger = build_train_items(ds, {n: dets[n] for n in tr_names if n in dets}, gt, tau)
    import pandas as pd
    pd.DataFrame(ledger).to_csv(f'{outdir}/train_crop_ledger.csv', index=False)
    classifier_dir = f'{outdir}/classifier'
    os.makedirs(classifier_dir, exist_ok=True)
    if inference_only():
        if not os.path.isfile(f'{classifier_dir}/best_model.pth'):
            raise FileNotFoundError('DIST external classifier missing; train first')
    else:
        best_acc, epochs_run, counts = train_classifier(ds, items, val_names, classifier_dir)
    from torchvision import models as tvm
    import torch.nn as nn
    clf = tvm.resnet18()
    clf.fc = nn.Linear(clf.fc.in_features, DS[ds]['n_cls'])
    clf.load_state_dict(torch.load(f'{classifier_dir}/best_model.pth', weights_only=True))
    cls_test = classify_dets(ds, clf, {n: dets[n] for n in te_names if n in dets})
    gt_list, pr_list = ([], [])
    for name in te_names:
        pts, cls = gt[name]
        for j, (c, k) in enumerate(zip(pts, cls)):
            gt_list.append(Instance(name, f'g{j}', c[0], c[1], int(k)))
        nuc = dets.get(name, [])
        for k2, n in enumerate(nuc, 1):
            pr_list.append(Instance(name, f'd{k2}', n['centroid'][0], n['centroid'][1], int(cls_test[name][k2 - 1])))
    ev = EvalInput(dataset=ds, tau=tau, gt=gt_list, pred=pr_list, image_ids=tuple(te_names), real_class_ids=real_classes(ds))
    res = evaluate(ev, mode='one_to_one')
    out_final = f'{FINAL}/dist_external_{ds}'
    os.makedirs(out_final, exist_ok=True)
    write_outputs(res, f'{out_final}/eval')
    row = {'dataset': ds, 'N_GT': res.n_gt, 'N_pred': res.n_pred, 'M': res.n_matched, 'missed': res.n_missed, 'spurious': res.n_spurious, 'cls_macroF1': round(res.macro('classification'), 4), 'plusFN_macroF1': round(res.macro('plus_missed_FN'), 4), 'uni_macroF1': round(res.macro('unified'), 4)}
    json.dump(row, open(f'{out_final}/result.json', 'w'), indent=2)
    log(f'[{ds}] {row}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['consep', 'crchisto'])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    outdir = f'{RUNS}/dist_scratch_{a.dataset}' + ('_smoke' if a.smoke else '')
    if not inference_only():
        outdir = train_dist(a.dataset, a.smoke)
    dets = infer_all(a.dataset, outdir, a.smoke)
    classify_and_eval(a.dataset, outdir, dets, a.smoke)
if __name__ == '__main__':
    main()
