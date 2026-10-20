"""Shared image lists, tiled inference, and ResNet-18 classifier helpers.

The within-fold Micro-Net scripts and the DIST external trainer import these helpers.
This file is not a separate reported Micro-Net result.
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource
import glob, hashlib, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
BASE = resource('project_root')
HERE = resource('scripts')
sys.path.insert(0, HERE)
import micronet_standalone
from unified_matcher import _assign_one_to_one
CONSEP = f"{resource('consep')}"
CRC_DIR = f"{resource('crchisto')}"
MANIFEST = f"{resource('split_manifest')}"
SEED = 42
TILE, STRIDE, MICRO_IN = (256, 128, 252)
MIN_SIZE, MARGIN = (50, 8)
DS = {'consep': dict(tau=5.0, n_cls=4, classes=['Inflammatory', 'Epithelial', 'Spindle', 'Misc']), 'crchisto': dict(tau=10.0, n_cls=4, classes=['inflammatory', 'epithelial', 'fibroblast', 'others'])}

def sha256(path, bs=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def load_gt(ds):
    """Reuse the fixed GT loading and class mapping from t08_external_indep_eval.py."""
    import scipy.io as sio
    gt = {}
    if ds == 'consep':
        cmap = {1: 3, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}
        for f in sorted(glob.glob(f'{CONSEP}/label/*.mat')):
            name = os.path.basename(f)[:-4]
            m = sio.loadmat(f)
            pts, cls = ([], [])
            for c, t in zip(m['inst_centroid'], m['inst_type'].ravel().astype(int)):
                if t == 0 or t not in cmap:
                    continue
                pts.append((float(c[0]), float(c[1])))
                cls.append(cmap[t])
            gt[name] = (np.array(pts) if pts else np.zeros((0, 2)), np.array(cls, int))
    else:
        cmap = {'inflammatory': 0, 'epithelial': 1, 'fibroblast': 2, 'others': 3}
        for f in sorted(glob.glob(f'{CRC_DIR}/crchisto_features_output/step0_preprocessed/img*_processed.json')):
            d = json.load(open(f))
            pts, cls = ([], [])
            for cl, coords in d.get('classification_data', {}).items():
                if cl in cmap and isinstance(coords, list):
                    for c in coords:
                        if isinstance(c, list) and len(c) >= 2:
                            pts.append((float(c[0]), float(c[1])))
                            cls.append(cmap[cl])
            gt[d['image_name']] = (np.array(pts) if pts else np.zeros((0, 2)), np.array(cls, int))
    return gt

def image_lists(ds):
    man = pd.read_csv(MANIFEST)
    sub = man[man.dataset == ('CoNSeP' if ds == 'consep' else 'CRCHisto')]
    tr = sorted(set(sub[sub.split == 'train'].image_id))
    te = sorted(set(sub[sub.split == 'test'].image_id))
    return (tr, te)

def image_path(ds, name):
    if ds == 'consep':
        return f'{CONSEP}/image/{name}.png'
    return f'{CRC_DIR}/CRCHisto_database/Detection/{name}/{name}.bmp'

def model_weight_sha(model):
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def infer_image(model, img):
    """Native-resolution tiled inference → whole-image nucleus probabilities → instance centroids/bounding boxes."""
    import cv2
    from scipy import ndimage
    from skimage import morphology
    H, W = img.shape[:2]
    ys = sorted(set(list(range(0, H - TILE + 1, STRIDE)) + [max(0, H - TILE)]))
    xs = sorted(set(list(range(0, W - TILE + 1, STRIDE)) + [max(0, W - TILE)]))
    canvas = np.zeros((H, W), np.float32)
    cover = np.zeros((H, W), np.float32)
    for oy in ys:
        for ox in xs:
            crop = img[oy:oy + TILE, ox:ox + TILE]
            resized = cv2.resize(crop, (MICRO_IN, MICRO_IN), interpolation=cv2.INTER_LINEAR)
            value = micronet_standalone.MicroNet.preproc(resized).astype(np.float32)
            x = torch.from_numpy(value.transpose(2, 0, 1)[None]).cuda()
            with torch.inference_mode():
                out = model(x)[0]
            prob = out[0, 1]  # MicroNet's output head already applies softmax.
            prob = torch.nn.functional.interpolate(prob[None, None], size=(TILE, TILE), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            canvas[oy:oy + TILE, ox:ox + TILE] += prob
            cover[oy:oy + TILE, ox:ox + TILE] += 1
    assert (cover > 0).all(), 'Uncovered image pixels'
    canvas /= cover
    fg = canvas >= 0.5
    inst = ndimage.label(fg)[0]
    inst = morphology.remove_small_objects(inst, min_size=MIN_SIZE)
    nuclei = []
    for iid in np.unique(inst):
        if iid == 0:
            continue
        sel = np.nonzero(inst == iid)
        y0, y1 = (int(sel[0].min()), int(sel[0].max()) + 1)
        x0, x1 = (int(sel[1].min()), int(sel[1].max()) + 1)
        nuclei.append({'centroid': [float(sel[1].mean()), float(sel[0].mean())], 'bbox': [x0, y0, x1, y1], 'area': int(len(sel[0]))})
    return nuclei

def detect_split(model, ds, names, savedir):
    identity = {'segmentation_sha256': model_weight_sha(model), 'dataset': ds, 'image_ids': sorted(names), 'tile': TILE, 'stride': STRIDE, 'min_size': MIN_SIZE, 'inference_code_sha256': sha256(__file__)}
    os.makedirs(savedir, exist_ok=True)
    identity_path = f'{savedir}/DETECTION_IDENTITY.json'
    if os.path.exists(identity_path):
        if json.load(open(identity_path)) != identity: raise ValueError('Micro-Net detection cache identity changed')
    elif glob.glob(f'{savedir}/*.json'):
        raise ValueError('Legacy detection cache lacks identity; configure a fresh output directory')
    else: json.dump(identity, open(identity_path, 'w'), indent=2)
    os.makedirs(savedir, exist_ok=True)
    out = {}
    t0 = time.time()
    for i, name in enumerate(names, 1):
        cache = f'{savedir}/{name}.json'
        if os.path.exists(cache):
            out[name] = json.load(open(cache))['nuclei']
            continue
        img = np.array(Image.open(image_path(ds, name)).convert('RGB'))
        nuc = infer_image(model, img)
        out[name] = nuc
        json.dump({'image_id': name, 'image_shape': list(img.shape), 'nuclei': nuc}, open(cache, 'w'))
        print(f'[{ds} detect] {i}/{len(names)} {name}: {len(nuc)} det', flush=True)
    print(f'[{ds} detect] DONE {len(names)} imgs in {time.time() - t0:.0f}s', flush=True)
    return out
_tf = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class CropDS(Dataset):
    """Generate 64×64 crops on demand from (image, bbox), with the same augmentation as t08_train_resnet18_v2.py."""

    def __init__(self, items, augment=False):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        ds, name, bb, cls = self.items[j]
        img = Image.open(image_path(ds, name)).convert('RGB')
        x0, y0, x1, y1 = bb
        x0 = max(0, x0 - MARGIN)
        y0 = max(0, y0 - MARGIN)
        W, H = img.size
        x1 = min(W, x1 + MARGIN)
        y1 = min(H, y1 + MARGIN)
        crop = img.crop((x0, y0, x1, y1)).resize((64, 64), Image.BILINEAR)
        x = _tf(crop)
        if self.augment:
            k = j % 4
            if k:
                x = torch.rot90(x, k, dims=(1, 2))
            if j % 2:
                x = torch.flip(x, dims=(2,))
        return (x, cls)

def build_train_items(ds, dets_train, gt, tau):
    """Match training-image detections one-to-one to GT using the fixed protocol threshold τ; return (bbox, GT label) for matched pairs."""
    items, ledger = ([], [])
    for name, nuclei in dets_train.items():
        pts, cls = gt[name]
        if len(nuclei) == 0 or len(pts) == 0:
            continue
        pr = np.array([[n['centroid'][0], n['centroid'][1]] for n in nuclei], np.float64)
        pairs = _assign_one_to_one(pts.astype(np.float64), pr, tau)
        for gi, pi in pairs:
            items.append((ds, name, nuclei[pi]['bbox'], int(cls[gi])))
            ledger.append({'image_id': name, 'gt_idx': gi, 'det_idx': pi, 'label': int(cls[gi])})
    return (items, pd.DataFrame(ledger))

def train_classifier(ds, items, val_names, outdir):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tr = [it for it in items if it[1] not in val_names]
    va = [it for it in items if it[1] in val_names]
    print(f'[{ds} cls] crops train={len(tr)} val={len(va)} (image-level split: {len(set((i[1] for i in tr)))}/{len(val_names)} images)', flush=True)
    dl_tr = DataLoader(CropDS(tr, True), batch_size=256, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    dl_va = DataLoader(CropDS(va, False), batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
    n_cls = DS[ds]['n_cls']
    model = models.resnet18(weights=None)
    model.load_state_dict(torch.load(resource('resnet18_imagenet_weight', required=True), map_location='cpu', weights_only=True), strict=True)
    model.fc = nn.Linear(model.fc.in_features, n_cls)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    counts = np.bincount([it[3] for it in tr], minlength=n_cls).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w = (w / w.sum() * n_cls).astype(np.float32)
    crit = nn.CrossEntropyLoss(weight=torch.FloatTensor(w).cuda())
    best, best_state, pat, ep = (0.0, None, 0, 0)
    for ep in range(1, 16):
        model.train()
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
        correct = total = 0
        with torch.no_grad():
            for x, y in dl_va:
                p = model(x.cuda()).argmax(1).cpu()
                correct += (p == y).sum().item()
                total += len(y)
        acc = correct / max(total, 1)
        print(f'[{ds} cls] ep {ep}/15 loss={tot / max(nb, 1):.4f} val_acc={acc:.4f}', flush=True)
        if acc > best:
            best, best_state, pat = (acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0)
        else:
            pat += 1
        if pat >= 4:
            break
    torch.save(best_state, f'{outdir}/best_model.pth')
    json.dump({'best_val_acc': round(best, 4), 'epochs_run': ep}, open(f'{outdir}/cls_stat.json', 'w'))
    return (best, ep, counts.tolist())

def classify_dets(ds, model, dets_test):
    model = model.cuda().eval()
    preds = {}
    items, keys = ([], [])
    for name, nuclei in dets_test.items():
        for k, n in enumerate(nuclei):
            items.append((ds, name, n['bbox'], 0))
            keys.append((name, k))
    dl = DataLoader(CropDS(items, False), batch_size=512, num_workers=8, pin_memory=True)
    out = []
    with torch.inference_mode():
        for x, _ in dl:
            out.extend(model(x.cuda()).argmax(1).cpu().tolist())
    for (name, k), c in zip(keys, out):
        preds.setdefault(name, {})[k] = int(c)
    return preds
