"""t08_train_maskrcnn.py — TorchVision Mask R-CNN within-fold training/inference (authors' option A).

Implementation identity: TorchVision maskrcnn_resnet50_fpn (ResNet50-FPN), with implementation differences from
Matterport TF1 (declared in the response letter). PanNuke NUM_CLASSES=6 (background+5 classes, labels 1-5 ↔ mask channels 0-4);
CoNSeP NUM_CLASSES=5 (background+Misc/Inf/Epi/Spindle). Anchors are reduced for small nuclei (8..128).
Within-fold protocol: reinitialize for each fold; train=two training folds; val=15% at image level within that training split (select best + freeze postprocessing parameters);
test=outer test fold, inference only. Export predictions in the unified schema → unified_matcher.
Usage:
  python t08_train_maskrcnn.py --dataset pannuke --test-fold 1 --train
  python t08_train_maskrcnn.py --dataset pannuke --test-fold 1 --infer
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import argparse
import glob
import hashlib
import hashlib
import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
BASE = resource('project_root')
PKG = resource('data_package')
MANIFEST = f"{resource('split_manifest')}"
CONSEP = f"{resource('consep')}"
CRC_DIR = f"{resource('crchisto')}"
OUTROOT = f"{resource('runs')}"
SEED = 42
import scipy.io as sio
from PIL import Image

class PanNukeDetect(Dataset):
    """masks.npy → torchvision detection targets (boxes/labels/masks)."""

    def __init__(self, fold_list, mode, split_rows=None, limit=0):
        assert mode in ('train', 'val', 'test')
        self.items = []
        for f in fold_list:
            imgs = np.load(f"{resource('pannuke')}/Fold {f}/images/fold{f}/images.npy", mmap_mode='r')
            msks = np.load(f"{resource('pannuke')}/Fold {f}/masks/fold{f}/masks.npy", mmap_mode='r')
            ids = range(imgs.shape[0])
            if split_rows is not None:
                want = set(split_rows.get(f, []))
                ids = [i for i in ids if i in want]
            for i in ids:
                self.items.append((f, i, imgs, msks))
            if limit:
                self.items = self.items[:limit]
        self.aug = mode == 'train'

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        f, i, imgs, msks = self.items[j]
        img = np.clip(np.asarray(imgs[i]), 0, 255).astype(np.uint8)
        m = np.asarray(msks[i])
        boxes, labels, masks = ([], [], [])
        for c in range(5):
            ch = m[..., c]
            for iid in np.unique(ch):
                if iid == 0:
                    continue
                sel = ch == iid
                ys, xs = np.nonzero(sel)
                if len(ys) < 3:
                    continue
                boxes.append([float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1])
                labels.append(c + 1)
                masks.append(sel.astype(np.uint8))
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, 256, 256), dtype=torch.uint8)
        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
        target = {'boxes': boxes, 'labels': labels, 'masks': masks, 'image_id': torch.tensor([i]), 'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)}
        return (img_t, target)

class CoNSePDetect(Dataset):
    """CoNSeP 27 training images (inst_map/type_map) → detection targets. NUM_CLASSES=5 (1-4 + background).

    Full-size 1000×1000 instance masks (~850 nuclei/image ≈ 800MB/sample) are impractical; resize all to 512×512
    (nearest-neighbor mask resizing, labels unchanged). split_of/val_exclude implement image-level train/val splitting;
    the 14 test images must not be used for validation-based model selection."""
    RESIZE = 512

    def __init__(self, split='train', limit=0, val_exclude=None, split_of=None):
        import scipy.io as sio
        cmap = {1: 3, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}
        if split == 'test':
            self.items = sorted(glob.glob(f'{CONSEP}/label/test_*.mat'))
        elif split_of == 'val':
            files = sorted(val_exclude)
            self.items = [f'{CONSEP}/label/{n}.mat' for n in files]
        else:
            self.items = [f for f in sorted(glob.glob(f'{CONSEP}/label/train_*.mat')) if os.path.basename(f)[:-4] not in (val_exclude or set())]
        if limit:
            self.items = self.items[:limit]
        self.cmap = cmap
        self.sio = sio

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        import cv2
        f = self.items[j]
        name = os.path.basename(f).replace('.mat', '')
        m = self.sio.loadmat(f)
        inst = (m['inst_map'] if 'inst_map' in m else m['inst_instancemap']).astype(np.int32)
        types = m['inst_type'].ravel().astype(int)
        img = np.array(Image.open(f'{CONSEP}/image/{name}.png').convert('RGB'))
        if img.shape[0] != self.RESIZE:
            r = self.RESIZE / img.shape[0]
            img = cv2.resize(img, (self.RESIZE, self.RESIZE), interpolation=cv2.INTER_LINEAR)
            inst = cv2.resize(inst, (self.RESIZE, self.RESIZE), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            keep = np.unique(inst)
        else:
            r = 1.0
            keep = np.unique(inst)
        boxes, labels, masks = ([], [], [])
        for iid in keep:
            if iid == 0:
                continue
            sel = inst == iid
            if iid - 1 >= len(types) or types[iid - 1] == 0 or types[iid - 1] not in self.cmap:
                continue
            ys, xs = np.nonzero(sel)
            if len(ys) < 3:
                continue
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1])
            labels.append(self.cmap[int(types[iid - 1])] + 1)
            masks.append(sel.astype(np.uint8))
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, self.RESIZE, self.RESIZE), dtype=torch.uint8)
        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
        target = {'boxes': boxes, 'labels': labels, 'masks': masks, 'image_id': torch.tensor([j]), 'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)}
        return (img_t, target)

class CRCHistoDetect(Dataset):
    """Native 500×500 CRCHisto images + disk-synthesized instances (r=8, same disclosure as the other CRCHisto baselines).
    NUM_CLASSES=5 (bg+4); image-level 68/12 split (same manifest as the shared classifier); 20 test images used only for inference."""

    def __init__(self, split, limit=0):
        import json as _json
        self.split = split
        self.CMAP = {'inflammatory': 0, 'epithelial': 1, 'fibroblast': 2, 'others': 3}
        base = f"{resource('crchisto')}"
        self.pre = f'{base}/crchisto_features_output/step0_preprocessed'
        sp = {'train_images': frozen_roles('crchisto')['train'], 'val_images': frozen_roles('crchisto')['val']}
        man = pd.read_csv(MANIFEST)
        te = set(man[(man.dataset == 'CRCHisto') & (man.split == 'test')].image_id)
        if split == 'train':
            self.names = sorted(sp['train_images'])
        elif split == 'val':
            self.names = sorted(sp['val_images'])
        else:
            self.names = sorted(te)
        if limit:
            self.names = self.names[:limit]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, j):
        import json as _json
        name = self.names[j]
        img = np.array(Image.open(f"{resource('crchisto')}/CRCHisto_database/Detection/{name}/{name}.bmp").convert('RGB'))
        H, W = img.shape[:2]
        d = _json.load(open(f'{self.pre}/{name}_processed.json'))
        yy, xx = np.mgrid[0:H, 0:W]
        boxes, labels, masks = ([], [], [])
        for cl, coords in d.get('classification_data', {}).items():
            if cl not in self.CMAP or not isinstance(coords, list):
                continue
            for c in coords:
                if not (isinstance(c, list) and len(c) >= 2):
                    continue
                cx, cy = (float(c[0]), float(c[1]))
                x0, x1 = (max(0, int(cx) - 9), min(W, int(cx) + 10))
                y0, y1 = (max(0, int(cy) - 9), min(H, int(cy) + 10))
                sel = np.zeros((H, W), bool)
                sel[y0:y1, x0:x1] = (xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2 <= 64
                ys2, xs2 = np.nonzero(sel)
                if len(ys2) < 3:
                    continue
                boxes.append([float(xs2.min()), float(ys2.min()), float(xs2.max()) + 1, float(ys2.max()) + 1])
                labels.append(self.CMAP[cl] + 1)
                masks.append(sel.astype(np.uint8))
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, H, W), dtype=torch.uint8)
        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
        target = {'boxes': boxes, 'labels': labels, 'masks': masks, 'image_id': torch.tensor([j]), 'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)}
        return (img_t, target)

def collate(batch):
    return tuple(zip(*batch))

def build_model(num_classes, small_anchors=True, initialize=True):
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    model = __import__('torchvision').models.detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=91)
    if initialize:
        model.load_state_dict(torch.load(resource('maskrcnn_coco_weight', required=True), map_location='cpu', weights_only=True), strict=True)
    if small_anchors:
        ap = model.rpn.anchor_generator
        ap.sizes = ((8,), (16,), (32,), (64,), (128,))
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    return model

def train(ds_name, test_fold, epochs, small_anchors):
    import torchvision
    from torch.utils.data import DataLoader
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = 'cuda'
    run_id = f'maskrcnn_{ds_name}_testfold{test_fold}'
    outdir = f'{OUTROOT}/{run_id}'
    os.makedirs(outdir, exist_ok=True)
    log = open(f'{outdir}/train.log', 'a', buffering=1)

    def P(s):
        print(f'[{run_id}] {s}', flush=True)
        log.write(s + '\n')
    num_classes = 6 if ds_name == 'pannuke' else 5
    if ds_name == 'pannuke':
        man = pd.read_csv(MANIFEST)
        pan = man[(man.dataset == 'PanNuke') & (man.outer_fold == f'fold{test_fold}')]
        assert pan.split.value_counts().to_dict() == {'train': 4458 if test_fold == 1 else 4571 if test_fold == 2 else 4402, 'val': 787 if test_fold == 1 else 807 if test_fold == 2 else 777, 'test': 2656 if test_fold == 1 else 2523 if test_fold == 2 else 2722}, f'split rows unexpected: {pan.split.value_counts().to_dict()}'
        tr_rows, va_rows = ({}, {})
        for s in ('train', 'val'):
            ids = pan[pan.split == s].image_id.tolist()
            tgt = tr_rows if s == 'train' else va_rows
            for x in ids:
                m = re.match('fold(\\d+)_image_(\\d+)', x)
                tgt.setdefault(int(m.group(1)), []).append(int(m.group(2)))
        for f in (1, 2, 3):
            assert not set(tr_rows.get(f, [])) & set(va_rows.get(f, [])), 'train/val overlap'
        ds_tr = PanNukeDetect([1, 2, 3], 'train', split_rows=tr_rows)
        ds_va = PanNukeDetect([1, 2, 3], 'val', split_rows=va_rows)
        assert len(ds_tr) == int((pan.split == 'train').sum()), (len(ds_tr), 'train size mismatch')
        assert len(ds_va) == int((pan.split == 'val').sum()), (len(ds_va), 'val size mismatch')
    elif ds_name == 'consep':
        val_set = set(frozen_roles('consep')['val'])
        ds_tr = CoNSePDetect('train', limit=0, val_exclude=val_set, split_of='train')
        ds_va = CoNSePDetect('train', limit=0, val_exclude=val_set, split_of='val')
    else:
        ds_tr = CRCHistoDetect('train')
        ds_va = CRCHistoDetect('val')

    def collate_fn(b):
        return tuple(zip(*b))
    dl_tr = DataLoader(ds_tr, batch_size=4, shuffle=True, num_workers=16, collate_fn=collate_fn, pin_memory=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=2, shuffle=False, num_workers=8, collate_fn=collate_fn, pin_memory=True, persistent_workers=True)
    model = build_model(num_classes, small_anchors).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 3, 1), gamma=0.1)

    def det_f1(model, loader, iou=0.5):
        """Single-class detection F1 (instance-level matching after merging all classes), used for validation-based selection."""
        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for imgs, targets in loader:
                imgs = [im.to(device) for im in imgs]
                outs = model(imgs)
                for t, o in zip(targets, outs):
                    keep = o['scores'] > 0.5
                    pd_boxes = o['boxes'][keep]
                    gt_boxes = t['boxes']
                    matched_gt = set()
                    for pb in pd_boxes:
                        best_iou, best_j = (0.0, -1)
                        for gj, gb in enumerate(gt_boxes):
                            if gj in matched_gt:
                                continue
                            xA = max(pb[0], gb[0])
                            yA = max(pb[1], gb[1])
                            xB = min(pb[2], gb[2])
                            yB = min(pb[3], gb[3])
                            inter = max(0, xB - xA) * max(0, yB - yA)
                            union = (pb[2] - pb[0]) * (pb[3] - pb[1]) + (gb[2] - gb[0]) * (gb[3] - gb[1]) - inter
                            i = inter / max(union, 1)
                            if i > best_iou:
                                best_iou, best_j = (i, gj)
                        if best_iou >= iou:
                            tp += 1
                            matched_gt.add(best_j)
                        else:
                            fp += 1
                    fn += len(gt_boxes) - len(matched_gt)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        return f1
    best, best_state = (-1.0, None)
    for ep in range(1, epochs + 1):
        model.train()
        tot = nb = 0
        for imgs, targets in dl_tr:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        f1 = det_f1(model, dl_va)
        print(f'[{run_id}] epoch {ep}/{epochs} loss={tot / max(nb, 1):.4f} val_detF1={f1:.4f}', flush=True)
        if f1 > best:
            best, best_state = (f1, {k: v.cpu().clone() for k, v in model.state_dict().items()})
            torch.save(best_state, f'{outdir}/best_model.pth')
    manifest = {'run_id': run_id, 'impl': 'torchvision maskrcnn_resnet50_fpn', 'dataset': ds_name, 'test_fold': test_fold, 'num_classes': num_classes, 'anchors': '8/16/32/64/128', 'epochs': epochs, 'seed': SEED, 'best_val_detF1': round(best, 4), 'train_source_sha256': hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(), 'split_manifest_sha256': hashlib.sha256(open(MANIFEST, 'rb').read()).hexdigest()}
    json.dump(manifest, open(f'{outdir}/run_manifest.json', 'w'), indent=2, ensure_ascii=False)
    P(f'DONE {json.dumps(manifest, ensure_ascii=False)}')

def infer(ds_name, test_fold, score_thr):
    import torchvision
    from torch.utils.data import DataLoader
    device = 'cuda'
    run_id = f'maskrcnn_{ds_name}_testfold{test_fold}'
    outdir = f'{OUTROOT}/{run_id}'
    model = build_model(6 if ds_name == 'pannuke' else 5, small_anchors=True, initialize=False)
    sd = torch.load(f'{outdir}/best_model.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    model.roi_heads.score_thresh = score_thr
    if ds_name == 'pannuke':
        ds = PanNukeDetect([test_fold], 'test')
        cls_of = lambda lab: int(lab) - 1
        scale = 1.0
    elif ds_name == 'consep':
        ds = CoNSePDetect('test')
        cls_of = lambda lab: int(lab) - 1
        scale = 1000 / CoNSePDetect.RESIZE
    else:
        ds = CRCHistoDetect('test')
        cls_of = lambda lab: int(lab) - 1
        scale = 1.0
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda b: tuple(zip(*b)))
    rows = []
    with torch.no_grad():
        for k, (imgs, targets) in enumerate(dl):
            outs = model([im.to(device) for im in imgs])
            for t, o in zip(targets, outs):
                img_id = int(t['image_id'].item())
                if ds_name == 'pannuke':
                    key = f'fold{test_fold}_image_{img_id}'
                elif ds_name == 'consep':
                    key = os.path.basename(ds.items[img_id]).replace('.mat', '')
                else:
                    key = ds.names[img_id]
                for box, lab, msk, sc in zip(o['boxes'].cpu(), o['labels'].cpu(), o['masks'].cpu(), o['scores'].cpu()):
                    m = (msk[0] > 0.5).numpy()
                    if m.sum() < 3:
                        continue
                    ys, xs = np.nonzero(m)
                    rows.append({'image_key': key, 'nucleus_id': sum((1 for r in rows if r['image_key'] == key)) + 1, 'cx': float(xs.mean()) * scale, 'cy': float(ys.mean()) * scale, 'pred': cls_of(int(lab)), 'score': float(sc)})
    df = pd.DataFrame(rows, columns=['image_key','nucleus_id','cx','cy','pred','score'])
    df.to_csv(f'{outdir}/pred_all.csv', index=False)
    print(f'[{run_id}] infer done: {len(df)} detections -> pred_all.csv', flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='pannuke', choices=['pannuke', 'consep', 'crchisto'])
    ap.add_argument('--test-fold', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--infer', action='store_true')
    ap.add_argument('--score-thr', type=float, default=0.5)
    a = ap.parse_args()
    if a.train:
        train(a.dataset, a.test_fold, a.epochs, True)
    if a.infer:
        infer(a.dataset, a.test_fold, a.score_thr)
