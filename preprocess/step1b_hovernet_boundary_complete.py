#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import glob
import numpy as np
import cv2
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, '/path/to/hovernet_segmentation')
sys.path.insert(0, '/path/to/hovernet_segmentation/models')
from models.hovernet.net_desc import create_model
from models.hovernet.post_proc import process as hn_process

PROJ = "/path/to/PanNuke_classification_final"
PAD = 46
IN, OUT = 256, 164
OFFS = [0, 92]


def load_model():
    m = create_model(mode='fast', nr_types=6)
    sd = torch.load('/path/to/hovernet_segmentation/hovernet_fast_pannuke_type_tf2pytorch.pth',
                    map_location='cpu', weights_only=False)
    m.load_state_dict(sd['desc'], strict=True)
    return m.cuda().eval()


def infer_padded(model, img):
    padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    cand = []
    for oy in OFFS:
        for ox in OFFS:
            crop = padded[oy:oy + IN, ox:ox + IN]
            x = torch.from_numpy(crop.astype(np.float32).transpose(2, 0, 1)[None]).cuda()
            with torch.no_grad():
                out = model(x)
            import torch.nn.functional as Fn
            np_out = Fn.softmax(out['np'], dim=1).cpu().numpy()[0].transpose(1, 2, 0)[..., 1:2]
            hv_out = out['hv'].cpu().numpy()[0].transpose(1, 2, 0)
            tp_map = Fn.softmax(out['tp'], dim=1).argmax(dim=1).cpu().numpy()[0].astype(np.float32)
            pred_map = np.concatenate([tp_map[..., None], np_out, hv_out], axis=-1)
            pred_inst, inst_info_dict = hn_process(pred_map, nr_types=6, return_centroids=True)
            if inst_info_dict is None:
                continue
            for inst_id, info in inst_info_dict.items():
                cx, cy = info['centroid']
                contour = np.array(info['contour'], dtype=np.float32)

                edge_cut = bool((contour[:, 0] <= 1).any() or (contour[:, 1] <= 1).any()
                                or (contour[:, 0] >= OUT - 2).any() or (contour[:, 1] >= OUT - 2).any())
                gx, gy = ox + (cx + 46), oy + (cy + 46)
                contour[:, 0] += ox + 46 - PAD
                contour[:, 1] += oy + 46 - PAD
                area = float(cv2.contourArea(contour.astype(np.int32)))
                cand.append({'contour': contour.tolist(),
                             'centroid': [gx - PAD, gy - PAD],
                             'type': int(info['type']) if info['type'] is not None else 0,
                             'type_prob': float(info.get('type_prob') or 0.0) if np.isscalar(info.get('type_prob') or 0.0) else 0.0,
                             'edge_cut': edge_cut, 'area': area})

    merged = []
    used = [False] * len(cand)
    cents = np.array([c['centroid'] for c in cand]) if cand else np.zeros((0, 2))
    for i in range(len(cand)):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        if len(cents):
            d = np.linalg.norm(cents - cents[i], axis=1)
            for j in np.where((d <= 8) & (d > 0))[0]:
                if not used[j]:
                    cluster.append(j)
                    used[j] = True
        pool = [cand[j] for j in cluster]
        intact = [c for c in pool if not c['edge_cut']]
        best = max(intact if intact else pool, key=lambda c: c['area'])
        merged.append(best)
    return merged


def infer_to_json(model, img):
    return infer_padded(model, img)


def full_run():
    model = load_model()
    import pandas as pd
    rows = []
    for fold in [1, 2, 3]:
        images = np.load(f"{PROJ}/PanNuke_dataset/Fold {fold}/images/fold{fold}/images.npy", mmap_mode='r')
        for i in range(images.shape[0]):
            img = np.asarray(images[i]).astype(np.uint8)
            insts = infer_padded(model, img)
            jp = f"{PROJ}/step1_hovernet_results/fold{fold}_image_{i}_segmentation.json"
            old = {}
            if os.path.exists(jp):
                d = json.load(open(jp))
                for t in d.get('tiles', [d]):
                    old.update(t.get('nuc', {}) or {})
            old_c = np.array([v['centroid'] for v in old.values()]) if old else np.zeros((0, 2))
            added = 0
            for v in insts:
                c = np.array(v['centroid'])
                if len(old_c) and cKDTree(old_c).query(c)[0] <= 8:
                    continue
                rows.append({'fold': fold, 'image_idx': i,
                             'centroid_x': c[0], 'centroid_y': c[1],
                             'type': v['type'],
                             'contour': json.dumps(v['contour'])})
                added += 1
            if i % 500 == 0:
                print(f"fold{fold} image {i}/{images.shape[0]} (cumulative new: {len(rows)})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('/path/to/CellNura_results/trained_models/padded_new_nuclei.csv', index=False)
    print(f"done: {len(df)} new nuclei added")


def main(sample_n=300):
    model = load_model()
    images = np.load(f"{PROJ}/PanNuke_dataset/Fold 1/images/fold1/images.npy", mmap_mode='r')
    masks = np.load(f"{PROJ}/PanNuke_dataset/Fold 1/masks/fold1/masks.npy", mmap_mode='r')
    sys.path.insert(0, '/path/to/CellNura/code/e2_segmentation_metrics')
    from seg_metrics import pannuke_gt_centroids

    idxs = np.linspace(0, images.shape[0] - 1, sample_n, dtype=int)
    n_new_inst, n_old_inst, n_recovered_gt, n_unmatched_gt = 0, 0, 0, 0
    for i in idxs:
        img = np.asarray(images[i]).astype(np.uint8)
        new = infer_padded(model, img)

        old = {}
        jp = f"{PROJ}/step1_hovernet_results/fold1_image_{i}_segmentation.json"
        if os.path.exists(jp):
            d = json.load(open(jp))
            for t in d.get('tiles', [d]):
                old.update(t.get('nuc', {}) or {})
        old_c = np.array([v['centroid'] for v in old.values()]) if old else np.zeros((0, 2))
        gt = pannuke_gt_centroids(np.asarray(masks[i]))
        gt_c = np.array([[x, y] for x, y, _ in gt]) if gt else np.zeros((0, 2))


        def covered(gt_c, det_c):
            if len(gt_c) == 0 or len(det_c) == 0:
                return np.zeros(len(gt_c), bool)
            tree = cKDTree(det_c)
            d_, _ = tree.query(gt_c)
            return d_ <= 12

        old_cov = covered(gt_c, old_c)
        n_unmatched_gt += int((~old_cov).sum())
        new_c = np.array([v['centroid'] for v in new]) if new else np.zeros((0, 2))
        new_cov = covered(gt_c, new_c)
        n_recovered_gt += int(((~old_cov) & new_cov).sum())
        n_old_inst += len(old)
        n_new_inst += len(new)
    print(f"sampled {len(idxs)} images: old detections {n_old_inst}, augmented {n_new_inst} (+{100*(n_new_inst-n_old_inst)/max(n_old_inst,1):.1f}%)")
    print(f"unmatched GT: {n_unmatched_gt}, recovered by boundary-complete inference: {n_recovered_gt} (recovery rate {100*n_recovered_gt/max(n_unmatched_gt,1):.1f}%)")


if __name__ == '__main__':
    main()
