"""t08_det_baseline_inference.py — StarDist full detection inference on PanNuke test folds.

StarDist 2D_versatile_he predicts instances and returns centroids. Classification is done later by the shared ResNet-18 classifier.
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(resource('scripts'))
BASE = Path(resource('project_root'))
PKG = Path(resource('data_package'))
OUTROOT = Path(resource('detections'))

def run_stardist(fold, limit=0):
    """Detection inference with StarDist 2D_versatile_he."""
    os.environ.setdefault('KERAS_HOME', str(Path(resource('cache') + '/keras')))
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    local = Path(resource('stardist_weight_dir'))
    model = StarDist2D(None, name=local.name, basedir=str(local.parent))
    imgs = np.load(str(Path(resource('pannuke')) / f'Fold {fold}/images/fold{fold}/images.npy'), mmap_mode='r')
    n = min(limit, imgs.shape[0]) if limit > 0 else imgs.shape[0]
    rows = []
    t0 = time.time()
    for i in range(n):
        img = np.asarray(imgs[i]).astype(np.uint8)
        labels, _ = model.predict_instances(normalize(img, 1, 99.8, axis=(0, 1)))
        for iid in np.unique(labels):
            if iid == 0:
                continue
            y, x = np.nonzero(labels == iid)
            rows.append({'image_key': f'fold{fold}_image_{i}_segmentation', 'nucleus_id': int(iid), 'cx': float(x.mean()), 'cy': float(y.mean())})
        if (i + 1) % 200 == 0:
            print(f'  StarDist fold{fold}: {i + 1}/{n} ({(i + 1) / (time.time() - t0):.1f}/s)', flush=True)
    print(f'  StarDist fold{fold} DONE: {len(rows)} detections ({time.time() - t0:.0f}s)', flush=True)
    return pd.DataFrame(rows, columns=['image_key', 'nucleus_id', 'cx', 'cy'])
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['stardist'])
    ap.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--limit', type=int, default=0, help='0=full fold')
    ap.add_argument('--output-dir', type=Path, required=True)
    a = ap.parse_args()
    if a.limit < 0 or any((f not in (1, 2, 3) for f in a.folds)):
        ap.error('Invalid limit/fold')
    outdir = a.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    import hashlib, json
    cfg_path = os.environ.get('CELLNURA_CONFIG_PATH')
    identity = {'method': a.method, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'config_sha256': hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest() if cfg_path else None}
    identity_path = outdir/'DETECTION_IDENTITY.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Detection output identity changed; configure a fresh detections directory')
    identity_path.write_text(json.dumps(identity, indent=2))
    for fold in a.folds:
        t0 = time.time()
        df = run_stardist(fold, a.limit)
        df.to_csv(outdir / f'det_fold{fold}.csv', index=False)
        total = np.load(Path(resource('pannuke')) / f'Fold {fold}/images/fold{fold}/images.npy', mmap_mode='r').shape[0]
        count = min(a.limit, total) if a.limit else total
        images = pd.DataFrame({'image_key': [f'fold{fold}_image_{i}_segmentation' for i in range(count)]})
        sizes = df.groupby('image_key').size()
        images['n_pred'] = images['image_key'].map(sizes).fillna(0).astype(int)
        images.to_csv(outdir / f'images_fold{fold}.csv', index=False)
        print(f'[{a.method} fold{fold}] saved {len(df)} detections ({time.time() - t0:.0f}s)', flush=True)
