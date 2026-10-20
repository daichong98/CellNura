"""t09_traditional_classifiers.py — Auxiliary traditional-classifier controls (Tier B controlled comparisons with shared segmentation).

Source identity: 8 sklearn models from 09_Traditional_Classifiers/source/baseline_models.py
(SVM_RBF/SVM_Linear/RandomForest/GradientBoosting/LogisticRegression/NaiveBayes/KNN/MLP_Sklearn,
with hyperparameters preserved); the README clarifies that these are not equivalent to the main-table baselines and that distance_transform_classifier is not DIST.
Protocol for this round (same comparable nucleus set as the T09 ablations):
  * Features: 17 morphological + 36 chromatin-ring features from v3.2 features.pkl (53-dimensional shallow set, same features as the T09 shallow variant)
  * Split: image-level train/val (outer-fold rows in split_manifest); test fold used only for evaluation
  * Frozen hyperparameters (original configuration), no tuning; no class weights (unchanged); SVC probability disabled (only labels are needed)
  * Evaluation: one-to-one unified_matcher at τ=12px, all test-fold images (including images with zero detections)
Purpose: comparison with T09 shallow (CellNura MLP 0.2825) isolates "classifier family" differences; comparison with full (0.4913)
isolates "representation" differences. These are controlled comparisons with shared segmentation (Tier B), excluded from Tier A rankings.
Usage: python t09_traditional_classifiers.py --model SVM_RBF --test-fold 1
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import argparse, hashlib, json, os, sys, time
import numpy as np
import pandas as pd
BASE = resource('project_root')
V32 = f"{resource('traditional_features')}"
MANIFEST = f"{resource('split_manifest')}"
PKG = resource('data_package')
OUTROOT = f"{resource('traditional_outputs')}"
sys.path.insert(0, f"{resource('scripts')}")
from unified_matcher import EvalInput, Instance, evaluate, write_outputs
from baseline_feature_columns import MORPH_COLS
SEED = 42

def log(s):
    print(s, flush=True)

def build_models():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier as SklearnMLP
    return {'SVM_RBF': SVC(kernel='rbf', random_state=SEED, cache_size=1000), 'SVM_Linear': SVC(kernel='linear', random_state=SEED, cache_size=1000), 'RandomForest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=16), 'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=SEED), 'LogisticRegression': LogisticRegression(random_state=SEED, max_iter=1000), 'NaiveBayes': GaussianNB(), 'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=16), 'MLP_Sklearn': SklearnMLP(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=SEED)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(build_models()))
    ap.add_argument('--test-fold', type=int, required=True)
    a = ap.parse_args()
    np.random.seed(SEED)
    t0 = time.time()
    outdir = f'{OUTROOT}/{a.model}/fold{a.test_fold}'
    os.makedirs(outdir, exist_ok=True)
    master = pd.read_pickle(f'{V32}/outer_fold{a.test_fold}/features.pkl')
    man = pd.read_csv(MANIFEST)
    g = man[(man.dataset == 'PanNuke') & (man.outer_fold == f'fold{a.test_fold}')]
    role = dict(zip(g.image_id, g.split))
    key = 'img_base' if 'img_base' in master.columns else 'original_image'
    master['img_id'] = master[key]
    master['role'] = master.img_id.map(role)
    lab = master.label.values
    matched = lab >= 0
    tr_mask = matched & (master.role == 'train').values
    va_mask = matched & (master.role == 'val').values
    mo_cols = [c for c in MORPH_COLS if c in master.columns]
    fr_cols = [c for c in master.columns if c.startswith('FR')]
    feat_cols = mo_cols + fr_cols
    X = master[feat_cols].values.astype(np.float32)
    y = lab
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X[tr_mask])
    Xtr, ytr = (scaler.transform(X[tr_mask]), y[tr_mask])
    Xva, yva = (scaler.transform(X[va_mask]), y[va_mask])
    model = build_models()[a.model]
    log(f'[{a.model}/fold{a.test_fold}] train={len(Xtr)} val={len(Xva)} feat={len(feat_cols)}')
    model.fit(Xtr, ytr)
    va_acc = (model.predict(Xva) == yva).mean() if len(Xva) else float('nan')
    log(f'[{a.model}/fold{a.test_fold}] val_acc={va_acc:.4f} ({time.time() - t0:.0f}s)')
    te_mask = master.role == 'test'
    test_rows = master[te_mask].copy()
    test_rows['pred'] = model.predict(scaler.transform(X[te_mask]))
    log(f'[{a.model}/fold{a.test_fold}] test rows={len(test_rows)}')
    masks = np.load(f"{resource('pannuke')}/Fold {a.test_fold}/masks/fold{a.test_fold}/masks.npy", mmap_mode='r')
    gt_list, pr_list = ([], [])
    for i in range(masks.shape[0]):
        m = np.asarray(masks[i])
        img_id = f'fold{a.test_fold}_image_{i}'
        for c in range(5):
            ch = m[..., c]
            for iid in np.unique(ch):
                if iid == 0:
                    continue
                ys, xs = np.nonzero(ch == iid)
                gt_list.append(Instance(img_id, f'g{c}_{int(iid)}', float(xs.mean()), float(ys.mean()), c))
    for k, r in enumerate(test_rows.itertuples(), 1):
        pr_list.append(Instance(r.img_id, f'd{k}', float(r.cx), float(r.cy), int(r.pred)))
    ev = EvalInput(dataset='PanNuke', tau=12.0, gt=gt_list, pred=pr_list, image_ids=tuple((f'fold{a.test_fold}_image_{i}' for i in range(masks.shape[0]))), class_ids=(0, 1, 2, 3, 4), real_class_ids=real_classes('PanNuke'))
    res = evaluate(ev, mode='one_to_one')
    write_outputs(res, f'{outdir}/eval')
    row = {'model': a.model, 'test_fold': a.test_fold, 'n_features': len(feat_cols), 'val_acc': round(float(va_acc), 4), 'cls_macroF1': round(res.macro('classification'), 4), 'plusFN_macroF1': round(res.macro('plus_missed_FN'), 4), 'uni_macroF1': round(res.macro('unified'), 4), 'seed': SEED, 'tier': 'B (same-segmentation controlled comparison)', 'code_sha256': hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()}
    json.dump(row, open(f'{outdir}/result.json', 'w'), indent=2, ensure_ascii=False)
    log(f"[{a.model}/fold{a.test_fold}] uni={row['uni_macroF1']:.4f} ({time.time() - t0:.0f}s)")
if __name__ == '__main__':
    main()
