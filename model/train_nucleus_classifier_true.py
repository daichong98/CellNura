#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import joblib

os.environ.setdefault('OMP_NUM_THREADS', '24')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_data import (load_match_lut, extract_fold, image_key_of,
                         CLASS_NAMES, CV_CONFIGS)
from run_cv_trained import load_static, prf, train_eval, predict_all, MORPH_COLS

PROJECT = "/path/to/PanNuke_classification_final"
RESULT = "/path/to/CellNura_results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True, choices=[1, 2, 3])
    args = ap.parse_args()
    fold = args.fold
    cv = [c for c in CV_CONFIGS if c['test_fold'] == fold][0]
    device = torch.device('cuda')
    torch.set_num_threads(24)

    fv = pd.read_csv(f"{RESULT}/trained_features/fvisual_gate_cv{fold}.csv")
    fv = fv.drop_duplicates(subset=['image_name', 'nucleus_id'])
    fv = fv.drop(columns=[c for c in ['fold'] if c in fv.columns])
    fg = pd.read_csv(f"{RESULT}/trained_features/fgraph_gate_cv{fold}.csv")
    fg = fg.drop_duplicates(subset=['image_name', 'nucleus_id'])
    master = fv.merge(fg, on=['image_name', 'nucleus_id'], how='inner')
    morph = load_static(f"{PROJECT}/step6_morphological/pannuke_morphological_features.csv",
                        ['image_name', 'nucleus_id'])
    master = master.merge(morph, on=['image_name', 'nucleus_id'], how='inner')
    ring = load_static(f"{PROJECT}/step7_ring/pannuke_ring_features.csv",
                       ['image_name', 'nucleus_id'])
    master = master.merge(ring, on=['image_name', 'nucleus_id'], how='inner')
    master['fold'] = master['original_image'].apply(extract_fold)
    master['image_key'] = master['image_name'].apply(image_key_of)

    lut = load_match_lut()
    keys = list(zip(master['image_key'], master['nucleus_id'].astype(int)))
    labels = np.array([lut.get(k, -1) for k in keys], dtype=int)
    folds_arr = master['fold'].values
    feat_cols = ([c for c in master.columns if c.startswith('coattention_feature_')]
                 + [c for c in MORPH_COLS if c in master.columns]
                 + [c for c in master.columns if c.startswith('FR')]
                 + [c for c in master.columns if c.startswith('graph_feature_')])
    print(f"[final_cv{fold}] master {master.shape}, feat_dim {len(feat_cols)}", flush=True)
    X = master[feat_cols].values.astype(np.float32)
    tr = (labels >= 0) & np.isin(folds_arr, cv['train_folds'])
    te = (labels >= 0) & (folds_arr == fold)
    allm = folds_arr == fold

    model, scaler, n_ep = train_eval(X[tr], labels[tr], X[te], labels[te], 'mlp', device)
    pred = predict_all(model, scaler, X[te], device)
    yte = labels[te]
    acc = float((pred == yte).mean())
    rows, f1s = [], []
    for ci, cname in enumerate(CLASS_NAMES):
        tp = int(((pred == ci) & (yte == ci)).sum())
        fp = int(((pred == ci) & (yte != ci)).sum())
        fn = int(((pred != ci) & (yte == ci)).sum())
        p, r, f1 = prf(tp, fp, fn)
        f1s.append(f1)
        rows.append({'config': 'final_gate', 'cv_fold': cv['name'], 'test_fold': fold,
                     'class': cname, 'P': p, 'R': r, 'F1': f1,
                     'support': int((yte == ci).sum()), 'accuracy': acc,
                     'macro_F1': float(np.mean(f1s)) if ci == 4 else np.nan,
                     'n_train': int(tr.sum()), 'n_eval': int(te.sum()), 'epochs': n_ep})
    pd.DataFrame(rows).to_csv(f"{RESULT}/e5final_cv{fold}_fold_metrics.csv", index=False)

    torch.save(model.state_dict(), f"{RESULT}/trained_models/mlp_final_cv{fold}.pth")
    joblib.dump(scaler, f"{RESULT}/trained_models/mlp_final_cv{fold}_scaler.pkl")
    pred_all = predict_all(model, scaler, X[allm], device)
    out = master.loc[allm, ['image_key', 'nucleus_id', 'fold', 'original_image']].copy()
    out['label'] = labels[allm]
    out['pred'] = pred_all
    out.to_csv(f"{RESULT}/trained_models/pred_final_cv{fold}.csv", index=False)
    print(f"[final_cv{fold}] acc={acc:.4f} macroF1={np.mean(f1s):.4f} epochs={n_ep}", flush=True)


if __name__ == '__main__':
    main()
