#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

os.environ.setdefault('OMP_NUM_THREADS', '32')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_data import (load_mobilevit, load_swin, load_match_lut,
                         extract_fold, image_key_of, CLASS_NAMES, CV_CONFIGS)

PROJECT = "/path/to/PanNuke_classification_final"
RESULT = "/path/to/CellNura_results"
LR, BATCH, MAX_EPOCH, PATIENCE, VAL_SPLIT, SEED = 1e-5, 64, 500, 30, 0.2, 42

CONFIGS = {
    'full_t':           {'feats': ['fv', 'morph', 'ring', 'fg'], 'head': 'mlp'},
    'concat_no_attn_t': {'feats': ['mv', 'swin', 'morph', 'ring', 'fg'], 'head': 'mlp'},
    'local_only_t':     {'feats': ['mv', 'morph', 'ring', 'fg'], 'head': 'mlp'},
    'global_only_t':    {'feats': ['swin', 'morph', 'ring', 'fg'], 'head': 'mlp'},
    'pca64_visual_t':   {'feats': ['fv', 'morph', 'ring', 'fg'], 'head': 'mlp', 'pca': 64},
    'linear_head_t':    {'feats': ['fv', 'morph', 'ring', 'fg'], 'head': 'linear'},
    'shallow_mlp_t':    {'feats': ['fv', 'morph', 'ring', 'fg'], 'head': 'shallow'},
}
AUX = {'image_name', 'nucleus_id', 'fold', 'original_image', 'image_key', 'img_idx'}


def load_static(path, key_cols):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=key_cols)
    drop = [c for c in df.columns if c.startswith(('original_image', 'image_name'))
            and c not in key_cols] + [c for c in ['fold', 'tile_x', 'tile_y', 'mag'] if c in df.columns]
    return df.drop(columns=drop)


def build_master(fold):
    fv = pd.read_csv(f"{RESULT}/trained_features/fvisual_cv{fold}.csv")
    fv = fv.drop_duplicates(subset=['image_name', 'nucleus_id'])
    fv = fv.drop(columns=[c for c in ['fold'] if c in fv.columns])
    master = fv
    fg = pd.read_csv(f"{RESULT}/trained_features/fgraph_cv{fold}.csv")
    fg = fg.drop_duplicates(subset=['image_name', 'nucleus_id'])
    master = master.merge(fg, on=['image_name', 'nucleus_id'], how='inner')
    morph = load_static(f"{PROJECT}/step6_morphological/pannuke_morphological_features.csv",
                        ['image_name', 'nucleus_id'])
    master = master.merge(morph, on=['image_name', 'nucleus_id'], how='inner')
    ring = load_static(f"{PROJECT}/step7_ring/pannuke_ring_features.csv",
                       ['image_name', 'nucleus_id'])
    master = master.merge(ring, on=['image_name', 'nucleus_id'], how='inner')
    mv, _ = load_mobilevit()
    mv = mv.drop(columns=[c for c in ['fold'] if c in mv.columns])
    mv = mv.drop(columns=[c for c in mv.columns if c.startswith('original_image')])
    master = master.merge(mv, on=['image_name', 'nucleus_id'], how='inner')
    sw, _ = load_swin()
    master = master.merge(sw, on='original_image', how='inner')
    master['fold'] = master['original_image'].apply(extract_fold)
    master['image_key'] = master['image_name'].apply(image_key_of)
    return master


def cols_of(master, group):
    if group == 'fv':
        return [c for c in master.columns if c.startswith('coattention_feature_')]
    if group == 'fg':
        return [c for c in master.columns if c.startswith('graph_feature_')]
    if group == 'mv':
        return [c for c in master.columns if c.startswith('mobilevit_feature_')]
    if group == 'swin':
        return [c for c in master.columns if c.startswith('swin_global_')]
    if group == 'morph':
        return [c for c in master.columns if c in MORPH_COLS]
    if group == 'ring':
        return [c for c in master.columns if c.startswith('FR')]
    return []


def _detect_morph_cols():
    hdr = pd.read_csv(f"{PROJECT}/step6_morphological/pannuke_morphological_features.csv", nrows=1)
    aux = {'image_name', 'original_image', 'nucleus_id', 'fold', 'tile_x', 'tile_y', 'mag'}
    return [c for c in hdr.columns if c not in aux]


MORPH_COLS = _detect_morph_cols()


def make_head(head, input_dim):
    if head == 'linear':
        return nn.Sequential(nn.Linear(input_dim, 5))
    if head == 'shallow':
        return nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256),
                             nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 5))
    layers, prev = [], input_dim
    for h in [512, 256, 128, 64]:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
        prev = h
    layers.append(nn.Linear(prev, 5))
    return nn.Sequential(*layers)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def train_eval(Xtr, ytr, Xte, yte, head, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Xi_tr, Xi_va, y_tr, y_va = train_test_split(
        Xtr, ytr, test_size=VAL_SPLIT, random_state=SEED, stratify=ytr)
    scaler = StandardScaler().fit(Xi_tr)
    t = lambda a: torch.FloatTensor(scaler.transform(a))
    Xtr_t, Xva_t, Xte_t = t(Xi_tr), t(Xi_va), t(Xte)
    ytr_t = torch.LongTensor(y_tr)
    model = make_head(head, Xtr.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    n = len(Xtr_t)
    best_acc, best_state, patience = 0.0, None, 0
    for epoch in range(MAX_EPOCH):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            bi = perm[i:i + BATCH]
            xb = Xtr_t[bi].to(device)
            opt.zero_grad()
            loss = crit(model(xb), ytr_t[bi].to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = (model(Xva_t.to(device)).argmax(1).cpu() == torch.LongTensor(y_va)).float().mean().item()
        if va > best_acc:
            best_acc, best_state, patience = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, epoch + 1


@torch.no_grad()
def predict_all(model, scaler, X, device):
    model.eval()
    preds = []
    for i in range(0, len(X), 8192):
        xb = torch.FloatTensor(scaler.transform(X[i:i + 8192])).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True, choices=[1, 2, 3])
    args = ap.parse_args()
    fold = args.fold
    cv = [c for c in CV_CONFIGS if c['test_fold'] == fold][0]
    train_folds = cv['train_folds']
    device = torch.device('cuda')
    torch.set_num_threads(32)

    print(f"[cv{fold}] building master table...", flush=True)
    master = build_master(fold)
    lut = load_match_lut()
    keys = list(zip(master['image_key'], master['nucleus_id'].astype(int)))
    labels = np.array([lut.get(k, -1) for k in keys], dtype=int)
    folds_arr = master['fold'].values
    print(f"[cv{fold}] master {master.shape}, labeled {int((labels>=0).sum())}", flush=True)

    tr_mask = (labels >= 0) & np.isin(folds_arr, train_folds)
    te_mask = (labels >= 0) & (folds_arr == fold)
    all_mask = folds_arr == fold

    for cfg_name, cfg in CONFIGS.items():
        feat_cols = []
        for grp in cfg['feats']:
            feat_cols += cols_of(master, grp)
        X = master[feat_cols].values.astype(np.float32)
        vis_idx = None
        if cfg.get('pca'):
            vis_idx = list(range(len(cols_of(master, 'fv'))))
        Xtr, ytr = X[tr_mask], labels[tr_mask]
        Xte, yte = X[te_mask], labels[te_mask]
        if vis_idx is not None:
            pca = PCA(n_components=cfg['pca'], random_state=SEED).fit(Xtr[:, vis_idx])
            keep = list(range(len(vis_idx), X.shape[1]))
            Xtr = np.concatenate([pca.transform(Xtr[:, vis_idx]), Xtr[:, keep]], 1)
            Xte = np.concatenate([pca.transform(Xte[:, vis_idx]), Xte[:, keep]], 1)
            Xall = np.concatenate([pca.transform(X[all_mask][:, vis_idx]), X[all_mask][:, keep]], 1)
        else:
            Xall = X[all_mask]

        model, scaler, n_ep = train_eval(Xtr, ytr, Xte, yte, cfg['head'], device)
        pred = predict_all(model, scaler, Xte, device)
        acc = float((pred == yte).mean())
        f1s = []
        rows = []
        for ci, cname in enumerate(CLASS_NAMES):
            tp = int(((pred == ci) & (yte == ci)).sum())
            fp = int(((pred == ci) & (yte != ci)).sum())
            fn = int(((pred != ci) & (yte == ci)).sum())
            p, r, f1 = prf(tp, fp, fn)
            f1s.append(f1)
            rows.append({'config': cfg_name, 'cv_fold': cv['name'], 'test_fold': fold,
                         'class': cname, 'P': p, 'R': r, 'F1': f1,
                         'support': int((yte == ci).sum()), 'accuracy': acc,
                         'macro_F1': float(np.mean(f1s)) if ci == 4 else np.nan,
                         'n_train': int(tr_mask.sum()), 'n_eval': int(te_mask.sum()),
                         'epochs': n_ep})
        pd.DataFrame(rows).to_csv(f"{RESULT}/e5_{cfg_name}_cv{fold}_fold_metrics.csv", index=False)
        print(f"[cv{fold}] {cfg_name}: acc={acc:.4f}, macroF1={np.mean(f1s):.4f}, epochs={n_ep}", flush=True)

        if cfg_name == 'full_t':
            os.makedirs(f"{RESULT}/trained_models", exist_ok=True)
            torch.save(model.state_dict(), f"{RESULT}/trained_models/mlp_full_cv{fold}.pth")
            joblib.dump(scaler, f"{RESULT}/trained_models/mlp_full_cv{fold}_scaler.pkl")

            pred_all = predict_all(model, scaler, Xall, device)
            out = master.loc[all_mask, ['image_key', 'nucleus_id', 'fold', 'original_image']].copy()
            out['label'] = labels[all_mask]
            out['pred'] = pred_all
            out.to_csv(f"{RESULT}/trained_models/pred_full_cv{fold}.csv", index=False)
            print(f"[cv{fold}] full_t model and predictions saved", flush=True)
        del X, Xtr, Xte, Xall
        torch.cuda.empty_cache()

    print(f"[cv{fold}] all configurations done", flush=True)


if __name__ == '__main__':
    main()
