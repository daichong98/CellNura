#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import glob
import json
import numpy as np
import pandas as pd

PROJECT = "/path/to/PanNuke_classification_final"
CLASS_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
CV_CONFIGS = [
    {'train_folds': [2, 3], 'test_fold': 1, 'name': 'CV_Fold_1'},
    {'train_folds': [1, 3], 'test_fold': 2, 'name': 'CV_Fold_2'},
    {'train_folds': [1, 2], 'test_fold': 3, 'name': 'CV_Fold_3'},
]


def extract_fold(name):
    m = re.search(r'fold(\d+)', str(name))
    return int(m.group(1)) if m else -1


def norm_img_name(s):
    return re.sub(r'image_0*(\d+)', r'image_\1', str(s))


def image_key_of(nucleus_image_name):
    return re.sub(r'_nucleus_\d+\.png', '_segmentation', str(nucleus_image_name))


def load_mobilevit():
    df = pd.read_csv(f"{PROJECT}/step3_mobilevit/pannuke_mobilevit_features.csv")
    df = df.drop_duplicates(subset=['image_name', 'nucleus_id'])
    feat_cols = [c for c in df.columns if c.startswith('mobilevit_feature_')]
    assert len(feat_cols) == 96
    return df, feat_cols


def load_swin():
    df = pd.read_csv(f"{PROJECT}/step4_swin_global/pannuke_swin_global_features.csv")
    df['original_image'] = df['image_name'].apply(norm_img_name)
    df = df.drop_duplicates(subset=['original_image'])
    feat_cols = [c for c in df.columns if c.startswith('swin_global_')]
    assert len(feat_cols) == 1024
    return df[['original_image'] + feat_cols], feat_cols


def load_match_lut():
    files = glob.glob(f"{PROJECT}/step8_centroid_revised1/*_matches.csv")
    parts = []
    for f in files:
        d = pd.read_csv(f)
        d['image_key'] = os.path.basename(f).replace('_matches.csv', '')
        parts.append(d)
    m = pd.concat(parts, ignore_index=True)
    return {(r.image_key, int(r.hovernet_id)): int(r.new_type) - 1
            for r in m.itertuples()}


def load_centroids():
    out = {}
    for jp in glob.glob(f"{PROJECT}/step1_hovernet_results/fold*_image_*_segmentation.json"):
        base = os.path.basename(jp)
        m = re.match(r'fold(\d+)_image_(\d+)_segmentation\.json', base)
        fold, idx = int(m.group(1)), int(m.group(2))
        with open(jp) as f:
            d = json.load(f)
        if 'tiles' in d:
            nuc = {}
            for t in d['tiles']:
                nuc.update(t.get('nuc', {}) or {})
        else:
            nuc = d.get('nuc', {}) or {}
        out[(fold, idx)] = {int(k): tuple(v['centroid']) for k, v in nuc.items()}
    return out


def load_tissue_types():
    types = {}
    names = None
    for fold in [1, 2, 3]:
        tp = f"{PROJECT}/PanNuke_dataset/Fold {fold}/images/fold{fold}/types.npy"
        if os.path.exists(tp):
            arr = np.load(tp, allow_pickle=True)
            for i, t in enumerate(arr):
                types[(fold, i)] = str(t)
    return types
