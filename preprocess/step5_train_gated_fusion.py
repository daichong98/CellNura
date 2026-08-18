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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

os.environ.setdefault('OMP_NUM_THREADS', '24')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_data import (load_mobilevit, load_swin, load_match_lut,
                         extract_fold, image_key_of, CV_CONFIGS)

RESULT = "/path/to/CellNura_results"
LR, BATCH, MAX_EPOCH, PATIENCE, SEED = 1e-3, 1024, 100, 10, 42


class CrossAttnBlock(nn.Module):
    def __init__(self, local_dim, ctx_dim, n_tokens, attn_dim):
        super().__init__()
        self.n_tokens, self.attn_dim = n_tokens, attn_dim
        self.W_q = nn.Linear(local_dim, attn_dim)
        self.W_k = nn.Linear(ctx_dim, n_tokens * attn_dim)
        self.W_v = nn.Linear(ctx_dim, n_tokens * attn_dim)

    def forward(self, x_l, g_tokens):

        q = self.W_q(x_l).unsqueeze(1)
        attn = torch.softmax(q @ g_tokens.transpose(-2, -1) / np.sqrt(self.attn_dim), -1)
        return (attn @ g_tokens).squeeze(1)


class FusionBig(nn.Module):
    def __init__(self, ld=96, gd=1024, T=32, A=128, out=1024):
        super().__init__()
        self.T, self.A = T, A
        self.W_q = nn.Linear(ld, A)
        self.W_k = nn.Linear(gd, T * A)
        self.W_v = nn.Linear(gd, T * A)
        self.W_o = nn.Linear(ld + A, out)
        self.norm = nn.LayerNorm(out)

    def forward(self, x_l, x_g):
        N = x_l.shape[0]
        q = self.W_q(x_l).unsqueeze(1)
        K = self.W_k(x_g).view(N, self.T, self.A)
        V = self.W_v(x_g).view(N, self.T, self.A)
        ctx = (torch.softmax(q @ K.transpose(-2, -1) / np.sqrt(self.A), -1) @ V).squeeze(1)
        return self.norm(self.W_o(torch.cat([x_l, ctx], -1)))


class FusionGate(nn.Module):
    def __init__(self, ld=96, gd=1024, out=1024):
        super().__init__()
        self.W_l = nn.Linear(ld, out)
        self.W_g = nn.Linear(gd, out)
        self.gate = nn.Linear(ld + gd, out)
        self.norm = nn.LayerNorm(out)

    def forward(self, x_l, x_g):
        g = torch.sigmoid(self.gate(torch.cat([x_l, x_g], -1)))
        return self.norm(self.W_l(x_l) + g * self.W_g(x_g))


class FusionHybrid(nn.Module):
    def __init__(self, ld=96, gd=1024, T=16, A=64, out=1024):
        super().__init__()
        self.T, self.A = T, A
        self.W1 = nn.Linear(ld, 256)
        self.W2 = nn.Linear(gd, 512)
        self.W_q = nn.Linear(ld, A)
        self.W_k = nn.Linear(gd, T * A)
        self.W_v = nn.Linear(gd, T * A)
        self.W_c = nn.Linear(A, 256)
        self.norm = nn.LayerNorm(out)

    def forward(self, x_l, x_g):
        N = x_l.shape[0]
        q = self.W_q(x_l).unsqueeze(1)
        K = self.W_k(x_g).view(N, self.T, self.A)
        V = self.W_v(x_g).view(N, self.T, self.A)
        ctx = (torch.softmax(q @ K.transpose(-2, -1) / np.sqrt(self.A), -1) @ V).squeeze(1)
        h = torch.cat([self.W1(x_l), self.W_c(ctx), self.W2(x_g)], -1)
        return self.norm(h)


class FusionDeep(nn.Module):
    def __init__(self, ld=96, gd=1024, T=16, A=64, out=1024):
        super().__init__()
        self.T, self.A = T, A
        self.proj_g1 = nn.Linear(gd, T * A)
        self.attn1 = CrossAttnBlock(ld, A, T, A)
        self.attn2 = CrossAttnBlock(ld + A, A, T, A)
        self.ffn = nn.Sequential(nn.Linear(ld + 2 * A, 512), nn.GELU(), nn.Linear(512, 512))
        self.W_o = nn.Linear(ld + 2 * A + 512, out)
        self.norm = nn.LayerNorm(out)

    def forward(self, x_l, x_g):
        N = x_l.shape[0]
        gt = self.proj_g1(x_g).view(N, self.T, self.A)
        c1 = self.attn1(x_l, gt)
        h = torch.cat([x_l, c1], -1)
        c2 = self.attn2(h, gt)
        h2 = torch.cat([h, c2], -1)
        return self.norm(self.W_o(torch.cat([h2, self.ffn(h2)], -1)))


VARIANTS = {'big': FusionBig, 'gate': FusionGate, 'hybrid': FusionHybrid, 'deep': FusionDeep}


class FusionClassifier(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.fusion = VARIANTS[variant]()
        self.head = nn.Linear(1024, 5)

    def forward(self, x_l, x_g):
        return self.head(self.fusion(x_l, x_g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=list(VARIANTS.keys()))
    ap.add_argument('--fold', type=int, required=True, choices=[1, 2, 3])
    args = ap.parse_args()
    cv = [c for c in CV_CONFIGS if c['test_fold'] == args.fold][0]
    train_folds = cv['train_folds']
    device = torch.device('cuda')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(24)
    tag = f"{args.variant}_cv{args.fold}"

    mv, mv_cols = load_mobilevit()
    sw, sw_cols = load_swin()
    df = mv.merge(sw, on='original_image', how='inner')
    df['fold'] = df['original_image'].apply(extract_fold)
    lut = load_match_lut()
    keys = list(zip(df['image_name'].apply(image_key_of), df['nucleus_id'].astype(int)))
    labels = np.array([lut.get(k, -1) for k in keys], dtype=int)

    Xl = df[mv_cols].values.astype(np.float32)
    Xg = df[sw_cols].values.astype(np.float32)
    folds = df['fold'].values
    tr = (labels >= 0) & np.isin(folds, train_folds)
    Xtr_l, Xtr_g, ytr = Xl[tr], Xg[tr], labels[tr]
    sc_l = StandardScaler().fit(Xtr_l)
    sc_g = StandardScaler().fit(Xtr_g)
    Xtr_l, Xtr_g = sc_l.transform(Xtr_l), sc_g.transform(Xtr_g)
    idx_tr, idx_va = train_test_split(np.arange(len(ytr)), test_size=0.2,
                                      random_state=SEED, stratify=ytr)
    V = lambda a, i: torch.FloatTensor(a[i]).to(device)
    Y = lambda a, i: torch.LongTensor(a[i]).to(device)

    model = FusionClassifier(args.variant).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    best_acc, best_state, patience = 0.0, None, 0
    ntr = len(idx_tr)
    for epoch in range(MAX_EPOCH):
        model.train()
        perm = np.random.permutation(ntr)
        for i in range(0, ntr, BATCH):
            bi = idx_tr[perm[i:i + BATCH]]
            opt.zero_grad()
            loss = crit(model(V(Xtr_l, bi), V(Xtr_g, bi)), Y(ytr, bi))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = (model(V(Xtr_l, idx_va), V(Xtr_g, idx_va)).argmax(1) == Y(ytr, idx_va)).float().mean().item()
        if va > best_acc:
            best_acc, best_state, patience = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    print(f"[{tag}] best val acc={best_acc:.4f}, epochs={epoch+1}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(df), 8192):
            xl = torch.FloatTensor(sc_l.transform(Xl[i:i + 8192])).to(device)
            xg = torch.FloatTensor(sc_g.transform(Xg[i:i + 8192])).to(device)
            outs.append(model.fusion(xl, xg).cpu().numpy().astype(np.float32))
    F = np.concatenate(outs)
    out = df[['image_name', 'original_image', 'nucleus_id', 'fold']].reset_index(drop=True)
    out = pd.concat([out, pd.DataFrame(F, columns=[f'coattention_feature_{i}' for i in range(1024)])], axis=1)
    path = f"{RESULT}/trained_features/fvisual_{tag}.csv"
    out.to_csv(path, index=False, float_format='%.6g')
    torch.save({'model': best_state, 'scaler_l': sc_l, 'scaler_g': sc_g,
                'best_val_acc': best_acc, 'epochs': epoch + 1, 'variant': args.variant},
               f"{RESULT}/trained_models/coattention_{tag}.pth")
    print(f"[{tag}] saved {path} {out.shape}", flush=True)


if __name__ == '__main__':
    main()
