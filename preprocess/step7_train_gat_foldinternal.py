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
import torch.nn.functional as Fn
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

os.environ.setdefault('OMP_NUM_THREADS', '32')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_data import (load_match_lut, load_centroids, extract_fold,
                         image_key_of, CV_CONFIGS)

RESULT = "/path/to/CellNura_results"
TAU = 40.0
IN_PROJ, HID, OUT_DIM, HEADS, DROPOUT = 100, 128, 256, 8, 0.4
LR, MAX_EPOCH, PATIENCE, SEED = 5e-3, 60, 8, 42
BATCH_GRAPHS = 64


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_f, out_f, dropout=DROPOUT, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_f, out_f, bias=False)
        self.a = nn.Linear(2 * out_f, 1, bias=False)
        self.leaky = nn.LeakyReLU(alpha)
        self.dropout = dropout

    def forward(self, h, adj):
        Wh = self.W(h)
        B, N, F = Wh.shape
        Wh_i = Wh.unsqueeze(2).expand(B, N, N, F)
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, F)
        e = self.leaky(self.a(torch.cat([Wh_i, Wh_j], -1)).squeeze(-1))
        e = e.masked_fill(adj == 0, -1e9)
        attn = Fn.dropout(torch.softmax(e, -1), self.dropout, self.training)
        return Fn.elu(attn @ Wh)


class CellularGAT(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()
        self.proj = nn.Linear(in_dim, IN_PROJ)
        self.heads = nn.ModuleList([GraphAttentionLayer(IN_PROJ, HID) for _ in range(HEADS)])
        self.out = GraphAttentionLayer(HID * HEADS, OUT_DIM)
        self.classifier = nn.Linear(OUT_DIM, 5)

    def embed(self, x, adj):
        h = Fn.elu(self.proj(x))
        h = torch.cat([head(h, adj) for head in self.heads], -1)
        h = Fn.dropout(h, DROPOUT, self.training)
        return self.out(h, adj)

    def forward(self, x, adj):
        return self.classifier(self.embed(x, adj))


def build_graphs(df, feat_cols, centroids, lut):
    graphs = []
    for (fold, img), g in df.groupby(['fold', 'img_idx']):
        cents = centroids.get((fold, img), {})
        ids = g['nucleus_id'].astype(int).tolist()
        coords = np.array([cents.get(i, (0.0, 0.0)) for i in ids])
        adj = np.ones((len(ids), len(ids)), dtype=np.float32)
        if len(ids) > 1:
            dist = cdist(coords, coords)
            adj = (dist < TAU).astype(np.float32)
            np.fill_diagonal(adj, 1)
        keys = zip(g['image_key'], g['nucleus_id'].astype(int))
        y = np.array([lut.get(k, -1) for k in keys], dtype=np.int64)
        graphs.append({'fold': fold, 'img': img,
                       'x': g[feat_cols].values.astype(np.float32),
                       'adj': adj, 'y': y, 'ids': ids})
    return graphs


def make_batches(graphs, max_nodes=400, max_graphs=BATCH_GRAPHS):
    batches, cur, cur_n = [], [], 0
    for g in graphs:
        n = len(g['ids'])
        if cur and (cur_n + n > max_nodes or len(cur) >= max_graphs):
            batches.append(cur)
            cur, cur_n = [], 0
        cur.append(g)
        cur_n += n
    if cur:
        batches.append(cur)
    return batches


def collate(graphs):
    xs, adjs, ys = [], [], []
    offs = 0
    blocks = []
    for g in graphs:
        n = len(g['ids'])
        xs.append(g['x'])
        ys.append(g['y'])
        blocks.append((offs, offs + n, g['adj']))
        offs += n
    x = np.concatenate(xs, 0)
    y = np.concatenate(ys, 0)
    adj = np.zeros((offs, offs), dtype=np.float32)
    for a, b, blk in blocks:
        adj[a:b, a:b] = blk
    return (torch.FloatTensor(x).unsqueeze(0).cuda(),
            torch.FloatTensor(adj).unsqueeze(0).cuda(),
            torch.LongTensor(y).cuda())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True, choices=[1, 2, 3])
    ap.add_argument('--variant', default='v1',
                    help='F_visual variant: v1=fvisual_cv{f}.csv; otherwise fvisual_{variant}_cv{f}.csv')
    args = ap.parse_args()
    VSUF = '' if args.variant == 'v1' else f"_{args.variant}"
    FTAG = f"fvisual{VSUF}_cv{args.fold}"
    GTAG = f"fgraph{VSUF}_cv{args.fold}"
    MTAG = f"gat{VSUF}_cv{args.fold}"
    cv = [c for c in CV_CONFIGS if c['test_fold'] == args.fold][0]
    train_folds = cv['train_folds']
    device = torch.device('cuda')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(32)

    print(f"[fold{args.fold}] loading F_visual and coordinates...", flush=True)
    fv = pd.read_csv(f"{RESULT}/trained_features/{FTAG}.csv")
    feat_cols = [c for c in fv.columns if c.startswith('coattention_feature_')]
    fv['img_idx'] = fv['original_image'].str.extract(r'image_(\d+)').astype(int)
    fv['image_key'] = fv['image_name'].apply(image_key_of)
    centroids = load_centroids()
    lut = load_match_lut()

    print(f"[fold{args.fold}] building graphs...", flush=True)
    graphs = build_graphs(fv, feat_cols, centroids, lut)
    tr_graphs = [g for g in graphs if g['fold'] in train_folds]
    idx = np.arange(len(tr_graphs))
    gi_tr, gi_va = train_test_split(idx, test_size=0.2, random_state=SEED)
    print(f"[fold{args.fold}] train images {len(gi_tr)}, val images {len(gi_va)}", flush=True)

    model = CellularGAT().to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_acc, best_state, patience, log = 0.0, None, 0, []
    for epoch in range(MAX_EPOCH):
        model.train()
        order = np.random.permutation(gi_tr)
        tot, cnt = 0.0, 0
        for batch in make_batches([tr_graphs[i] for i in order]):
            x, adj, y = collate(batch)
            mask = y >= 0
            if mask.sum() == 0:
                continue
            opt.zero_grad()
            logits = model(x, adj).squeeze(0)
            loss = crit(logits[mask], y[mask])
            loss.backward()
            opt.step()
            tot += loss.item()
            cnt += 1
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in make_batches([tr_graphs[i] for i in gi_va]):
                x, adj, y = collate(batch)
                mask = y >= 0
                pred = model(x, adj).squeeze(0).argmax(1)
                correct += (pred[mask] == y[mask]).sum().item()
                total += int(mask.sum())
        va = correct / max(total, 1)
        log.append({'epoch': epoch + 1, 'train_loss': tot / max(cnt, 1), 'val_acc': va})
        print(f"[fold{args.fold}] epoch {epoch+1}: loss={tot/max(cnt,1):.4f} val_acc={va:.4f}", flush=True)
        if va > best_acc:
            best_acc, best_state, patience = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    print(f"[fold{args.fold}] training done: best val acc={best_acc:.4f}, epochs={epoch+1}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in make_batches(graphs):
            xs, adjs, metas = [], [], []
            offs = 0
            adj_tot = None
            blocks = []
            for g in batch:
                n = len(g['ids'])
                xs.append(g['x'])
                blocks.append((offs, offs + n, g['adj'], g))
                offs += n
            x = np.concatenate(xs, 0)
            adj = np.zeros((offs, offs), dtype=np.float32)
            for a, b, blk, _ in blocks:
                adj[a:b, a:b] = blk
            xt = torch.FloatTensor(x).unsqueeze(0).to(device)
            adjt = torch.FloatTensor(adj).unsqueeze(0).to(device)
            emb = model.embed(xt, adjt).squeeze(0).cpu().numpy().astype(np.float32)
            for a, b, _, g in blocks:
                img_name = f"fold{g['fold']}_image_{g['img']}"
                d = pd.DataFrame({'image_name': [f"{img_name}_nucleus_{i}.png" for i in g['ids']],
                                  'nucleus_id': g['ids']})
                d = pd.concat([d.reset_index(drop=True),
                               pd.DataFrame(emb[a:b], columns=[f'graph_feature_{i}' for i in range(OUT_DIM)])], axis=1)
                rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    path = f"{RESULT}/trained_features/{GTAG}.csv"
    out.to_csv(path, index=False, float_format='%.6g')
    torch.save({'model': best_state, 'best_val_acc': best_acc, 'epochs': epoch + 1,
                'tau_px': TAU}, f"{RESULT}/trained_models/{MTAG}.pth")
    pd.DataFrame(log).to_csv(f"{RESULT}/trained_models/{MTAG}_log.csv", index=False)
    print(f"[fold{args.fold}] features saved {path} {out.shape}", flush=True)


if __name__ == '__main__':
    main()
