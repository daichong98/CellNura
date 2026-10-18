"""t06_v32_infer_per_outer_fold.py — v3.2: consistent inference on all 7,901 images for each outer fold.

Correct cross-fold feature contamination (audit issue 3): all input images for outer fold f (all train/val/test images)
are processed using the testfold{f} segmenter, which has never seen the test images in fold f.

For each outer fold f ∈ {1,2,3}:
  - Load hovernet_v3_fast_pannuke_testfold{f}/best_model.pth
  - Run inference on all images from all three source folds (7,901 images)
  - Write outputs to cellnura_v32/outer_fold{f}/step1/source_fold{j}/

GPU memory usage ~1.5GB (fp16, batch=8); can run concurrently with TSFD (33GB).
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import argparse, glob, json, os, re, sys, time
import cv2
import numpy as np
import torch
BASE = resource('project_root')
PKG = resource('data_package')
RUNS = f"{resource('runs')}"
OUTROOT = f"{resource('hovernet_segmentation')}"
sys.path.insert(0, f"{resource('vendor_hovernet_original')}")
sys.path.insert(0, f"{resource('vendor_hovernet')}")
sys.path.insert(0, f"{resource('vendor_hovernet')}/models")
from models.hovernet.net_desc import create_model
from models.hovernet.post_proc import process as hn_process
PAD, IN = (46, 256)

def infer_image(model, img_f32):
    """Must be called within a torch.no_grad() context."""
    'Sliding-window inference (reflect-pad 46, 2×2 offset windows); return merged probability/type/hv maps.'
    H, W = img_f32.shape[:2]
    padded = np.pad(img_f32, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='reflect')
    ph = (16 - padded.shape[0] % 16) % 16
    pw = (16 - padded.shape[1] % 16) % 16
    if ph or pw:
        padded = np.pad(padded, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    Hp, Wp = padded.shape[:2]
    prob_sum = np.zeros((Hp, Wp), np.float32)
    hv_sum = np.zeros((Hp, Wp, 2), np.float32)
    tp_prob_sum = np.zeros((6, Hp, Wp), np.float32)
    cov = np.zeros((Hp, Wp), np.float32)
    windows = []
    for oy in range(0, Hp - IN + 1, 164):
        for ox in range(0, Wp - IN + 1, 164):
            windows.append((oy, ox))
    if windows and windows[-1][0] + IN < Hp:
        windows += [(Hp - IN, ox) for oy, ox in windows if oy == windows[-1][0]]
    extra = []
    for oy, ox in windows:
        if ox + IN < Wp:
            extra.append((oy, Wp - IN))
    windows += [w for w in extra if w not in windows]
    for oy, ox in windows:
        crop = padded[oy:oy + IN, ox:ox + IN]
        x = torch.from_numpy(crop.transpose(2, 0, 1)[None]).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(x)
        prob = torch.softmax(out['np'].float(), 1)[0, 1].cpu().numpy()
        hv = out['hv'].float()[0].permute(1, 2, 0).cpu().numpy()
        tp_prob = torch.softmax(out['tp'].float(), 1)[0].cpu().numpy()
        o = 46
        prob_sum[oy + o:oy + o + 164, ox + o:ox + o + 164] += prob
        hv_sum[oy + o:oy + o + 164, ox + o:ox + o + 164] += hv
        tp_prob_sum[:, oy + o:oy + o + 164, ox + o:ox + o + 164] += tp_prob
        cov[oy + o:oy + o + 164, ox + o:ox + o + 164] += 1
    cov = np.maximum(cov, 1)
    prob_full = (prob_sum / cov)[PAD:PAD + H, PAD:PAD + W]
    hv_full = (hv_sum / cov[..., None])[PAD:PAD + H, PAD:PAD + W]
    tp_argmax = np.argmax(tp_prob_sum, axis=0)
    tp_full = tp_argmax[PAD:PAD + H, PAD:PAD + W].astype(np.float32)
    return np.concatenate([tp_full[..., None], prob_full[..., None], hv_full], axis=-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outer-fold', type=int, required=True, choices=[1, 2, 3])
    ap.add_argument('--test-only', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='debug: limit images')
    args = ap.parse_args()
    f = args.outer_fold
    ckpt_path = f'{RUNS}/hovernet_v3_fast_pannuke_testfold{f}/best_model.pth'
    model = create_model(mode='fast', nr_types=6)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['desc'] if 'desc' in ck else ck, strict=True)
    model = model.cuda().eval()
    print(f"[outer_fold{f}] loaded testfold{f} weights (epoch {ck.get('epoch', '?')}, dice {ck.get('val_dice', 0):.4f})", flush=True)
    import hashlib
    ck_sha = hashlib.sha256(open(ckpt_path, 'rb').read()).hexdigest()[:16]
    total_done = 0
    t0 = time.time()
    for src_fold in (f,) if args.test_only else (1,2,3):
        outdir = f'{OUTROOT}/outer_fold{f}/step1/source_fold{src_fold}'
        os.makedirs(outdir, exist_ok=True)
        existing = len(glob.glob(f'{outdir}/fold{src_fold}_image_*_segmentation.json'))
        for cached in glob.glob(f'{outdir}/fold{src_fold}_image_*_segmentation.json'):
            meta = json.load(open(cached)).get('_meta', {})
            if meta.get('seg_ckpt_sha256_prefix') != ck_sha or meta.get('outer_fold') != f or meta.get('source_fold') != src_fold:
                raise ValueError(f'HoVer-Net segmentation cache identity mismatch: {cached}; configure a fresh hovernet_segmentation directory')
        imgs = np.load(f"{resource('pannuke')}/Fold {src_fold}/images/fold{src_fold}/images.npy", mmap_mode='r')
        n = imgs.shape[0]
        if existing == n:
            print(f'  [outer{f}/src{src_fold}] already done ({n} files)', flush=True)
            total_done += n
            continue
        print(f'  [outer{f}/src{src_fold}] {existing}/{n} done, inferring...', flush=True)
        fold_t0 = time.time()
        for i in range(n):
            img_id = f'fold{src_fold}_image_{i}'
            outp = f'{outdir}/{img_id}_segmentation.json'
            if os.path.exists(outp):
                continue
            img = np.asarray(imgs[i]).astype(np.float32)
            with torch.no_grad():
                pred_map = infer_image(model, img)
            _, info = hn_process(pred_map, nr_types=6, return_centroids=True)
            nuc = {}
            for k, (iid, inf) in enumerate(sorted(info.items()), start=1):
                nuc[str(k)] = {'contour': np.asarray(inf['contour']).tolist(), 'centroid': [float(inf['centroid'][0]), float(inf['centroid'][1])], 'type': int(inf['type']) if inf['type'] is not None else 0, 'type_prob': float(inf['type_prob']) if inf['type_prob'] else 0.0}
            meta = {'outer_fold': f, 'source_fold': src_fold, 'seg_ckpt_sha256_prefix': ck_sha, 'n_detections': len(nuc)}
            with open(outp, 'w') as fo:
                json.dump({'tiles': [{'mag': 'tile', 'nuc': nuc}], '_meta': meta}, fo)
            total_done += 1
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - fold_t0)
                print(f'  [outer{f}/src{src_fold}] {i + 1}/{n} ({rate:.1f}/s)', flush=True)
        print(f'  [outer{f}/src{src_fold}] DONE {n} imgs ({time.time() - fold_t0:.0f}s)', flush=True)
    print(f'[outer_fold{f}] ALL DONE: {total_done} total inferences ({time.time() - t0:.0f}s), GPU mem: {torch.cuda.max_memory_allocated() / 1000000000.0:.1f}GB', flush=True)
if __name__ == '__main__':
    main()
