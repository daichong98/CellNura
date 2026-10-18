"""t06_consep_infer.py — CoNSeP inference with a within-fold segmenter (all 41 images → v3.2 step1 JSON format).

The sliding-window protocol is identical to t06_v32_infer_per_outer_fold.py (fast mode, PAD 46 / IN 256 / OUT 164 /
stride 164 / probability accumulation followed by argmax), nr_types=5 (0=bg + 4 CoNSeP classes).
Output: one segmentation JSON per image, named {image_name}_segmentation.json.
Usage: python t06_consep_infer.py [--smoke]
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import argparse, glob, hashlib, json, os, sys, time
import numpy as np
import torch
BASE = resource('project_root')
CONSEP = f"{resource('consep')}"
RUNS = f"{resource('runs')}"
OUTROOT = str(Path(resource('hovernet_external_detections_consep')).parent)
sys.path.insert(0, f"{resource('vendor_hovernet_original')}")
sys.path.insert(0, f"{resource('vendor_hovernet')}")
sys.path.insert(0, f"{resource('vendor_hovernet')}/models")
from models.hovernet.net_desc import create_model
from models.hovernet.post_proc import process as hn_process
PAD, IN, OUTP, NR_TYPES = (46, 256, 164, 5)
CKPT = f'{RUNS}/hovernet_fast_consep_v33/best_model.pth'

def infer_image(model, img_f32):
    """Sliding-window inference (reflect-pad 46, probability accumulation); return stitched tp(argmax)+np+hv maps."""
    H, W = img_f32.shape[:2]
    padded = np.pad(img_f32, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='reflect')
    ph = (16 - padded.shape[0] % 16) % 16
    pw = (16 - padded.shape[1] % 16) % 16
    if ph or pw:
        padded = np.pad(padded, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    Hp, Wp = padded.shape[:2]
    prob_sum = np.zeros((Hp, Wp), np.float32)
    hv_sum = np.zeros((Hp, Wp, 2), np.float32)
    tp_prob_sum = np.zeros((NR_TYPES, Hp, Wp), np.float32)
    cov = np.zeros((Hp, Wp), np.float32)
    windows = []
    for oy in range(0, Hp - IN + 1, OUTP):
        for ox in range(0, Wp - IN + 1, OUTP):
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
        o = PAD
        prob_sum[oy + o:oy + o + OUTP, ox + o:ox + o + OUTP] += prob
        hv_sum[oy + o:oy + o + OUTP, ox + o:ox + o + OUTP] += hv
        tp_prob_sum[:, oy + o:oy + o + OUTP, ox + o:ox + o + OUTP] += tp_prob
        cov[oy + o:oy + o + OUTP, ox + o:ox + o + OUTP] += 1
    cov = np.maximum(cov, 1)
    prob_full = (prob_sum / cov)[PAD:PAD + H, PAD:PAD + W]
    hv_full = (hv_sum / cov[..., None])[PAD:PAD + H, PAD:PAD + W]
    hv_full[prob_full < 0.5] = 0.0
    tp_full = np.argmax(tp_prob_sum, axis=0)[PAD:PAD + H, PAD:PAD + W].astype(np.float32)
    return np.concatenate([tp_full[..., None], prob_full[..., None], hv_full], axis=-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0)
    a = ap.parse_args()
    outdir = f'{OUTROOT}/step1'
    os.makedirs(outdir, exist_ok=True)
    sd = torch.load(CKPT, map_location='cpu', weights_only=True)
    sha = hashlib.sha256(open(CKPT, 'rb').read(1 << 22)).hexdigest()[:12]
    model = create_model(mode='fast', nr_types=NR_TYPES)
    model.load_state_dict(sd['desc'], strict=True)
    model = model.cuda().eval()
    imgs = sorted(glob.glob(f'{CONSEP}/image/*.png'))
    if a.smoke:
        imgs = imgs[:a.smoke]
    t0 = time.time()
    done = 0
    for ip in imgs:
        name = os.path.basename(ip)[:-4]
        outp = f'{outdir}/{name}_segmentation.json'
        if os.path.exists(outp):
            done += 1
            continue
        from PIL import Image
        img = np.array(Image.open(ip).convert('RGB')).astype(np.float32)
        with torch.no_grad():
            pred_map = infer_image(model, img)
        _, info = hn_process(pred_map, nr_types=NR_TYPES, return_centroids=True)
        nuc = {}
        for k, (iid, inf) in enumerate(sorted(info.items()), start=1):
            nuc[str(k)] = {'contour': np.asarray(inf['contour']).tolist(), 'centroid': [float(inf['centroid'][0]), float(inf['centroid'][1])], 'type': int(inf['type']) if inf['type'] is not None else 0, 'type_prob': float(inf['type_prob']) if inf['type_prob'] else 0.0}
        json.dump({'tiles': [{'mag': 'tile', 'nuc': nuc}], '_meta': {'dataset': 'CoNSeP', 'image': name, 'seg_ckpt_sha256_prefix': sha, 'n_detections': len(nuc)}}, open(outp, 'w'))
        done += 1
        print(f'[consep infer] {done}/{len(imgs)} {name}: {len(nuc)} det ({time.time() - t0:.0f}s)', flush=True)
    print(f'[consep infer] DONE {done} imgs in {time.time() - t0:.0f}s', flush=True)
if __name__ == '__main__':
    main()
