"""CellViT++ SAM-H + official CoNSeP head; full images and auditable unified evaluation."""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import argparse, json, sys, time, hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
HERE = Path(resource('scripts'))
REV = resource('project_root')
ROOT = Path(resource('project_root'))
BASE = Path(resource('baselines'))
sys.path.insert(0, str(Path(resource('vendor_cellvitpp'))))
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.classifier.linear_classifier import LinearClassifier
from unified_matcher import Instance, EvalInput, evaluate, write_outputs
HEAD_MAP = {0: 3, 1: 0, 2: 1, 3: 2}
GT_MAP = {1: 3, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        while (b := f.read(4 * 1024 * 1024)):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0, help='0=all14 test images')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.limit < 0:
        ap.error('--limit must be nonnegative')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'instances').mkdir()
    torch.set_num_threads(4)
    backbone_path = Path(resource('cellvit_sam_weight'))
    head_path = Path(resource('cellvitpp_head_weight'))
    source = Path(resource('consep'))
    model = CellViTSAM(model_path=None, num_nuclei_classes=6, num_tissue_classes=19, vit_structure='SAM-H')
    ck = torch.load(backbone_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    del ck
    hk = torch.load(head_path, map_location='cpu', weights_only=True)
    state = hk['model_state_dict']
    hidden, embed = state['fc1.weight'].shape
    classes = state['fc2.weight'].shape[0]
    if embed != model.embed_dim or classes != 4:
        raise ValueError(f'Backbone/head mismatch: backbone={model.embed_dim}, head={embed}, classes={classes}')
    head = LinearClassifier(embed_dim=embed, hidden_dim=hidden, num_classes=classes, drop_rate=0)
    head.load_state_dict(state, strict=True)
    model = model.to(args.device).eval()
    head = head.to(args.device).eval()
    paths = sorted((source / 'image').glob('test_*.png'), key=lambda p: int(p.stem.split('_')[-1]))
    if len(paths) != 14:
        raise ValueError(f'Expected14 CoNSeP test images, found{len(paths)}')
    if args.limit:
        paths = paths[:args.limit]
    all_gt = []
    all_pred = []
    audit = []
    images = []
    tick = time.time()
    for path in paths:
        name = path.stem
        image = np.asarray(Image.open(path).convert('RGB'))
        h, w = image.shape[:2]
        ph, pw = (-h % 16, -w % 16)
        padded = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode='reflect')
        tensor = torch.from_numpy(((padded.astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)[None]).to(args.device)
        with torch.inference_mode():
            out = model(tensor, retrieve_tokens=True)
            tokens = out['tokens'][0]
            assert tokens.shape[0] == embed and torch.isfinite(tokens).all()
            for key in ('nuclei_binary_map', 'nuclei_type_map'):
                out[key] = torch.softmax(out[key], 1)
            maps, infos = model.calculate_instance_map(out, magnification=40)
            inst = np.asarray(maps[0].cpu() if torch.is_tensor(maps) else maps[0], dtype=np.int32)[:h, :w]
            embeddings = []
            cell_records = []
            padding_only = 0
            for iid, cell in infos[0].items():
                yy, xx = np.nonzero(inst == iid)
                if not len(xx):
                    padding_only += 1
                    continue
                bbox = np.asarray(cell['bbox'], dtype=float) / model.patch_size
                lo = np.floor(bbox[0]).astype(int)
                hi = np.ceil(bbox[1]).astype(int)
                lo = np.maximum(lo, 0)
                hi = np.minimum(hi, tokens.shape[1:])
                token = tokens[:, lo[0]:hi[0], lo[1]:hi[1]]
                if not token.numel():
                    raise ValueError(f'Empty token region {name}/{iid}')
                embeddings.append(token.mean((1, 2)))
                cell_records.append((int(iid), float(xx.mean()), float(yy.mean())))
            if embeddings:
                feats = torch.stack(embeddings)
                probs = torch.softmax(head(feats), 1).cpu().numpy()
                if not np.isfinite(probs).all():
                    raise ValueError('Nonfinite class probabilities')
            else:
                probs = np.zeros((0, 4), dtype=np.float32)
        pred = []
        for (iid, x, y), prob in zip(cell_records, probs):
            raw = int(prob.argmax())
            cls = HEAD_MAP[raw]
            pred.append(Instance(name, str(iid), x, y, cls))
            audit.append(dict(image_id=name, instance_id=str(iid), raw_head_class=raw, unified_class=cls, probabilities=prob.tolist()))
        mat = loadmat(source / 'label' / (name + '.mat'))
        gt = [Instance(name, str(i), float(xy[0]), float(xy[1]), GT_MAP[int(c)]) for i, (xy, c) in enumerate(zip(mat['inst_centroid'], mat['inst_type'].ravel())) if int(c) in GT_MAP]
        all_gt.extend(gt)
        all_pred.extend(pred)
        images.append(dict(image_id=name, width=w, height=h, padded_shape=list(padded.shape[:2]), tokens_shape=list(tokens.shape), n_gt=len(gt), n_pred=len(pred), padding_only_instances=padding_only))
        np.savez_compressed(root / 'instances' / (name + '.npz'), instances=inst)
        print(json.dumps(images[-1]), flush=True)
        del out, tensor, tokens, maps, infos
    ids = tuple((p.stem for p in paths))
    for kind in ('typed', 'detection'):
        gt = all_gt if kind == 'typed' else [Instance(v.image_id, v.instance_id, v.x, v.y, 0) for v in all_gt]
        pred = all_pred if kind == 'typed' else [Instance(v.image_id, v.instance_id, v.x, v.y, 0) for v in all_pred]
        result = evaluate(EvalInput('CoNSeP', 5, gt, pred, image_ids=ids, class_ids=tuple(range(4)) if kind == 'typed' else (0,)))
        write_outputs(result, str(root / kind))
    for name, values in [('predictions', all_pred), ('ground_truth', all_gt)]:
        (root / (name + '.json')).write_text(json.dumps([asdict(v) for v in values], indent=2))
    (root / 'classification_audit.json').write_text(json.dumps(audit, indent=2))
    manifest = dict(method='CellViT++', dataset='CoNSeP', typed_predictions=True, backbone=str(backbone_path), backbone_sha256=sha(backbone_path), head=str(head_path), head_sha256=sha(head_path), strict_load_backbone=True, strict_load_head=True, backbone_embed_dim=model.embed_dim, head_input_dim=embed, head_hidden_dim=hidden, head_classes=classes, head_class_mapping=HEAD_MAP, official_head_order=['Miscellaneous', 'Inflammatory', 'Epithelium', 'Spindle-Shaped'], image_ids=ids, images=images, n_gt=len(all_gt), n_pred=len(all_pred), seconds=time.time() - tick, inference_protocol='native1000 reflect-pad to1008; full-image forward; bbox token mean; crop instance mask to1000 then geometric centroid', protocol_caveat='project native-resolution adapter; not a claim of exact official1024 patch/stitch benchmark reproduction', tau=5, paper_main_table_eligible=False, independence='PanNuke-contaminated public backbone; CoNSeP head exact train/validation provenance still to verify', script_sha256=sha(Path(__file__)), evaluator_sha256=sha(HERE / 'unified_matcher.py'))
    (root / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({'completed': True, 'n_images': len(ids), 'n_pred': len(all_pred), 'seconds': manifest['seconds']}), flush=True)
if __name__ == '__main__':
    main()
