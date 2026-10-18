"""t08_hovernet_external_eval.py — Unified external-dataset evaluation of within-fold retrained HoVer-Net (Tier A).

Reuse step1 detection JSON files (centroids+types) from the within-fold segmenters already trained for the CellNura v3.3 external pipeline
(hovernet_fast_consep_v33 / hovernet_fast_crchisto_v33, both with type heads); no retraining is needed:
  * CoNSeP: trained from scratch on 27 training images (val Dice 0.8265), step1 = cellnura_v33_external/consep/step1
  * CRCHisto: trained from scratch on 80 training images (disk GT disclosed, Dice 0.6279), step1 = cellnura_v33_external/crchisto/step1
Types: model tp channels 1-4 → unified 0-3; type=0 → -1 unknown (consistent treatment).
Evaluation: one-to-one unified_matcher at τ=5/10px, including all image_ids.
Usage: python t08_hovernet_external_eval.py --dataset consep|crchisto
"""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
import argparse, glob, json, os, sys
import numpy as np
BASE = resource('project_root')
HERE = Path(resource('scripts'))
sys.path.insert(0, HERE)
from unified_matcher import EvalInput, Instance, evaluate, write_outputs
CFG = {'consep': dict(tau=5.0, step1=f"{resource('hovernet_external_detections_consep')}", test_prefix='test_', out=f"{resource('evaluation')}/hovernet_indep_consep"), 'crchisto': dict(tau=10.0, step1=f"{resource('hovernet_external_detections_crchisto')}", test_prefix=None, out=f"{resource('evaluation')}/hovernet_indep_crchisto")}

def load_gt(ds):
    gt = {}
    if ds == 'consep':
        import scipy.io as sio
        for f in sorted(glob.glob(f"{resource('consep')}/label/test_*.mat")):
            name = os.path.basename(f)[:-4]
            m = sio.loadmat(f)
            pts, cls = ([], [])
            for c, t in zip(m['inst_centroid'], m['inst_type'].ravel().astype(int)):
                if t == 0 or t not in {1: 3, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}:
                    continue
                pts.append((float(c[0]), float(c[1])))
                cls.append({1: 3, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}[t])
            gt[name] = (pts, cls)
    else:
        CMAP = {'inflammatory': 0, 'epithelial': 1, 'fibroblast': 2, 'others': 3}
        man = None
        import pandas as pd
        man = pd.read_csv(f"{resource('split_manifest')}")
        te = sorted(set(man[(man.dataset == 'CRCHisto') & (man.split == 'test')].image_id))
        for name in te:
            d = json.load(open(f"{resource('crchisto')}/crchisto_features_output/step0_preprocessed/{name}_processed.json"))
            pts, cls = ([], [])
            for cl, coords in d.get('classification_data', {}).items():
                if cl in CMAP and isinstance(coords, list):
                    for c in coords:
                        if isinstance(c, list) and len(c) >= 2:
                            pts.append((float(c[0]), float(c[1])))
                            cls.append(CMAP[cl])
            gt[name] = (pts, cls)
    return gt

def test_names(ds):
    cfg = CFG[ds]
    if cfg['test_prefix']:
        return sorted((os.path.basename(p).replace('_segmentation.json', '') for p in glob.glob(f"{cfg['step1']}/{cfg['test_prefix']}*_segmentation.json")))
    import pandas as pd
    man = pd.read_csv(f"{resource('split_manifest')}")
    return sorted(set(man[(man.dataset == 'CRCHisto') & (man.split == 'test')].image_id))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['consep', 'crchisto'])
    a = ap.parse_args()
    ds = a.dataset
    cfg = CFG[ds]
    gt = load_gt(ds)
    names = test_names(ds)
    gt_list, pr_list, n_unk = ([], [], 0)
    for name in names:
        d = json.load(open(f"{cfg['step1']}/{name}_segmentation.json"))
        nuc = d['tiles'][0]['nuc']
        pts, cls = gt[name]
        for j, (c, k) in enumerate(zip(pts, cls)):
            gt_list.append(Instance(name, f'g{j}', c[0], c[1], int(k)))
        for iid, v in nuc.items():
            t = int(v['type']) - 1
            cx, cy = (float(v['centroid'][0]), float(v['centroid'][1]))
            pr_list.append(Instance(name, f'd{int(iid)}', cx, cy, t))
            if t < 0:
                n_unk += 1
    ev = EvalInput(dataset=ds, tau=cfg['tau'], gt=gt_list, pred=pr_list, image_ids=tuple(names), real_class_ids=real_classes(ds))
    res = evaluate(ev, mode='one_to_one')
    os.makedirs(cfg['out'], exist_ok=True)
    write_outputs(res, f"{cfg['out']}/eval")
    row = {'dataset': ds, 'N_GT': res.n_gt, 'N_pred': res.n_pred, 'M': res.n_matched, 'missed': res.n_missed, 'spurious': res.n_spurious, 'cls_macroF1': round(res.macro('classification'), 4), 'plusFN_macroF1': round(res.macro('plus_missed_FN'), 4), 'uni_macroF1': round(res.macro('unified'), 4), 'unknown_type_preds': n_unk}
    print(str(row), flush=True)
    json.dump(row, open(f"{cfg['out']}/result.json", 'w'), indent=2)
if __name__ == '__main__':
    main()
