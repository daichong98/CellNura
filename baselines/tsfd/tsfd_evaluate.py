"""Evaluate TSFD disk records, with full image coverage and geometry checks."""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
from pathlib import Path
import csv, json, hashlib, sys
import numpy as np
from unified_matcher import Instance, EvalInput, evaluate, write_outputs
NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']

def run(root, folds=(1, 2, 3), limit=0):
    root = Path(root)
    config = json.loads((root / 'configuration.json').read_text())
    summaries = []
    per_class = []
    all_gt = []
    all_pred = []
    all_ids = []
    total_empty = 0
    validated_masks = 0
    for fold in folds:
        expected = [f'fold{f}_image_{i}' for f, i in config['keys'] if f == fold]
        paths = list((root / f'fold{fold}' / 'records').glob('*.json'))
        assert {p.stem for p in paths} == set(expected), 'Incomplete/extra image records'
        gt = []
        pred = []
        image_rows = []
        for name in expected:
            rec = json.loads((root / f'fold{fold}' / 'records' / (name + '.json')).read_text())
            assert rec['image_id'] == name
            inst = np.load(root / f'fold{fold}' / 'instances' / (name + '.npz'))['instances']
            assert inst.shape == (256, 256, 5)
            expected_instances = {(c, int(i)) for c in range(5) for i in np.unique(inst[..., c]) if i}
            actual_instances = {(p['cls'], int(p['instance_id'].split('_')[-1])) for p in rec['pred']}
            assert actual_instances == expected_instances
            for item in rec['pred']:
                c = item['cls']
                iid = int(item['instance_id'].split('_')[-1])
                yy, xx = np.nonzero(inst[..., c] == iid)
                assert np.isclose(xx.mean(), item['x'], atol=1e-10) and np.isclose(yy.mean(), item['y'], atol=1e-10)
            for side in ('gt', 'pred'):
                assert len({v['instance_id'] for v in rec[side]}) == len(rec[side])
                assert all((v['image_id'] == name and 0 <= v['x'] < 256 and (0 <= v['y'] < 256) and (v['cls'] in range(5)) for v in rec[side]))
            gt.extend((Instance(**v) for v in rec['gt']))
            pred.extend((Instance(**v) for v in rec['pred']))
            image_rows.append(dict(image_id=name, n_gt=len(rec['gt']), n_pred=len(rec['pred'])))
            total_empty += not bool(rec['pred'])
            validated_masks += 1
        all_gt += gt
        all_pred += pred
        all_ids += expected
        res = evaluate(EvalInput('PanNuke', 12, gt, pred, image_ids=tuple(expected), class_ids=(0, 1, 2, 3, 4), real_class_ids=real_classes('PanNuke')))
        assert res.n_gt == res.n_matched + res.n_missed and res.n_pred == res.n_matched + res.n_spurious
        assert res.n_duplicate_claims == 0 and len(res.per_image) == len(expected)
        write_outputs(res, str(root / f'eval_fold{fold}'))
        row = dict(fold=fold, n_images=len(expected), N_gt=res.n_gt, N_pred=res.n_pred, matched=res.n_matched, missed=res.n_missed, spurious=res.n_spurious, matched_classification_macroF1=res.macro('classification'), plus_missed_macroF1=res.macro('plus_missed_FN'), unified_macroF1=res.macro('unified'), detection_F1=2 * res.n_matched / (res.n_gt + res.n_pred) if res.n_gt + res.n_pred else 0)
        summaries.append(row)
        for cls in range(5):
            d = res.classes[cls]
            per_class.append(dict(fold=fold, class_id=cls, class_name=NAMES[cls], **d['counts'], P=d['unified']['P'], R=d['unified']['R'], F1=d['unified']['F1']))
        with (root / f'fold{fold}/images.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=image_rows[0].keys())
            w.writeheader()
            w.writerows(image_rows)
        print(json.dumps(row), flush=True)
    for name, rows in [('fold_summary', summaries), ('per_class_summary', per_class)]:
        with (root / (name + '.csv')).open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    combined = evaluate(EvalInput('PanNuke', 12, all_gt, all_pred, image_ids=tuple(all_ids), class_ids=(0, 1, 2, 3, 4), real_class_ids=real_classes('PanNuke')))
    write_outputs(combined, str(root / 'eval_pooled'))
    metrics = {}
    for key in ('matched_classification_macroF1', 'plus_missed_macroF1', 'unified_macroF1', 'detection_F1'):
        values = [row[key] for row in summaries]
        metrics[key] = dict(mean=float(np.mean(values)), std_sample=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    result = dict(passed=True, images=validated_masks, empty_prediction_images=total_empty, all_instance_masks_and_centroids_checked=True, complete_coverage=True, folds=summaries, fold_mean_std=metrics, pooled=dict(N_gt=combined.n_gt, N_pred=combined.n_pred, matched=combined.n_matched, missed=combined.n_missed, spurious=combined.n_spurious, unified_macroF1=combined.macro('unified')), paper_main_table_eligible=False, limitation='Public task checkpoint split unverified; these are reference, not independent fold-trained results', evaluator_sha256=hashlib.sha256(Path(__file__).resolve().parents[1] / 'shared' / 'unified_matcher.py'.read_bytes()).hexdigest())
    (root / 'ACCEPTANCE.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('EVALUATION AND COVERAGE PASSED', flush=True)
    return result
if __name__ == '__main__':
    root = Path(sys.argv[1])
    config = json.loads((root / 'configuration.json').read_text())
    run(root, sorted({k[0] for k in config['keys']}))
