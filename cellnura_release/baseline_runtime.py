import csv
import json
import hashlib
import os
from pathlib import Path
from .paths import load_config, project_root, resolve_path

DEFAULTS = {
    'scripts': 'baselines/shared', 'baselines': 'baselines',
    'vendor_cellvit': 'third_party/cellvit', 'vendor_cellvitpp': 'third_party/cellvitpp',
    'vendor_hovernet': 'third_party/hovernet', 'vendor_hovernet_original': 'third_party/hovernet_original',
    'vendor_dist_port': 'third_party/dist_port', 'tsfd_source': 'third_party/tsfd',
    'runs': 'outputs/baselines/runs', 'evaluation': 'outputs/baselines/evaluation',
    'detections': 'outputs/baselines/detections', 'cache': 'outputs/baselines/cache',
    'tsfd_cache': 'outputs/baselines/cache/tsfd', 'traditional_outputs': 'outputs/baselines/traditional',
    'hovernet_segmentation': 'outputs/baselines/hovernet_segmentation',
    'hovernet_detections': 'outputs/baselines/hovernet_detections',
    'hovernet_external_detections_consep': 'outputs/baselines/hovernet_external/consep/step1',
    'hovernet_external_detections_crchisto': 'outputs/baselines/hovernet_external/crchisto/step1',
    'official_hovernet_detections': 'outputs/baselines/official_hovernet',
    'tphead_runs': 'outputs/baselines/tphead',
}
EXTERNAL = {
    'pannuke', 'consep', 'crchisto', 'data_package', 'split_manifest',
    'stardist_weight_dir', 'cellvit_sam_weight',
    'cellvit256_official_weight', 'cellvitpp_head_weight', 'tsfd_public_weight',
    'hovernet_official_weight', 'traditional_features', 'resnet18_imagenet_weight', 'maskrcnn_coco_weight', 'tsfd_boundary_reference',
}

def resource(key, config=None, required=False):
    config = load_config() if config is None else config
    if key == 'project_root':
        return str(project_root())
    if key not in DEFAULTS and key not in EXTERNAL:
        raise KeyError(f'Unknown baseline resource: {key}')
    value = config.get('baseline_paths', {}).get(key)
    if value in (None, '') and key == 'split_manifest':
        value = config.get('paths', {}).get('split_manifest')
    if value in (None, '') and key in DEFAULTS:
        path = project_root()/DEFAULTS[key]
    elif value in (None, '') and key == 'data_package':
        # Older wrappers still declare PKG, but PanNuke reads are rewritten directly.
        path = Path(resource('pannuke', config)).parent.parent
    elif value in (None, ''):
        raise ValueError(f'Configure baseline_paths.{key} explicitly in {config.get("_config_path", "configuration")}')
    else:
        path = resolve_path(value, config)
    if required and not path.exists():
        raise FileNotFoundError(f'Baseline resource {key} does not exist: {path}')
    return str(path)

def real_classes(dataset):
    ds = str(dataset).lower()
    if ds == 'pannuke':
        return (0, 1, 2, 3, 4)
    if ds in ('consep', 'crchisto'):
        return (0, 1, 2, 3)
    raise ValueError(f'Unknown dataset: {dataset}')

def test_image_ids(dataset, fold=None):
    ds = {'pannuke': 'PanNuke', 'consep': 'CoNSeP', 'crchisto': 'CRCHisto'}.get(str(dataset).lower())
    if ds is None:
        raise ValueError(f'Unknown dataset: {dataset}')
    rows = list(csv.DictReader(open(resource('split_manifest', required=True), encoding='utf-8-sig')))
    ids = {r['image_id'] for r in rows if r['dataset'] == ds and r['split'] == 'test'
           and (ds != 'PanNuke' or fold is None or r['outer_fold'] == f'fold{int(fold)}')}
    if not ids:
        raise ValueError(f'No frozen test images for {ds} fold={fold}')
    return tuple(sorted(ids))

def setting(key, default=None):
    cfg = load_config()
    return cfg.get('baselines', {}).get(os.environ.get('CELLNURA_BASELINE_NAME', ''), {}).get(key, default)

def inference_only():
    return os.environ.get('CELLNURA_BASELINE_ACTION') == 'infer'

def frozen_roles(dataset, config=None):
    ds = {'consep':'CoNSeP', 'crchisto':'CRCHisto'}.get(dataset)
    if ds is None: raise ValueError('frozen_roles requires an external dataset')
    with open(resource('split_manifest', config, required=True), encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['dataset'] == ds]
    result = {k: sorted({r['image_id'] for r in rows if r['split']==k}) for k in ('train','val','test')}
    if not result['val']:
        file = project_root()/'baselines/external_validation_splits.json'
        if hashlib.sha256(file.read_bytes()).hexdigest() != 'd8a1d5773e76acd7393708526bd0033938e769f13607a9f3ee6488b1f2a4f255': raise ValueError('Frozen external validation SHA mismatch')
        external = json.loads(file.read_text())[dataset]
        if set(external['train_images']) | set(external['val_images']) != set(result['train']):
            raise ValueError('Frozen external validation does not partition official training images')
        result['train'], result['val'] = sorted(external['train_images']), sorted(external['val_images'])
    if any(not v for v in result.values()): raise ValueError(f'Incomplete frozen split for {ds}')
    if set(result['train']) & set(result['val']) or set(result['test']) & (set(result['train'])|set(result['val'])):
        raise ValueError(f'Image leakage in frozen split for {ds}')
    return result


def install_baseline_paths():
    """Put each algorithm directory on sys.path so cross-folder imports keep working."""
    import sys
    root = Path(resource('baselines'))
    for name in ('shared', 'maskrcnn', 'dist', 'micronet', 'hovernet', 'cellvitpp', 'traditional', 'tsfd', 'cellnura_cellvit'):
        folder = root / name
        if folder.is_dir():
            entry = str(folder)
            if entry not in sys.path:
                sys.path.insert(0, entry)


install_baseline_paths()

