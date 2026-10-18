"""CPU-only TSFD preprocessing/postprocessing; original functions extracted via AST."""
from pathlib import Path
from cellnura_release.baseline_runtime import resource, real_classes, test_image_ids, frozen_roles, setting, inference_only
from pathlib import Path
import ast, copy, json, os, types
import numpy as np
import cv2
from scipy import linalg, ndimage as ndi
from scipy.special import expit, softmax
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology
from skimage.segmentation import find_boundaries
ROOT = Path(resource('project_root'))
REV = Path(resource('project_root'))
SOURCE = Path(resource('tsfd_source') + '/slide_inference/utils.py')
DATA = Path(resource('pannuke'))
_IMAGES = {}
_MASKS = {}
UTILS = None

def load_utils():
    tree = ast.parse(SOURCE.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))]

    class Compat(ast.NodeTransformer):

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'np' and (node.attr == 'bool'):
                node.attr = 'bool_'
            return self.generic_visit(node)
    tree = ast.fix_missing_locations(Compat().visit(tree))
    module = types.ModuleType('tsfd_original_cpu')
    module.__dict__.update(np=np, cv2=cv2, linalg=linalg, ndi=ndi, measure=measure, morphology=morphology, binary_fill_holes=binary_fill_holes, find_boundaries=find_boundaries, copy=copy)
    exec(compile(tree, str(SOURCE), 'exec'), module.__dict__)

    def one_hot(values, classes):
        values = np.asarray(values, dtype=np.int32)
        result = np.zeros(values.shape + (classes,), dtype=np.float32)
        valid = (values >= 0) & (values < classes)
        idx = np.nonzero(valid)
        result[idx + (values[valid],)] = 1
        return result
    module.gray2encoded = one_hot
    module.sigmoid_activation = expit
    module.softmax_activation = lambda x: softmax(x, axis=-1)
    return module

def initialize():
    global UTILS
    cv2.setNumThreads(1)
    UTILS = load_utils()

def arrays(fold):
    if fold not in _IMAGES:
        _IMAGES[fold] = np.load(DATA / f'Fold {fold}/images/fold{fold}/images.npy', mmap_mode='r')
        _MASKS[fold] = np.load(DATA / f'Fold {fold}/masks/fold{fold}/masks.npy', mmap_mode='r')
    return (_IMAGES[fold], _MASKS[fold])

def prepare(key):
    fold, index = key
    image = np.clip(np.asarray(arrays(fold)[0][index]), 0, 255).astype(np.uint8)
    with np.errstate(divide='ignore', invalid='ignore'):
        stains = UTILS.deconv_stains(image, UTILS.her_from_rgb)
    rgb = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    stains = cv2.resize(stains, (512, 512), interpolation=cv2.INTER_LINEAR)
    return np.concatenate([rgb, stains], -1)

def points(mask, image_id, prefix):
    result = []
    for cls in range(5):
        channel = mask[..., cls]
        for iid in np.unique(channel):
            if iid == 0:
                continue
            y, x = np.nonzero(channel == iid)
            result.append(dict(image_id=image_id, instance_id=f'{prefix}{cls}_{int(iid)}', x=float(x.mean()), y=float(y.mean()), cls=cls))
    return result

def finish(item):
    fold, index, seg, boundary, output = item
    image_id = f'fold{fold}_image_{index}'
    seg, boundary = UTILS.decode_predictions(seg[None], boundary[None])
    instances = UTILS.seprate_instances(seg, boundary, 6, True, 3)[..., :5]
    if instances.shape != (256, 256, 5) or not np.isfinite(instances).all():
        raise ValueError(f'Invalid instance output {image_id}')
    if instances.max() >= 65536:
        raise ValueError('Instance ID overflow')
    instances = instances.astype(np.uint16)
    gt = points(arrays(fold)[1][index], image_id, 'g')
    pred = points(instances, image_id, 'p')
    record = dict(image_id=image_id, fold=fold, index=index, width=256, height=256, gt=gt, pred=pred)
    folder = Path(output) / f'fold{fold}'
    path = folder / 'instances' / f'{image_id}.npz'
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, instances=instances)
    tmp.replace(path)
    path = folder / 'records' / f'{image_id}.json'
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(record, separators=(',', ':')))
    tmp.replace(path)
    return dict(image_id=image_id, n_gt=len(gt), n_pred=len(pred))
