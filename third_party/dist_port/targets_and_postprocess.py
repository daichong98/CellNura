"""Original CDT target and dynamic watershed with explicit modern API fixes."""
import numpy as np
from scipy.ndimage import distance_transform_cdt
from skimage.measure import label, regionprops
from skimage.morphology import reconstruction, dilation, erosion, square
from skimage.segmentation import watershed


def instance_distances(instances):
    instances = np.asarray(instances)
    if instances.ndim != 2 or not np.issubdtype(instances.dtype, np.integer) or np.any(instances < 0):
        raise ValueError("Expected nonnegative 2D integer instance IDs")
    result = np.zeros(instances.shape, dtype=np.uint8)
    for iid in np.unique(instances):
        if iid == 0:
            continue
        mask = instances == iid
        dist = distance_transform_cdt(mask, metric="chessboard")
        if np.any(dist[mask] < 0) or np.any(dist[mask] > 255):
            raise ValueError("Distance not representable by original uint8 target")
        result[mask] = dist[mask]
    return result


def pannuke_instances(mask):
    if mask.ndim != 3 or mask.shape[-1] != 6:
        raise ValueError("Expected PanNuke HxWx6 mask")
    result = np.zeros(mask.shape[:2], dtype=np.int32)
    next_id = 1
    for cls in range(5):
        for iid in np.unique(mask[..., cls]):
            if iid == 0:
                continue
            region = mask[..., cls] == iid
            if np.any(result[region]):
                raise ValueError("Overlapping GT instances across class channels")
            result[region] = next_id
            next_id += 1
    return result


def _reconstruct(img, h):
    return reconstruction(np.minimum(255, img.astype(np.float64)+h),
                          img, method="erosion").astype(np.uint8)


def postprocess(distance, h=1, threshold=1.0):
    """Dist.py clips/truncates to 0..255 before PostProcess.
    Cast explicitly to uint8 to avoid version-dependent img_as_ubyte(int64).
    Preserve original largest-region-as-background and watershed-line removal.
    h/threshold here are diagnostic defaults, NOT validation-selected parameters.
    """
    distance = np.asarray(distance)
    if distance.ndim != 2 or not np.isfinite(distance).all():
        raise ValueError("Expected finite 2D distance prediction")
    if not 0 <= h <= 255 or threshold < 0:
        raise ValueError("Invalid watershed parameters")
    quantized = np.clip(distance, 0, 255).astype(np.uint8)
    foreground = quantized > threshold
    if not foreground.any():
        return np.zeros(quantized.shape, dtype=np.int32)
    inverse = 255 - quantized
    reconstructed = _reconstruct(inverse, h)
    maxima = _reconstruct(reconstructed, 1) - reconstructed
    maxima[~foreground] = 0
    markers = label(maxima)
    ws = watershed(reconstructed, markers, mask=foreground)
    values, counts = np.unique(ws, return_counts=True)
    arranged = label(ws, background=values[np.argmax(counts)])
    eroded = arranged.copy()
    eroded[eroded == 0] = eroded.max()+1
    eroded = erosion(eroded, square(3))
    eroded[arranged == 0] = 0
    boundary = dilation(arranged, square(3)) - eroded
    boundary[arranged == 0] = 0
    arranged[boundary > 0] = 0
    return arranged.astype(np.int32)


def centroids_xy(instances):
    return np.asarray([(p.centroid[1], p.centroid[0]) for p in regionprops(instances)], dtype=np.float64).reshape(-1, 2)
