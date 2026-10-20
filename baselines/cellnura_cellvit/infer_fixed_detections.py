
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HEADER = ["image_id", "instance_id", "cx", "cy", "class_id"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["image_id"]:
            raise ValueError("Image list must have one image_id column")
        ids = [row["image_id"] for row in reader]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Image list is empty or has duplicate IDs")
    return ids


def frozen_detections(path: Path, allowed_ids: set[str]):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADER:
            raise ValueError(f"Frozen predictions need columns {HEADER}")
        rows = list(reader)
    grouped: dict[str, list[int]] = defaultdict(list)
    seen = set()
    for index, row in enumerate(rows):
        image_id, instance_id = row["image_id"], row["instance_id"]
        if image_id not in allowed_ids or not instance_id:
            raise ValueError(f"Unknown image or empty instance ID: {image_id}/{instance_id}")
        key = image_id, instance_id
        if key in seen:
            raise ValueError(f"Duplicate instance: {key}")
        seen.add(key)
        if not all(math.isfinite(float(row[c])) for c in ("cx", "cy")):
            raise ValueError(f"Nonfinite centroid: {key}")
        grouped[image_id].append(index)
    return rows, grouped


def image_array(dataset: str, root: Path, image_id: str, fold: int, pannuke_images) -> np.ndarray:
    if dataset == "pannuke":
        prefix = f"fold{fold}_image_"
        if not image_id.startswith(prefix) or not image_id[len(prefix):].isdigit():
            raise ValueError(f"Unexpected PanNuke image ID: {image_id}")
        index = int(image_id[len(prefix):])
        if index >= len(pannuke_images):
            raise ValueError(f"PanNuke image index outside source array: {image_id}")
        image = np.asarray(pannuke_images[index])
    elif dataset == "consep":
        image = np.asarray(Image.open(root / "image" / f"{image_id}.png").convert("RGB"))
    else:
        image = np.asarray(
            Image.open(root / "CRCHisto_database" / "Detection" / image_id / f"{image_id}.bmp").convert("RGB")
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image: {image_id}")
    image = np.clip(image, 0, 255).astype(np.uint8)
    if dataset != "pannuke":
        ph, pw = (-image.shape[0]) % 16, (-image.shape[1]) % 16
        if ph or pw:
            image = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return image


def assign_classes(rows: list[dict], indices: list[int], type_argmax: np.ndarray,
                   n_real_classes: int):
    """Round-and-clip centroid lookup on the forced-commit type map.

    D065: type_argmax is the argmax over the real-class channels 1..N-1
    only, so its values 0..N-2 map directly to manuscript classes and no
    unknown exists.
    """
    if type_argmax.ndim != 2:
        raise ValueError("Type argmax must be a 2-D map")
    height, width = type_argmax.shape
    for index in indices:
        row = rows[index]
        px = min(max(int(round(float(row["cx"]))), 0), width - 1)
        py = min(max(int(round(float(row["cy"]))), 0), height - 1)
        cls = int(type_argmax[py, px])
        if cls < -1 or cls >= n_real_classes:
            raise ValueError(f"Unexpected CellViT channel at {row['image_id']}/{row['instance_id']}: {cls + 1}")
        row["class_id"] = str(cls)


def run(args) -> dict:
    import torch

    release_root = Path(__file__).resolve().parents[2]
    vendor_root = release_root / "third_party" / "cellvit"
    sys.path.insert(0, str(vendor_root))
    from models.segmentation.cell_segmentation.cellvit import CellViT256

    checkpoint = args.checkpoint.resolve()
    digest = sha256(checkpoint)
    if args.checkpoint_sha256 and digest != args.checkpoint_sha256:
        raise ValueError(f"Classifier checkpoint SHA mismatch: {digest}")
    ids = image_ids(args.image_list)
    rows, grouped = frozen_detections(args.frozen_detections, set(ids))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    ntypes, ntissues = (6, 19) if args.dataset == "pannuke" else (5, 4)
    model = CellViT256(model256_path=None, num_nuclei_classes=ntypes, num_tissue_classes=ntissues)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    pannuke_images = None
    if args.dataset == "pannuke":
        if args.fold not in (1, 2, 3):
            raise ValueError("PanNuke needs --fold 1, 2, or 3")
        image_file = args.dataset_root / f"Fold {args.fold}" / "images" / f"fold{args.fold}" / "images.npy"
        pannuke_images = np.load(image_file, mmap_mode="r")

    torch.set_num_threads(args.torch_threads)
    for image_id in ids:
        if not grouped.get(image_id):
            continue
        image = image_array(args.dataset, args.dataset_root, image_id, args.fold, pannuke_images)
        tensor = torch.from_numpy(
            ((image.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        ).to(device)
        with torch.inference_mode():
            output = model(tensor)
            type_map = torch.softmax(output["nuclei_type_map"].float(), 1)[0, 1:].argmax(0).cpu().numpy()
        assign_classes(rows, grouped[image_id], type_map, ntypes - 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "protocol": "D065 CellNuraCellViT forced-commit fixed-detection type-head lookup",
        "lookup_rule": ("argmax over type channels 1..N only (background excluded); "
                        "every detection receives manuscript class 0..N-1; no unknown"),
        "dataset": args.dataset,
        "fold": args.fold if args.dataset == "pannuke" else None,
        "ground_truth_read_during_inference": False,
        "frozen_detections_sha256": sha256(args.frozen_detections),
        "classifier_checkpoint_sha256": digest,
        "prediction_sha256": sha256(args.output),
        "prediction_rows": len(rows),
        "images_including_zero_detection": len(ids),
        "unknown_predictions": sum(row["class_id"] == "-1" for row in rows),
    }
    receipt_path = args.output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("pannuke", "consep", "crchisto"))
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--frozen-detections", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
