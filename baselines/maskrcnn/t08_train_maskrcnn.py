#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train or run the TorchVision Mask R-CNN baseline.

Paths come from configs/release.json through cellnura_release.baseline_runtime.
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from cellnura_release.baseline_runtime import frozen_roles, resource

SEED = 42
PANNUKE_SPLIT_COUNTS = {
    1: {"train": 4458, "val": 787, "test": 2656},
    2: {"train": 4571, "val": 807, "test": 2523},
    3: {"train": 4402, "val": 777, "test": 2722},
}
# Official CoNSeP types mapped to the four manuscript classes.
CONSEP_CLASS_MAP = {1: 3, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}
CRCHISTO_CLASS_MAP = {"inflammatory": 0, "epithelial": 1, "fibroblast": 2, "others": 3}


def pannuke_root():
    return resource("pannuke")


def consep_root():
    return resource("consep")


def crchisto_root():
    return resource("crchisto")


def split_manifest_path():
    return resource("split_manifest")


def run_root():
    return resource("runs")


def detection_target(boxes, labels, masks, image_id, empty_hw):
    """Pack one image for a TorchVision detection model."""
    if boxes:
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
    else:
        height, width = empty_hw
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return {
        "boxes": boxes,
        "labels": labels,
        "masks": masks,
        "image_id": torch.tensor([image_id]),
        "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
        "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
    }


def collate(batch):
    return tuple(zip(*batch))


class PanNukeDetect(Dataset):
    """PanNuke instance masks as boxes, labels, and binary masks.

    Channel c is manuscript class c and TorchVision label c + 1.
    split_rows keeps only the image indices of one split role.
    """

    def __init__(self, fold_list, mode, split_rows=None, limit=0):
        if mode not in ("train", "val", "test"):
            raise ValueError(f"Unknown PanNuke split mode: {mode}")
        self.items = []
        root = pannuke_root()
        for fold in fold_list:
            images = np.load(
                os.path.join(root, f"Fold {fold}", "images", f"fold{fold}", "images.npy"),
                mmap_mode="r",
            )
            masks = np.load(
                os.path.join(root, f"Fold {fold}", "masks", f"fold{fold}", "masks.npy"),
                mmap_mode="r",
            )
            indices = range(images.shape[0])
            if split_rows is not None:
                wanted = set(split_rows.get(fold, []))
                indices = [index for index in indices if index in wanted]
            for index in indices:
                self.items.append((fold, index, images, masks))
            if limit:
                self.items = self.items[:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        _fold, index, images, masks = self.items[item]
        image = np.clip(np.asarray(images[index]), 0, 255).astype(np.uint8)
        mask = np.asarray(masks[index])
        boxes, labels, instances = [], [], []
        for class_index in range(5):
            channel = mask[..., class_index]
            for instance_id in np.unique(channel):
                if instance_id == 0:
                    continue
                selected = channel == instance_id
                ys, xs = np.nonzero(selected)
                if len(ys) < 3:
                    continue
                boxes.append([float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1])
                labels.append(class_index + 1)
                instances.append(selected.astype(np.uint8))
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        return image_tensor, detection_target(boxes, labels, instances, index, (256, 256))


class CoNSePDetect(Dataset):
    """CoNSeP images resized to 512, with nearest-neighbor instance masks.

    Unknown official types are left out of the training target.
    """

    RESIZE = 512

    def __init__(self, names):
        import scipy.io as sio

        self.names = list(names)
        self.cmap = CONSEP_CLASS_MAP
        self.sio = sio
        self.root = consep_root()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, item):
        import cv2

        name = self.names[item]
        mat = self.sio.loadmat(os.path.join(self.root, "label", f"{name}.mat"))
        if "inst_map" in mat:
            instances = mat["inst_map"].astype(np.int32)
        else:
            instances = mat["inst_instancemap"].astype(np.int32)
        types = mat["inst_type"].ravel().astype(int)
        image = np.array(Image.open(os.path.join(self.root, "image", f"{name}.png")).convert("RGB"))
        if image.shape[0] != self.RESIZE:
            image = cv2.resize(image, (self.RESIZE, self.RESIZE), interpolation=cv2.INTER_LINEAR)
            instances = cv2.resize(instances, (self.RESIZE, self.RESIZE), interpolation=cv2.INTER_NEAREST)
            instances = instances.astype(np.int32)
        boxes, labels, masks = [], [], []
        for instance_id in np.unique(instances):
            if instance_id == 0:
                continue
            selected = instances == instance_id
            type_index = int(instance_id) - 1
            if type_index >= len(types) or types[type_index] == 0 or int(types[type_index]) not in self.cmap:
                continue
            ys, xs = np.nonzero(selected)
            if len(ys) < 3:
                continue
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1])
            labels.append(self.cmap[int(types[type_index])] + 1)
            masks.append(selected.astype(np.uint8))
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        return image_tensor, detection_target(boxes, labels, masks, item, (self.RESIZE, self.RESIZE))


class CRCHistoDetect(Dataset):
    """CRCHisto images with synthetic radius-8 disks as instance supervision.

    CRCHisto publishes centroids only. Test metrics use the original centroids.
    """

    def __init__(self, names):
        self.names = list(names)
        self.cmap = CRCHISTO_CLASS_MAP
        self.root = crchisto_root()
        self.preprocessed = os.path.join(self.root, "crchisto_features_output", "step0_preprocessed")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, item):
        name = self.names[item]
        image_path = os.path.join(self.root, "CRCHisto_database", "Detection", name, f"{name}.bmp")
        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        with open(os.path.join(self.preprocessed, f"{name}_processed.json"), encoding="utf-8") as handle:
            record = json.load(handle)
        yy, xx = np.mgrid[0:height, 0:width]
        boxes, labels, masks = [], [], []
        for class_name, coordinates in record.get("classification_data", {}).items():
            if class_name not in self.cmap or not isinstance(coordinates, list):
                continue
            for coordinate in coordinates:
                if not (isinstance(coordinate, list) and len(coordinate) >= 2):
                    continue
                cx, cy = float(coordinate[0]), float(coordinate[1])
                x0, x1 = max(0, int(cx) - 9), min(width, int(cx) + 10)
                y0, y1 = max(0, int(cy) - 9), min(height, int(cy) + 10)
                selected = np.zeros((height, width), bool)
                selected[y0:y1, x0:x1] = (
                    (xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2 <= 64
                )
                ys, xs = np.nonzero(selected)
                if len(ys) < 3:
                    continue
                boxes.append([float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1])
                labels.append(self.cmap[class_name] + 1)
                masks.append(selected.astype(np.uint8))
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        return image_tensor, detection_target(boxes, labels, masks, item, (height, width))


def build_model(num_classes, small_anchors=True, coco_weights=None):
    """Build Mask R-CNN and optionally load the pinned COCO checkpoint.

    COCO weights require 91 classes, so those heads are created first and then
    replaced for this task.
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=91)
    if coco_weights:
        if not os.path.isfile(coco_weights):
            raise FileNotFoundError(
                "COCO Mask R-CNN checkpoint not found: "
                f"{coco_weights}. Set baseline_paths.maskrcnn_coco_weight."
            )
        state = torch.load(coco_weights, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        elif isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
        model.load_state_dict(state)
    if small_anchors:
        model.rpn.anchor_generator.sizes = ((8,), (16,), (32,), (64,), (128,))
    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, num_classes)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, num_classes)
    return model


def pannuke_rows(test_fold):
    """Return train and validation image indices for one PanNuke outer fold.

    Only rows for this outer fold are used. Rows stored under another outer
    fold include images from the current test fold.
    """
    manifest = pd.read_csv(split_manifest_path())
    rows = manifest[(manifest.dataset == "PanNuke") & (manifest.outer_fold == f"fold{test_fold}")]
    counts = rows.split.value_counts().to_dict()
    expected = PANNUKE_SPLIT_COUNTS[test_fold]
    if counts != expected:
        raise AssertionError(f"Unexpected PanNuke split counts for fold {test_fold}: {counts}")
    grouped = {"train": {}, "val": {}}
    for role in ("train", "val"):
        for image_id in rows[rows.split == role].image_id.tolist():
            match = re.match(r"fold(\d+)_image_(\d+)", image_id)
            if match is None:
                raise ValueError(f"Unrecognized PanNuke image id: {image_id}")
            fold = int(match.group(1))
            grouped[role].setdefault(fold, []).append(int(match.group(2)))
    for fold in (1, 2, 3):
        overlap = set(grouped["train"].get(fold, [])) & set(grouped["val"].get(fold, []))
        if overlap:
            raise AssertionError(f"PanNuke train/validation overlap in fold {fold}")
    return grouped["train"], grouped["val"], rows


def make_datasets(dataset_name, test_fold, for_inference):
    if dataset_name == "pannuke":
        if for_inference:
            return PanNukeDetect([test_fold], "test")
        train_rows, val_rows, manifest_rows = pannuke_rows(test_fold)
        # Validation images are stored in the other two fold arrays.
        train_set = PanNukeDetect([1, 2, 3], "train", split_rows=train_rows)
        val_set = PanNukeDetect([1, 2, 3], "val", split_rows=val_rows)
        if len(train_set) != int((manifest_rows.split == "train").sum()):
            raise AssertionError((len(train_set), "PanNuke training size mismatch"))
        if len(val_set) != int((manifest_rows.split == "val").sum()):
            raise AssertionError((len(val_set), "PanNuke validation size mismatch"))
        return train_set, val_set
    if dataset_name == "consep":
        roles = frozen_roles("consep")
        if for_inference:
            return CoNSePDetect(roles["test"])
        return CoNSePDetect(roles["train"]), CoNSePDetect(roles["val"])
    if dataset_name == "crchisto":
        roles = frozen_roles("crchisto")
        if for_inference:
            return CRCHistoDetect(roles["test"])
        return CRCHistoDetect(roles["train"]), CRCHistoDetect(roles["val"])
    raise ValueError(f"Unknown dataset: {dataset_name}")


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("Mask R-CNN training and inference require a CUDA GPU.")
    return "cuda"


def detection_f1(model, loader, device, iou=0.5):
    """Class-agnostic detection F1 used to select the checkpoint."""
    model.eval()
    true_positive = false_positive = false_negative = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            for target, output in zip(targets, outputs):
                predicted = output["boxes"][output["scores"] > 0.5]
                ground_truth = target["boxes"]
                matched = set()
                for box in predicted:
                    best_iou, best_index = 0.0, -1
                    for index, reference in enumerate(ground_truth):
                        if index in matched:
                            continue
                        x0 = max(box[0], reference[0])
                        y0 = max(box[1], reference[1])
                        x1 = min(box[2], reference[2])
                        y1 = min(box[3], reference[3])
                        intersection = max(0, x1 - x0) * max(0, y1 - y0)
                        union = (
                            (box[2] - box[0]) * (box[3] - box[1])
                            + (reference[2] - reference[0]) * (reference[3] - reference[1])
                            - intersection
                        )
                        overlap = intersection / max(union, 1)
                        if overlap > best_iou:
                            best_iou, best_index = overlap, index
                    if best_iou >= iou:
                        true_positive += 1
                        matched.add(best_index)
                    else:
                        false_positive += 1
                false_negative += len(ground_truth) - len(matched)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def train(dataset_name, test_fold, epochs, small_anchors):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = require_cuda()
    run_id = f"maskrcnn_{dataset_name}_testfold{test_fold}"
    output_dir = os.path.join(run_root(), run_id)
    os.makedirs(output_dir, exist_ok=True)
    log = open(os.path.join(output_dir, "train.log"), "a", buffering=1)

    def write_log(message):
        print(f"[{run_id}] {message}", flush=True)
        log.write(message + "\n")

    num_classes = 6 if dataset_name == "pannuke" else 5
    train_set, val_set = make_datasets(dataset_name, test_fold, for_inference=False)
    train_loader = DataLoader(
        train_set, batch_size=4, shuffle=True, num_workers=16,
        collate_fn=collate, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=2, shuffle=False, num_workers=8,
        collate_fn=collate, pin_memory=True, persistent_workers=True,
    )
    model = build_model(
        num_classes, small_anchors, coco_weights=resource("maskrcnn_coco_weight")
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)

    best_score = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = batches = 0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        scheduler.step()
        score = detection_f1(model, val_loader, device)
        print(
            f"[{run_id}] epoch {epoch}/{epochs} loss={total_loss / max(batches, 1):.4f} val_detF1={score:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, os.path.join(output_dir, "best_model.pth"))
    manifest_path = split_manifest_path()
    manifest = {
        "run_id": run_id,
        "impl": "torchvision maskrcnn_resnet50_fpn",
        "dataset": dataset_name,
        "test_fold": test_fold,
        "num_classes": num_classes,
        "anchors": "8/16/32/64/128",
        "epochs": epochs,
        "seed": SEED,
        "best_val_detF1": round(best_score, 4),
        "train_source_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(open(manifest_path, "rb").read()).hexdigest(),
    }
    with open(os.path.join(output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    write_log(f"DONE {json.dumps(manifest)}")
    log.close()


def image_key(dataset_name, dataset, image_id, test_fold):
    if dataset_name == "pannuke":
        return f"fold{test_fold}_image_{image_id}"
    return dataset.names[image_id]


def infer(dataset_name, test_fold, score_threshold, detection_cap=100):
    device = require_cuda()
    run_id = f"maskrcnn_{dataset_name}_testfold{test_fold}"
    output_dir = os.path.join(run_root(), run_id)
    model = build_model(6 if dataset_name == "pannuke" else 5, small_anchors=True)
    checkpoint = os.path.join(output_dir, "best_model.pth")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model = model.to(device).eval()
    model.roi_heads.score_thresh = score_threshold
    # The constructor argument is box_detections_per_img. The runtime attribute
    # is detections_per_img. Setting the constructor name leaves the default of 100.
    model.roi_heads.detections_per_img = detection_cap
    dataset = make_datasets(dataset_name, test_fold, for_inference=True)
    # CoNSeP predictions are made at 512 px and mapped back to the 1000 px image.
    scale = 1000 / CoNSePDetect.RESIZE if dataset_name == "consep" else 1.0
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate)
    rows = []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                key = image_key(dataset_name, dataset, image_id, test_fold)
                for label, mask, score in zip(
                    output["labels"].cpu(), output["masks"].cpu(), output["scores"].cpu()
                ):
                    binary = (mask[0] > 0.5).numpy()
                    if binary.sum() < 3:
                        continue
                    ys, xs = np.nonzero(binary)
                    rows.append({
                        "image_key": key,
                        "nucleus_id": sum(row["image_key"] == key for row in rows) + 1,
                        "cx": float(xs.mean()) * scale,
                        "cy": float(ys.mean()) * scale,
                        "pred": int(label) - 1,
                        "score": float(score),
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(output_dir, "pred_all.csv"), index=False)
    print(f"[{run_id}] infer done: {len(frame)} detections -> pred_all.csv", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run the Mask R-CNN baseline.")
    parser.add_argument("--dataset", default="pannuke", choices=["pannuke", "consep", "crchisto"])
    parser.add_argument("--test-fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument(
        "--det-cap", type=int, default=100,
        help="Maximum detections per image (RoIHeads.detections_per_img). "
             "The validation-selected cap is 1000 for every dataset.",
    )
    args = parser.parse_args()
    if args.train:
        train(args.dataset, args.test_fold, args.epochs, True)
    if args.infer:
        infer(args.dataset, args.test_fold, args.score_thr, args.det_cap)
