#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a within-fold Micro-Net detector on CRCHisto from random initialization.

Protocol:
  * Use the 80 CRCHisto training images. The image-level 68/12 split is the
    frozen validation split shared with the classifier.
  * Ground truth has centroids only. Each centroid becomes a disk of radius 8.
    Overlapping pixels stay with the instance drawn first. The segmentation
    target is the binary disk mask.
  * Input: random 256 crop, resized to 252, then MicroNet.preproc.
  * Loss: 0.5 BCE + 0.5 Dice on the main output, plus 0.25 times auxiliary CE.
  * Optimization: Adam 5e-5 for 60 epochs, batch 16, AMP, seed 42. The learning
    rate stays at 5e-5. The checkpoint is the best validation Dice.

Usage: python t06_train_micronet_crchisto.py [--smoke]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cellnura_release.baseline_runtime import frozen_roles, resource

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import micronet_standalone

CRC_DIR = resource("crchisto")
OUTROOT = resource("runs")
RUN_ID = "micronet_retrain_crchisto"
SEED = 42
TILE, MICRO_IN = 256, 252
DISK_R = 8
PATCHES_PER_IMG = 48
EPOCHS = 60
CLASS_MAP = {"inflammatory": 0, "epithelial": 1, "fibroblast": 2, "others": 3}


def load_split():
    roles = frozen_roles("crchisto")
    return roles["train"], roles["val"]


def load_binary_target(name):
    """Build a binary mask from radius-8 disks. Earlier disks keep overlapping pixels."""
    from PIL import Image
    record = json.load(open(
        f"{CRC_DIR}/crchisto_features_output/step0_preprocessed/{name}_processed.json",
        encoding="utf-8",
    ))
    image = np.array(Image.open(f"{CRC_DIR}/CRCHisto_database/Detection/{name}/{name}.bmp").convert("RGB"))
    height, width = image.shape[:2]
    annotation = np.zeros((height, width), dtype=np.int32)
    yy, xx = np.mgrid[0:height, 0:width]
    radius2 = DISK_R ** 2
    nxt = 1
    for label, coords in record.get("classification_data", {}).items():
        if label not in CLASS_MAP or not isinstance(coords, list):
            continue
        for coord in coords:
            if not (isinstance(coord, list) and len(coord) >= 2):
                continue
            cx, cy = float(coord[0]), float(coord[1])
            x0, x1 = max(0, int(cx) - DISK_R - 1), min(width, int(cx) + DISK_R + 2)
            y0, y1 = max(0, int(cy) - DISK_R - 1), min(height, int(cy) + DISK_R + 2)
            selected = (xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2 <= radius2
            selected &= annotation[y0:y1, x0:x1] == 0
            annotation[y0:y1, x0:x1][selected] = nxt
            nxt += 1
    return image.astype(np.uint8), (annotation > 0).astype(np.uint8)


class ImageCache:
    def __init__(self, names):
        self.data = {name: load_binary_target(name) for name in names}

    def __getitem__(self, name):
        return self.data[name]


class CRCPatch(Dataset):
    def __init__(self, cache, names, train, patches_per_img=PATCHES_PER_IMG, seed=SEED):
        self.cache, self.names, self.train = cache, names, train
        self.ppi = patches_per_img
        self.rng = np.random.RandomState(seed if train else seed + 1)

    def __len__(self):
        return len(self.names) * self.ppi

    def __getitem__(self, index):
        name = self.names[index % len(self.names)]
        image, target = self.cache[name]
        height, width = target.shape
        if self.train:
            top = self.rng.randint(0, height - TILE)
            left = self.rng.randint(0, width - TILE)
        else:
            rng = np.random.RandomState(index)
            top, left = rng.randint(0, height - TILE), rng.randint(0, width - TILE)
        image = image[top:top + TILE, left:left + TILE]
        target = target[top:top + TILE, left:left + TILE]
        if self.train:
            turns = self.rng.randint(0, 4)
            if turns:
                image, target = np.rot90(image, turns), np.rot90(target, turns)
            if self.rng.rand() < 0.5:
                image, target = image[:, ::-1], target[:, ::-1]
            if self.rng.rand() < 0.5:
                image, target = image[::-1], target[::-1]
        image = np.ascontiguousarray(image)
        target = np.ascontiguousarray(target)
        image = cv2.resize(image, (MICRO_IN, MICRO_IN), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (MICRO_IN, MICRO_IN), interpolation=cv2.INTER_NEAREST)
        value = micronet_standalone.MicroNet.preproc(image.astype(np.float32))
        return (
            torch.from_numpy(value.transpose(2, 0, 1).astype(np.float32)),
            torch.from_numpy((target > 0).astype(np.float32))[None],
        )


def dice_loss(prob, true, eps=1.0):
    numerator = 2 * (prob * true).sum(dim=(1, 2, 3)) + eps
    denominator = (prob * prob).sum(dim=(1, 2, 3)) + (true * true).sum(dim=(1, 2, 3)) + eps
    return (1 - numerator / denominator).mean()


def loss_fn(outputs, target):
    main, auxiliary = outputs[0], outputs[1:]
    prob = main[:, 1:2].clamp(1e-6, 1 - 1e-6)
    main_loss = 0.5 * F.binary_cross_entropy(prob, target) + 0.5 * dice_loss(prob, target)
    aux_loss = 0.0
    for head in auxiliary:
        probability = head.clamp(1e-6, 1 - 1e-6)
        aux_loss = aux_loss + (-torch.log(probability.gather(1, target.long().clamp(max=1)))).mean()
    return main_loss + 0.25 * aux_loss, {"main": main_loss.item(), "aux": float(aux_loss)}


@torch.no_grad()
def validate(model, loader):
    model.eval()
    intersection = predicted = truth = 0.0
    for image, target in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(image.cuda(non_blocking=True))
        prediction = (output[0].float()[:, 1] > 0.5).float().cpu()
        true = target[:, 0]
        intersection += (prediction * true).sum().item()
        predicted += prediction.sum().item()
        truth += true.sum().item()
    return 2 * intersection / max(predicted + truth, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    outdir = f"{OUTROOT}/{RUN_ID}"
    os.makedirs(outdir, exist_ok=True)
    log = open(f"{outdir}/train.log", "a", buffering=1, encoding="utf-8")

    def report(message):
        print(f"[{RUN_ID}] {message}", flush=True)
        log.write(message + "\n")

    train_names, val_names = load_split()
    epochs, patches = EPOCHS, PATCHES_PER_IMG
    if args.smoke:
        train_names, val_names = train_names[:3], val_names[:1]
        epochs, patches = 2, 8
    report(f"train={len(train_names)} images val={len(val_names)} images (image-level 68/12, r={DISK_R} disk targets)")
    cache = ImageCache(train_names + val_names)
    train_set = CRCPatch(cache, train_names, True, patches_per_img=patches)
    val_set = CRCPatch(cache, val_names, False, patches_per_img=16 if not args.smoke else 4)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=16, num_workers=4, pin_memory=True, persistent_workers=True)

    model = micronet_standalone.MicroNet(num_input_channels=3, num_output_channels=2).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    best = 0.0
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = batches = 0
        parts = {"main": 0.0, "aux": 0.0}
        for image, target in train_loader:
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(image)
            outputs = [item.float() for item in outputs]
            loss, parts = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        dice = validate(model, val_loader)
        report(
            f"ep {epoch}/{epochs} loss={total / max(batches, 1):.4f} "
            f"(main={parts['main']:.4f} aux={parts['aux']:.4f}) "
            f"val_dice={dice:.4f} ({time.time() - started:.0f}s)"
        )
        if dice > best:
            best = dice
            torch.save({"desc": model.state_dict()}, f"{outdir}/best_model.pth")
    manifest = {
        "run_id": RUN_ID,
        "impl": "MicroNet from scratch on CRCHisto training images, binary nucleus target from r=8 disks",
        "split": "image-level 68/12 frozen validation split",
        "input": f"random {TILE} crop -> resize {MICRO_IN} -> MicroNet.preproc",
        "loss": "main 0.5BCE+0.5Dice, aux CE x0.25 deep supervision",
        "lr": "5e-5 fixed",
        "epochs": epochs,
        "batch": 16,
        "patches_per_img": patches,
        "seed": SEED,
        "best_val_dice": round(best, 4),
        "train_images": train_names,
        "val_images": val_names,
        "code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
    }
    with open(f"{outdir}/run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    report(f"DONE best_val_dice={best:.4f}")
    log.close()


if __name__ == "__main__":
    main()
