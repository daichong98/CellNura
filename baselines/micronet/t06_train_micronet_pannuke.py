#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a within-fold Micro-Net detector on PanNuke from random initialization.

Protocol:
  * For each outer fold, train only on that fold's training images. Validation
    uses the frozen image-level split shared with the classifier.
  * Target: any positive pixel in the first five mask channels. PanNuke patches
    are 256 pixels.
  * Input: resize the whole patch from 256 to 252, then MicroNet.preproc.
    Rotations and flips are applied to the image and the target together.
  * Loss: 0.5 BCE + 0.5 Dice on the main output, plus 0.25 times auxiliary CE.
  * Optimization: Adam 1e-4, reduced to 1e-5 at epoch 48; 60 epochs; batch 32;
    AMP; seed 42; cuDNN deterministic. The checkpoint is the best validation Dice.

Usage: python t06_train_micronet_pannuke.py [--folds 1 2 3] [--smoke]
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cellnura_release.baseline_runtime import resource

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import micronet_standalone

PANNUKE = resource("pannuke")
OUTROOT = resource("runs")
SEED = 42
MICRO_IN = 252
EPOCHS = 60
IMAGE_ID = re.compile(r"^fold([123])_image_(\d+)$")
EXPECTED_COUNTS = {1: (4458, 787, 2656), 2: (4571, 807, 2523), 3: (4402, 777, 2722)}


def split_for(test_fold):
    """Return train and validation image IDs for one PanNuke outer fold."""
    path = resource("split_manifest")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["dataset"] == "PanNuke" and row["outer_fold"] == f"fold{test_fold}"
        ]
    roles = {row["image_id"]: row["split"] for row in rows}
    if len(roles) != len(rows) or len(roles) != 7901:
        raise ValueError(f"PanNuke fold {test_fold} split is incomplete or duplicated")
    train_ids = sorted(image_id for image_id, role in roles.items() if role == "train")
    val_ids = sorted(image_id for image_id, role in roles.items() if role == "val")
    test_ids = sorted(image_id for image_id, role in roles.items() if role == "test")
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise ValueError(f"PanNuke fold {test_fold} image roles overlap")
    expected = EXPECTED_COUNTS[test_fold]
    actual = (len(train_ids), len(val_ids), len(test_ids))
    if actual != expected:
        raise ValueError(f"Unexpected PanNuke fold {test_fold} counts: {actual}")
    return train_ids, val_ids


def parse_image_id(image_id):
    match = IMAGE_ID.match(image_id)
    if match is None:
        raise ValueError(f"Unrecognized PanNuke image id: {image_id}")
    return int(match.group(1)), int(match.group(2))


def load_arrays():
    images, masks = {}, {}
    for fold in (1, 2, 3):
        images[fold] = np.load(f"{PANNUKE}/Fold {fold}/images/fold{fold}/images.npy", mmap_mode="r")
        masks[fold] = np.load(f"{PANNUKE}/Fold {fold}/masks/fold{fold}/masks.npy", mmap_mode="r")
    return images, masks


class PanNukeWhole(Dataset):
    """Resize a 256 PanNuke patch to 252. Training applies geometric augmentation."""

    def __init__(self, image_ids, images, masks, train, seed=SEED):
        self.items = [parse_image_id(image_id) for image_id in image_ids]
        self.images, self.masks, self.train = images, masks, train
        self.rng = np.random.RandomState(seed if train else seed + 1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        fold, item = self.items[index]
        image = np.clip(np.asarray(self.images[fold][item]), 0, 255).astype(np.uint8)
        mask = np.asarray(self.masks[fold][item])
        target = (mask[..., :5] > 0).any(-1).astype(np.uint8)
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
    """Main output is 0.5 BCE + 0.5 Dice. Auxiliary heads use 0.25 CE."""
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
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    epochs = 2 if args.smoke else EPOCHS
    for test_fold in args.folds:
        run_id = f"micronet_retrain_pannuke_fold{test_fold}"
        outdir = f"{OUTROOT}/{run_id}"
        os.makedirs(outdir, exist_ok=True)
        log = open(f"{outdir}/train.log", "a", buffering=1, encoding="utf-8")

        def report(message, _run_id=run_id, _log=log):
            print(f"[{_run_id}] {message}", flush=True)
            _log.write(message + "\n")

        checkpoint = f"{outdir}/best_model.pth"
        manifest_path = f"{outdir}/run_manifest.json"
        if os.path.exists(checkpoint) and os.path.exists(manifest_path) and not args.smoke:
            report(f"fold {test_fold} already has best_model.pth and run_manifest.json; skipping")
            log.close()
            continue

        train_ids, val_ids = split_for(test_fold)
        report(f"fold {test_fold}: train={len(train_ids)} val={len(val_ids)}")
        images, masks = load_arrays()
        train_set = PanNukeWhole(train_ids, images, masks, True)
        val_set = PanNukeWhole(val_ids, images, masks, False)
        train_loader = DataLoader(
            train_set, batch_size=32, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True,
        )
        val_loader = DataLoader(
            val_set, batch_size=64, num_workers=6, pin_memory=True, persistent_workers=True,
        )

        model = micronet_standalone.MicroNet(num_input_channels=3, num_output_channels=2).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        best = 0.0
        started = time.time()
        for epoch in range(1, epochs + 1):
            if epoch == 48 and epochs == EPOCHS:
                for group in optimizer.param_groups:
                    group["lr"] = 1e-5
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
                torch.save({"desc": model.state_dict()}, checkpoint)
        manifest = {
            "run_id": run_id,
            "impl": "MicroNet from scratch on PanNuke outer-fold training images, binary nucleus target",
            "split": f"outer fold {test_fold} frozen image-level split",
            "input": "whole 256 patch -> resize 252 -> MicroNet.preproc",
            "loss": "main 0.5BCE+0.5Dice, aux CE x0.25 deep supervision",
            "epochs": epochs,
            "batch": 32,
            "seed": SEED,
            "best_val_dice": round(best, 4),
            "train_images": train_ids,
            "val_images": val_ids,
            "code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        report(f"DONE fold {test_fold} best_val_dice={best:.4f}")
        log.close()
        del model, train_set, val_set, train_loader, val_loader
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
