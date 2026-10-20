from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, RandomSampler, WeightedRandomSampler

from cellnura_release.baseline_runtime import resource
from preflight import HERE, RAW, SOURCE, MANIFEST, read_roles, raw_paths, sha256, source_fold

sys.path.insert(0, str(SOURCE))
import torchmetrics.functional as torchmetrics_functional  # noqa: E402

# This archived CellViT trainer imports torchmetrics.functional.dice, removed
# in the installed torchmetrics 1.8. For binary 0/1 masks, foreground Dice is
# the binary F1 score. Keep the compatibility alias local to this entry point.
if not hasattr(torchmetrics_functional, "dice"):
    def _binary_foreground_dice(preds, target, ignore_index=0):
        if ignore_index != 0:
            raise ValueError("Compatibility Dice alias only supports ignore_index=0")
        return torchmetrics_functional.classification.binary_f1_score(preds, target)

    torchmetrics_functional.dice = _binary_foreground_dice

from cell_segmentation.datasets.pannuke import PanNukeDataset  # noqa: E402
from cell_segmentation.experiments.experiment_cellvit_pannuke import ExperimentCellVitPanNuke  # noqa: E402
from cell_segmentation.utils.metrics import remap_label  # noqa: E402
from models.segmentation.cell_segmentation.cellvit import CellViT256  # noqa: E402


OFFICIAL_CONFIG = (SOURCE / "logs_paper/PanNuke/CellViTHV/ViT256/Best-Setting/Fold-1/config.yaml")
COUNT_ROOT = SOURCE / "configs/datasets/PanNuke"
DATASET_CONFIG_ROOT = COUNT_ROOT


class ManifestRawPanNuke(Dataset):
    """Read only the supplied image IDs from original PanNuke arrays.

    Targets match the official PanNukeDataset item structure. Memmaps are opened
    lazily in each DataLoader worker; neither validation nor test arrays are
    scanned to construct training statistics.
    """

    def __init__(self, image_ids: list[str], transforms=None):
        self.image_ids = list(image_ids)
        self.transforms = transforms
        self._arrays = {}
        self._tissues = {}
        if len(self.image_ids) != len(set(self.image_ids)):
            raise ValueError("Duplicate image ID in dataset")
        for fold in {source_fold(item)[0] for item in self.image_ids}:
            self._tissues[fold] = np.load(raw_paths(fold)[2], allow_pickle=True)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        fold, image_index = source_fold(image_id)
        if fold not in self._arrays:
            image_path, mask_path, _ = raw_paths(fold)
            self._arrays[fold] = (np.load(image_path, mmap_mode="r"),
                                  np.load(mask_path, mmap_mode="r"))
        images, masks = self._arrays[fold]
        image = images[image_index].astype(np.uint8)
        source_mask = masks[image_index]

        instance = np.zeros((256, 256), dtype=np.int32)
        type_map = np.zeros((256, 256), dtype=np.int32)
        offset = 0
        for cls in range(5):
            layer = remap_label(source_mask[:, :, cls].astype(np.int32))
            instance = np.where(layer > 0, layer + offset, instance)
            offset += int(layer.max())
            type_map[source_mask[:, :, cls] > 0] = cls + 1
        instance = remap_label(instance)
        mask = np.stack((instance, type_map), axis=-1)
        if self.transforms is not None:
            result = self.transforms(image=image, mask=mask)
            image, mask = result["image"], result["mask"]

        instance = mask[:, :, 0].copy()
        type_map = mask[:, :, 1].copy()
        binary = (instance > 0).astype(np.int64)
        hv = PanNukeDataset.gen_instance_hv_map(instance)
        tensor = torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32).permute(2, 0, 1)
        if tensor.max() >= 5:
            tensor = tensor / 255
        targets = {
            "instance_map": torch.as_tensor(np.ascontiguousarray(instance), dtype=torch.int64),
            "nuclei_type_map": torch.as_tensor(np.ascontiguousarray(type_map), dtype=torch.int64),
            "nuclei_binary_map": torch.as_tensor(np.ascontiguousarray(binary), dtype=torch.int64),
            "hv_map": torch.as_tensor(np.ascontiguousarray(hv), dtype=torch.float32),
        }
        return tensor, targets, str(self._tissues[fold][image_index]), image_id


def train_only_weights(image_ids: list[str], gamma: float) -> torch.Tensor:
    """Recompute official cell+tissue sampling formula from train IDs only."""
    if not 0 <= gamma <= 1:
        raise ValueError("sampling_gamma must lie in [0, 1]")
    counts = {}
    for fold in {source_fold(item)[0] for item in image_ids}:
        counts[fold] = pd.read_csv(COUNT_ROOT / f"fold{fold - 1}/cell_count.csv", index_col=0)
    cells = []
    tissues = []
    tissue_arrays = {fold: np.load(raw_paths(fold)[2], allow_pickle=True)
                     for fold in counts}
    for image_id in image_ids:
        fold, index = source_fold(image_id)
        row = counts[fold].loc[f"{fold - 1}_{index}.png"].to_numpy(dtype=np.float64)
        cells.append(np.clip(row, 0, 1))
        tissues.append(str(tissue_arrays[fold][index]))
    presence = np.asarray(cells, dtype=np.float64)
    class_frequency = presence.sum(axis=0)
    class_total = class_frequency.sum()
    cell_factor = class_total / (gamma * class_frequency + (1 - gamma) * class_total)
    cell_weights = (1 - gamma) * presence.max(axis=1) + gamma * (presence * cell_factor).sum(axis=1)
    positive = cell_weights[cell_weights > 0]
    cell_weights[cell_weights == 0] = positive.min() if len(positive) else 1.0
    tissue_frequency = Counter(tissues)
    n = len(tissues)
    tissue_weights = np.asarray([n / (gamma * tissue_frequency[t] + (1 - gamma) * n)
                                 for t in tissues], dtype=np.float64)
    weights = tissue_weights / tissue_weights.max() + cell_weights / cell_weights.max()
    return torch.as_tensor(weights, dtype=torch.float32)


class FrozenCellViT256Experiment(ExperimentCellVitPanNuke):
    def __init__(self, configuration: dict, roles: dict[str, list[str]], scratch_init: bool):
        self.frozen_roles = roles
        self.scratch_init = scratch_init
        super().__init__(default_conf=configuration)

    def get_datasets(self, train_transforms=None, val_transforms=None):
        train_ids = self.frozen_roles["train"]
        val_ids = self.frozen_roles["val"]
        assert not set(train_ids) & set(val_ids)
        return (ManifestRawPanNuke(train_ids, train_transforms),
                ManifestRawPanNuke(val_ids, val_transforms))

    def get_sampler(self, train_dataset, strategy="random", gamma=1):
        if strategy.lower() == "random":
            generator = torch.Generator().manual_seed(self.default_conf["random_seed"])
            return RandomSampler(train_dataset, generator=generator)
        if strategy.lower() != "cell+tissue":
            raise ValueError("Supported sampling strategies: random, cell+tissue")
        weights = train_only_weights(train_dataset.image_ids, gamma)
        generator = torch.Generator().manual_seed(self.default_conf["random_seed"])
        return WeightedRandomSampler(weights, num_samples=len(train_dataset),
                                     replacement=True, generator=generator)

    def get_train_model(self, pretrained_encoder=None, pretrained_model=None,
                        backbone_type="ViT256", shared_decoders=False,
                        regression_loss=False):
        if not self.scratch_init:
            return super().get_train_model(
                pretrained_encoder=pretrained_encoder, pretrained_model=pretrained_model,
                backbone_type=backbone_type, shared_decoders=shared_decoders,
                regression_loss=regression_loss)
        if shared_decoders or regression_loss or backbone_type.lower() != "vit256":
            raise ValueError("Scratch initialization supports only the unshared ViT256 configuration")
        model = CellViT256(model256_path=None, num_nuclei_classes=6,
                           num_tissue_classes=19,
                           drop_rate=self.run_conf["training"].get("drop_rate", 0),
                           attn_drop_rate=self.run_conf["training"].get("attn_drop_rate", 0),
                           drop_path_rate=self.run_conf["training"].get("drop_path_rate", 0))
        return model


def build_config(args, roles):
    config = yaml.safe_load(OFFICIAL_CONFIG.read_text())
    config["data"]["dataset_path"] = str(DATASET_CONFIG_ROOT)
    config["data"]["train_folds"] = [f - 1 for f in (1, 2, 3) if f != args.outer_fold]
    config["data"]["val_folds"] = list(config["data"]["train_folds"])
    config["data"]["test_folds"] = [args.outer_fold - 1]
    config["logging"]["mode"] = "disabled"
    config["logging"]["project"] = "CellViT256-frozen-PanNuke"
    config["logging"]["notes"] = f"Frozen outer fold {args.outer_fold}; manifest image-level train/val"
    config["logging"]["log_comment"] = f"CellViT256-frozen-fold{args.outer_fold}"
    config["logging"]["wandb_dir"] = str(args.output_root / "wandb")
    config["logging"]["log_dir"] = str(args.output_root / f"outer_fold{args.outer_fold}" / "runs")
    config["gpu"] = args.gpu
    config["run_sweep"] = False
    config["agent"] = None
    config["model"].pop("pretrained", None)
    if args.smoke_run or args.init == "scratch":
        config["model"].pop("pretrained_encoder", None)
        config["training"]["unfreeze_epoch"] = 0
        config["logging"]["log_comment"] += "-SCRATCH"
    elif args.encoder_checkpoint:
        config["model"]["pretrained_encoder"] = str(args.encoder_checkpoint.resolve())
    if args.smoke_run:
        config["training"].update(epochs=1, batch_size=2, unfreeze_epoch=0,
                                  sampling_strategy="cell+tissue", mixed_precision=False,
                                  eval_every=1, early_stopping_patience=None)
        config["logging"]["log_comment"] += "-SMOKE"
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-fold", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--init", choices=("hipt", "scratch"), default="scratch",
                        help="HIPT teacher or independent random initialization for full training")
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(resource("runs"))/"cellvit256")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke-run", action="store_true",
                        help="One random-init epoch on two train and two val images")
    parser.add_argument("--launch-full", action="store_true",
                        help="Explicitly launch 130-epoch training with audited encoder")
    args = parser.parse_args()
    if sum((args.validate_only, args.smoke_run, args.launch_full)) != 1:
        parser.error("Choose exactly one of --validate-only, --smoke-run, --launch-full")
    if args.launch_full and args.init == "hipt" and not args.encoder_checkpoint:
        parser.error("HIPT training requires --encoder-checkpoint; audit its training data first")
    if args.init == "scratch" and args.encoder_checkpoint:
        parser.error("Scratch initialization must not receive --encoder-checkpoint")
    if args.encoder_checkpoint and not args.encoder_checkpoint.is_file():
        parser.error(f"Encoder checkpoint missing: {args.encoder_checkpoint}")

    roles = read_roles(args.outer_fold)
    if args.smoke_run:
        # Archived trainer uses torch.squeeze without a dim on the type map;
        # batch size 1 loses the batch axis. Two images exercise the real
        # trainer while avoiding that unrelated upstream bug.
        roles = {"train": roles["train"][:2], "val": roles["val"][:2],
                 "test": roles["test"]}
    config = build_config(args, roles)
    manifest_frame = pd.read_csv(MANIFEST)
    one_fold = manifest_frame.loc[(manifest_frame["dataset"] == "PanNuke") &
                                  (manifest_frame["outer_fold"] == f"fold{args.outer_fold}")]
    report = {
        "outer_fold": args.outer_fold,
        "manifest_sha256": sha256(MANIFEST),
        "role_counts": {key: len(value) for key, value in roles.items()},
        "test_gt_loaded_during_training": False,
        "encoder_checkpoint": str(args.encoder_checkpoint) if args.encoder_checkpoint else None,
        "encoder_sha256": sha256(args.encoder_checkpoint) if args.encoder_checkpoint else None,
        "initialization": "scratch" if args.smoke_run or args.init == "scratch" else "hipt",
        "smoke_random_init": bool(args.smoke_run),
        "training_sampler": config["training"]["sampling_strategy"],
        "status": "validated_only" if args.validate_only else "starting",
    }
    if args.validate_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    outer_dir = args.output_root / f"outer_fold{args.outer_fold}"
    outer_dir.mkdir(parents=True, exist_ok=True)
    one_fold.to_csv(outer_dir / "split_manifest_snapshot.csv", index=False)
    with (outer_dir / "role_image_ids.json").open("w") as handle:
        json.dump(roles, handle, ensure_ascii=False, indent=2)
    with (outer_dir / "resolved_config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    with (outer_dir / "run_manifest.json").open("w") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    experiment = FrozenCellViT256Experiment(
        config, roles, scratch_init=args.smoke_run or args.init == "scratch")
    run_dir = experiment.run_experiment()
    checkpoint=Path(run_dir)/"checkpoints/latest_checkpoint.pth"
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    report.update(status="training_finished", run_dir_relative=str(Path(run_dir).relative_to(args.output_root)), checkpoint_relative=str(checkpoint.relative_to(args.output_root)), checkpoint_sha256=sha256(checkpoint))
    with (outer_dir / "run_manifest.json").open("w") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
