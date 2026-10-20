from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


from cellnura_release.baseline_runtime import resource
HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = Path(resource('project_root'))
WORKSPACE = EXPERIMENT_ROOT
SOURCE = Path(resource('vendor_cellvit'))
MANIFEST = Path(resource('split_manifest'))
RAW = Path(resource('pannuke'))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_fold(image_id: str) -> tuple[int, int]:
    prefix, separator, index = image_id.partition("_image_")
    if separator != "_image_" or prefix not in {"fold1", "fold2", "fold3"}:
        raise ValueError(f"Unexpected PanNuke image_id: {image_id}")
    return int(prefix.removeprefix("fold")), int(index)


def read_roles(outer_fold: int) -> dict[str, list[str]]:
    frame = pd.read_csv(MANIFEST)
    frame = frame.loc[(frame["dataset"] == "PanNuke") &
                      (frame["outer_fold"] == f"fold{outer_fold}")].copy()
    assert len(frame) == frame["image_id"].nunique() == 7901
    roles = {role: frame.loc[frame["split"] == role, "image_id"].tolist()
             for role in ("train", "val", "test")}
    sets = [set(roles[role]) for role in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert {source_fold(item)[0] for item in roles["test"]} == {outer_fold}
    assert not {source_fold(item)[0] for item in roles["train"] + roles["val"]} & {outer_fold}
    assert sum(map(len, roles.values())) == 7901
    return roles


def raw_paths(fold: int) -> tuple[Path, Path, Path]:
    root = RAW / f"Fold {fold}"
    return (root / f"images/fold{fold}/images.npy",
            root / f"masks/fold{fold}/masks.npy",
            root / f"images/fold{fold}/types.npy")


def write_one_sample(image_id: str, target_root: Path, dataset_fold: int) -> str:
    """Convert one original array entry using the official PanNuke mask convention."""
    original_fold, index = source_fold(image_id)
    image_path, mask_path, tissue_path = raw_paths(original_fold)
    for path in (image_path, mask_path, tissue_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    image = np.load(image_path, mmap_mode="r")[index]
    mask = np.load(mask_path, mmap_mode="r")[index]
    tissue = str(np.load(tissue_path, allow_pickle=True)[index])
    if image.shape != (256, 256, 3) or mask.shape != (256, 256, 6):
        raise ValueError(f"Unexpected shapes for {image_id}: {image.shape}, {mask.shape}")

    sys.path.insert(0, str(SOURCE))
    from cell_segmentation.utils.metrics import remap_label

    instance_map = np.zeros((256, 256), dtype=np.int32)
    type_map = np.zeros((256, 256), dtype=np.int32)
    offset = 0
    for cls in range(5):
        layer = remap_label(mask[:, :, cls].astype(np.int32))
        instance_map = np.where(layer > 0, layer + offset, instance_map)
        offset += int(layer.max())
        type_map[mask[:, :, cls] > 0] = cls + 1
    instance_map = remap_label(instance_map)

    fold_root = target_root / f"fold{dataset_fold}"
    (fold_root / "images").mkdir(parents=True, exist_ok=True)
    (fold_root / "labels").mkdir(parents=True, exist_ok=True)
    filename = f"{dataset_fold}_{index}.png"
    Image.fromarray(image.astype(np.uint8)).save(fold_root / "images" / filename)
    np.save(fold_root / "labels" / filename.replace(".png", ".npy"),
            {"inst_map": instance_map, "type_map": type_map})
    with (fold_root / "types.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("img", "type"))
        writer.writerow((filename, tissue))
    return filename


def run_smoke(outer_fold: int, device: str) -> dict:
    import torch
    import torch.nn.functional as functional

    sys.path.insert(0, str(SOURCE))
    from cell_segmentation.datasets.pannuke import PanNukeDataset
    from models.segmentation.cell_segmentation.cellvit import CellViT256

    roles = read_roles(outer_fold)
    torch.set_num_threads(4)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cellvit256_smoke_") as temporary:
        root = Path(temporary)
        train_name = write_one_sample(roles["train"][0], root, 0)
        val_name = write_one_sample(roles["val"][0], root, 1)
        train = PanNukeDataset(root, folds=[0])
        val = PanNukeDataset(root, folds=[1])
        image, target, tissue, loaded_name = train[0]
        assert loaded_name == train_name and len(val) == 1
        assert target["nuclei_type_map"].shape == (256, 256)
        assert target["hv_map"].shape == (2, 256, 256)
        assert int(target["nuclei_type_map"].max()) <= 5

        model = CellViT256(model256_path=None, num_nuclei_classes=6,
                           num_tissue_classes=19).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        x = ((image.unsqueeze(0).to(device) - 0.5) / 0.5)
        output = model(x)
        losses = {
            "binary": functional.cross_entropy(output["nuclei_binary_map"],
                                               target["nuclei_binary_map"].unsqueeze(0).to(device)),
            "type": functional.cross_entropy(output["nuclei_type_map"],
                                             target["nuclei_type_map"].unsqueeze(0).to(device)),
            "hv": functional.mse_loss(output["hv_map"],
                                      target["hv_map"].unsqueeze(0).to(device)),
        }
        loss = sum(losses.values())
        assert torch.isfinite(loss).item()
        loss.backward()
        gradient_count = sum(p.grad is not None for p in model.parameters())
        assert gradient_count > 0
        optimizer.step()
        model.eval()
        val_image, val_target, _, loaded_val_name = val[0]
        assert loaded_val_name == val_name
        with torch.inference_mode():
            val_out = model(((val_image.unsqueeze(0).to(device) - 0.5) / 0.5))
        assert val_out["nuclei_type_map"].shape == (1, 6, 256, 256)
        assert val_target["nuclei_type_map"].shape == (256, 256)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return {
            "status": "PASS", "scope": "one real train/val image; random-init technical smoke only",
            "outer_fold": outer_fold, "train_image_id": roles["train"][0],
            "val_image_id": roles["val"][0], "test_images_loaded": 0,
            "train_loss": float(loss.detach().cpu()),
            "loss_components": {key: float(value.detach().cpu()) for key, value in losses.items()},
            "gradient_parameter_tensors": gradient_count,
            "type_output_shape": list(val_out["nuclei_type_map"].shape),
            "device": device, "torch_version": torch.__version__,
            "seconds": round(time.monotonic() - start, 2),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-fold", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--smoke", action="store_true", help="Run one real train/val gradient step")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "preflight_20260924.json")
    args = parser.parse_args()

    all_roles = {f"fold{fold}": {role: len(items) for role, items in read_roles(fold).items()}
                 for fold in (1, 2, 3)}
    report = {
        "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "roles": all_roles, "source": str(SOURCE),
        "public_x40_checkpoint_is_not_independent_for_pannuke": True,
    }
    if args.smoke:
        report["smoke"] = run_smoke(args.outer_fold, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
