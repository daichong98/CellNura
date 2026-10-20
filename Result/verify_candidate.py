#!/usr/bin/env python3
"""Verify the selected-method prediction-only Scheme-2 data candidate.

This verifies frozen bytes, table schemas, image coverage, class IDs and the
same-detection control. GT is deliberately absent and is not checked here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HEADER = ["image_id", "instance_id", "cx", "cy", "class_id"]


def safe_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest path: {value}")
    path = ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing file or symlink: {value}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image_list(path: Path) -> list[str]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image_id"]:
            raise ValueError(f"Wrong image-list header: {path}")
        images = [row["image_id"] for row in reader]
    if len(images) != len(set(images)) or not all(images):
        raise ValueError(f"Duplicate or empty image ID: {path}")
    return images


def verify_prediction(task: dict, image_ids: set[str]) -> int:
    path = safe_path(task["prediction"])
    if sha256(path) != task["prediction_sha256"]:
        raise ValueError(f"SHA mismatch: {task['job_id']}")
    allowed = set(task["allowed_class_ids"])
    seen = set()
    rows = 0
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != HEADER:
            raise ValueError(f"Wrong prediction schema: {task['job_id']}")
        for row in reader:
            image_id = row["image_id"]
            key = (image_id, row["instance_id"])
            if image_id not in image_ids or not row["instance_id"] or key in seen:
                raise ValueError(f"Unknown image/duplicate instance: {task['job_id']} {key}")
            seen.add(key)
            if not (math.isfinite(float(row["cx"])) and math.isfinite(float(row["cy"]))):
                raise ValueError(f"Non-finite coordinate: {task['job_id']} {key}")
            if int(row["class_id"]) not in allowed:
                raise ValueError(f"Invalid class ID: {task['job_id']} {key}")
            rows += 1
    if rows != task["prediction_rows"] or rows != task["expected"]["N_pred"]:
        raise ValueError(f"Row count mismatch: {task['job_id']}")
    return rows


def verify_same_detection(candidate: dict, base: dict) -> None:
    with safe_path(candidate["prediction"]).open(newline="") as ca, \
            safe_path(base["prediction"]).open(newline="") as ba:
        a, b = csv.DictReader(ca), csv.DictReader(ba)
        for index, (ar, br) in enumerate(zip_longest(a, b), 1):
            if ar is None or br is None:
                raise ValueError(f"Fixed-detection row count differs: {candidate['job_id']}")
            for field in ("image_id", "instance_id"):
                if ar[field] != br[field]:
                    raise ValueError(f"Fixed-detection {field} differs: {candidate['job_id']} row {index}")
            for field in ("cx", "cy"):
                if abs(float(ar[field]) - float(br[field])) > 1e-9:
                    raise ValueError(f"Fixed-detection coordinate differs: {candidate['job_id']} row {index}")


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    expected_ab = manifest["method_comparison_task_count"]
    expected_c = manifest["same_segmentation_attribution_task_count"]
    expected_total = expected_ab + expected_c
    if len(tasks) != expected_total or len({task["job_id"] for task in tasks}) != expected_total:
        raise ValueError(f"Expected {expected_total} unique tasks")
    if not (manifest["ground_truth_included"] is False and
            manifest["raw_images_included"] is False and
            manifest["weights_included"] is False):
        raise ValueError("The candidate must contain predictions only")

    image_lists = {}
    for name, item in manifest["images"].items():
        path = safe_path(item["path"])
        if path.name != name or sha256(path) != item["sha256"]:
            raise ValueError(f"Image list SHA mismatch: {name}")
        image_lists[item["path"]] = set(read_image_list(path))
        if len(image_lists[item["path"]]) != item["rows"]:
            raise ValueError(f"Image list size mismatch: {name}")

    split_register = manifest["training_split_manifest"]
    split_file = safe_path(split_register["path"])
    if sha256(split_file) != split_register["sha256"]:
        raise ValueError("Training split manifest SHA mismatch")
    role_sets = defaultdict(set)
    with split_file.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["dataset", "outer_fold", "image_id", "split", "seed", "note"]:
            raise ValueError("Wrong training split manifest schema")
        split_rows = 0
        for row in reader:
            if row["split"] not in {"train", "val", "test"}:
                raise ValueError("Unknown image role")
            key = (row["dataset"], row["outer_fold"], row["split"])
            if row["image_id"] in role_sets[key]:
                raise ValueError(f"Duplicate image role: {key}/{row['image_id']}")
            role_sets[key].add(row["image_id"])
            split_rows += 1
    if split_rows != split_register["rows"]:
        raise ValueError("Training split manifest row count differs")
    role_partitions = {(dataset, fold) for dataset, fold, role in role_sets}
    if len(role_partitions) != 5:
        raise ValueError("Expected five training role partitions")
    for dataset, fold in role_partitions:
        a, b, c = (role_sets[(dataset, fold, role)] for role in ("train", "val", "test"))
        if not (a and b and c) or a & b or a & c or b & c:
            raise ValueError(f"Image train/val/test overlap or empty role: {dataset}/{fold}")
        image_name = f"pannuke_{fold}.csv" if dataset == "PanNuke" else dataset.lower() + ".csv"
        if c != image_lists[f"images/{image_name}"]:
            raise ValueError(f"Test image inventory differs from role manifest: {dataset}/{fold}")

    checkpoint_register = manifest["segmentation_checkpoint_identities"]
    checkpoint_file = safe_path(checkpoint_register["path"])
    if sha256(checkpoint_file) != checkpoint_register["sha256"]:
        raise ValueError("Segmentation checkpoint identity table SHA mismatch")
    with checkpoint_file.open(newline="", encoding="utf-8") as stream:
        checkpoints = list(csv.DictReader(stream))
    if len(checkpoints) != 5 or len(checkpoints) != checkpoint_register["rows"]:
        raise ValueError("Expected five segmentation checkpoint identities")
    if {(row["dataset"], row["outer_fold"]) for row in checkpoints} != role_partitions:
        raise ValueError("Checkpoint and image-role fold registrations differ")
    for row in checkpoints:
        ds, fold = row["dataset"], row["outer_fold"]
        for role, field in (("train", "train_image_count"), ("val", "val_image_count"),
                            ("test", "test_image_count")):
            if int(row[field]) != len(role_sets[(ds, fold, role)]):
                raise ValueError(f"Checkpoint {role} count differs: {ds}/{fold}")
        if int(row["image_id_overlap_train_val_test"]) != 0:
            raise ValueError(f"Checkpoint role overlap: {ds}/{fold}")
        for field in ("checkpoint_sha256", "source_run_manifest_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", row[field]):
                raise ValueError(f"Invalid checkpoint evidence SHA: {ds}/{fold}/{field}")

    public_gt = manifest["expected_public_GT"]["files"]
    if set(public_gt) != set(manifest["images"]):
        raise ValueError("GT identity scope differs from the five test-image lists")
    reconstruction = manifest["ground_truth_rebuild_script"]
    if sha256(safe_path(reconstruction["path"])) != reconstruction["sha256"]:
        raise ValueError("GT reconstruction script SHA mismatch")
    registration_path = safe_path(reconstruction["expected_registration_path"])
    if sha256(registration_path) != reconstruction["expected_registration_sha256"]:
        raise ValueError("GT identity registration SHA mismatch")
    registered_gt = json.loads(registration_path.read_text(encoding="utf-8"))
    if registered_gt["files"] != public_gt:
        raise ValueError("GT identity registration differs from portable manifest")
    if manifest.get("local_gt_replay_validation"):
        raise ValueError("The pre-reduction replay record is not part of this candidate")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest["evaluator_sha256"]):
        raise ValueError("Invalid unified evaluator SHA")

    tier_counts = Counter()
    source_counts = Counter()
    rows_total = 0
    by_id = {task["job_id"]: task for task in tasks}
    for task in tasks:
        images = image_lists[task["image_list"]]
        if task["image_list_sha256"] != manifest["images"][Path(task["image_list"]).name]["sha256"]:
            raise ValueError(f"Task image-list SHA mismatch: {task['job_id']}")
        rows_total += verify_prediction(task, images)
        tier_counts["C" if task["tier"].startswith("C") else "AB"] += 1
        source_counts[task["source_group"]] += 1

    if tier_counts != {"AB": expected_ab, "C": expected_c}:
        raise ValueError(f"Wrong A/B/C task scope: {tier_counts}")
    if source_counts["frozen_task"] != manifest["historical_frozen_tasks_retained"]:
        raise ValueError("Historical prediction scope differs from manifest")
    if sum(count for source, count in source_counts.items() if source != "frozen_task") != manifest["new_or_corrected_tasks"] + expected_c:
        raise ValueError("Current and corrected prediction scope differs from manifest")
    if any(task["method_name"] in {"CellViT-256 scratch", "CellViT-256 official weights", "TSFD-Net public weights"} for task in tasks):
        raise ValueError("An excluded standalone method is still registered")
    method_groups = {(task["dataset"], task["method_name"], task["tier"]) for task in tasks}
    inventory = manifest["selected_method_inventory"]
    inventory_path = safe_path(inventory["path"])
    if sha256(inventory_path) != inventory["sha256"]:
        raise ValueError("Selected-method inventory SHA differs")
    with inventory_path.open(newline="", encoding="utf-8") as stream:
        inventory_rows = list(csv.DictReader(stream))
    if len(inventory_rows) != inventory["rows"] or len(inventory_rows) != len(method_groups):
        raise ValueError("Selected-method inventory row count differs")
    seen_inventory_groups = set()
    for row in inventory_rows:
        group = (row["dataset"], row["method_name"], row["tier"])
        if group not in method_groups or group in seen_inventory_groups:
            raise ValueError(f"Inventory has a missing or duplicate method group: {group}")
        seen_inventory_groups.add(group)
        group_tasks = [task for task in tasks if (task["dataset"], task["method_name"], task["tier"]) == group]
        if (int(row["n_prediction_tasks"]) != len(group_tasks) or
                set(row["job_ids"].split(";")) != {task["job_id"] for task in group_tasks}):
            raise ValueError(f"Selected-method inventory task IDs differ: {group}")
    if seen_inventory_groups != method_groups:
        raise ValueError("Selected-method inventory omits a prediction group")
    if any(not task.get("source_visibility") for task in tasks):
        raise ValueError("A prediction lacks source-table visibility registration")
    for task in tasks:
        if task["dataset"] == "PanNuke" and task["method"] == "cellvit_sam_B":
            note = task.get("source_alias", "")
            if not note.startswith("CellViT++") or "CellViT-SAM-H-x40" not in note:
                raise ValueError("PanNuke CellViT++ official-weight identity is not explicitly disclosed")

    for suffix in (
        "pannuke_fold1",
        "pannuke_fold2",
        "pannuke_fold3",
        "consep",
        "crchisto",
    ):
        verify_same_detection(by_id[f"CellNura_V3.7_{suffix}"], by_id[f"Cellvit_{suffix}"])

    with (ROOT / "expected_task_results.csv").open(newline="") as stream:
        aggregate = list(csv.DictReader(stream))
    if len(aggregate) != expected_total or {r["job_id"] for r in aggregate} != set(by_id):
        raise ValueError("Expected task result scope differs from manifest")
    for row in aggregate:
        task = by_id[row["job_id"]]
        expected = task["expected"]
        for field in ("N_gt", "N_pred", "M", "missed", "spurious"):
            if int(row[field]) != expected[field]:
                raise ValueError(f"Aggregate {field} differs: {task['job_id']}")
        if int(row["n_images"]) != len(image_lists[task["image_list"]]):
            raise ValueError(f"Aggregate image count differs: {task['job_id']}")
        for field, keys in {
            "cls_F1": ("cls_macro_real", "cls_F1"),
            "plusFN_F1": ("plusFN_macro_real", "plusFN_F1"),
            "unified_F1": ("uni_macro_real", "unified_F1", "uni_F1"),
        }.items():
            value = next(expected[key] for key in keys if key in expected)
            if abs(float(row[field]) - value) > 1e-12:
                raise ValueError(f"Aggregate {field} differs: {task['job_id']}")
    with (ROOT / "expected_per_class.csv").open(newline="") as stream:
        per_class = list(csv.DictReader(stream))
    expected_classes = sum(len(task["real_class_ids"]) for task in tasks)
    if len(per_class) != expected_classes:
        raise ValueError(f"Expected {expected_classes} method/fold-class rows, got {len(per_class)}")
    if any(r["prediction_sha256"] != by_id[r["job_id"]]["prediction_sha256"] for r in per_class):
        raise ValueError("Per-class prediction SHA differs from manifest")
    classes_per_task = defaultdict(set)
    for row in per_class:
        task = by_id[row["job_id"]]
        class_id = int(row["class_id"])
        if class_id not in task["real_class_ids"] or class_id in classes_per_task[task["job_id"]]:
            raise ValueError(f"Missing/duplicate per-class result: {task['job_id']} class {class_id}")
        classes_per_task[task["job_id"]].add(class_id)
        counts = {field: int(row[field]) for field in
                  ("N_gt_class", "N_pred_class", "TP", "FP_mis", "FP_spur", "FN_mis", "FN_miss")}
        tp = counts["TP"]
        if counts["N_gt_class"] != tp + counts["FN_mis"] + counts["FN_miss"]:
            raise ValueError(f"GT-class conservation failed: {task['job_id']} class {class_id}")
        if counts["N_pred_class"] != tp + counts["FP_mis"] + counts["FP_spur"]:
            raise ValueError(f"Pred-class conservation failed: {task['job_id']} class {class_id}")
        for prefix, fp, fn in (
            ("cls_", counts["FP_mis"], counts["FN_mis"]),
            ("plusFN_", counts["FP_mis"], counts["FN_mis"] + counts["FN_miss"]),
            ("uni_", counts["FP_mis"] + counts["FP_spur"], counts["FN_mis"] + counts["FN_miss"]),
        ):
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if any(abs(float(row[prefix + key]) - value) > 1e-10 for key, value in
                   (("P", precision), ("R", recall), ("F1", f1))):
                raise ValueError(f"Per-class metric/count mismatch: {task['job_id']} class {class_id} {prefix}")
        if task["tier"].startswith("C"):
            reference = task["expected"]["per_class"][str(class_id)]
            for key in ("P", "R", "F1"):
                if abs(float(row["uni_" + key]) - reference[key]) > 0.000051:
                    raise ValueError(f"C-layer saved metric mismatch: {task['job_id']} class {class_id}")
    if any(classes_per_task[task["job_id"]] != set(task["real_class_ids"]) for task in tasks):
        raise ValueError("Per-class result coverage differs from the registered class maps")

    forbidden = {"ground_truth", "raw_images", "weights", "eval_match_ledger", "__pycache__"}
    files = [p for p in ROOT.rglob("*") if p.is_file()]
    if any(part in forbidden for path in files for part in path.relative_to(ROOT).parts):
        raise ValueError("Forbidden data directory exists inside candidate")
    if any(path.suffix == ".pyc" for path in files):
        raise ValueError("Generated bytecode exists inside candidate")
    forbidden_suffixes = {".pth", ".pt", ".ckpt", ".npy", ".npz", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".mat"}
    if any(path.suffix.lower() in forbidden_suffixes for path in files):
        raise ValueError("Raw image, label, or model-weight file exists inside candidate")
    if any(p.is_symlink() for p in ROOT.rglob("*")):
        raise ValueError("Candidate contains symlinks")

    checksums = ROOT / "checksums.txt"
    if not checksums.is_file():
        raise ValueError("checksums.txt is missing")
    listed = {}
    for line in checksums.read_text(encoding="utf-8").splitlines():
        checksum, relative = line.split("  ", 1)
        if relative in listed or sha256(safe_path(relative)) != checksum:
            raise ValueError(f"Whole-package checksum mismatch: {relative}")
        listed[relative] = checksum
    generated = {checksums, ROOT / "verification_report.json"}
    actual = {path.relative_to(ROOT).as_posix() for path in files if path not in generated}
    if set(listed) != actual:
        raise ValueError(f"Whole-package file list differs: missing={actual-set(listed)}, extra={set(listed)-actual}")

    report = {
        "status": "PASS",
        "method_comparison_tasks": tier_counts["AB"],
        "same_segmentation_tasks": tier_counts["C"],
        "prediction_rows": rows_total,
        "image_lists": len(image_lists),
        "image_disjoint_role_partitions": len(role_partitions),
        "segmentation_checkpoint_identities": len(checkpoints),
        "per_class_rows": len(per_class),
        "gt_reconstructed_or_scored": False,
        "note": "Prediction, class-count, image-role and fixed-detection integrity checked. Use bundled rebuild_ground_truth.py and replay_candidate.py for GT-based numerical reproduction."
    }
    (ROOT / "verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
