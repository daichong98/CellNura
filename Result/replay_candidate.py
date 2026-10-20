#!/usr/bin/env python3
"""Re-evaluate every current Scheme-2 prediction against reconstructed public GT.

No GT instances, per-image matches, or per-instance ledgers are written.
Outputs are aggregate task, class, and method tables in a caller-chosen directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True  # Keep the detachable candidate free of __pycache__.
from verify_candidate import ROOT, read_image_list, safe_path, sha256


TASK_FIELDS = ["job_id", "dataset", "tier", "method_name", "fold", "N_gt", "N_pred",
               "M", "missed", "spurious", "n_images", "cls_F1", "plusFN_F1", "unified_F1"]
CLASS_FIELDS = ["job_id", "dataset", "tier", "fold", "class_id", "class_name",
                "N_gt_class", "N_pred_class", "TP", "FP_mis", "FP_spur", "FN_mis", "FN_miss",
                "cls_P", "cls_R", "cls_F1", "plusFN_P", "plusFN_R", "plusFN_F1",
                "uni_P", "uni_R", "uni_F1"]
METHOD_FIELDS = ["dataset", "tier", "method_name", "n_test_folds", "job_ids", "N_gt",
                 "N_pred", "M", "missed", "spurious", "M_div_N_gt", "cls_F1_fold_mean",
                 "plusFN_F1_fold_mean", "unified_F1_fold_mean", "unified_F1_fold_sample_SD",
                 "unified_F1_display", "unified_F1_SD_display"]


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_instances(path: Path, instance_cls: type) -> list:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image_id", "instance_id", "cx", "cy", "class_id"]:
            raise ValueError(f"Wrong instance CSV schema: {path}")
        return [instance_cls(row["image_id"], row["instance_id"], float(row["cx"]),
                             float(row["cy"]), int(row["class_id"])) for row in reader]


def reference_scores(expected: dict) -> dict[str, float]:
    return {
        "cls_F1": expected.get("cls_macro_real", expected.get("cls_F1")),
        "plusFN_F1": expected.get("plusFN_macro_real", expected.get("plusFN_F1")),
        "unified_F1": expected.get("uni_macro_real", expected.get("unified_F1", expected.get("uni_F1"))),
    }


def evaluate_task(task: dict, gt: list, image_ids: list[str], expected_class: dict,
                  eval_input_cls: type, evaluate_fn, instance_cls: type) -> tuple[dict, list[dict]]:
    pred_path = safe_path(task["prediction"])
    if sha256(pred_path) != task["prediction_sha256"]:
        raise ValueError(f"Prediction SHA differs: {task['job_id']}")
    pred = read_instances(pred_path, instance_cls)
    if len(pred) != task["prediction_rows"]:
        raise ValueError(f"Prediction row count differs: {task['job_id']}")
    real = tuple(task["real_class_ids"])
    result = evaluate_fn(eval_input_cls(
        dataset=task["dataset"], tau=float(task["matching_radius_px"]),
        gt=gt, pred=pred, image_ids=tuple(image_ids),
        class_ids=tuple(task["allowed_class_ids"]), real_class_ids=real
    ), mode="one_to_one")
    counts = {"N_gt": result.n_gt, "N_pred": result.n_pred, "M": result.n_matched,
              "missed": result.n_missed, "spurious": result.n_spurious}
    for key, value in counts.items():
        if value != task["expected"][key]:
            raise ValueError(f"{task['job_id']} {key}: observed={value} expected={task['expected'][key]}")
    scores = {"cls_F1": result.macro("classification", real_class_ids=real),
              "plusFN_F1": result.macro("plus_missed_FN", real_class_ids=real),
              "unified_F1": result.macro("unified", real_class_ids=real)}
    for key, value in scores.items():
        reference = reference_scores(task["expected"])[key]
        if abs(value - reference) > 0.000051:
            raise ValueError(f"{task['job_id']} {key}: observed={value} expected={reference}")
    task_row = {"job_id": task["job_id"], "dataset": task["dataset"], "tier": task["tier"],
                "method_name": task["method_name"], "fold": task["fold"] or "",
                **counts, "n_images": len(image_ids), **scores}
    class_rows = []
    names = MANIFEST["class_maps"][task["dataset"]]["class_names"]
    for class_id in real:
        item = result.classes[class_id]
        c = item["counts"]
        registered = expected_class.get((task["job_id"], class_id))
        if registered is None:
            raise ValueError(f"Unregistered class: {task['job_id']} class {class_id}")
        class_row = {"job_id": task["job_id"], "dataset": task["dataset"], "tier": task["tier"],
                     "fold": task["fold"] or "", "class_id": class_id, "class_name": names[str(class_id)],
                     "N_gt_class": c["N_gt"], "N_pred_class": c["N_pred"],
                     "TP": c["TP"], "FP_mis": c["FP_mis"], "FP_spur": c["FP_spur"],
                     "FN_mis": c["FN_mis"], "FN_miss": c["FN_miss"]}
        for source, prefix in (("classification", "cls_"), ("plus_missed_FN", "plusFN_"),
                               ("unified", "uni_")):
            class_row.update({prefix + key: item[source][key] for key in ("P", "R", "F1")})
        for key in ("N_gt_class", "N_pred_class", "TP", "FP_mis", "FP_spur", "FN_mis", "FN_miss"):
            if class_row[key] != int(registered[key]):
                raise ValueError(f"{task['job_id']} class {class_id} integer {key} differs")
        for prefix in ("cls_", "plusFN_", "uni_"):
            for key in ("P", "R", "F1"):
                if abs(class_row[prefix + key] - float(registered[prefix + key])) > 1e-10:
                    raise ValueError(f"{task['job_id']} class {class_id} {prefix}{key} differs")
        class_rows.append(class_row)
    # The evaluator keeps per-image and per-instance records in RAM, but this
    # script intentionally writes only aggregate rows after validation.
    del result
    return task_row, class_rows


def method_summary(task_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in task_rows:
        groups[(row["dataset"], row["tier"], row["method_name"])].append(row)
    output = []
    for (dataset, tier, name), folds in groups.items():
        folds.sort(key=lambda row: str(row["fold"]))
        n = len(folds)
        if n != (3 if dataset == "PanNuke" else 1):
            raise ValueError(f"Incomplete method/fold group: {dataset}/{name}: {n}")
        aggregated = {key: sum(row[key] for row in folds)
                      for key in ("N_gt", "N_pred", "M", "missed", "spurious")}
        f1s = [row["unified_F1"] for row in folds]
        mean_f1 = statistics.mean(f1s)
        sd_f1 = statistics.stdev(f1s) if n > 1 else None
        output.append({"dataset": dataset, "tier": tier, "method_name": name,
                       "n_test_folds": n, "job_ids": ";".join(row["job_id"] for row in folds),
                       **aggregated, "M_div_N_gt": aggregated["M"] / aggregated["N_gt"],
                       "cls_F1_fold_mean": statistics.mean(row["cls_F1"] for row in folds),
                       "plusFN_F1_fold_mean": statistics.mean(row["plusFN_F1"] for row in folds),
                       "unified_F1_fold_mean": mean_f1, "unified_F1_fold_sample_SD": sd_f1,
                       "unified_F1_display": f"{mean_f1:.4f}",
                       "unified_F1_SD_display": f"{sd_f1:.4f}" if sd_f1 is not None else ""})
    return sorted(output, key=lambda row: (row["dataset"], row["tier"], row["method_name"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", required=True, type=Path,
                        help="Directory containing five SHA-matched reconstructed public GT CSVs")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Write aggregate replay outputs here, outside this prediction package")
    parser.add_argument("--code-root", type=Path,
                        default=ROOT.parent,
                        help="CellNura code release root, the parent of this Result directory")
    args = parser.parse_args()
    for label, directory in (("GT", args.gt_dir), ("output", args.out_dir)):
        if directory.resolve() == ROOT or ROOT in directory.resolve().parents:
            raise ValueError(f"{label} directory must be outside the prediction-only candidate")
    if args.gt_dir.resolve() == args.out_dir.resolve():
        raise ValueError("The replay output directory must differ from the reconstructed GT directory")

    global MANIFEST
    MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    tasks = MANIFEST["tasks"]
    expected_ab = MANIFEST["method_comparison_task_count"]
    expected_c = MANIFEST["same_segmentation_attribution_task_count"]
    expected_total = expected_ab + expected_c
    expected_class_count = sum(len(task["real_class_ids"]) for task in tasks)
    if len(tasks) != expected_total:
        raise ValueError(f"Expected exactly {expected_total} selected prediction tasks")
    matcher = args.code_root / "cellnura_release" / "evaluation" / "unified_matcher.py"
    if not matcher.is_file() or sha256(matcher) != MANIFEST["evaluator_sha256"]:
        raise ValueError("Companion unified matcher missing or SHA-256 differs")
    sys.path.insert(0, str(args.code_root))
    from cellnura_release.evaluation.unified_matcher import EvalInput, Instance, evaluate

    gt_cache = {}
    for filename, registered in MANIFEST["expected_public_GT"]["files"].items():
        path = args.gt_dir / filename
        if not path.is_file() or path.stat().st_size != registered["bytes"] or sha256(path) != registered["sha256"]:
            raise ValueError(f"Public GT missing or SHA-256 differs: {path}")
        gt_cache[filename] = read_instances(path, Instance)
        if len(gt_cache[filename]) != registered["rows"]:
            raise ValueError(f"Public GT row count differs: {filename}")

    with (ROOT / "expected_per_class.csv").open(newline="", encoding="utf-8") as stream:
        expected_rows = list(csv.DictReader(stream))
    expected_class = {(row["job_id"], int(row["class_id"])): row for row in expected_rows}
    if len(expected_rows) != expected_class_count or len(expected_class) != expected_class_count:
        raise ValueError(f"Expected {expected_class_count} unique task-class registrations")

    task_rows, class_rows = [], []
    for index, task in enumerate(tasks, 1):
        image_path = safe_path(task["image_list"])
        if sha256(image_path) != task["image_list_sha256"]:
            raise ValueError(f"Image inventory SHA-256 differs: {task['job_id']}")
        images = read_image_list(image_path)
        gt_name = image_path.name
        row, classes = evaluate_task(task, gt_cache[gt_name], images, expected_class,
                                     EvalInput, evaluate, Instance)
        task_rows.append(row)
        class_rows.extend(classes)
        print(f"[{index:02d}/{expected_total}] {task['job_id']} PASS", flush=True)
    if len(class_rows) != expected_class_count:
        raise ValueError(f"Expected {expected_class_count} evaluated task-class rows, got {len(class_rows)}")

    methods = method_summary(task_rows)
    ab = [row for row in methods if not row["tier"].startswith("C")]
    c = [row for row in methods if row["tier"].startswith("C")]
    inventory = MANIFEST["selected_method_inventory"]
    inventory_path = safe_path(inventory["path"])
    if sha256(inventory_path) != inventory["sha256"]:
        raise ValueError("Selected-method inventory SHA differs")
    with inventory_path.open(newline="", encoding="utf-8") as stream:
        inventory_rows = list(csv.DictReader(stream))
    expected_ab = sum(not row["tier"].startswith("C") for row in inventory_rows)
    expected_c = sum(row["tier"].startswith("C") for row in inventory_rows)
    if len(ab) != expected_ab or len(c) != expected_c:
        raise ValueError(f"Expected {expected_ab} A/B and {expected_c} C method rows, got {len(ab)}/{len(c)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "task_results.csv", TASK_FIELDS, task_rows)
    write_csv(args.out_dir / "per_class_results.csv", CLASS_FIELDS, class_rows)
    write_csv(args.out_dir / "method_results_AB.csv", METHOD_FIELDS, ab)
    write_csv(args.out_dir / "Table2_PanNuke_AB.csv", METHOD_FIELDS,
              [row for row in ab if row["dataset"] == "PanNuke"])
    write_csv(args.out_dir / "Table4_CoNSeP_AB.csv", METHOD_FIELDS,
              [row for row in ab if row["dataset"] == "CoNSeP"])
    write_csv(args.out_dir / "Table5_CRCHisto_AB.csv", METHOD_FIELDS,
              [row for row in ab if row["dataset"] == "CRCHisto"])
    write_csv(args.out_dir / "same_segmentation_C.csv", METHOD_FIELDS, c)
    report = {"status": "PASS", "task_count": len(task_rows), "per_class_rows": len(class_rows),
              "method_rows_AB": len(ab), "method_rows_C": len(c),
              "matched_protocol": "class-independent max-cardinality/min-distance one-to-one centroid matching",
              "matcher_sha256": MANIFEST["evaluator_sha256"],
              "public_gt_sha256": {name: item["sha256"] for name, item in MANIFEST["expected_public_GT"]["files"].items()},
              "gt_or_match_ledger_written": False,
              "note": "This replays prediction scoring. It does not prove segmentation checkpoint or training split independence."}
    (args.out_dir / "replay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
