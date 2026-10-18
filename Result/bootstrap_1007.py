#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
from replay_candidate import read_instances
from verify_candidate import ROOT, read_image_list, safe_path, sha256, verify_prediction


COMPARISONS = [
    ("pannuke_micronet_retrain", "PanNuke", "A", "Micro-Net", "pannuke_micronet_retrain", (0.2238, 0.2172, 0.2303)),
    ("pannuke_stardist", "PanNuke", "A", "StarDist", "pannuke_stardist", (0.1678, 0.1587, 0.1759)),
    ("pannuke_hovernet", "PanNuke", "A", "HoVer-Net v3", "pannuke_hovernet_v3", (0.1336, 0.1263, 0.1405)),
    ("pannuke_dist", "PanNuke", "A", "DIST", "pannuke_dist", (0.1024, 0.0956, 0.1091)),
    ("pannuke_tsfd", "PanNuke", "A", "TSFD-Net", "pannuke_tsfd_retrain", (0.0875, 0.0787, 0.0959)),
    ("pannuke_maskrcnn", "PanNuke", "A", "Mask R-CNN", "pannuke_maskrcnn", (0.0559, 0.0488, 0.0645)),
    ("pannuke_cellvit", "PanNuke", "A", "CellViT", "Cellvit_pannuke", (0.0275, 0.0215, 0.0330)),
    ("pannuke_hovernet_official", "PanNuke", "B", "HoVer-Net official weights", "pannuke_tphead", (0.1475, 0.1388, 0.1559)),
    ("pannuke_cellvitpp", "PanNuke", "B", "CellViT++ official weights", "pannuke_cellvit_sam_B", (-0.1954, -0.2063, -0.1869)),
    ("consep_maskrcnn", "CoNSeP", "A", "Mask R-CNN", "consep_maskrcnn", (0.3522, 0.2387, 0.3904)),
    ("consep_micronet_retrain", "CoNSeP", "A", "Micro-Net", "consep_micronet_retrain", (0.2369, 0.1304, 0.2753)),
    ("consep_hovernet", "CoNSeP", "A", "HoVer-Net", "consep_hovernet_foldtrain", (0.1381, 0.0383, 0.1839)),
    ("consep_cellvit", "CoNSeP", "A", "CellViT", "Cellvit_consep", (0.0868, 0.0188, 0.1190)),
    ("consep_dist", "CoNSeP", "A", "DIST", "consep_dist", (0.0930, -0.0058, 0.1349)),
    ("consep_hovernet_official", "CoNSeP", "B", "HoVer-Net official weights", "consep_hovernet_official", (-0.1196, -0.1670, -0.0908)),
    ("consep_cellvitpp", "CoNSeP", "B", "CellViT++ official weights", "consep_cellvitpp", (-0.1380, -0.1776, -0.1122)),
    ("crchisto_micronet_retrain", "CRCHisto", "A", "Micro-Net", "crchisto_micronet_retrain", (0.2902, 0.1937, 0.3557)),
    ("crchisto_maskrcnn", "CRCHisto", "A", "Mask R-CNN", "crchisto_maskrcnn", (0.2891, 0.1853, 0.3415)),
    ("crchisto_dist", "CRCHisto", "A", "DIST", "crchisto_dist", (0.2162, 0.1314, 0.2697)),
    ("crchisto_cellvit", "CRCHisto", "A", "CellViT", "Cellvit_crchisto", (0.0498, 0.0021, 0.0848)),
    ("crchisto_hovernet", "CRCHisto", "A", "HoVer-Net", "crchisto_hovernet_foldtrain", (0.0293, -0.0064, 0.0552)),
    ("crchisto_hovernet_official", "CRCHisto", "B", "HoVer-Net official weights", "crchisto_hovernet_official", (0.1986, 0.1241, 0.2484)),
    ("crchisto_cellvitpp", "CRCHisto", "B", "CellViT++ official weights", "crchisto_cellvitpp", (0.1926, 0.0978, 0.2451)),
]
STATISTICS_SHA256 = "c406732f53e3e1b3a9d80649a81bab7d0d1e76fc8e8f64e98c560834482b61de"
CSV_FIELDS = ["comparison_id", "dataset", "tier", "opponent", "base_job_ids", "opponent_job_ids",
              "n_images", "point_delta", "ci_low", "ci_high", "bootstrap_fraction_positive",
              "reference_point", "reference_low", "reference_high", "matches_reference_4dp",
              "iterations", "seed"]


def job_ids(dataset: str, name: str) -> list[str]:
    if dataset == "PanNuke":
        return [f"{name}_fold{fold}" for fold in (1, 2, 3)]
    return [name]


def base_ids(dataset: str) -> list[str]:
    """CellNura V3.7."""
    if dataset == "PanNuke":
        return [f"CellNura_V3.7_pannuke_fold{fold}" for fold in (1, 2, 3)]
    if dataset == "CoNSeP":
        return ["CellNura_V3.7_consep"]
    return ["CellNura_V3.7_crchisto"]


def image_order(dataset: str, ids: list[str]) -> list[str]:
    if dataset == "PanNuke":
        return sorted(ids, key=lambda value: int(value.rsplit("_", 1)[1]))
    return sorted(ids)


def matrix_for_task(task: dict, gt: list, image_ids: list[str], eval_input_cls, evaluate_fn,
                    instance_cls) -> np.ndarray:
    """Recreate statistics.ledger_matrix from an in-memory one-to-one evaluation."""
    prediction = read_instances(safe_path(task["prediction"]), instance_cls)
    result = evaluate_fn(eval_input_cls(
        dataset=task["dataset"], tau=float(task["matching_radius_px"]),
        gt=gt, pred=prediction, image_ids=tuple(image_ids),
        class_ids=tuple(task["allowed_class_ids"]),
        real_class_ids=tuple(task["real_class_ids"])), mode="one_to_one")
    wanted = task["expected"]
    for observed, key in ((result.n_gt, "N_gt"), (result.n_pred, "N_pred"),
                          (result.n_matched, "M"), (result.n_missed, "missed"),
                          (result.n_spurious, "spurious")):
        if observed != wanted[key]:
            raise ValueError(f"{task['job_id']} {key}: observed {observed} != registered {wanted[key]}")
    real = tuple(task["real_class_ids"])
    registered_unified = wanted.get("uni_macro_real", wanted.get("unified_F1", wanted.get("uni_F1")))
    if registered_unified is None or abs(result.macro("unified", real_class_ids=real) - registered_unified) > 0.000051:
        raise ValueError(f"{task['job_id']} unified macro F1 differs from its registered result")
    index = {image_id: i for i, image_id in enumerate(image_ids)}
    classes = {class_id: i for i, class_id in enumerate(real)}
    matrix = np.zeros((len(image_ids), len(real), 3), dtype=np.int64)
    for row in result.match_ledger:
        i = index[row["image_id"]]
        gc = row["gt_cls"]
        pc = row["pred_cls"]
        if gc in classes:
            matrix[i, classes[gc], 1] += 1
        if pc in classes:
            matrix[i, classes[pc], 2] += 1
        if row["match_status"] == "matched-correct" and gc in classes and gc == pc:
            matrix[i, classes[gc], 0] += 1
    for class_id, ci in classes.items():
        counts = result.classes[class_id]["counts"]
        actual = tuple(int(value) for value in matrix[:, ci, :].sum(0))
        expected = (counts["TP"], counts["N_gt"], counts["N_pred"])
        if actual != expected:
            raise ValueError(f"{task['job_id']} class {class_id} ledger totals {actual} != {expected}")
    ground_truth_by_image = np.zeros(matrix.shape[:2], dtype=np.int64)
    for item in gt:
        if item.cls in classes:
            ground_truth_by_image[index[item.image_id], classes[item.cls]] += 1
    if not np.array_equal(matrix[:, :, 1], ground_truth_by_image):
        raise ValueError(f"{task['job_id']} per-image GT counts differ from public annotations")
    del result, prediction
    return matrix


def four_decimals_match(computed: dict, reference: tuple[float, float, float] | None) -> bool | None:
    if reference is None:
        return None
    return all(f"{computed[key]:.4f}" == f"{value:.4f}" for key, value in
               zip(("point_delta", "ci_low", "ci_high"), reference))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", required=True, type=Path,
                        help="Five public-annotation GT CSVs rebuilt outside this candidate")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Write aggregate CI CSV/JSON outside this candidate")
    parser.add_argument("--code-root", type=Path,
                        default=ROOT.parent)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", action="append", choices=[item[0] for item in COMPARISONS],
                        help="Run only the named comparison; repeat for multiple comparisons")
    parser.add_argument("--require-reference-match", action="store_true",
                        help="Return nonzero if any four-decimal manuscript value differs")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    gt_dir, out_dir = args.gt_dir.resolve(), args.out_dir.resolve()
    for label, path in (("GT", gt_dir), ("output", out_dir)):
        if path == ROOT or ROOT in path.parents:
            parser.error(f"{label} directory must be outside the prediction candidate")
    if gt_dir == out_dir:
        parser.error("GT and output directories must differ")
    code_root = args.code_root.resolve()
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    tasks = {task["job_id"]: task for task in manifest["tasks"]}
    if len(tasks) != 46 or len(manifest["tasks"]) != 46:
        raise ValueError("The candidate must contain 46 distinct registered tasks")
    matcher = code_root / "cellnura_release/evaluation/unified_matcher.py"
    statistics = code_root / "cellnura_release/evaluation/statistics.py"
    if not matcher.is_file() or sha256(matcher) != manifest["evaluator_sha256"]:
        raise ValueError("Companion unified matcher missing or SHA-256 differs")
    if not statistics.is_file() or sha256(statistics) != STATISTICS_SHA256:
        raise ValueError("Companion image-paired statistics code missing or SHA-256 differs")
    sys.path.insert(0, str(code_root))
    from cellnura_release.evaluation.statistics import paired_bootstrap
    from cellnura_release.evaluation.unified_matcher import EvalInput, Instance, evaluate

    # Read and authenticate every registered prediction task, including methods
    # outside the selected manuscript comparisons. This ties the CI run to one bundle.
    inventories = {}
    for task in manifest["tasks"]:
        name = task["image_list"]
        if name not in inventories:
            path = safe_path(name)
            if sha256(path) != task["image_list_sha256"]:
                raise ValueError(f"Image-inventory SHA differs: {name}")
            inventories[name] = read_image_list(path)
        verify_prediction(task, set(inventories[name]))
    expected_gt = manifest["expected_public_GT"]["files"]
    for name, identity in expected_gt.items():
        path = gt_dir / name
        if not path.is_file() or path.stat().st_size != identity["bytes"] or sha256(path) != identity["sha256"]:
            raise ValueError(f"Registered public GT missing or SHA-256 differs: {name}")

    selected = [item for item in COMPARISONS if args.only is None or item[0] in args.only]
    matrix_cache = {}
    gt_cache = {}

    def get_matrix(job_id: str) -> np.ndarray:
        if job_id in matrix_cache:
            return matrix_cache[job_id]
        task = tasks[job_id]
        image_file = Path(task["image_list"]).name
        if image_file not in gt_cache:
            gt_cache[image_file] = read_instances(gt_dir / image_file, Instance)
            if len(gt_cache[image_file]) != expected_gt[image_file]["rows"]:
                raise ValueError(f"Registered public GT row count differs: {image_file}")
        images = image_order(task["dataset"], inventories[task["image_list"]])
        matrix = matrix_for_task(task, gt_cache[image_file], images, EvalInput, evaluate, Instance)
        matrix_cache[job_id] = matrix
        return matrix

    rows = []
    for number, (slug, dataset, tier, label, opponent, reference) in enumerate(selected, 1):
        paired_base_ids = base_ids(dataset)
        other_ids = job_ids(dataset, opponent)
        base, other = [], []
        for b_id, o_id in zip(paired_base_ids, other_ids):
            if b_id not in tasks or o_id not in tasks:
                raise ValueError(f"Missing paired jobs: {b_id}, {o_id}")
            b_task, o_task = tasks[b_id], tasks[o_id]
            if b_task["method_name"] != "CellNura V3.7":
                raise ValueError(f"Bootstrap base is not CellNura V3.7: {b_id}")
            if (b_task["dataset"] != dataset or o_task["dataset"] != dataset or
                    not str(b_task["tier"]).startswith("C") or o_task["tier"] != tier or
                    b_task["image_list"] != o_task["image_list"] or
                    b_task["real_class_ids"] != o_task["real_class_ids"] or
                    b_task["matching_radius_px"] != o_task["matching_radius_px"]):
                raise ValueError(f"Pair protocol differs: {b_id}, {o_id}")
            base.append(get_matrix(b_id))
            other.append(get_matrix(o_id))
        result = paired_bootstrap(base, other, iterations=args.iterations, seed=args.seed)
        comparable = args.iterations == 10000 and args.seed == 42
        matched = four_decimals_match(result, reference) if comparable else None
        row = {"comparison_id": slug, "dataset": dataset, "tier": tier, "opponent": label,
               "base_job_ids": ";".join(paired_base_ids), "opponent_job_ids": ";".join(other_ids),
               "n_images": sum(len(item) for item in base), **result,
               "reference_point": "" if reference is None else reference[0],
               "reference_low": "" if reference is None else reference[1],
               "reference_high": "" if reference is None else reference[2],
               "matches_reference_4dp": matched,
               "iterations": args.iterations, "seed": args.seed}
        rows.append(row)
        print(f"[{number:02d}/{len(selected):02d}] {slug}: Δ={result['point_delta']:.4f} "
              f"CI=[{result['ci_low']:.4f}, {result['ci_high']:.4f}] "
              f"reference_match={matched}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "paired_bootstrap_1007.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    referenced = [row for row in rows if row["reference_point"] != ""]
    all_match = bool(referenced) and all(row["matches_reference_4dp"] is True for row in referenced)
    full_scope = len(selected) == len(COMPARISONS) and len(referenced) == len(COMPARISONS)
    status = ("PASS" if full_scope else "PARTIAL_PASS") if all_match else "UNVERIFIED_OR_DIFFERENT"
    report = {"status": status, "scope": "image-paired real-class Unified macro-F1 delta",
              "comparison_count": len(rows), "source_prediction_tasks_sha_verified": len(tasks),
              "iterations": args.iterations, "seed": args.seed,
              "resampling_unit": "image/tile, paired between models; PanNuke stratified by outer fold",
              "reference_checked": args.iterations == 10000 and args.seed == 42,
              "full_comparison_scope": full_scope,
              "comparison_count_defined": len(COMPARISONS),
              "reference_matches_4dp": sum(row["matches_reference_4dp"] is True for row in rows),
              "mismatches": [row["comparison_id"] for row in rows if row["matches_reference_4dp"] is False],
              "matcher_sha256": sha256(matcher), "statistics_sha256": sha256(statistics),
              "public_gt_sha256": {name: identity["sha256"] for name, identity in expected_gt.items()},
              "gt_or_match_ledger_written": False}
    (out_dir / "bootstrap_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.require_reference_match and not all_match:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
