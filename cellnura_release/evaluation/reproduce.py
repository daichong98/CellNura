"""Recompute Tables 2/4/5 and statistics from explicitly supplied centroid artifacts."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import ArtifactError, load_catalog, load_input, sha256
from .unified_matcher import evaluate, write_outputs

DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


def metrics(result):
    return {"N_gt": result.n_gt, "N_pred": result.n_pred, "M": result.n_matched,
            "missed": result.n_missed, "spurious": result.n_spurious,
            "duplicate_claims": result.n_duplicate_claims,
            "n_images": len(result.per_image),
            "cls_macro_real": result.macro("classification"),
            "plusFN_macro_real": result.macro("plus_missed_FN"),
            "uni_macro_real": result.macro("unified")}


def check_expected(actual, expected, job_id):
    for key, value in expected.items():
        if key not in actual or abs(actual[key] - value) > 1e-12:
            raise ArtifactError(f"Frozen result mismatch for {job_id}/{key}: {actual.get(key)} != {value}")


def _evaluate_worker(payload):
    artifact_root, job, output = payload
    result = evaluate(load_input(artifact_root, job))
    values = metrics(result)
    check_expected(values, job["expected"], job["job_id"])
    folder = Path(output) / "jobs" / job["job_id"]
    folder.mkdir(parents=True, exist_ok=True)
    prefix = folder / "eval"
    write_outputs(result, str(prefix))
    (folder / "metrics.json").write_text(json.dumps(values, indent=2) + "\n")
    row = {"job_id": job["job_id"], "dataset": job["dataset"], "method": job["method"],
           "method_name": job["method_name"], "tier": job["tier"], "fold": job["fold"], **values}
    per_class = []
    for c in job["real_class_ids"]:
        data = result.classes[c]
        per_class.append({"job_id": job["job_id"], "dataset": job["dataset"], "method": job["method"],
                          "method_name": job["method_name"], "tier": job["tier"], "fold": job["fold"],
                          "class_id": c, "class_name": job["class_names"][str(c)],
                          "cls_F1": data["classification"]["F1"],
                          "plusFN_F1": data["plus_missed_FN"]["F1"],
                          "uni_F1": data["unified"]["F1"], **data["counts"]})
    files = {name: {"path": str((folder / filename).relative_to(output)),
                    "sha256": sha256(folder / filename)}
             for name, filename in (("ledger", "eval_match_ledger.csv"),
                                    ("per_image", "eval_per_image.csv"),
                                    ("per_class", "eval_per_class.csv"),
                                    ("metrics", "metrics.json"))}
    return row, per_class, files


def _sample_sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def write_tables(tables, output):
    """A primary comparisons and B references are never jointly ranked."""
    output = Path(output)
    for number, dataset in ((2, "PanNuke"), (4, "CoNSeP"), (5, "CRCHisto")):
        subset = tables[tables.dataset == dataset]
        primary = subset[subset.tier == "A"].sort_values("uni_macro_real_mean", ascending=False)
        references = subset[subset.tier != "A"].sort_values("method_name")
        table = pd.concat([primary, references], ignore_index=True)
        table.to_csv(output / f"Table{number}.csv", index=False)
        primary.to_csv(output / f"Table{number}_primary_A.csv", index=False)
        references.to_csv(output / f"Table{number}_reference_B.csv", index=False)
        lines = [f"# Table {number}: {dataset}", "", "True-class macro F1; unweighted fold means.",
                 "Primary comparisons and disclosed references are separated; no shared A/B ranking.", ""]
        for title, group in (("Primary comparison (tier A)", primary),
                             ("Disclosed references (tier B or unverified training split)", references)):
            lines.extend([f"## {title}", "", "| Method | Tier | Folds | Unified F1 | Sample SD |",
                          "|---|---|---:|---:|---:|"])
            for r in group.itertuples():
                sd = f"{r.uni_macro_real_sample_sd:.4f}" if r.n_folds > 1 else "—"
                lines.append(f"| {r.method_name} | {r.tier} | {r.n_folds} | {r.uni_macro_real_mean:.4f} | {sd} |")
            lines.append("")
        (output / f"Table{number}.md").write_text("\n".join(lines) + "\n")


def run_evaluation(artifact_root, catalog, output, workers=4, selected=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = catalog["jobs"]
    if selected:
        unknown = set(selected) - {j["job_id"] for j in jobs}
        if unknown:
            raise ArtifactError(f"Unknown job IDs: {sorted(unknown)}")
        jobs = [j for j in jobs if j["job_id"] in selected]
    rows, per_class, manifest = [], [], {}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for row, classes, files in pool.map(_evaluate_worker,
                [(str(artifact_root), j, str(output)) for j in jobs]):
            rows.append(row)
            per_class.extend(classes)
            manifest[row["job_id"]] = files
    df = pd.DataFrame(rows).sort_values(["dataset", "method", "fold"])
    df.to_csv(output / "per_fold_results.csv", index=False)
    table_rows = []
    for (dataset, method), group in df.groupby(["dataset", "method"], sort=True):
        entry = {"dataset": dataset, "method": method, "method_name": group.method_name.iloc[0],
                 "tier": group.tier.iloc[0], "n_folds": len(group),
                 "n_gt_total": int(group.N_gt.sum()), "n_pred_total": int(group.N_pred.sum()),
                 "n_matched_total": int(group.M.sum())}
        for key in ("cls_macro_real", "plusFN_macro_real", "uni_macro_real"):
            entry[key + "_mean"] = float(group[key].mean())
            entry[key + "_sample_sd"] = _sample_sd(group[key].to_numpy())
        table_rows.append(entry)
    tables = pd.DataFrame(table_rows)
    write_tables(tables, output)

    classes = pd.DataFrame(per_class).sort_values(["dataset", "method", "fold", "class_id"])
    classes.to_csv(output / "per_class_per_fold.csv", index=False)
    class_rows = []
    for keys, group in classes.groupby(["dataset", "method", "class_id"], sort=True):
        row = {"dataset": keys[0], "method": keys[1], "class_id": int(keys[2]),
               "class_name": group.class_name.iloc[0], "tier": group.tier.iloc[0], "n_folds": len(group)}
        for key in ("cls_F1", "plusFN_F1", "uni_F1"):
            row[key + "_mean"] = float(group[key].mean())
            row[key + "_sample_sd"] = _sample_sd(group[key].to_numpy())
        class_rows.append(row)
    pd.DataFrame(class_rows).to_csv(output / "per_class_mean_sample_sd.csv", index=False)
    em = {"schema_version": 1, "catalog_sha256": catalog["_catalog_sha256"],
          "job_count": len(jobs), "frozen_metrics_matched": len(rows),
          "metric_basis": "integer TP/FP/FN counts; macro restricted to each job's real_class_ids",
          "rounding_before_mean_or_sd": False, "sample_sd_ddof": 1,
          "aggregation": "unweighted mean of full-precision per-fold real-class macro F1",
          "jobs": manifest}
    (output / "evaluation_manifest.json").write_text(json.dumps(em, indent=2) + "\n")
    return {"jobs": len(jobs), "frozen_metrics_matched": len(rows), "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "evaluate", "statistics"):
        p = sub.add_parser(name)
        p.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
        if name != "verify":
            p.add_argument("--output", type=Path,
                           default=Path("reproduced") / ("evaluation" if name == "evaluate" else name))
            p.add_argument("--workers", type=int, default=4)
        if name == "evaluate":
            p.add_argument("--job", action="append", dest="selected")
            p.add_argument("--verify-only", action="store_true")
        if name == "statistics":
            p.add_argument("--evaluation-dir", type=Path, default=Path("reproduced/evaluation"))
            p.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args(argv)
    try:
        legacy_manifest = args.artifacts / "artifact_manifest.json"
        if args.command == "evaluate" and args.verify_only and not legacy_manifest.is_file():
            script = Path(__file__).resolve().parents[2] / "Result" / "verify_candidate.py"
            if not script.is_file():
                raise ArtifactError("Result/verify_candidate.py is missing")
            completed = subprocess.run([sys.executable, str(script)], cwd=script.parent)
            return completed.returncode
        pinned_manifest = DEFAULT_ARTIFACTS / "artifact_manifest.json"
        if pinned_manifest.is_file() and sha256(args.artifacts / "artifact_manifest.json") != sha256(pinned_manifest):
            raise ArtifactError("Supplementary manifest differs from this code release")
        catalog = load_catalog(args.artifacts)
        if args.command == "verify" or (args.command == "evaluate" and args.verify_only):
            result = {"verified": True, "jobs": len(catalog["jobs"])}
        elif args.command == "evaluate":
            result = run_evaluation(
                args.artifacts, catalog, args.output, args.workers, args.selected)
        else:
            from .statistics import run_statistics
            result = run_statistics(args.artifacts, catalog, args.evaluation_dir, args.output,
                                    args.iterations, args.workers)
    except (ArtifactError, OSError, KeyError) as error:
        parser.exit(2, f"Evaluation refused: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
