"""Paired image bootstrap and matching/sensitivity statistics; CPU only."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .artifacts import ArtifactError, checked_file, load_input, read_image_ids, read_instances
from .unified_matcher import evaluate


def macro_counts(tp, gt, pred):
    denominator = gt + pred
    return float(np.divide(2 * tp, denominator, out=np.zeros(len(tp), dtype=float),
                           where=denominator > 0).mean())


def paired_bootstrap(base, opponent, iterations=10000, seed=42):
    """Per-fold real-class macro F1, then unweighted fold mean.

    Inputs contain aligned (TP,GT,pred) arrays, shape images x real classes x 3.
    Images, rather than nuclei, are resampled together for both fixed models.
    """
    if iterations < 1 or len(base) != len(opponent) or not base:
        raise ArtifactError("Invalid bootstrap input or iteration count")
    point_base, point_opponent = [], []
    for b, o in zip(base, opponent):
        if b.shape != o.shape or len(b) == 0:
            raise ArtifactError("Bootstrap models have different image universes")
        if not np.array_equal(b[:, :, 1], o[:, :, 1]):
            raise ArtifactError("Paired models must have identical per-image GT")
        point_base.append(macro_counts(*b.sum(0).T))
        point_opponent.append(macro_counts(*o.sum(0).T))
    point = float(np.mean(point_base) - np.mean(point_opponent))
    rng = np.random.RandomState(seed)
    deltas = np.empty(iterations, dtype=float)
    for t in range(iterations):
        fold_deltas = []
        for b, o in zip(base, opponent):
            idx = rng.randint(0, len(b), len(b))
            # Integer multiplicities preserve the exact randint experiment;
            # weighted integer sums avoid allocating three sampled count arrays.
            weights = np.bincount(idx, minlength=len(b)).astype(np.int64)
            bc = np.tensordot(weights, b, axes=(0, 0))
            oc = np.tensordot(weights, o, axes=(0, 0))
            fold_deltas.append(macro_counts(*bc.T) - macro_counts(*oc.T))
        deltas[t] = np.mean(fold_deltas)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"point_delta": point, "ci_low": float(low), "ci_high": float(high),
            "bootstrap_fraction_positive": float((deltas > 0).mean())}


def ledger_matrix(artifact_root, job, evaluation_root, evaluation_manifest):
    entry = evaluation_manifest["jobs"][job["job_id"]]["ledger"]
    path = checked_file(evaluation_root, entry)
    ids = read_image_ids(artifact_root, job["images"])
    if job["dataset"] == "PanNuke":
        ids = tuple(sorted(ids, key=lambda x: int(x.rsplit("_", 1)[1])))
    else:
        ids = tuple(sorted(ids))
    real = tuple(job["real_class_ids"])
    im_index = {im: i for i, im in enumerate(ids)}
    class_index = {c: i for i, c in enumerate(real)}
    result = np.zeros((len(ids), len(real), 3), dtype=np.int64)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["image_id"] not in im_index:
                raise ArtifactError("Ledger contains an image outside the frozen manifest")
            i = im_index[row["image_id"]]
            gc = int(row["gt_cls"]) if row["gt_cls"] != "" else None
            pc = int(row["pred_cls"]) if row["pred_cls"] != "" else None
            if gc in class_index:
                result[i, class_index[gc], 1] += 1
            if pc in class_index:
                result[i, class_index[pc], 2] += 1
            if row["match_status"].startswith("matched") and gc == pc and gc in class_index:
                result[i, class_index[gc], 0] += 1
    gt_reference = np.zeros(result.shape[:2], dtype=np.int64)
    for g in read_instances(artifact_root, job["ground_truth"], ids, real, ground_truth=True):
        gt_reference[im_index[g.image_id], class_index[g.cls]] += 1
    if not np.array_equal(result[:, :, 1], gt_reference):
        raise ArtifactError("Ledger dropped or duplicated GT instances")
    return result


def _bootstrap_worker(payload):
    comparison, base, opponent, iterations, seed = payload
    row = paired_bootstrap(base, opponent, iterations, seed)
    row.update({"dataset": comparison["dataset"], "opponent": comparison["opponent"],
                "opponent_tier": comparison["tier"], "iterations": iterations, "seed": seed})
    return row


def _matching_worker(payload):
    root, job = payload
    ev = load_input(root, job)
    old_new = []
    for mode in ("one_to_one", "old_nn", "legacy_r1"):
        result = evaluate(ev, mode)
        old_new.append({"method": job["method"], "dataset": job["dataset"],
                        "fold": job["fold"], "mode": mode, "N_gt": result.n_gt,
                        "N_pred": result.n_pred, "M": result.n_matched,
                        "missed": result.n_missed, "spurious": result.n_spurious,
                        "dup_claims": result.n_duplicate_claims,
                        "cls_macroF1": result.macro("classification"),
                        "uni_macroF1": result.macro("unified")})
    sensitivity = []
    for tau in job["tau_values"]:
        ev.tau = tau
        result = evaluate(ev)
        sensitivity.append({"method": job["method"], "dataset": job["dataset"],
                            "fold": job["fold"], "tau": tau, "n_gt": result.n_gt,
                            "n_pred": result.n_pred, "M": result.n_matched,
                            "missed": result.n_missed, "spurious": result.n_spurious,
                            "cls_macro_real": result.macro("classification"),
                            "plusFN_macro_real": result.macro("plus_missed_FN"),
                            "uni_macro_real": result.macro("unified")})
    return old_new, sensitivity


def validate_reference(catalog, bootstrap, iterations):
    """Compare to the archived paper reference without altering computed values.

    Point estimates and CI endpoints must match at the published four decimals.
    Positive-difference fractions are diagnostics; record any difference and
    allow at most one of the requested bootstrap draws. The archived CoNSeP
    DIST ledger has two predicted types distributed differently across images
    from the registered prediction source; this cancels in global scores but
    changes one positive bootstrap draw. It is a source consolidation difference,
    not a hypothesis-test P value and not an excuse to change model outputs.
    """
    mismatches, fraction_differences = [], []
    checked = iterations == catalog["bootstrap_reference_iterations"]
    if checked:
        for comparison, row in zip(catalog["bootstrap_comparisons"], bootstrap):
            for key, expected in comparison["expected"].items():
                if round(float(row[key]), 4) == expected:
                    continue
                difference = {"dataset": row["dataset"], "opponent": row["opponent"],
                              "metric": key, "computed": float(row[key]), "expected": expected}
                if key == "bootstrap_fraction_positive" and abs(row[key] - expected) <= 1 / iterations + 1e-12:
                    fraction_differences.append(difference)
                else:
                    mismatches.append(difference)
    return {"reference_checked": checked, "reference_mismatches": mismatches,
            "reference_fraction_tolerance": 1 / iterations,
            "reference_fraction_differences": fraction_differences,
            "reference_all_values_identical_4dp": checked and not mismatches and not fraction_differences,
            "all_point_estimates_and_ci_endpoints_match_4dp": checked and not any(
                r["metric"] != "bootstrap_fraction_positive" for r in mismatches)}


def validate_matching_references(artifact_root, old, sensitivity):
    result = {}
    for name, actual, keys, columns in (
        ("old_new_comparison_v2.csv", old, ["dataset", "method", "fold", "mode"],
         ["N_gt", "N_pred", "M", "missed", "spurious", "dup_claims", "cls_macroF1", "uni_macroF1"]),
        ("tau_sensitivity.csv", sensitivity, ["dataset", "method", "fold", "tau"],
         ["n_gt", "n_pred", "M", "missed", "spurious", "cls_macro_real", "plusFN_macro_real", "uni_macro_real"]),
    ):
        reference = pd.read_csv(Path(artifact_root) / "reference" / name)
        actual = actual.copy()
        actual["fold"] = actual["fold"].astype(str)
        reference["fold"] = reference["fold"].astype(str)
        a = actual.set_index(keys).sort_index()
        b = reference.set_index(keys).sort_index()
        if not a.index.equals(b.index):
            raise ArtifactError(f"Task coverage differs from matching reference: {name}")
        mismatches = []
        for key in a.index:
            for column in columns:
                value = float(a.loc[key, column])
                expected = float(b.loc[key, column])
                if round(value, 4) != expected:
                    mismatches.append({"key": str(key), "column": column, "computed": value, "expected": expected})
        result[name] = {"rows": len(a), "matches_at_published_precision": not mismatches, "mismatches": mismatches}
        if mismatches:
            raise ArtifactError(f"Matching differs from frozen reference: {name}: {mismatches[:3]}")
    return result


def run_statistics(artifact_root, catalog, evaluation_root, output, iterations=10000, workers=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evaluation_root = Path(evaluation_root)
    manifest_path = evaluation_root / "evaluation_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactError("Run evaluate first, or pass --evaluation-dir containing its outputs")
    em = json.loads(manifest_path.read_text(encoding="utf-8"))
    if em["catalog_sha256"] != catalog["_catalog_sha256"]:
        raise ArtifactError("Evaluation outputs were generated from a different frozen catalog")
    jobs = {j["job_id"]: j for j in catalog["jobs"]}
    cache = {}
    payloads = []
    for comparison in catalog["bootstrap_comparisons"]:
        b, o = [], []
        for bj, oj in zip(comparison["base_jobs"], comparison["opponent_jobs"]):
            for jid in (bj, oj):
                if jid not in cache:
                    cache[jid] = ledger_matrix(artifact_root, jobs[jid], evaluation_root, em)
            b.append(cache[bj])
            o.append(cache[oj])
        payloads.append((comparison, b, o, iterations, catalog["bootstrap_seed"]))
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        bootstrap = list(pool.map(_bootstrap_worker, payloads))
    pd.DataFrame(bootstrap).to_csv(output / "paired_bootstrap.csv", index=False)

    old_new, sensitivity = [], []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for old, tau in pool.map(_matching_worker, [(str(artifact_root), j) for j in catalog["jobs"]]):
            old_new.extend(old)
            sensitivity.extend(tau)
    old = pd.DataFrame(old_new).sort_values(["dataset", "method", "fold", "mode"])
    old.to_csv(output / "old_new_comparison.csv", index=False)
    summary = old.pivot_table(index=["dataset", "method"], columns="mode",
                              values="uni_macroF1", aggfunc="mean")
    summary["d_1v1_vs_oldnn"] = summary.one_to_one - summary.old_nn
    summary["d_1v1_vs_legacy"] = summary.one_to_one - summary.legacy_r1
    summary.to_csv(output / "old_new_summary.csv")
    sensitivity_df = pd.DataFrame(sensitivity).sort_values(["dataset", "method", "fold", "tau"])
    sensitivity_df.to_csv(
        output / "tau_sensitivity.csv", index=False)

    reference = validate_reference(catalog, bootstrap, iterations)
    matching_reference = validate_matching_references(artifact_root, old, sensitivity_df)
    check = {"bootstrap_comparisons": len(bootstrap), "iterations": iterations,
             **reference, "old_new_rows": len(old_new),
             "matching_reference_checks": matching_reference,
             "tau_rows": len(sensitivity),
             "legacy_duplicate_claims": int(old.loc[old['mode'] == 'legacy_r1', 'dup_claims'].sum()),
             "uncertainty_unit": "image/tile, paired within fold, fixed trained models",
             "macro_vocabulary": "job real_class_ids: PanNuke 5; CoNSeP/CRCHisto 4",
             "rounding_before_fold_mean": False,
             "reference_comparison": "exact counts and four-decimal per-fold macro values; means and sample SD use full-precision scores",
             "fraction_positive_is_two_sided_p_value": False}
    (output / "statistics_validation.json").write_text(json.dumps(check, indent=2) + "\n")
    if reference["reference_mismatches"]:
        raise ArtifactError(f"Bootstrap differs from frozen reference: {reference['reference_mismatches']}")
    return check
