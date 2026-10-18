#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""unified_matcher.py — One-to-one matching and complete detection/classification evaluation (T04.2).

Protocol: docs/EVALUATION_PROTOCOL.md
  * legal edges: centroid Euclidean distance <= tau, same image only, class-independent
  * one-to-one assignment: max cardinality -> min total distance -> deterministic tie-break
    (input sorted by (image_id, instance_id); scipy.linear_sum_assignment is deterministic)
  * error accounting: matched-correct TP; misclassified paired FP(pred)/FN(true);
    unmatched GT FN(true) [missed]; unmatched pred FP(pred) [spurious]
  * conservation asserts: N_gt = M + missed ; N_pred = M + spurious
  * empty predictions still emit an image row; missing prediction FILE raises

Tiers: classification-only / plus_missed_FN / unified (FN+FP), computed per class.
Also provides legacy nearest-neighbour mode (mode="old_nn") for T10.1 comparison.
"""
from __future__ import annotations

import csv
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------- data model

@dataclass
class Instance:
    image_id: str
    instance_id: str
    x: float
    y: float
    cls: int


@dataclass
class Failure:
    image_id: str
    instance_id: str
    stage: str            # e.g. "segmentation", "feature", "classifier"
    reason: str


@dataclass
class EvalInput:
    dataset: str
    tau: float
    gt: List[Instance]
    pred: List[Instance]                       # final predictions WITH class (post feature-filter)
    failures: List[Failure] = field(default_factory=list)
    raw_pred_count: Optional[int] = None       # N_raw (segmentation output before feature stage)
    ignore_gt_labels: Tuple[int, ...] = ()     # ignored before N_gt, counted separately
    real_class_ids: Optional[Tuple[int, ...]] = None  # F3: macro vocabulary (real classes); None=all observed classes (legacy convention)
    image_ids: Tuple[str, ...] = ()           # includes genuinely empty images
    class_ids: Tuple[int, ...] = ()           # explicit dataset vocabulary


# ---------------------------------------------------------------- matching

def _assign_one_to_one(gt: np.ndarray, pr: np.ndarray, tau: float) -> List[Tuple[int, int]]:
    """Max-cardinality, min-distance one-to-one assignment over legal (<=tau) edges.

    Augmented cost matrix of size (n+m, n+m):
      real x real : distance if <= tau else BIG
      real x dummy / dummy x real : C = (min(n,m)+1)*max(tau,1) (one unmatched node)
      dummy x dummy : 0
    2*C exceeds any change in the total cost of all legal real-real edges.
    Therefore cardinality has priority even along long alternating paths.
    Among maximum-cardinality solutions total distance is minimized.
    Determinism: caller sorts gt/pred by (image_id, instance_id) beforehand.
    Returns list of (gt_index, pred_index).
    """
    n, m = len(gt), len(pr)
    if n == 0 or m == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    d = np.linalg.norm(gt[:, None, :2] - pr[None, :, :2], axis=2)
    C = (min(n, m) + 1) * max(float(tau), 1.0)
    BIG = 4.0 * (n + m + 1) * C
    cost = np.full((n + m, m + n), BIG, dtype=np.float64)
    # real gt x real pred
    legal = d <= tau
    cost[:n, :m][legal] = d[legal]
    # gt x dummy_pred (each gt may stay unmatched)
    cost[:n, m:] = C
    np.fill_diagonal(cost[:n, m:], C)          # explicit; full block already C
    # dummy_gt x pred
    cost[n:, :m] = C
    # dummy x dummy
    cost[n:, m:] = 0.0
    r, c = linear_sum_assignment(cost)
    pairs = []
    for ri, ci in zip(r, c):
        if ri < n and ci < m:                  # real-real assignment
            if d[ri, ci] <= tau:               # must be legal (cost matrix guarantees)
                pairs.append((int(ri), int(ci)))
    return pairs


def _assign_old_nearest(gt: np.ndarray, pr: np.ndarray, tau: float) -> List[Tuple[int, int]]:
    """Legacy R1 behaviour: each detection independently takes its nearest GT (<=tau).
    Multiple detections may claim the same GT; duplicate claims counted by caller."""
    if len(gt) == 0 or len(pr) == 0:
        return []
    d = np.linalg.norm(gt[:, None, :2] - pr[None, :, :2], axis=2)
    pairs = []
    for j in range(len(pr)):                   # iterate predictions (detection-side), like step8
        i = int(np.argmin(d[:, j]))
        if d[i, j] <= tau:
            pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------- evaluation

@dataclass
class EvalResult:
    dataset: str
    tau: float
    mode: str                                  # "one_to_one" | "old_nn"
    n_gt: int = 0
    n_pred: int = 0
    n_ignored_gt: int = 0
    n_matched: int = 0
    n_missed: int = 0
    n_spurious: int = 0
    n_duplicate_claims: int = 0    # old_nn: count duplicate claims as spurious; legacy_r1: retain duplicate matches                # old_nn only: extra preds claiming a taken GT
    n_feature_failures: int = 0
    classes: Dict[int, dict] = field(default_factory=dict)
    per_image: List[dict] = field(default_factory=list)
    match_ledger: List[dict] = field(default_factory=list)

    # derived tiers -------------------------------------------------------
    def macro(self, tier: str, key: str = "F1", real_class_ids=None) -> float:
        # Absent GT but present predictions still incur false-positive penalties.
        # When real_class_ids=None, inherit the vocabulary stored from EvalInput.real_class_ids
        # (real-class convention); the legacy "all observed classes" convention requires macro_all().
        ids = real_class_ids if real_class_ids is not None else None
        if ids is None:
            ids = getattr(self, "_real_class_ids", None)
        if ids is None:
            ids = list(self.classes.keys())
        vals = [self.classes[c][tier][key] for c in ids
                if c in self.classes and self.classes[c]["counts"]["N_gt"] + self.classes[c]["counts"]["N_pred"] > 0]
        return float(np.mean(vals)) if vals else 0.0

    def macro_all(self, tier: str, key: str = "F1") -> float:
        """Legacy macro over all observed classes (including -1/unknown and out-of-vocabulary labels); diagnostics only."""
        vals = [self.classes[c][tier][key] for c in self.classes
                if self.classes[c]["counts"]["N_gt"] + self.classes[c]["counts"]["N_pred"] > 0]
        return float(np.mean(vals)) if vals else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"P": p, "R": r, "F1": f}


def evaluate(ev: EvalInput, mode: str = "one_to_one") -> EvalResult:
    if mode not in {"one_to_one", "old_nn", "legacy_r1"}:
        raise ValueError(f"Unknown matching mode: {mode}")
    if not np.isfinite(ev.tau) or ev.tau < 0:
        raise ValueError("tau must be finite and nonnegative")
    for side, items in (("gt", ev.gt), ("pred", ev.pred)):
        keys = [(a.image_id, a.instance_id) for a in items]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate {side} instance identifiers")
        for a in items:
            if not np.isfinite([a.x, a.y]).all():
                raise ValueError(f"Non-finite coordinates: {side} {a.instance_id}")
            if ev.image_ids and a.image_id not in ev.image_ids:
                raise ValueError(f"Instance outside explicit image manifest: {a.image_id}")
            if ev.class_ids and a.cls not in ev.class_ids and not (side == "gt" and a.cls in ev.ignore_gt_labels):
                raise ValueError(f"Invalid {side} class: {a.cls}")
    res = EvalResult(dataset=ev.dataset, tau=ev.tau, mode=mode)
    res._real_class_ids = ev.real_class_ids

    # 0. drop ignored GT labels BEFORE N_gt, count them
    gt_all = [g for g in ev.gt]
    gt = [g for g in gt_all if g.cls not in ev.ignore_gt_labels]
    res.n_ignored_gt = len(gt_all) - len(gt)
    pred = list(ev.pred)
    res.n_feature_failures = len(ev.failures)

    gt_sorted = sorted(gt, key=lambda g: (g.image_id, g.instance_id))
    pr_sorted = sorted(pred, key=lambda p: (p.image_id, p.instance_id))
    res.n_gt, res.n_pred = len(gt_sorted), len(pr_sorted)

    classes = sorted(set(ev.class_ids) | {g.cls for g in gt_sorted} | {p.cls for p in pr_sorted})

    # per-image grouping (matching never crosses images)
    images = sorted(set(ev.image_ids) | {g.image_id for g in gt_sorted} | {p.image_id for p in pr_sorted})
    g_by_img: Dict[str, List[int]] = {im: [] for im in images}
    p_by_img: Dict[str, List[int]] = {im: [] for im in images}
    for i, g in enumerate(gt_sorted):
        g_by_img[g.image_id].append(i)
    for j, p in enumerate(pr_sorted):
        p_by_img[p.image_id].append(j)

    matched_gt, matched_pr = set(), set()
    per_class = {c: {"TP": 0, "FP_mis": 0, "FN_mis": 0, "FN_miss": 0, "FP_spur": 0,
                     "N_gt": 0, "N_pred": 0} for c in classes}
    for g in gt_sorted:
        per_class[g.cls]["N_gt"] += 1
    for p in pr_sorted:
        per_class[p.cls]["N_pred"] += 1

    for im in images:
        gi, pi = g_by_img[im], p_by_img[im]
        if mode == "legacy_r1":
            # F6: reproduce original R1 behavior: each detection independently claims its nearest GT (≤τ); duplicate claims remain matched,
            # rather than being deduplicated into spurious FPs (unlike old_nn).
            gt_arr = np.array([[gt_sorted[i].x, gt_sorted[i].y] for i in gi], dtype=float)
            pr_arr = np.array([[pr_sorted[j].x, pr_sorted[j].y] for j in pi], dtype=float)
            pairs = _assign_old_nearest(gt_arr, pr_arr, ev.tau) if len(gt_arr) and len(pr_arr) else []
            claimed_gt_local = set()
            for gi_i, pi_j in pairs:
                g_ins, p_ins = gt_sorted[gi[gi_i]], pr_sorted[pi[pi_j]]
                matched_gt.add(gi[gi_i])
                matched_pr.add(pi[pi_j])
                claimed_gt_local.add(gi_i)
                tp = g_ins.cls == p_ins.cls
                cid = g_ins.cls
                per_class[cid]["TP"] += int(tp)
                per_class[cid]["N_gt"] += 0  # already counted above
                if not tp:
                    per_class[cid]["FN_mis"] += 1
                    per_class[p_ins.cls]["FP_mis"] += 1
                res.match_ledger.append(_row(im, g_ins, p_ins, "legacy-matched", None))
            # unmatched gt
            for i_idx in gi:
                if i_idx not in matched_gt:
                    g_ins = gt_sorted[i_idx]
                    per_class[g_ins.cls]["FN_miss"] += 1
                    res.match_ledger.append(_row(im, g_ins, None, "missed", None))
            # legacy_r1: a detection with no claimable GT within τ is spurious (as in original R1 when the nearest GT lies beyond τ).
            n_spur_local = 0
            for j_idx in pi:
                if j_idx not in matched_pr:
                    p_ins = pr_sorted[j_idx]
                    per_class[p_ins.cls]["FP_spur"] += 1
                    res.match_ledger.append(_row(im, None, p_ins, "spurious", None))
                    n_spur_local += 1
            # Duplicate-claim metadata: number of claim pairs minus number of claimed GT nuclei.
            res.n_duplicate_claims += len(pairs) - len(claimed_gt_local)
            # Image row: M=claim pairs (R1 semantics retain duplicates); missed=unclaimed GT;
            # spur=unclaimed detections. Duplicate claims imply N_gt≠M+missed; the difference is their count, inherent to R1.
            res.per_image.append(_img_row(im, len(gi), len(pi), len(pairs),
                                          len(gi) - len(claimed_gt_local), n_spur_local))
            continue
        if not gi:
            # predictions only -> all spurious
            for j in pi:
                per_class[pr_sorted[j].cls]["FP_spur"] += 1
                res.match_ledger.append(_row(im, None, pr_sorted[j], "spurious", None))
            res.per_image.append(_img_row(im, 0, len(pi), 0, 0, len(pi)))
            continue
        if not pi:
            for i in gi:
                per_class[gt_sorted[i].cls]["FN_miss"] += 1
                res.match_ledger.append(_row(im, gt_sorted[i], None, "missed", None))
            res.per_image.append(_img_row(im, len(gi), 0, 0, len(gi), 0))
            continue

        g_arr = np.array([[gt_sorted[i].x, gt_sorted[i].y] for i in gi])
        p_arr = np.array([[pr_sorted[j].x, pr_sorted[j].y] for j in pi])
        if mode == "one_to_one":
            pairs = _assign_one_to_one(g_arr, p_arr, ev.tau)
        else:
            pairs = _assign_old_nearest(g_arr, p_arr, ev.tau)

        local_matched_gt = set()
        m = 0
        for a, b in pairs:
            gi_i, pi_j = gi[a], pi[b]
            g_ins, p_ins = gt_sorted[gi_i], pr_sorted[pi_j]
            if mode == "old_nn" and gi_i in local_matched_gt:
                # duplicate claim: older pipeline kept such detections as matched-to-same-GT;
                # count once as spurious FP and mark as processed (not double-counted below)
                res.n_duplicate_claims += 1
                matched_pr.add(pi_j)
                per_class[p_ins.cls]["FP_spur"] += 1
                res.match_ledger.append(_row(im, None, p_ins, "spurious(duplicate-claim)", 0.0))
                continue
            local_matched_gt.add(gi_i)
            matched_gt.add(gi_i)
            matched_pr.add(pi_j)
            m += 1
            dist = float(np.linalg.norm([g_ins.x - p_ins.x, g_ins.y - p_ins.y]))
            if g_ins.cls == p_ins.cls:
                per_class[g_ins.cls]["TP"] += 1
                res.match_ledger.append(_row(im, g_ins, p_ins, "matched-correct", dist))
            else:
                per_class[g_ins.cls]["FN_mis"] += 1
                per_class[p_ins.cls]["FP_mis"] += 1
                res.match_ledger.append(_row(im, g_ins, p_ins, "matched-wrong-class", dist))
        for i in gi:                            # unmatched GT -> missed
            if i not in matched_gt:
                per_class[gt_sorted[i].cls]["FN_miss"] += 1
                res.match_ledger.append(_row(im, gt_sorted[i], None, "missed", None))
        for j in pi:                            # unmatched pred -> spurious
            if j not in matched_pr:
                per_class[pr_sorted[j].cls]["FP_spur"] += 1
                res.match_ledger.append(_row(im, None, pr_sorted[j], "spurious", None))
        res.per_image.append(_img_row(im, len(gi), len(pi), m, len(gi) - m, len(pi) - m))

    # ---------------- conservation (spec §4.2) ----------------
    res.n_matched = len(matched_pr) if mode == "legacy_r1" else len(matched_gt)
    res.n_missed = res.n_gt - len(matched_gt)
    res.n_spurious = res.n_pred - res.n_matched
    tot_tp = sum(v["TP"] for v in per_class.values())
    tot_fn = sum(v["FN_mis"] + v["FN_miss"] for v in per_class.values())
    tot_fp = sum(v["FP_mis"] + v["FP_spur"] for v in per_class.values())
    if mode != "legacy_r1":
        assert res.n_gt == res.n_matched + res.n_missed, "GT-side conservation broken"
    else:
        assert res.n_gt == len(matched_gt) + res.n_missed, "legacy_r1 GT conservation broken"
    assert res.n_pred == res.n_matched + res.n_spurious, "pred-side conservation broken"
    if mode != "legacy_r1":
        assert tot_tp + tot_fn == res.n_gt, f"class conservation broken: TP+FN={tot_tp+tot_fn} != N_gt={res.n_gt}"
    else:
        # F6: legacy_r1 permits duplicate claims (TP+FN ≥ N_gt), so conservation is relaxed.
        assert tot_tp + tot_fn >= res.n_gt, f"legacy_r1 conservation broken: TP+FN={tot_tp+tot_fn} < N_gt={res.n_gt}"
    assert tot_tp + tot_fp == res.n_pred, f"class conservation broken: TP+FP={tot_tp+tot_fp} != N_pred={res.n_pred}"

    # ---------------- three tiers ----------------
    for c, v in per_class.items():
        cls_tier = _prf(v["TP"], v["FP_mis"], v["FN_mis"])
        plus_tier = _prf(v["TP"], v["FP_mis"], v["FN_mis"] + v["FN_miss"])
        uni_tier = _prf(v["TP"], v["FP_mis"] + v["FP_spur"], v["FN_mis"] + v["FN_miss"])
        res.classes[c] = {"counts": v, "classification": cls_tier,
                          "plus_missed_FN": plus_tier, "unified": uni_tier}
    return res


def _row(im, g, p, status, dist):
    return {"image_id": im,
            "gt_id": g.instance_id if g else "", "gt_cls": g.cls if g else "",
            "pred_id": p.instance_id if p else "", "pred_cls": p.cls if p else "",
            "distance": round(dist, 3) if dist is not None else "",
            "match_status": status}


def _img_row(im, n_gt, n_pred, m, missed, spur):
    return {"image_id": im, "N_gt": n_gt, "N_pred": n_pred, "M": m,
            "missed": missed, "spurious": spur}


# ---------------------------------------------------------------- reporting

def write_outputs(res: EvalResult, out_prefix: str) -> None:
    # F3: when a real-class vocabulary is specified, append real-class macro rows (MACRO_REAL) to summary/CSV; retain legacy MACRO for traceability.
    with open(f"{out_prefix}_per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "N_gt", "N_pred", "TP", "FP_misclassified", "FP_spurious",
                    "FN_misclassified", "FN_missed",
                    "cls_P", "cls_R", "cls_F1", "plusFN_P", "plusFN_R", "plusFN_F1",
                    "uni_P", "uni_R", "uni_F1"])
        for c, d in sorted(res.classes.items()):
            v = d["counts"]
            w.writerow([c, v["N_gt"], v["N_pred"], v["TP"], v["FP_mis"], v["FP_spur"],
                        v["FN_mis"], v["FN_miss"],
                        *(round(d[t][k], 4) for t in ("classification", "plus_missed_FN", "unified")
                          for k in ("P", "R", "F1"))])
        w.writerow(["MACRO", res.n_gt, res.n_pred, "", "", "", "", "",
                    *(round(res.macro(t, k), 4) for t in ("classification", "plus_missed_FN", "unified")
                      for k in ("P", "R", "F1"))])
    with open(f"{out_prefix}_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "N_gt", "N_pred", "M", "missed", "spurious"])
        w.writeheader()
        w.writerows(res.per_image)
    with open(f"{out_prefix}_match_ledger.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "gt_id", "gt_cls", "pred_id", "pred_cls",
                                          "distance", "match_status"])
        w.writeheader()
        w.writerows(res.match_ledger)
    with open(f"{out_prefix}_summary.txt", "w") as f:
        f.write(f"dataset={res.dataset} mode={res.mode} tau={res.tau}\n"
                f"N_gt={res.n_gt} N_pred={res.n_pred} M={res.n_matched} "
                f"missed={res.n_missed} spurious={res.n_spurious} "
                f"duplicate_claims={res.n_duplicate_claims} "
                f"ignored_gt={res.n_ignored_gt} feature_failures={res.n_feature_failures}\n"
                f"macro cls_F1={res.macro('classification'):.4f} "
                f"plusFN_F1={res.macro('plus_missed_FN'):.4f} "
                f"unified_F1={res.macro('unified'):.4f}\n")

    _append_macro_real(res, out_prefix)


def _append_macro_real(res, out_prefix):
    """F3: append real-class macro values to the summary and the per_class CSV MACRO_REAL row."""
    rc = getattr(res, "_real_class_ids", None)
    if not rc:
        return
    suf = ("classification", "plus_missed_FN", "unified")
    macro_real = {t: res.macro(t, real_class_ids=rc) for t in suf}
    # Append to summary.
    sp = f"{out_prefix}_summary.txt"
    if os.path.exists(sp):
        with open(sp, "a") as f:
            f.write(f"real_class_macro cls_F1={macro_real['classification']:.4f} "
                    f"plusFN_F1={macro_real['plus_missed_FN']:.4f} unified_F1={macro_real['unified']:.4f} "
                    f"(real_class_ids={list(rc)})\n")
    # Append a per_class CSV row, aligned exactly with the 17-column header:
    # class,N_gt,N_pred,TP,FP_mis,FP_spur,FN_mis,FN_miss,cls_P,cls_R,cls_F1,
    # plusFN_P,plusFN_R,plusFN_F1,uni_P,uni_R,uni_F1)
    cp = f"{out_prefix}_per_class.csv"
    if os.path.exists(cp):
        fields = ["MACRO_REAL", "", "", "", "", "", "", "", "", "",
                  f"{macro_real['classification']:.4f}", "", "",
                  f"{macro_real['plus_missed_FN']:.4f}", "", "",
                  f"{macro_real['unified']:.4f}"]
        assert len(fields) == 17, f"MACRO_REAL row must have 17 fields, got {len(fields)}"
        with open(cp, "a") as f:
            f.write(",".join(fields) + "\n")
