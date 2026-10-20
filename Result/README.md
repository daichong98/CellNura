# Tables 2/4/5 prediction package

This directory is `Result/` inside the CellNura code release. It holds the frozen predictions used to rebuild the manuscript and supplementary tables, including the main tables, per-class tables, and bootstrap comparisons. It has **not** been uploaded or published as a separate archive. It contains no raw images, instance-level GT, reconstructed GT, model weights, or per-instance GT matching ledgers.

There are **41 distinct A/B method-by-test-fold prediction tasks** and **five complete CellNura V3.7 prediction tasks retained in the existing file group**: **46 task predictions in total**. These yield **23 distinct A/B method rows plus three CellNura V3.7 result rows in that C grouping** and **214 task-by-real-class result rows**. The five CellNura V3.7 predictions preserve all three PanNuke folds and the CoNSeP and CRCHisto test sets.

**CellNura V3.7 is the complete CellNura method described in the manuscript.** Its representation combines learned visual features and feature fusion, explicit morphology/chromatin descriptors, and GAT-derived neighborhood features, followed by the final MLP classifier.

The row named **Cellvit** corresponds to `cellvit` / `Cellnura-cellvit` in `CellNura_Code_Release/docs/BASELINES.md`: a CellViT encoder and type head trained from scratch, evaluated at the same CellNura detection centroids. Cellvit is the controlled classifier comparison that changes the classification module while preserving detection instance IDs and coordinates. It is not a standalone CellViT detection or segmentation evaluation. No standalone CellViT scratch prediction, result row, checkpoint, or file is included in this candidate.

## Selected method scope

- The package retains every distinct method used in the ranking tables and the registered public-weight references. The fixed TIAToolbox Micro-Net detector predictions are not included. `manifest.json` registers each task's source visibility.
- On PanNuke, the official public CellViT++ weight is the CellViT-SAM-H-x40 checkpoint. The package keeps one three-fold prediction series for that weight. The two names refer to the same checkpoint and the same prediction, so the result is reported once and is not counted as two methods.
- The public-weight CoNSeP/CRCHisto CellViT-SAM results have class-mapping limits that belong in the manuscript or supplementary material. The package preserves their predictions for diagnosis, without promoting them to same-protocol A-layer evidence.

## Files

- `predictions/*.csv`: all 46 selected tasks in one folder, including the five complete CellNura V3.7 predictions that share detection coordinates with Cellvit and remain in the existing C file group. Each CSV has `image_id,instance_id,cx,cy,class_id`.
- `images/*.csv`: all five complete test-image inventories, including images with zero predictions.
- `manifest.json`: task IDs, fold, layer, prediction SHA-256, matching radius, manuscript class map, source visibility, the PanNuke note that the official CellViT++ weight is CellViT-SAM-H-x40, and expected aggregate results.
- `SOURCE_SCOPE.csv`: one row per distinct retained method and dataset, showing layer, task IDs, where it is reported in the manuscript or supplementary tables, and the PanNuke note that the official CellViT++ weight is CellViT-SAM-H-x40.
- `expected_task_results.csv` and `expected_per_class.csv`: expected task totals and 214 task-class TP/FP/FN, P/R/F records. These contain aggregate counts but no GT nucleus coordinates or per-instance matching records.
- `training_split_manifest.csv`: image-level train/validation/test roles for five partitions. The roles are disjoint; each test role matches its complete image inventory.
- `segmentation_checkpoint_identities.csv`: five HoVer-Net segmentation checkpoint identities checked by `verify_candidate.py`. The weight files are absent. Broader final-model identities, including CellViT, Micro-Net, Mask R-CNN, DIST, the shared PanNuke classifier, and TSFD-Net, are in `baselines/final_checkpoint_identities.csv`. Pretrained initialization hashes do not identify those final models. CellNura fusion, GAT, MLP, and scaler identities are in that same file. Their weight files are absent.
- `rebuild_ground_truth.py` and `expected_ground_truth.json`: recreate and SHA-check five canonical GT CSVs from the original public PanNuke, CoNSeP and CRCHisto annotations outside this package.
- `verify_candidate.py`: checks files, hashes, task/class completeness, image-level split integrity, and exact fixed-detection coordinates without GT.
- `replay_candidate.py`: independently evaluates all selected predictions against the reconstructed public GT, checks task and per-class counts, and writes aggregate tables outside the package. It requires the companion CellNura code release's SHA-pinned unified matcher.
- `bootstrap.py`: registers 23 paired comparisons (PanNuke 9, CoNSeP 7, CRCHisto 7), including StarDist. It can recompute those intervals outside this package from public annotations, with 10,000 draws and seed 42. The intervals stored in the script are archived reference values, not a scoring run by themselves. `verify_candidate.py` does not perform that scoring.
- `checksums.txt`: whole-package checksum list. `verify_candidate.py` writes `verification_report.json` only after a local check; that report is not part of the checksum list.

## Verification and reproduction

From this directory, check the package:

```bash
python verify_candidate.py
```

This checks file hashes, 46 prediction tasks, five image inventories, five image-disjoint role partitions, five segmentation checkpoint identities, 214 task-class rows, and exact detection coordinates between Cellvit and CellNura V3.7.

A PASS from `verify_candidate.py` shows that these files match the registered records. It does not itself score predictions against the official annotations.

The numerical replay of this 46-task package is complete. All 46 tasks and all 214 class-count registrations were scored against the reconstructed public annotations, and the scores matched `expected_task_results.csv` and `expected_per_class.csv`. The paired bootstrap is also complete: all 23 comparisons were recomputed with 10,000 image-level draws and seed 42, and each interval matched the reference now stored in `bootstrap.py`.

The commands below are the procedure for that completed replay and bootstrap. `replay_candidate.py` and `bootstrap.py` default `--code-root` to the parent CellNura release. They write `replay_report.json` and `bootstrap_report.json` outside this directory.

```bash
python rebuild_ground_truth.py \
  --artifact-root . \
  --pannuke-root /path/to/PanNuke \
  --consep-root /path/to/CoNSeP \
  --crchisto-root /path/to/CRCHistoPhenotypes \
  --output /tmp/scheme2_public_gt
python replay_candidate.py \
  --gt-dir /tmp/scheme2_public_gt \
  --out-dir /tmp/scheme2_replay_results
python bootstrap.py \
  --gt-dir /tmp/scheme2_public_gt \
  --out-dir /tmp/scheme2_bootstrap_results \
  --require-reference-match
```

Replay checks every selected task and all 214 class-count registrations. It writes `task_results.csv`, `per_class_results.csv`, `method_results_AB.csv`, the A/B table CSVs, `same_segmentation_C.csv`, and `replay_report.json` outside this directory. It refuses byte-different GT or unified-matcher code, and does not write GT or per-instance matches here. Matching is class-independent and one-to-one: maximum cardinality, then minimum distance, at 12 px on PanNuke, 5 px on CoNSeP, and 10 px on CRCHisto.

Bootstrap writes `paired_bootstrap.csv` and `bootstrap_report.json` outside this directory. It recomputes the 23 paired intervals, CellNura V3.7 minus each opponent, with 10,000 draws and seed 42. The completed run matched the references stored in `bootstrap.py`. A 95% interval containing zero does not establish equivalence, and the fraction of positive draws is not a two-sided P value.
