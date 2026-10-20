# Unified evaluation protocol

## Complete population and matching

Each job fixes its test-image list independently of prediction rows. Ground truth is complete; a zero-detection image remains an image with missed nuclei. A missing prediction file is an error, not an empty prediction. Identical dataset/GT/tau/class vocabularies are used for all methods in each comparison group.

Canonical CSV columns are `image_id,instance_id,cx,cy,class_id`. Coordinates use x/y pixels in the original image. IDs need only be unique within an image; their deterministic string ordering is fixed. File hashes, row counts and vocabularies are checked before matching. Invalid identities, nonfinite values, duplicate IDs, unexpected images and corrupted files are rejected.

Legal edges link centroids from the same image with Euclidean distance **≤τ**. Maximum-cardinality one-to-one matching is solved first, then minimum total geometric distance among those matchings. Class predictions do not affect assignment. Ground truth and predictions are sorted by `(image_id,instance_id)` before SciPy assignment; this makes the implemented tie behavior explicit. No duplicated GT claims are permitted in the formal evaluation.

| Dataset  |                            Test tiles | Real classes                                                 |    τ |
| -------- | ------------------------------------: | ------------------------------------------------------------ | ----: |
| PanNuke  | fold1 2,656; fold2 2,523; fold3 2,722 | 0 Neoplastic;1 Inflammatory;2 Connective;3 Dead;4 Epithelial | 12 px |
| CoNSeP   |                                    14 | 0 Inflammatory;1 Epithelial;2 Spindle;3 Others               |  5 px |
| CRCHisto |                                    20 | 0 Inflammatory;1 Epithelial;2 Fibroblast;3 Others            | 10 px |

The graph radius (<40 px) and the training-label nearest-neighbor assignment are separate operations. Neither should be substituted for the test matching radius or assignment.

## Accounting and macro F1

Matched correct pairs add TP to the true class. A wrong matched class adds FP to the predicted class and FN to the true class. An unmatched GT adds FN to its true class; an unmatched prediction adds FP to its predicted class. Geometric totals obey `N_gt=M+missed` and `N_pred=M+spurious`.

Three diagnostic levels are reported from the same assignment:

1. `classification`: matched pairs only.
2. `plus_missed_FN`: additionally includes every unmatched GT.
3. `unified`: includes both every unmatched GT and every unmatched prediction; this is the complete-pipeline primary endpoint.

F1 for a class is `2TP/(2TP+FP+FN)` with zero denominator handled as zero. Primary macro averages use the real-class vocabulary; a class absent on both GT and prediction sides is omitted by the implementation. All real classes are represented in the complete reported datasets/folds, so no extra class is silently added to their denominator.

## Reporting groups and statistics

For PanNuke use the unweighted mean across three outer folds and sample SD with `ddof=1`, including the per-class means/SD. External datasets have one fixed holdout evaluation; no artificial fold SD is assigned. Paired bootstrap resamples entire images for both methods together, within each PanNuke fold, and recomputes the fold-average real-class macro difference. Seed=42, iterations=10,000, percentile CI=2.5/97.5. Bootstrap sampling does not split nuclei from the same image. The positive fraction is not a two-sided P value or a multiplicity-adjusted test.

This 46-task release, with 214 real-class records, reports `one_to_one` matching at the registered τ values only. Diagnostic matching modes retained in the matcher code are not part of the reported results. Any additional τ sensitivity analysis must be run separately on the same frozen predictions and complete ground truth, without selecting τ by test accuracy.

## Source consolidation note

The registered DIST CoNSeP raw prediction snapshot differs from an older formal matching ledger at two nucleus class labels. Aggregate main-table counts stay equal, but per-image bootstrap inputs can differ. The current 23 registered paired comparisons (PanNuke 9, CoNSeP 7, CRCHisto 7, including StarDist) must be recomputed from the SHA-registered predictions in `Result/`, the public annotations reconstructed outside `Result/`, and this release's unified matcher. The numbers stored in `bootstrap.py` are archived reference intervals, not a fresh scoring run. The candidate's `bootstrap.py` builds per-image TP/GT/pred matrices in memory and calls the SHA-pinned `statistics.paired_bootstrap` routine.
