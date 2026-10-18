# Portable baselines

See [BASELINES](../docs/BASELINES.md) for the complete configuration, method-specific training/inference, weight sources, and limitations. Executable adapters are organized by algorithm: `maskrcnn/`, `dist/`, `micronet/`, `hovernet/`, `cellvitpp/`, `traditional/`, `tsfd/`, and `cellnura_cellvit/`. StarDist detection and the shared classification scripts are in `shared/`. Frozen predictions for Tables 2, 4, and 5 are in `Result/` at the repository root.

## Unknown class (−1) and real-class macro F1

The prediction-file value `class_id = -1` is retained as supplied and is not relabelled as a real class. It is not an MLP output. The CellNura MLP takes `argmax` over K real classes, yielding 0 to K−1. The `evaluation_table` function in `cellnura_release/inference.py` writes −1 only when a detection has no corresponding MLP row, so the same table may contain both −1 and 0–4.

These −1 predictions participate in class-independent one-to-one matching and in the total spurious-detection accounting. GT in the formal evaluation contains only the target real classes.

- An unknown prediction matched to a GT nucleus: record an FN for the real class and a misclassification FP for the unknown class. This FN contributes to real-class macro F1.
- An unknown prediction unmatched to any GT nucleus: record an FP only for the unknown class and include it in the total spurious-detection count. Real-class macro F1 averages only PanNuke 0–4 or external-dataset 0–3, so these spurious detections do not enter the denominator of the primary metric.

Thus, unified macro F1 does not penalize every spurious detection in the same way. The same rule is documented in `docs/BASELINES.md`. All methods use the same matching and real-class vocabulary. Tables 2, 4, and 5 use the registered files in `Result/predictions/`.
