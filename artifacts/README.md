# Code-only split and image metadata

This directory contains five complete test-image lists and the image-level train/validation/test manifests used by training. It does not contain prediction CSVs, ground-truth CSVs, or result tables.

The 46-task registry, its 214 real-class records, and the frozen predictions are in `Result/`. Evaluation annotations are reconstructed outside `Result/` from the public PanNuke, CoNSeP, and CRCHisto annotations.

`training_split_provenance.json` maps the historical split hash `3c1b4d69…` to the two published CSVs. `training_split_manifest.csv` is `1afcdb0c…`. `split_manifest.csv` is `38b39de7…`. The historical hash remains in the completed HoVer-Net run records and in `third_party/dist_port/preflight/TRAINING_PREFLIGHT.json`.
