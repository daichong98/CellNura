# Baseline implementations and reproducible recipes

This package contains the algorithm source code used for the current second revision, fixed source snapshots, training and inference entry points, and unified-evaluation adapters. Datasets, GPU environments, and large model weights are external resources. The reported manuscript metrics should be recomputed from the frozen predictions using the package evaluation subcommand; the training commands below produce new models and predictions and do not guarantee bitwise reproduction of the existing training results.

## Scope, independence, and evaluation units

The train/val/test image counts for PanNuke outer folds 1/2/3 are 4458/787/2656, 4571/807/2523, and 4402/777/2722, respectively. The CoNSeP and CRCHisto sections of the original file fix only the official train/test partitions of 27/14 and 80/20. Internal image-level validation is fixed in `baselines/external_validation_splits.json`, giving train/val/test counts of 22/5/14 and 68/12/20. These assignments come from the classifier_split files of the completed formal training runs and are not randomly resampled. The fixed SHA of this JSON is recorded in the catalog; changes to partition membership are rejected at runtime. Other CellNura modules use the explicitly derived training_split_manifest; the two manifests serve different purposes.

Unified testing uses all test images, all GT nuclei, all detected nuclei, and one-to-one spatial matching, with τ=12/5/10px for PanNuke/CoNSeP/CRCHisto. Unknown predictions remain eligible for matching, but real-class macro averaging includes only PanNuke0..4 or external0..3. The CoNSeP GT mapping is 1→3, 2→0, 3/4→1, 5/6/7→2; CRCHisto inflammatory/epithelial/fibroblast/others→0/1/2/3. CRCHisto provides centroids only. Methods requiring instance supervision are trained with synthetic disks of radius 8 pixels, while test metrics use the original centroid GT. This is not a fully equivalent reproduction of studies using true instance-mask supervision.

### Unknown predictions and predictions outside the target class set

Detections predicted as unknown (−1) remain eligible for class-independent one-to-one geometric matching. GT in the formal evaluation contains only the target real classes, without an unknown class. When an unknown prediction matches an annotated nucleus of a real class, it contributes a misclassification FN (`FN_misclassified`) to that real class and a misclassification FP (`FP_misclassified`) to the unknown class. When unmatched, it contributes a spurious-detection FP (`FP_spurious`) to the unknown class and is included in the total spurious-detection count.

If a method outputs a label outside the target dataset's real-class set but registered in the allowed scoring vocabulary (`class_ids`), that prediction also remains eligible for geometric matching and is counted using the rules above.

Macro F1 is computed only over the predefined real-class vocabulary (`real_class_ids`). Unknown and other registered non-target labels are not added to the averaging set. Their FPs do not directly enter real-class macro F1, but a match to a real nucleus still affects macro F1 through the FN of that real class. All methods use the same matching, counting, and macro-averaging rules.

## Configuration and basic commands

Run from the root of the submitted package:

```bash
python -m cellnura_release baseline list
python -m cellnura_release baseline check hovernet --sources-only
python -m cellnura_release baseline check hovernet --config configs/release.json --dataset pannuke --environment --action train
python -m cellnura_release baseline run hovernet train --config configs/release.json --dataset pannuke --fold 1
python -m cellnura_release baseline run hovernet infer --config configs/release.json --dataset pannuke --fold 1
```

`check --sources-only` verifies only the package source code and third-party snapshots; it does not verify data, weights, or GPU inference. A full check verifies input files, fixed partitions, and public-weight SHA values; `--environment` imports dependencies and checks CUDA. `run` also checks trained checkpoints and executes actual subprocesses in order. Missing data or weights, an incorrect architecture, or an invalid method/dataset causes a nonzero exit; infer does not implicitly start training. `--dry-run` prints the exact argv and explicitly reports `executed:false`. Additional arguments such as `run ... -- --smoke` are passed only to the first subcommand; debug multi-stage pipelines by running the corresponding scripts directly as listed in the registry.

`configs/release.json`: paths are resolved relative to `path_base` (the package root by default). `baseline_paths` lists dataset roots, weights, caches, and output directories. `baselines.<name>.python` specifies the method's Python interpreter, while `classifier_python` separately specifies the shared PyTorch classifier environment. When empty, the CLI interpreter is used; checks report unmet requirements and do not install dependencies automatically. `weight_sha256` contains either a single-file SHA or a file-to-SHA mapping within the StarDist directory; the actual SHA values of the existing weights are registered by default. Dataset layouts are listed below, and each root may point to an external mount.

```json
{
  "path_base": "..",
  "baseline_paths": {
    "pannuke": "data/PanNuke_dataset",
    "consep": "data/CoNSeP",
    "crchisto": "data/CRCHisto",
    "split_manifest": "artifacts/split_manifest.csv",
    "runs": "outputs/baselines/runs"
  },
  "baselines": {
    "stardist": {"python": "envs/stardist/bin/python", "classifier_python": "envs/torch/bin/python"}
  }
}
```

This example is a fragment to merge into the complete configuration, not a replacement for the complete release.json. PanNuke layout: `Fold N/images/foldN/images.npy, types.npy` and `Fold N/masks/foldN/masks.npy`; CoNSeP layout: `image/{train,test}_*.png` and `label/*.mat`; CRC layout: `CRCHisto_database/Detection/imgN/imgN.bmp` and `crchisto_features_output/step0_preprocessed/imgN_processed.json`.

## Execution order

First configure the external datasets and weights and create the corresponding environments according to BASELINE_ENVIRONMENTS; then run the method check. For tier A models, run train→infer separately for each PanNuke outer fold (the three folds may run in parallel on different GPUs with separate output trees). Run train→infer separately for CoNSeP and CRC.

```bash
# Independently trained within-split models (method environments and datasets must be configured)
python -m cellnura_release baseline run maskrcnn train --dataset consep --config configs/release.json
python -m cellnura_release baseline run maskrcnn infer --dataset consep --config configs/release.json
python -m cellnura_release baseline run dist train --dataset crchisto --config configs/release.json
python -m cellnura_release baseline run dist infer --dataset crchisto --config configs/release.json
python -m cellnura_release baseline run cellnura_cellvit train --dataset pannuke --fold 1 --config configs/release.json
python -m cellnura_release baseline run cellnura_cellvit infer --dataset pannuke --fold 1 --config configs/release.json
# 
python -m cellnura_release baseline run micronet train --dataset pannuke --fold 1 --config configs/release.json
python -m cellnura_release baseline run micronet infer --dataset consep --config configs/release.json
python -m cellnura_release baseline run micronet infer --dataset crchisto --config configs/release.json
python -m cellnura_release baseline run stardist train --dataset pannuke --fold 1 --config configs/release.json
python -m cellnura_release baseline run stardist infer --dataset pannuke --fold 1 --config configs/release.json
#
python -m cellnura_release baseline check tsfd --config configs/release.json --dataset pannuke --action train --environment
python -m cellnura_release baseline run tsfd train --config configs/release.json --dataset pannuke --fold 1
python -m cellnura_release baseline run tsfd infer --config configs/release.json --dataset pannuke --fold 1
#
python -m cellnura_release baseline run cellvitpp infer --dataset consep --config configs/release.json
python -m cellnura_release baseline run traditional train --dataset pannuke --fold 1 --config configs/release.json
```

### `maskrcnn` — Mask R-CNN (torchvision adaptation)

- Initialization/training: COCO detector with task heads replaced; no PanNuke task pretraining; 30 epochs for PanNuke / 60 for external datasets; image-level validation; dense-nucleus anchors; synthetic r=8 CRC masks are disclosed.
- Preprocessing/adaptation: native256 for PanNuke; CoNSeP RGB1000→512 with bilinear interpolation / instance masks with nearest-neighbor interpolation, inverse centroid scaling ×1000/512; native500 CRC with synthetic radius8 disks for training only; torchvision RGB normalization.
- Class mapping: task class+1 with background0.
- Package source files: `baselines/maskrcnn/t08_train_maskrcnn.py`, `baselines/shared/evaluate_csv.py`.
- Paper/author source: [He et al. Mask R-CNN, ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html).

### `dist` — DIST / DRFNS PyTorch project port + classifier v3

- Initialization/training: randomly initialized distance model; separate ImageNet ResNet18 classifier; 20 epochs of PanNuke distance regression and 300 for external datasets; v3 PanNuke classifier for 15 epochs and external classifier for 10; port of the original author architecture with adapted targets/optimizer.
- Preprocessing/adaptation: 396x396 reflection-padded window ->212 distance output; native pixel range; full-image distance stitching; fixed64 center crop for the PanNuke classifier and external bbox+8 resized64.
- Class mapping: untyped distance detector; shared classifier with PanNuke0..4/external0..3.
- Package source files: `baselines/dist/t08_train_dist.py`, `baselines/dist/t08_train_dist_external.py`, `baselines/shared/t08_classifier_v3.py`, `baselines/micronet/t08_micronet_external.py`.
- Paper/author source: [Naylor et al. Segmentation of nuclei in histopathology images by deep regression of the distance map, IEEE TMI 2019](https://github.com/PeterJackNaylor/DRFNS).

### `micronet` — Micro-Net within-fold retraining + shared classifier

The Micro-Net row, like the other comparison methods, refers to Micro-Net retrained within the split: the detector is trained from scratch on the training images of the corresponding dataset, then combined with the shared ResNet-18 classifier. This setup is used for PanNuke, CoNSeP, and CRCHisto. The command entry points are `t08_micronet_pannuke_retrain.py`, `t08_micronet_consep_retrain.py`, and `t08_micronet_crchisto_retrain.py`. The PanNuke training-from-scratch script `t06_train_micronet_pannuke.py` is not included in this package; the entry points above stop if the corresponding retrained weights are unavailable.

- Package source files: `baselines/micronet/micronet_standalone.py`, `t08_micronet_pannuke_retrain.py`, `t08_micronet_consep_retrain.py`, `t08_micronet_crchisto_retrain.py`, `t08_micronet_external.py` (providing only the image-list, tiled-inference, and classifier functions shared by within-split retraining and DIST), and `baselines/shared/t08_classifier_v3.py`. `baselines/shared/t08_det_baseline_inference.py` serves StarDist only.
- Paper/author source: [Raza et al. Micro-Net: A unified model for segmentation of various objects in microscopy images, Medical Image Analysis 2019](https://github.com/TissueImageAnalytics/tiatoolbox).

### `stardist` — StarDist versatile HE fixed detector + classifier v3

- Initialization/training: fixed author-pretrained 2D_versatile_he detector; the train action trains the shared PyTorch classifier for 15 epochs, without retraining the StarDist detector.
- Preprocessing/adaptation: csbdeep percentile RGB normalization and pretrained StarDist postprocessing; fixed64 center-crop shared classifier.
- Class mapping: binary detection followed by a PanNuke0..4 classifier.
- Package source files: `baselines/shared/t08_det_baseline_inference.py`, `baselines/shared/t08_classifier_v3.py`.
- Paper/author source: [Schmidt et al. Cell Detection with Star-Convex Polygons, MICCAI 2018](https://github.com/stardist/stardist).

### `hovernet` — HoVer-Net independent fast v3

- Initialization/training: random initialization for each outer fold / external dataset; PanNuke50 epochs and external60; foreground HV MSE + NP CE/Dice + TP CE at the native image scale; external CRC uses the FP32 selection path.
- Preprocessing/adaptation: float RGB0..255; network-internal /255 applied exactly once; fast256→164 with reflection halo46 and probability stitching; retain type0 as unknown.
- Class mapping: native task head type1..5->PanNuke0..4; external1..4->0..3; 0->-1.
- Package source files: `baselines/hovernet/train_hovernet_pannuke_v3.py`, `baselines/hovernet/t06_v32_infer_per_outer_fold.py`, `baselines/hovernet/t08_hovernet_v3_indep.py`, `baselines/hovernet/t06_train_hovernet_consep.py`, `baselines/hovernet/t06_train_hovernet_crchisto.py`, `baselines/hovernet/hovernet_external_infer.py`.
- Paper/author source: [Graham et al. HoVer-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images, Medical Image Analysis 2019](https://github.com/vqdang/hover_net).

### `cellvit` -Cellnura-cellvit

- Datasets/tier: shared-detection classifier comparison on PanNuke, CoNSeP, and CRCHisto; supported actions: train, infer. The method name in the manifest is Cellvit.
- Initialization/training: the CellViT-256 encoder and type head are randomly initialized and trained separately for each of the three PanNuke test folds and independently for CoNSeP/CRCHisto, all with image-level train/val partitions. The historical original run directory name `scratch` records this provenance.
- Preprocessing/adaptation: RGB/255 followed by `(x-.5)/.5`; native 256 for PanNuke; external images are reflection-padded to multiples of 16; `nuclei_type_map` argmax is read at shared detection centroids. IDs and coordinates remain unchanged.
- Class mapping: channel 0→unknown -1; PanNuke channels 1..5→manuscript 0..4; external channels 1..4→0..3. Unknown predictions remain eligible for geometric matching.
- Package source files: `baselines/cellnura_cellvit/train_pannuke.py`, `train_external.py`, `preflight.py`, `infer_fixed_detections.py`. Identities and SHA-256 values are recorded in `baselines/WEIGHTS_EXTERNAL.json` and agree with `baselines.cellnura_cellvit.checkpoint_sha256` in `configs/release.json`.
- CLI registry: `baselines/catalog.json`.
- Paper/author source: [Hörst et al. CellViT, Medical Image Analysis 2024](https://github.com/TIO-IKIM/CellViT) provides the architecture.

### `tsfd` — TSFD-Net fold-specific scratch

- Datasets/tier: {'pannuke': 'A'}; supported actions: train, infer.
- Initialization/training: random initialization, without ef4.h5 or a public task model; 150 epochs; seed42+fold; TF2.15 SGDmomentum.9 wd5e-4 lr.16 warmup3 cosine; author-corrected losses; best full validation total loss; BF16 training / FP32 exported inference.
- Preprocessing/adaptation: RGB+HED6 channels; deterministic per-image cache; 256->512 model input with inverse centroid scaling.
- Class mapping: PanNuke5 foreground channels; all spatial predictions retained.
- Package source files: `baselines/tsfd/tsfd_runtime.py`, `baselines/tsfd/tsfd_evaluate.py`, `baselines/tsfd/common.py`, `baselines/tsfd/data.py`, `baselines/tsfd/training.py`, `baselines/tsfd/train.py`, `baselines/tsfd/prepare_cache.py`, `baselines/tsfd/evaluate_fold.py`.
- Paper/author source: [TSFD-Net author implementation](https://github.com/Mr-TalhaIlyas/TSFD).

### `cellvitpp` — CellViT++ public-weight references and dataset-specific classification heads

- Initialization/training: CellViT++ official public backbone weights + official 1024-input CoNSeP classification head, with strict shape/load checks; inference only; exact head train/val provenance is unverified; tier B is excluded from the independent ranking.
- Preprocessing/adaptation: native1000 reflection-padded to1008 full image; bbox mean token pooling; geometric centroid of the cropped instance mask; this project adapter differs from the official1024 stitched benchmark.
- Class mapping: official head0misc->3,1infl->0,2epi->1,3spindle->2.
- Package source files: `baselines/cellvitpp/t08_cellvitpp_consep.py`.
- Paper/author source: [CellViT++](https://github.com/TIO-IKIM/CellViT-plus-plus).

### `hovernet_official` — HoVer-Net official CoNSeP public reference

- Initialization/training: identified official CoNSeP checkpoint; the exact source/test separation must be disclosed; inference only, public reference B.
- Preprocessing/adaptation: native original270→80 or fast256→164 selected to match the checkpoint; float0..255 with internal /255; project stitching.
- Class mapping: official1misc->3,2infl->0,3epi->1,4spindle->2;0unknown->-1.
- Package source files: `baselines/hovernet/hovernet_external_infer.py`, `baselines/hovernet/t06_consep_infer.py`, `baselines/hovernet/t08_hovernet_external_eval.py`.
- Paper/author source: [HoVer-Net](https://github.com/vqdang/hover_net).

### `hovernet_tphead` — HoVer-Net legacy frozen backbone / folded type-head reference

- Initialization/training: fold-specific type head with a historical shared segmentation backbone whose PanNuke independence is unverified; tier B reference only; this upload does not provide a new type-head trainer.
- Preprocessing/adaptation: fast256→164, four windows with per-window lookup at the supplied official detection centers.
- Class mapping: type-1, unknown retained.
- Package source files: `baselines/hovernet/t08_hovernet_tphead_eval.py`.
- Paper/author source: [HoVer-Net project adaptation](https://github.com/vqdang/hover_net).

### `traditional` — 8 sklearn controlled shallow classifiers

- Initialization/training: sklearn random seed42; shared independent HoVer detections/features; fixed eight canonical hyperparameter sets; StandardScaler fits training data only; image-level validation; testing always uses fullGT.
- Preprocessing/adaptation: 17 morphology +36 FR =53 shallow features; unmatched nuclei excluded from train/val; all test detections receive predictions.
- Class mapping: PanNuke0..4.
- Package source files: `baselines/traditional/t09_traditional_classifiers.py`, `baselines/traditional/baseline_feature_columns.py`.
