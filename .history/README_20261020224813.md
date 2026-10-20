# CellNura

A nucleus representation-aware deep learning framework for nucleus classification in histopathology.

## Introduction

CellNura classifies nuclei in histopathology images from an instance segmentation. The classifier combines:

- **Global visual features.** A frozen Swin-Base backbone, `swin_base_patch4_window12_384`, reads a 384-pixel input and produces a 1024-dimensional embedding.
- **Local visual features.** A frozen MobileViT-XS backbone produces the nucleus-level embedding.
- **Gated cross-scale fusion.** A trained gate injects the global tissue context into the nucleus representation.
- **Morphological features.** Seventeen geometric descriptors of each nucleus.
- **Chromatin ring features.** Thirty-six radial chromatin measurements, `FR1_2` and `FR3`–`FR37`.
- **Graph features.** A graph attention network summarizes the spatial neighborhood of each nucleus.
- **MLP classifier.** The concatenated representation is classified by an MLP. Dropout is 0.3. Class weights are `1/sqrt(n_k)`.

The pipeline is segmentation followed by classification. It is not one end-to-end network. Instance segmentation isolates each nucleus. The feature module then builds the visual, morphological, chromatin, and neighborhood representation. The MLP assigns the nucleus class.

## Repository layout

```text
CellNura/
├── cellnura_release/     # train, infer, match, evaluate
├── Result/               # 46 frozen predictions for Tables 2, 4, and 5
├── baselines/            # comparison-method adapters
├── third_party/          # upstream snapshots used by those adapters
├── artifacts/            # image-level splits and test-image lists
├── configs/              # paths and recorded weight hashes
├── docs/                 # baseline notes and evaluation protocol
├── provenance/           # MobileViT identity files
├── tests/
├── data/README.md        # no images
└── requirements-*.txt    # one file per environment
```

`Result/` contains 46 prediction tasks and 214 real-class records. It does not contain raw images, official annotations, or model weights.

`third_party/` contains the upstream snapshots used by the comparison adapters: CellViT, CellViT++, HoVer-Net (`hovernet` and `hovernet_original`), DIST (`dist_port`), and StarDist. Reproducing Tables 2, 4, and 5 reads the frozen predictions in `Result/` and does not use these trees. TSFD is fetched separately with `python third_party/fetch_source.py tsfd`. Mask R-CNN, Micro-Net, and the from-scratch CellViT-256 adapter are under `baselines/`.

## Installation

```bash
git clone https://github.com/daichong98/CellNura.git
cd CellNura
pip install -r requirements-cellnura.txt
```

Install a compatible PyTorch and torchvision pair before that requirements file. Evaluation-only scoring needs `requirements-evaluation.txt`. Baseline adapters use the additional `requirements-baselines-*.txt` files that match the method.

Python 3.10 or newer. The CellNura environment uses NumPy, pandas, SciPy, scikit-learn, scikit-image, OpenCV, Pillow, timm, and transformers.

## Datasets

The images and official nucleus annotations are not in this repository. Download them from the owners after requesting access:

- [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/)
- [CoNSeP](https://github.com/vqdang/hover_net)
- [CRCHisto](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/)

Point `configs/release.json` at the local copies. Do not commit the images or annotations.

## Backbones and weights

CellNura uses these frozen backbones. The weight files are not shipped in this repository. Their expected SHA-256 values are in `configs/release.json`.

- Swin-Base, ImageNet-22k: [Swin Transformer](https://github.com/microsoft/Swin-Transformer), saved as `weights/swin_base_patch4_window12_384_22k.pth`
- MobileViT-XS: [apple/mobilevit-x-small](https://huggingface.co/apple/mobilevit-x-small), saved locally as `weights/mobilevitv3_xs`

Comparison-method checkpoints, including [HoVer-Net](https://github.com/vqdang/hover_net), CellViT, and StarDist, are also obtained separately. `docs/BASELINES.md` and `baselines/WEIGHTS_EXTERNAL.json` record which file each row uses. The official CellViT++ weight on PanNuke is the CellViT-SAM-H x40 checkpoint.

## Reproduce Tables 2, 4, and 5

From `Result/`:

```bash
python verify_candidate.py
```

This checks that the 46 prediction files, image lists, and 214 class records still match the manifest. It does not itself score predictions against the official annotations.

The numerical replay of this 46-task package is complete. All 46 tasks and all 214 class-count registrations were scored against the reconstructed public annotations, and the scores matched the registered task and class records. The paired bootstrap is also complete: all 23 comparisons were recomputed with 10,000 image-level draws and seed 42, and each interval matched the reference in `Result/bootstrap.py`.

The commands below are the procedure for that completed replay and bootstrap. They use the parent directory as the code root.

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

The completed replay matched the registered task and class scores. The completed bootstrap matched the 23 paired intervals in `Result/bootstrap.py`, including StarDist. Matching is class-independent and one-to-one: maximum cardinality, then minimum distance, at 12 px on PanNuke, 5 px on CoNSeP, and 10 px on CRCHisto.

## Train and run CellNura

```bash
python -m cellnura_release train --help
python -m cellnura_release infer --help
python -m cellnura_release baseline --help
python -m unittest discover -s tests -v
```

Training reads the image-level split in `artifacts/`. File checks and numerical replay for Tables 2, 4, and 5 are the `Result/` commands above. They use `Result/manifest.json` and `Result/checksums.txt`.

`python -m cellnura_release verify` and `python -m cellnura_release evaluate --verify-only` run that same `Result/` file check. This release has no `release_manifest.json` or `artifacts/artifact_manifest.json`. Numerical scoring remains `Result/replay_candidate.py`.

## License

This directory contains the CellNura code, baseline adapters, frozen predictions, and the upstream snapshots under `third_party/`. Those snapshots keep the upstream license files shipped with them (`LICENSE` or `LICENSE.upstream`). TSFD is not included.
