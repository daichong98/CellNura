# CellNura

A nucleus representation-aware deep learning framework for nucleus classification in histopathology.

Release identifier: `cellnura-second-revision-v3`.

## Introduction

CellNura classifies nuclei in histopathology images from an instance segmentation. The classifier combines:

- **Global visual features.** A frozen Swin-Base backbone, `swin_base_patch4_window12_384`, reads a 384-pixel input and produces a 1024-dimensional embedding.
- **Local visual features.** A frozen MobileViT-XS backbone produces the nucleus-level embedding.
- **Gated cross-scale fusion.** A trained gate injects the global tissue context into the nucleus representation.
- **Morphological features.** Seventeen geometric descriptors of each nucleus.
- **Chromatin ring features.** Thirty-six radial chromatin measurements, `FR1_2` and `FR3`–`FR37`.
- **Graph features.** A graph attention network summarizes the spatial neighborhood of each nucleus.
- **MLP classifier.** The concatenated representation is classified by an MLP. Dropout is 0.3. Class weights are `1/sqrt(n_k)`.
<img width="4086" height="1926" alt="Figure1" src="https://github.com/user-attachments/assets/b3e4f4ac-9b48-4949-b661-bdeffd269f4b" />
Overview of the CellNura architecture. (a) Instance segmentation provides individual nuclear masks and centroids. (b) The multi-view feature extraction module then integrates four distinct feature streams: visual representations from Swin-Base and MobileViT branches combined through a trainable gated cross-scale fusion module, morphological descriptors, radial chromatin intensity profiles derived from annular sampling, and learned spatial-neighborhood embeddings produced by a graph attention network. (c) Finally, these multi-dimensional features are concatenated and fed into an MLP classifier for fine-grained nuclear classification.

## Repository layout

| Path                            | Contents                                                            |
| ------------------------------- | ------------------------------------------------------------------- |
| `cellnura_release/`           | Training, inference, matching, and evaluation commands              |
| `Result/`                     | Frozen predictions for Tables 2, 4, and 5, plus the scoring scripts |
| `baselines/`                  | Comparison-method adapters                                          |
| `docs/BASELINES.md`           | What each comparison row trains and how it is named                 |
| `docs/EVALUATION_PROTOCOL.md` | Matching radius, class maps, and bootstrap rule                     |
| `artifacts/`                  | Test-image lists and the image-level train/validation/test split    |
| `configs/release.json`        | Paths, backbone identities, and recorded weight hashes              |
| `third_party/`                | Vendored baseline source. Each tree keeps its own upstream license  |

`Result/` contains 46 prediction tasks and 214 real-class records. It does not contain raw images, official nucleus annotations, reconstructed ground truth, or model weights.

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
- MobileViT-XS: `weights/mobilevitv3_xs`

Comparison-method checkpoints, including HoVer-Net, CellViT, and StarDist, are also obtained separately. `docs/BASELINES.md` and `baselines/WEIGHTS_EXTERNAL.json` record which file each row uses. The official CellViT++ weight on PanNuke is the CellViT-SAM-H x40 checkpoint.

## Reproduce Tables 2, 4, and 5

From `Result/`:

```bash
python verify_candidate.py
```

This checks that the 46 prediction files, image lists, and 214 class records still match the manifest. It does not score predictions against annotations.

After the three datasets are available locally, rebuild the evaluation annotations outside `Result/` and score the frozen predictions. The two scoring commands use the parent directory as the code root.

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
python bootstrap_1007.py \
  --gt-dir /tmp/scheme2_public_gt \
  --out-dir /tmp/scheme2_bootstrap_results \
  --require-reference-match
```

A passing replay reprints the registered task and class scores. A passing bootstrap reprints the 23 paired intervals, including StarDist. Matching is class-independent and one-to-one: maximum cardinality, then minimum distance, at 12 px on PanNuke, 5 px on CoNSeP, and 10 px on CRCHisto.

## Train and run CellNura

```bash
python -m cellnura_release train --help
python -m cellnura_release infer --help
python -m cellnura_release evaluate --help
python -m cellnura_release baseline --help
python -m unittest discover -s tests -v
```

Training reads the image-level split in `artifacts/`. 

## License

Third-party baseline code keeps the license files shipped under `third_party/`. This release does not add a license to an upstream repository that has none.
