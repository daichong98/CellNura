# CellNura

CellNura combines learned visual embeddings with explicit morphology, chromatin, and spatial context to classify nuclei in histopathology images.

Revision package: `cellnura-second-revision-v3`.

## Installation

Use Python 3.10 or newer and install a compatible PyTorch and torchvision pair, then:

```bash
git clone https://github.com/daichong98/CellNura.git
cd CellNura
pip install -r requirements-cellnura.txt
```

Evaluation-only dependencies are in `requirements-evaluation.txt`. Baselines use the corresponding `requirements-baselines-*.txt` files.

## Data and weights

Raw images, original annotations, and model weights are obtained separately:

- [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/)
- [CoNSeP](https://github.com/vqdang/hover_net)
- [CRCHisto](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/)

Set local paths in configs/release.json, which also records backbone identities and weight hashes. Baseline protocols and weight requirements are documented in docs/BASELINES.md and baselines/WEIGHTS_EXTERNAL.json.

## Repository layout

| Path | Contents |
| --- | --- |
| `cellnura_release/` | Training, inference, and evaluation commands |
| `baselines/` | Comparison-method adapters |
| `third_party/` | Third-party source and licenses |
| `artifacts/` | Image lists and image-level data splits |
| `Result/` | Frozen predictions and evaluation scripts for Tables 2, 4, and 5 |
| `docs/` | Baseline and evaluation protocols |

## Result verification

From the repository root:

```bash
cd Result
python verify_candidate.py
```

This checks the integrity of 46 prediction tasks, image lists, and 214 real-class records. It does not score predictions against annotations.

For ground-truth reconstruction, numerical replay, and bootstrap comparisons, follow Result/README.md. These steps require the original dataset annotations. Matching and class mappings are described in docs/EVALUATION_PROTOCOL.md.

## Training and inference

From the repository root:

```bash
python -m cellnura_release train --help
python -m cellnura_release infer --help
python -m cellnura_release evaluate --help
python -m cellnura_release baseline --help
```

Training uses the image-level splits recorded in `artifacts/`.
