# CellNura
A nucleus representation-aware deep learning for cell classification in histopathology
## Introduction

**CellNura** is a comprehensive deep learning pipeline designed for precise segmentation and classification of nuclei in pathology images. It leverages a multi-feature fusion strategy, combining:
* **Local Features**: Extracted via MobileViT.
* **Global Features**: Extracted via Swin-Base.
* **Gated Cross-Scale Fusion**: A trainable gating mechanism that injects global tissue context into the nucleus-level representation.
* **Morphological Features**: Geometric properties of the nuclei.
* **Ring Features**: Radial chromatin-distribution profiles from concentric-ring sampling.
* **Graph Features**: Spatial microenvironment topology captured by a Graph Attention Network (GAT) trained on the fused representation.
<img width="4086" height="1926" alt="Figure1" src="https://github.com/user-attachments/assets/6e7d5bb8-d8a4-4b3c-9b23-0d6bf91f42a2" />
Overview of the CellNura architecture. (a) Instance segmentation provides individual nuclear masks and centroids. (b) The multi-view feature extraction module then integrates four distinct feature streams: visual representations from Swin-Base and MobileViT branches combined through a trainable gated cross-scale fusion module, morphological descriptors, radial chromatin intensity profiles derived from annular sampling, and learned spatial-neighborhood embeddings produced by a graph attention network. (c) Finally, these multi-dimensional features are concatenated and fed into an MLP classifier for fine-grained nuclear classification.


## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/CellNura.git
    cd CellNura
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    
## Model weight 

- [MobileViT](https://huggingface.co/apple/mobilevit-x-small)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [HoVer-Net](https://github.com/vqdang/hover_net) 

  
## Dataset 

- [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/)
- [CoNSeP](https://github.com/vqdang/hover_net?tab=readme-ov-file)
- [CRCHisto](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/)

## Usage

Run the scripts in numerical order. Ensure you have configured the paths in each script to point to your dataset location.

```bash
# 1. Preprocess Data
python step0_data_preprocessor.py

# 2. Run Segmentation
python step1_hovernet_batch.py

# ... (Run steps 2 through 9)

# 10. Train Classifier
python train_nucleus_classifier_true.py
```

## Requirements

*   Python 3.10+
*   PyTorch
*   Torchvision
*   NumPy
*   Pandas
*   OpenCV (opencv-python)
*   Scikit-learn
*   Scikit-image
*   Transformers (Hugging Face)
*   Timm
*   Matplotlib
*   Seaborn
*   Tqdm

## License

[MIT License](LICENSE)
