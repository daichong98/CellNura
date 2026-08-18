# CellNura
A nucleus representation-aware deep learning for cell classification in histopathology
## Introduction

**CellNura** is a comprehensive deep learning pipeline designed for precise segmentation and classification of nuclei in pathology images. It leverages a multi-feature fusion strategy, combining:
* **Local Features**: Extracted via MobileViT.
* **Global Features**: Extracted via Swin Transformer.
* **Gated Cross-Scale Fusion**: A trainable gating mechanism that injects global tissue context into the nucleus-level representation.
* **Morphological Features**: Geometric properties of the nuclei.
* **Ring Features**: Radial chromatin-distribution profiles from concentric-ring sampling.
* **Graph Features**: Spatial microenvironment topology captured by a Graph Attention Network (GAT) trained on the fused representation.
<img width="6129" height="2889" alt="model" src="https://github.com/user-attachments/assets/77cd698c-2f05-4464-ac9d-83b0b96188a2" />
The workflow of the CellNura framework. The pipeline begins with (a) instance segmentation to isolate single nuclei and their masks. (b) The multi-view feature extraction module then integrates four distinct feature streams: deep visual representations (Swin Transformer global and MobileViT local branches combined by a fully trained gated cross-scale fusion), morphological descriptors, chromatin distribution profiles derived from annular sampling, and spatial microenvironmental topology features modeled by a graph attention network. (c) Finally, these multi-dimensional features are concatenated and fed into an MLP classifier for fine-grained nuclear classification.


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
