# CellNura
 A Nucleus Representation-Aware Deep  Learning Model for Cell Classification  in Histopathological Images
## Introduction

**CellNura** is a comprehensive deep learning pipeline designed for precise segmentation and classification of nuclei in pathology images. It leverages a multi-feature fusion strategy, combining:
*   **Local Features**: Extracted via MobileViT.
*   **Global Features**: Extracted via Swin Transformer.
*   **Cross-Attention Mechanism**: To effectively fuse local and global visual features.
*   **Morphological Features**: Geometric properties of the nuclei.
*   **Ring Features**: Texture information from the nuclear boundary.
*   **Graph Features**: Spatial relationships captured by Graph Attention Networks (GAT).
<img width="6129" height="2889" alt="model" src="https://github.com/user-attachments/assets/7233f6dc-ba84-42a3-962f-bb7f939748d6" />
The workflow of the CellNura framework. The pipeline begins with (a) instance segmentation to isolate single nuclei and their masks. (b) The Multi-View Feature Extraction module then integrates four distinct feature streams: deep visual representations (combining Swin and Vision Transformers via cross-attention), morphological descriptors, chromatin distribution profiles derived from annular sampling, and microenvironmental topology features modeled by a Graph Attention Network. (c) Finally, these multi-dimensional features are fused and fed into an MLP classifier to achieve fine-grained nuclear classification..


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
- [Swin-T](https://github.com/microsoft/Swin-Transformer)
  
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
