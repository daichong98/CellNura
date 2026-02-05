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
(a) Preprocessing Module. Raw histopathology images are first processed by a nuclear segmentation network to delineate nuclear boundaries and extract individual nucleus instances, yielding high-quality nuclear masks and cropped patches that serve as the input for downstream analysis. (b) Classification Module. For each segmented nucleus, CellNura constructs a multi-source representation by combining four complementary groups of features: Context-aware Appearance Features (CAF) obtained via cross-scale cross-attention between tile-level and nucleus-level representations, Morphological Structure Features (MSF) derived from contour-based morphometric descriptors, Chromatin Distribution Features (CDF) capturing intra-nuclear and perinuclear chromatin distribution through multi-scale annular sampling, and Microenvironment Graph Features (MGF) modeling local tissue topology using a GAT. These features are concatenated and fed into a multilayer perceptron classifier, which is trained with a cross-entropy loss to achieve accurate and robust nuclear cell type classification.

## Pipeline Overview

The project is structured into sequential steps:

1.  **Step 0: Data Preprocessing** (`step0_data_preprocessor.py`)
    *   Prepares the dataset for processing.
2.  **Step 1: Segmentation** (`step1_hovernet_batch.py`)
    *   Runs HoverNet to generate instance segmentation masks for all images.
3.  **Step 2: Nuclei Extraction** (`step2_extract_nuclei.py`)
    *   Crops individual nucleus images based on segmentation masks.
4.  **Step 3: Local Feature Extraction** (`step3_batch_mobilevit.py`)
    *   Uses MobileViT to extract local visual features from nucleus crops.
5.  **Step 4: Global Feature Extraction** (`step4_batch_swin.py`)
    *   Uses Swin Transformer to extract global context features from whole slide images (or large patches).
6.  **Step 5: Feature Fusion** (`step5_batch_cross_attention.py`)
    *   Applies a Cross-Attention mechanism to fuse MobileViT and Swin features.
8.  **Step 6: Morphological Features** (`step6_batch_morphological.py`)
    *   Calculates geometric features (area, perimeter, eccentricity, etc.).
9.  **Step 7: Graph Features** (`step7_gat_integrated.py`)
    *   Constructs a cell graph and extracts features using GAT.
10.  **Step 8: Ring Features** (`step8_batch_ring.py`)
    *   Extracts intensity patterns around the nuclear boundary.
11. **Step 9: Centroid Matching** (`step9_train_centroid_matcher.py`)
    *   Matches predicted centroids with ground truth centroids to assign labels.
12. **Training** (`train_nucleus_classifier_true.py`)
    *   Trains the final MLP classifier using the fused feature set.

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
