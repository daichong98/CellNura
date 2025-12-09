import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import scipy.io as sio
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from skimage import measure
import gc


class PanNukePreprocessor:

    def __init__(self, pannuke_root: str, output_dir: str):
        """
        Initialize Preprocessor

        Args:
            pannuke_root: PanNuke dataset root directory
            output_dir: Output directory
        """
        self.pannuke_root = Path(pannuke_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # PanNuke Class Mapping
        self.class_mapping = {
            1: "Neoplastic",
            2: "Inflammatory",
            3: "Connective",
            4: "Dead",
            5: "Epithelial"
        }

        print(f"PanNuke dataset path: {self.pannuke_root}")
        print(f"Output directory: {self.output_dir}")

    def extract_nuclei_from_fold(self, fold_num: int) -> Dict:
        """
        Extract nucleus info from specified fold

        Args:
            fold_num: Fold number (1, 2, 3)

        Returns:
            nuclei_data: Nucleus data dictionary
        """
        print(f" Extracting nucleus info from PanNuke Fold {fold_num}...")

        nuclei_data = {
            'images': [],
            'nuclei': [],
            'statistics': {}
        }

        # Path to masks.npy and images.npy
        # Assuming standard PanNuke structure
        fold_dir = self.pannuke_root / f"Fold {fold_num}"
        masks_path = fold_dir / "masks" / f"fold{fold_num}" / "masks.npy"
        images_path = fold_dir / "images" / f"fold{fold_num}" / "images.npy"

        if not masks_path.exists() or not images_path.exists():
            print(f" Fold {fold_num} data files not found")
            return nuclei_data

        # Load data
        try:
            masks = np.load(masks_path)
            images = np.load(images_path)
        except Exception as e:
            print(f" Failed to load data: {e}")
            return nuclei_data

        total_images = images.shape[0]
        print(f" Found {total_images} images in Fold {fold_num}")
        class_counts = {i: 0 for i in range(1, 6)}
        total_nuclei = 0

        for idx in range(total_images):
            image_name = f"fold{fold_num}_image_{idx}"
            
            # Process mask
            mask = masks[idx]  # (256, 256, 6)
            
            # Extract nuclei for each class
            image_nuclei = []
            
            for class_idx in range(5): # 0-4 channels
                class_id = class_idx + 1
                instance_mask = mask[:, :, class_idx]
                
                if np.sum(instance_mask) == 0:
                    continue
                    
                unique_instances = np.unique(instance_mask)
                unique_instances = unique_instances[unique_instances > 0]
                
                for inst_id in unique_instances:
                    # Create binary mask for this instance
                    inst_binary = (instance_mask == inst_id).astype(np.uint8)
                    
                    # Find contours
                    contours, _ = cv2.findContours(inst_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if not contours:
                        continue
                        
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    if largest_contour.shape[0] < 3:
                        continue
                        
                    # Bounding box and centroid
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cX = M["m10"] / M["m00"]
                        cY = M["m01"] / M["m00"]
                    else:
                        cX, cY = x + w/2, y + h/2
                        
                    contour_points = largest_contour.squeeze()
                    if contour_points.ndim != 2:
                        continue
                        
                    nucleus_info = {
                        'image_name': image_name,
                        'nucleus_id': int(inst_id), # Note: IDs might overlap between classes if not careful, but usually distinct in instance map? 
                                                    # Actually PanNuke instance map is per channel. So IDs are local to channel.
                                                    # We might need a global ID or just keep class info.
                        'label': class_id,
                        'label_name': self.class_mapping[class_id],
                        'contour': contour_points.tolist(),
                        'centroid': [cX, cY],
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'area': float(cv2.contourArea(largest_contour)),
                        'fold': fold_num
                    }
                    
                    image_nuclei.append(nucleus_info)
                    class_counts[class_id] += 1
                    total_nuclei += 1
            
            nuclei_data['images'].append({
                'name': image_name,
                'shape': images[idx].shape,
                'num_nuclei': len(image_nuclei),
                'fold': fold_num
            })
            
            nuclei_data['nuclei'].extend(image_nuclei)

        # Statistics
        nuclei_data['statistics'] = {
            'fold': fold_num,
            'total_images': len(nuclei_data['images']),
            'total_nuclei': total_nuclei,
            'class_distribution': class_counts,
            'class_mapping': self.class_mapping
        }

        print(f" Fold {fold_num} extraction complete:")
        print(f"  - Images: {nuclei_data['statistics']['total_images']}")
        print(f"  - Nuclei: {nuclei_data['statistics']['total_nuclei']}")
        print(f"  - Class Distribution: {nuclei_data['statistics']['class_distribution']}")

        return nuclei_data

    def process_all_folds(self):
        """Process all 3 folds"""
        all_data = {
            'images': [],
            'nuclei': [],
            'statistics': {}
        }
        
        total_class_counts = {i: 0 for i in range(1, 6)}
        
        for fold in [1, 2, 3]:
            fold_data = self.extract_nuclei_from_fold(fold)
            
            if not fold_data['images']:
                continue
                
            all_data['images'].extend(fold_data['images'])
            all_data['nuclei'].extend(fold_data['nuclei'])
            
            # Merge stats
            for cls, count in fold_data['statistics']['class_distribution'].items():
                total_class_counts[cls] += count
                
        all_data['statistics'] = {
            'total_images': len(all_data['images']),
            'total_nuclei': len(all_data['nuclei']),
            'class_distribution': total_class_counts,
            'class_mapping': self.class_mapping
        }
        
        print(f" Overall extraction complete:")
        print(f"  - Total Images: {all_data['statistics']['total_images']}")
        print(f"  - Total Nuclei: {all_data['statistics']['total_nuclei']}")
        
        # Save to JSON
        output_file = self.output_dir / "pannuke_nuclei_data.json"
        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=2) # Use default encoder for numpy types if needed, but we converted to list/float
            
        print(f" Data saved to: {output_file}")


if __name__ == "__main__":
    # Example usage
    PANNUKE_ROOT = "/path/to/PanNuke_dataset"
    OUTPUT_DIR = "./output/step0_preprocessed"
    
    preprocessor = PanNukePreprocessor(PANNUKE_ROOT, OUTPUT_DIR)
    preprocessor.process_all_folds()