
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import MobileViTForImageClassification, MobileViTImageProcessor
from PIL import Image
import torch
import re
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from torch.cuda.amp import autocast


class PanNukeMobileViTExtractor:
    def __init__(self, model_path, output_dir, device=None, batch_size=64):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        # Preload model, all threads share the same model instance
        print(f"Loading MobileViT model (Device: {self.device})...")
        self.model, self.feature_extractor = self.load_model_and_feature_extractor()
        print("Model loading complete")

        # Clean GPU cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def load_model_and_feature_extractor(self):
        """Load MobileViT model and feature extractor"""
        model = MobileViTForImageClassification.from_pretrained(
            self.model_path, output_hidden_states=True
        )

        if self.device == 'cuda':
            model = model.half()  # Use half precision

        model = model.to(self.device)
        model.eval()
        feature_extractor = MobileViTImageProcessor.from_pretrained(
            self.model_path)
        return model, feature_extractor

    def extract_nucleus_info(self, file_name):
        """Extract image ID, nucleus ID and fold info from nucleus image filename"""
        match = re.match(
            r"fold(\d+)_image_(\d+)_nucleus_(\d+)\.png", file_name)

        if match:
            fold_idx = int(match.group(1))
            image_idx = int(match.group(2))
            nucleus_idx = int(match.group(3))
            return fold_idx, image_idx, nucleus_idx

        return None, None, None

    def process_nucleus_batch(self, nucleus_files, image_names):
        """Batch process multiple nucleus images, extract MobileViT features"""
        try:
            # Prepare batch data
            images = []
            valid_indices = []

            # Preprocess all images
            for i, nucleus_file in enumerate(nucleus_files):
                try:
                    image = Image.open(nucleus_file).convert("RGB")
                    image = image.resize((64, 64), resample=Image.BILINEAR)
                    images.append(image)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Cannot load image {nucleus_file}: {e}")

            if not images:
                return []

            # Batch preprocessing
            with torch.no_grad():
                inputs = self.feature_extractor(
                    images=images, return_tensors="pt")

                if self.device != 'cpu':
                    inputs = {key: val.to(self.device)
                              for key, val in inputs.items()}

                # Use mixed precision acceleration
                with autocast(enabled=(self.device == 'cuda')):
                    outputs = self.model(**inputs, output_hidden_states=True)

                # Get last hidden state
                last_hidden_state = outputs.hidden_states[-1]
                global_features = torch.mean(last_hidden_state, dim=(2, 3))

                # Move features to CPU and convert to numpy
                if self.device != 'cpu':
                    global_features = global_features.cpu()

                all_features = global_features.numpy()

            # Build feature dictionary list
            results = []
            for idx, feature_idx in enumerate(valid_indices):
                nucleus_file = nucleus_files[feature_idx]
                image_name = image_names[feature_idx]
                nucleus_filename = os.path.basename(nucleus_file)
                fold_idx, image_idx, nucleus_idx = self.extract_nucleus_info(
                    nucleus_filename)

                if fold_idx is None:
                    continue

                # Build feature dictionary
                feature_dict = {
                    "image_name": nucleus_filename,
                    "original_image": image_name,
                    "nucleus_id": nucleus_idx,
                    "fold": fold_idx
                }

                # Add feature vector
                for i, val in enumerate(all_features[idx]):
                    feature_dict[f"mobilevit_feature_{i}"] = val

                results.append(feature_dict)

            return results

        except Exception as e:
            print(f"Batch processing failed: {e}")
            return []

    def process_pannuke_dataset_batch(self, nuclei_folder, max_workers=16, gpu_batch_size=64):
        """Use multi-threading to batch process PanNuke dataset (all folds)"""

        self.batch_size = gpu_batch_size
        start_time = time.time()

        nuclei_folder = Path(nuclei_folder)
        if not nuclei_folder.exists():
            print(f"Nuclei folder does not exist: {nuclei_folder}")
            return False

        # Get all nucleus files
        fold1_nucleus_files = list(nuclei_folder.glob("fold1_*_nucleus_*.png"))
        fold2_nucleus_files = list(nuclei_folder.glob("fold2_*_nucleus_*.png"))
        fold3_nucleus_files = list(nuclei_folder.glob("fold3_*_nucleus_*.png"))
        all_nucleus_files = fold1_nucleus_files + \
            fold2_nucleus_files + fold3_nucleus_files

        # Extract unique image names
        image_names = set()
        for nucleus_file in all_nucleus_files:
            match = re.match(
                r"(fold\d+_image_\d+)_nucleus_\d+\.png", nucleus_file.name)

            if match:
                image_names.add(match.group(1))

        image_names = sorted(list(image_names))

        fold1_images = [
            name for name in image_names if name.startswith("fold1_")]
        fold2_images = [
            name for name in image_names if name.startswith("fold2_")]
        fold3_images = [
            name for name in image_names if name.startswith("fold3_")]

        print(f"Found images:")
        print(
            f"  Fold 1: {len(fold1_images)} images, {len(fold1_nucleus_files)} nuclei")
        print(
            f"  Fold 2: {len(fold2_images)} images, {len(fold2_nucleus_files)} nuclei")
        print(
            f"  Fold 3: {len(fold3_images)} images, {len(fold3_nucleus_files)} nuclei")
        print(f"  Total: {len(image_names)} images, {len(all_nucleus_files)} nuclei")

        all_features = []

        # Batch process by image, avoid collecting all nuclei at once
        print(f"Start processing {len(image_names)} images...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed_nuclei = 0
            total_nuclei = len(all_nucleus_files)  # Total nuclei count

            # Process images by batch
            for batch_idx in range(0, len(image_names), 50):  # 50 images per batch
                batch_images = image_names[batch_idx:batch_idx+50]

                # Collect nucleus files only for current batch images
                batch_files = []
                batch_img_names = []
                for image_name in batch_images:
                    nucleus_pattern = os.path.join(
                        nuclei_folder, f"{image_name}_nucleus_*.png")
                    nucleus_files = glob.glob(nucleus_pattern)

                    for nucleus_file in nucleus_files:
                        batch_files.append(nucleus_file)
                        batch_img_names.append(image_name)

                print(
                    f"Batch {batch_idx//50 + 1}/{(len(image_names)+49)//50}: Collected {len(batch_files)} nuclei")

                # Split by GPU batch
                futures = []
                for i in range(0, len(batch_files), gpu_batch_size):
                    end_idx = min(i + gpu_batch_size, len(batch_files))
                    sub_batch_files = batch_files[i:end_idx]
                    sub_batch_imgs = batch_img_names[i:end_idx]

                    futures.append(
                        executor.submit(self.process_nucleus_batch,
                                        sub_batch_files, sub_batch_imgs)
                    )

                # Process results
                batch_features = []
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"Processing batch {batch_idx//50 + 1}"):
                    result = future.result()
                    batch_features.extend(result)

                processed_nuclei += len(batch_files)
                all_features.extend(batch_features)

                # Calculate speed
                elapsed_time = time.time() - start_time
                nuclei_per_sec = processed_nuclei / elapsed_time
                remaining_time = (total_nuclei - processed_nuclei) / \
                    nuclei_per_sec if nuclei_per_sec > 0 else 0

                print(
                    f"Total progress: {processed_nuclei}/{total_nuclei} ({processed_nuclei/total_nuclei*100:.1f}%)")
                print(
                    f"Processing speed: {nuclei_per_sec:.1f} nuclei/sec | Elapsed time: {elapsed_time/60:.1f} minutes | Remaining: {remaining_time/60:.1f} minutes")

                # Periodically save interim results
                if len(all_features) >= 10000 and len(all_features) % 10000 < 200:
                    self.save_interim_features(all_features, processed_nuclei)

        # Save all features
        if all_features:
            self.save_features(all_features)
            return True
        else:
            return False

    def save_interim_features(self, features_list, processed_count):
        """Save interim feature results"""
        interim_dir = self.output_dir / "interim"
        interim_dir.mkdir(exist_ok=True)

        interim_file = interim_dir / f"features_interim_{processed_count}.csv"
        df = pd.DataFrame(features_list)
        df.to_csv(interim_file, index=False)

    def save_features(self, features_list):
        """Save features to CSV file"""
        df = pd.DataFrame(features_list)

        # Sort by fold, original image and nucleus ID
        df.sort_values(by=["fold", "original_image",
                       "nucleus_id"], inplace=True)

        # Save complete feature file
        output_csv = self.output_dir / "pannuke_mobilevit_features.csv"
        df.to_csv(output_csv, index=False)

        # Processing statistics
        fold1_count = len(df[df['fold'] == 1])
        fold2_count = len(df[df['fold'] == 2])
        fold3_count = len(df[df['fold'] == 3])
        feature_dim = len(
            [col for col in df.columns if col.startswith('mobilevit_feature')])

        print(f"Processing statistics:")
        print(f"  Fold 1: {fold1_count} nuclei")
        print(f"  Fold 2: {fold2_count} nuclei")
        print(f"  Fold 3: {fold3_count} nuclei")
        print(f"  Total: {len(df)} nuclei")
        print(f"  Feature dimension: {feature_dim}")

        # Save simplified version (excluding image name column, for training)
        training_df = df.drop(columns=["image_name", "original_image"]).copy()
        training_csv = self.output_dir / "pannuke_mobilevit_features_training.csv"
        training_df.to_csv(training_csv, index=False)


def main():
    """Main function"""
    # Configure paths
    MODEL_PATH = "/path/to/mobilevitv3_xs_weight"
    NUCLEI_FOLDER = "./output/step2_nuclei_images"
    OUTPUT_DIR = "./output/step3_mobilevit"

    # Thread count and batch size - Optimized for your hardware
    max_workers = 32  # Fully utilize 32 threads
    gpu_batch_size = 4096  # Increase GPU batch size to better utilize 24GB VRAM


    # Create feature extractor
    extractor = PanNukeMobileViTExtractor(
        MODEL_PATH, OUTPUT_DIR, batch_size=gpu_batch_size)

    # Batch process dataset
    success = extractor.process_pannuke_dataset_batch(
        NUCLEI_FOLDER, max_workers, gpu_batch_size)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
