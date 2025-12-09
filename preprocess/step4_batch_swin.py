
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import timm
from PIL import Image
from torchvision import transforms
from typing import List, Tuple
import cv2


class PanNukeSwinExtractor:
    def __init__(self, model_path, output_dir):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Swin model configuration
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.WINDOW_SIZE = (384, 384) 
        self.STEP_SIZE = 256 

        # Load model
        self.model = self.load_swin_model()

    def load_swin_model(self):
        """Load Swin-Base model"""
        model = timm.create_model(
            "Swin-Base model",  ####need
            pretrained=False,
            features_only=True,
            num_classes=0
        )

        # Load pre-trained weights
        state_dict = torch.load(self.model_path, map_location="cpu")
        state_dict = {k: v for k, v in state_dict.items() if "head" not in k}
        model.load_state_dict(state_dict, strict=False)
        model.to(self.DEVICE).eval()

        return model

    def pad_image(self, image: Image.Image, window_size: Tuple[int, int], mode='reflect') -> Image.Image:
        """Pad image to make its size a multiple of window_size"""
        w, h = image.size
        win_w, win_h = window_size

        # Calculate size to pad to
        new_w = ((w + win_w - 1) // win_w) * win_w
        new_h = ((h + win_h - 1) // win_h) * win_h

        if new_w == w and new_h == h:
            return image

        # Convert to numpy array for padding
        img_np = np.array(image)

        # Calculate padding amount
        top = bottom = right = left = 0
        if h % win_h != 0:
            pad_h = new_h - h
            top = pad_h // 2
            bottom = pad_h - top
        if w % win_w != 0:
            pad_w = new_w - w
            left = pad_w // 2
            right = pad_w - left

        padded_img = cv2.copyMakeBorder(
            img_np, top, bottom, left, right,
            borderType=cv2.BORDER_REFLECT
        )
        return Image.fromarray(padded_img)

    def sliding_window(self, image: Image.Image, window_size: Tuple[int, int], step: int) -> List[Image.Image]:
        """Use sliding window to split large image into small patches"""
        w, h = image.size
        patches = []

        for y in range(0, h - window_size[1] + 1, step):
            for x in range(0, w - window_size[0] + 1, step):
                box = (x, y, x + window_size[0], y + window_size[1])
                patch = image.crop(box)
                patches.append(patch)

        return patches

    def preprocess_patch(self, patch: Image.Image) -> torch.Tensor:
        """Single image patch preprocessing"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        return transform(patch).unsqueeze(0).to(self.DEVICE)

    def extract_global_features(self, image_array):
        """Extract global features from single HE image"""
        try:
            # Convert NumPy array to PIL image
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)

            # Ensure RGB format
            if len(image_array.shape) == 2:  # Grayscale image
                image_array = np.stack(
                    [image_array, image_array, image_array], axis=-1)
            elif image_array.shape[-1] == 1:  # Single channel
                image_array = np.concatenate(
                    [image_array, image_array, image_array], axis=-1)
            elif image_array.shape[-1] > 3:  # More than 3 channels
                image_array = image_array[:, :, :3]

            img = Image.fromarray(image_array)

            # Check image size, resize if smaller than WINDOW_SIZE
            if img.width < self.WINDOW_SIZE[0] or img.height < self.WINDOW_SIZE[1]:
                print(
                    f" Image size ({img.width}x{img.height}) smaller than model requirement ({self.WINDOW_SIZE[0]}x{self.WINDOW_SIZE[1]}), resizing...")
                img = img.resize(self.WINDOW_SIZE, Image.Resampling.LANCZOS)

            # Pad image
            padded_img = self.pad_image(img, window_size=self.WINDOW_SIZE)

            # Patch processing
            patches = self.sliding_window(
                padded_img, window_size=self.WINDOW_SIZE, step=self.STEP_SIZE)

            # Extract features for all patches
            all_features = []
            for patch in patches:
                input_tensor = self.preprocess_patch(patch)

                with torch.no_grad():
                    features = self.model(input_tensor)
                    # (1, 1024, 12, 12)
                    last_layer_feat = features[-1].permute(0, 3, 1, 2)
                    pooled_feat = torch.mean(
                        last_layer_feat, dim=(2, 3))  # (1, 1024)
                    all_features.append(pooled_feat.cpu().numpy().flatten())

            # Merge features of all patches (take mean)
            final_feature = np.mean(all_features, axis=0)  # (1024,)

            return final_feature

        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def process_pannuke_dataset(self, pannuke_root):
        """Process entire PanNuke dataset (all folds)"""
        print("🏥 Start processing PanNuke dataset - Swin Transformer global feature extraction")

        pannuke_root = Path(pannuke_root)
        all_features = []

        # Process each fold
        for fold_idx in range(1, 4):
            fold_dir = pannuke_root / f"Fold {fold_idx}"
            images_path = fold_dir / "images" / \
                f"fold{fold_idx}" / "images.npy"

            if not images_path.exists():
                print(f"Cannot find Fold {fold_idx} image file: {images_path}")
                continue

            try:
                # Load image data
                images = np.load(str(images_path), mmap_mode='r')
                print(f"Loaded Fold {fold_idx} images: {images.shape}")

                # Process each image
                for img_idx in range(len(images)):
                    image_name = f"fold{fold_idx}_image_{img_idx:04d}"
                    print(f"Processing {image_name}...")

                    # Extract global features
                    global_features = self.extract_global_features(
                        images[img_idx])

                    if global_features is not None:
                        # Build feature dictionary
                        feature_dict = {
                            "image_name": image_name,
                            "fold": fold_idx
                        }

                        # Add feature vector
                        for j, val in enumerate(global_features):
                            feature_dict[f"swin_global_{j}"] = val

                        all_features.append(feature_dict)
                        print(f"Feature extraction successful, dimension: {len(global_features)}")
                    else:
                        print(f"Feature extraction failed")

            except Exception as e:
                print(f"Failed to process Fold {fold_idx}: {e}")

    def save_features(self, features_list):
        """Save features to CSV file"""
        df = pd.DataFrame(features_list)

        # Sort by fold and image name
        df.sort_values(by=["fold", "image_name"], inplace=True)

        # Save feature file
        output_csv = self.output_dir / "pannuke_swin_global_features.csv"
        df.to_csv(output_csv, index=False)

        print(f"\n Swin global features saved: {output_csv}")

        # Processing statistics
        fold1_count = len(df[df['fold'] == 1])
        fold2_count = len(df[df['fold'] == 2])
        fold3_count = len(df[df['fold'] == 3])
        feature_dim = len(
            [col for col in df.columns if col.startswith('swin_global')])

        print(f" Processing statistics:")
        print(f"  Fold 1: {fold1_count} images")
        print(f"  Fold 2: {fold2_count} images")
        print(f"  Fold 3: {fold3_count} images")
        print(f"  Total: {len(df)} images")
        print(f"  Feature dimension: {feature_dim}")


def main():
    """Main function"""
    # Configure paths
    MODEL_PATH = "/path/to/swin_base_weight.pth"
    PANNUKE_ROOT = "/path/to/PanNuke_dataset"
    OUTPUT_DIR = "/path/to/output_directory"

    # Create feature extractor
    extractor = PanNukeSwinExtractor(MODEL_PATH, OUTPUT_DIR)

    # Process dataset
    success = extractor.process_pannuke_dataset(PANNUKE_ROOT)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

