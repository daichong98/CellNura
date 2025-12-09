
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import random
import re


class CoAttention(nn.Module):
    """Co-Attention Module Definition"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_local: torch.Tensor, x_global: torch.Tensor):
        """
        x_local: (N, D), Local features for each nucleus
        x_global: (1, D), Global features for corresponding image
        Returns: (N, D), Co-attention features for each nucleus
        """
        q = self.query(x_local)       # (N, D)
        k = self.key(x_global)        # (1, D)
        v = self.value(x_local)       # (N, D)

        attn_weights = torch.matmul(
            q, k.transpose(-2, -1)) * (1.0 / np.sqrt(self.dim))  # (N, 1)
        attn_weights = self.softmax(attn_weights)  # (N, 1)

        output = v * attn_weights.expand_as(v)  # (N, D)
        return output


class PanNukeCoAttentionExtractor:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device configuration
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds
        self.set_random_seeds()

    def set_random_seeds(self, seed=42):
        """Set random seeds"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_features(self, csv_path: str):
        """Load feature file"""
        if not os.path.exists(csv_path):
            print(f" Feature file does not exist: {csv_path}")
            return None, None

        df = pd.read_csv(csv_path)

        # Check if image name column exists
        if 'image_name' in df.columns:
            image_names = df['image_name'].values
            # Remove non-feature columns
            feature_df = df.drop(columns=[col for col in [
                                 'image_name', 'original_image', 'nucleus_id', 'fold'] if col in df.columns])
        else:
            # If no image name column, use index
            image_names = df.index.values
            feature_df = df.copy()

        # Check if remaining columns are numeric
        non_numeric_cols = []
        for col in feature_df.columns:
            try:
                pd.to_numeric(feature_df[col])
            except (ValueError, TypeError):
                non_numeric_cols.append(col)

        if non_numeric_cols:
            print(f"Found non-numeric columns: {non_numeric_cols}")
            feature_df = feature_df.drop(columns=non_numeric_cols)
            print(f"Shape after removing non-numeric columns: {feature_df.shape}")

        # Convert to numeric type
        try:
            features = feature_df.values.astype(np.float32)
        except Exception as e:
            print(f"Failed to convert features to float32: {e}")
            print(f"Check first few rows of data:")
            print(feature_df.head())
            return None, None

        return image_names, features

    def extract_image_name(self, nucleus_filename):
        """Extract original image name from nucleus filename"""
        match = re.match(
            r"(fold\d+_image_\d+)_nucleus_\d+\.png", nucleus_filename)
        if match:
            return match.group(1)
        return None

    def process_coattention_features(self, mobilevit_csv, swin_csv):
        """Process co-attention features (all folds)"""
        print("🏥 Start processing PanNuke dataset - Co-attention feature fusion")

        # Load MobileViT local features
        print("📊 Load MobileViT local features...")
        mobilevit_names, mobilevit_features = self.load_features(mobilevit_csv)
        if mobilevit_features is None:
            return False

        # Load Swin global features
        swin_names, swin_features = self.load_features(swin_csv)
        if swin_features is None:
            return False
        # Read MobileViT CSV to get complete info
        mobilevit_df = pd.read_csv(mobilevit_csv)

        # Create lookup dictionary for Swin features
        swin_dict = {}
        swin_df = pd.read_csv(swin_csv)

        # Output sample image names from Swin feature file


        # Normalize image name format and create lookup dictionary
        for _, row in swin_df.iterrows():
            original_name = row['image_name']

            # Extract fold and image number (ignore format differences)
            match = re.match(r'fold(\d+)_image_(\d+)', original_name)
            if match:
                fold_num = match.group(1)
                img_num = match.group(2)

                # Add original format
                swin_dict[original_name] = row.drop(['image_name', 'fold'] if 'fold' in row else [
                                                    'image_name']).values.astype(np.float32)

                # Add format without leading zeros (fold1_image_0)
                standard_name = f"fold{fold_num}_image_{int(img_num)}"
                swin_dict[standard_name] = row.drop(['image_name', 'fold'] if 'fold' in row else [
                                                    'image_name']).values.astype(np.float32)

        print(f" Swin feature dictionary contains {len(swin_dict)} images")

        # Match corresponding global features for each nucleus
        nucleus_global_pairs = []
        for _, row in mobilevit_df.iterrows():
            nucleus_filename = row['image_name']
            original_image = row['original_image']

            # Determine corresponding global feature filename based on original_image
            if original_image.startswith('fold'):
                global_image_name = original_image
            else:
                print(f"Cannot recognize image type: {original_image}")
                continue

            if global_image_name in swin_dict:
                # Extract local features (remove non-feature columns)
                feature_cols = [col for col in mobilevit_df.columns
                                if col not in ['image_name', 'original_image', 'nucleus_id', 'fold']]
                local_features = row[feature_cols].values.astype(np.float32)
                global_features = swin_dict[global_image_name]

                nucleus_global_pairs.append({
                    'nucleus_filename': nucleus_filename,
                    'original_image': original_image,
                    'nucleus_id': row['nucleus_id'],
                    'fold': row['fold'],
                    'local_features': local_features,
                    'global_features': global_features
                })
            else:
                print(f" Global features for image {global_image_name} not found")


        if not nucleus_global_pairs:
            print(" No matched feature pairs")
            return False

        # Count for each fold
        fold1_pairs = [p for p in nucleus_global_pairs if p['fold'] == 1]
        fold2_pairs = [p for p in nucleus_global_pairs if p['fold'] == 2]
        fold3_pairs = [p for p in nucleus_global_pairs if p['fold'] == 3]
        print(
            f" Match statistics: Fold 1: {len(fold1_pairs)}, Fold 2: {len(fold2_pairs)}, Fold 3: {len(fold3_pairs)}")

        # Convert to tensor
        local_features_list = [pair['local_features']
                               for pair in nucleus_global_pairs]
        global_features_list = [pair['global_features']
                                for pair in nucleus_global_pairs]

        local_tensor = torch.tensor(
            np.array(local_features_list), dtype=torch.float32).to(self.device)
        global_tensor = torch.tensor(
            np.array(global_features_list), dtype=torch.float32).to(self.device)

        print(f" Local feature tensor: {local_tensor.shape}")
        print(f" Global feature tensor: {global_tensor.shape}")

        # Get unified dimension (take max)
        unified_dim = max(local_tensor.shape[-1], global_tensor.shape[-1])

        # If dimensions inconsistent, do linear projection
        if local_tensor.shape[-1] != global_tensor.shape[-1]:
            print(
                f" Dimension alignment: Local({local_tensor.shape[-1]}) -> Global({global_tensor.shape[-1]}) -> Unified({unified_dim})")

            local_projection = nn.Linear(
                local_tensor.shape[-1], unified_dim).to(self.device)
            global_projection = nn.Linear(
                global_tensor.shape[-1], unified_dim).to(self.device)

            local_tensor = local_projection(local_tensor)
            global_tensor = global_projection(global_tensor)

        # Build Co-Attention module
        print(" Calculate Co-attention features...")
        co_attn = CoAttention(unified_dim).to(self.device)

        with torch.no_grad():
            # For each nucleus, use corresponding global features
            coattention_features_list = []
            for i in range(local_tensor.shape[0]):
                local_single = local_tensor[i:i+1]  # (1, D)
                global_single = global_tensor[i:i+1]  # (1, D)

                coattention_single = co_attn(local_single, global_single)
                coattention_features_list.append(
                    coattention_single.cpu().numpy())

            coattention_features = np.concatenate(
                coattention_features_list, axis=0)

        # Save results
        self.save_coattention_features(
            nucleus_global_pairs, coattention_features)
        return True

    def save_coattention_features(self, nucleus_global_pairs, coattention_features):
        """Save co-attention features"""
        # Build result DataFrame
        results = []
        for i, pair in enumerate(nucleus_global_pairs):
            result_dict = {
                'image_name': pair['nucleus_filename'],
                'original_image': pair['original_image'],
                'nucleus_id': pair['nucleus_id'],
                'fold': pair['fold']
            }

            # Add co-attention features
            for j, feature_val in enumerate(coattention_features[i]):
                result_dict[f'coattention_feature_{j}'] = feature_val

            results.append(result_dict)

        df_result = pd.DataFrame(results)

        # Sort by fold, original image and nucleus ID
        df_result.sort_values(
            by=['fold', 'original_image', 'nucleus_id'], inplace=True)

        # Save complete feature file
        output_csv = self.output_dir / "pannuke_coattention_features.csv"
        df_result.to_csv(output_csv, index=False)


        # Save simplified version for training
        training_df = df_result.drop(
            columns=['image_name', 'original_image']).copy()
        training_csv = self.output_dir / "pannuke_coattention_features_training.csv"
        training_df.to_csv(training_csv, index=False)

        print(f"✅ Training feature file: {training_csv}")


def main():
    """Main function"""
    # Configure paths
    MOBILEVIT_CSV = "/path/to/step3_mobilevit/pannuke_mobilevit_features.csv"
    SWIN_CSV = "/path/to/step4_swin_global/pannuke_swin_global_features.csv"
    OUTPUT_DIR = "./output/step5_coattention"

    # Create feature extractor
    extractor = PanNukeCoAttentionExtractor(OUTPUT_DIR)
    # Process co-attention features
    success = extractor.process_coattention_features(MOBILEVIT_CSV, SWIN_CSV)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
