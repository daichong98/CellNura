
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE


class GraphAttentionLayer(nn.Module):
    """Implementation of Graph Attention Layer"""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.4,
                 alpha: float = 0.2, concat: bool = True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Linear transformation weight matrix
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W)

        # Attention mechanism weight vector
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a)

        # LeakyReLU activation function
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation

        Parameters:
            h: Node feature matrix [N, in_features]
            adj: Adjacency matrix [N, N]

        Returns:
            Output features [N, out_features]
        """
        # Linear transformation
        Wh = torch.mm(h, self.W)

        # Calculate attention scores
        e = self._prepare_attentional_mechanism_input(Wh)

        # Mask non-adjacent cells based on adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # Apply softmax and dropout
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Weighted aggregation of neighbor features
        h_prime = torch.matmul(attention, Wh)

        # Apply non-linear activation function
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """Prepare input for attention mechanism"""
        # Calculate attention scores between all node pairs
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        # Broadcast and concatenate
        e = Wh1 + Wh2.transpose(0, 1)

        return self.leakyrelu(e)


class MultiHeadGAT(nn.Module):
    """Multi-Head Graph Attention Network"""

    def __init__(self, in_features: int, hidden_features: int, out_features: int,
                 n_heads: int = 8, dropout: float = 0.4, alpha: float = 0.2):
        super(MultiHeadGAT, self).__init__()

        # First layer multi-head attention
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(
                in_features=in_features,
                out_features=hidden_features,
                dropout=dropout,
                alpha=alpha,
                concat=True
            ) for _ in range(n_heads)
        ])

        # Output layer attention
        self.out_att = GraphAttentionLayer(
            in_features=hidden_features * n_heads,
            out_features=out_features,
            dropout=dropout,
            alpha=alpha,
            concat=False
        )

        # Save dropout rate
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation

        Parameters:
            x: Node feature matrix [N, in_features]
            adj: Adjacency matrix [N, N]

        Returns:
            Output features [N, out_features]
        """
        # First layer multi-head attention
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        # Output layer
        x = self.out_att(x, adj)

        return x


class SpatialGraphBuilder:
    """Build spatial relationship graph between nuclei"""

    def __init__(self, distance_threshold: float = 40.0):
        """
        Initialize spatial graph builder

        Parameters:
            distance_threshold: Maximum distance (pixels) for two nuclei to be considered neighbors
        """
        self.distance_threshold = distance_threshold

    def build_adjacency_matrix(self, cell_coords: np.ndarray) -> np.ndarray:
        """
        Build adjacency matrix based on nucleus coordinates

        Parameters:
            cell_coords: Nucleus center coordinates, shape [N, 2]

        Returns:
            Adjacency matrix, shape [N, N]
        """
        cell_num = cell_coords.shape[0]

        # Calculate Euclidean distance between all cell pairs
        distances = cdist(cell_coords, cell_coords, metric='euclidean')

        # Create adjacency matrix: if distance < threshold, cells are neighbors
        adj_matrix = np.zeros((cell_num, cell_num))
        adj_matrix[distances < self.distance_threshold] = 1

        # Ensure diagonal values are 1 (self-connection)
        np.fill_diagonal(adj_matrix, 1)

        return adj_matrix

    def build_batch_adjacency_matrix(self, batch_coordinates: torch.Tensor) -> torch.Tensor:
        """Build adjacency matrix for batch data"""
        batch_size = batch_coordinates.shape[0]
        num_cells = batch_coordinates.shape[1]

        # Initialize batch adjacency matrix
        batch_adj = torch.zeros(batch_size, num_cells, num_cells)

        # Build adjacency matrix for each sample
        for b in range(batch_size):
            coords = batch_coordinates[b].cpu().numpy()
            adj_matrix = self.build_adjacency_matrix(coords)
            batch_adj[b] = torch.from_numpy(adj_matrix).float()

        return batch_adj


class CellularGAT(nn.Module):
    """Graph Attention Network model for nucleus classification"""

    def __init__(self, input_dim: int = 100, hidden_dim: int = 128, output_dim: int = 256,
                 n_heads: int = 8, dropout: float = 0.4):
        super(CellularGAT, self).__init__()

        # Graph Attention Network
        self.gat = MultiHeadGAT(
            in_features=input_dim,
            hidden_features=hidden_dim,
            out_features=output_dim,
            n_heads=n_heads,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation

        Parameters:
            x: Node feature matrix [B, N, F]
            adj: Adjacency matrix [B, N, N]

        Returns:
            Output features [B, N, output_dim]
        """
        batch_size, num_cells, _ = x.shape
        output = torch.zeros(batch_size, num_cells,
                             self.gat.out_att.out_features, device=x.device)

        # Apply GAT to each sample in batch separately
        for b in range(batch_size):
            output[b] = self.gat(x[b], adj[b])

        return output


class HoVerNetDataProcessor:
    """Process HoVerNet segmentation results, extract nucleus positions and features"""

    def __init__(self, hovernet_dir: str):
        self.hovernet_dir = hovernet_dir
        self.json_files = self._get_json_files()

    def _get_json_files(self) -> List[str]:
        """Get all JSON segmentation result files"""
        json_files = []
        for file in os.listdir(self.hovernet_dir):
            if file.endswith("_segmentation.json"):
                json_files.append(os.path.join(self.hovernet_dir, file))
        return json_files

    def extract_nuclei_data(self, json_file: str) -> Dict[int, Tuple[float, float]]:
        """Extract nucleus ID and centroid coordinates from single JSON file"""
        with open(json_file, 'r') as f:
            data = json.load(f)

        nuclei_data = {}
        for tile in data.get('tiles', []):
            for nuc_id, nuc_info in tile.get('nuc', {}).items():
                if 'centroid' in nuc_info:
                    nuclei_data[int(nuc_id)] = tuple(nuc_info['centroid'])

        return nuclei_data

    def process_all_files(self) -> Dict[str, Dict[int, Tuple[float, float]]]:
        """Process all JSON files, extract nucleus data"""
        all_nuclei_data = {}

        for json_file in tqdm(self.json_files, desc="Processing segmentation results"):
            image_name = os.path.basename(
                json_file).replace("_segmentation.json", "")
            nuclei_data = self.extract_nuclei_data(json_file)
            all_nuclei_data[image_name] = nuclei_data

        return all_nuclei_data


class FeatureProcessor:
    """Process and merge various features"""

    def __init__(self, feature_files: Dict[str, str]):
        self.feature_files = feature_files
        self.features = self._load_features()

    def _load_features(self) -> Dict[str, pd.DataFrame]:
        """Load all feature files"""
        features = {}
        for feature_type, file_path in self.feature_files.items():
            if os.path.exists(file_path):
                features[feature_type] = pd.read_csv(file_path)
            else:
                print(f"Warning: Feature file does not exist: {file_path}")

        return features

    def get_features_by_image(self, image_name: str) -> Dict[str, pd.DataFrame]:
        """Get all features for specified image"""
        image_features = {}

        for feature_type, df in self.features.items():
            # First check 'original_image' column
            if 'original_image' in df.columns:
                # Filter features for specific image
                image_df = df[df['original_image'] == image_name]
                if not image_df.empty:
                    image_features[feature_type] = image_df
                    continue

            # If no 'original_image' column, try 'image_name' column
            if 'image_name' in df.columns:
                # Filter features for specific image
                image_df = df[df['image_name'] == image_name]
                if not image_df.empty:
                    image_features[feature_type] = image_df
            else:
                print(f"Warning: No image_name or original_image column in {feature_type} features")

        return image_features

    def merge_features(self, image_name: str) -> pd.DataFrame:
        """Merge all features for specified image"""
        image_features = self.get_features_by_image(image_name)

        if not image_features:
            print(f"Warning: Image {image_name} has no available features")
            return pd.DataFrame()

        # Use first feature as base
        base_feature_type = list(image_features.keys())[0]
        merged_df = image_features[base_feature_type].copy()

        # Merge other features
        for feature_type, df in image_features.items():
            if feature_type != base_feature_type:
                # Ensure columns for merging exist
                if 'nucleus_id' in merged_df.columns and 'nucleus_id' in df.columns:
                    # Exclude duplicate columns
                    columns_to_use = [
                        col for col in df.columns if col not in merged_df.columns or col == 'nucleus_id']
                    if columns_to_use:
                        merged_df = pd.merge(
                            merged_df, df[columns_to_use], on='nucleus_id', how='inner')
                else:
                    print(f"Warning: Cannot merge {feature_type} features, missing nucleus_id column")

        return merged_df


class GATFeatureGenerator:
    """Use GAT to generate features considering spatial relationships"""

    def __init__(self,
                 input_dim: int = 100,
                 hidden_dim: int = 128,
                 output_dim: int = 256,
                 n_heads: int = 8,
                 dropout: float = 0.4,
                 distance_threshold: float = 40.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # Initialize GAT model
        self.gat_model = CellularGAT(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_heads=n_heads,
            dropout=dropout
        ).to(device)

        # Initialize graph builder
        self.graph_builder = SpatialGraphBuilder(
            distance_threshold=distance_threshold)

    def generate_features(self,
                          nuclei_data: Dict[str, Dict[int, Tuple[float, float]]],
                          feature_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate GAT features for all images"""
        enhanced_features = {}
        self.gat_model.eval()

        for image_name, nuclei_info in tqdm(nuclei_data.items(), desc="Generating GAT features"):
            if image_name not in feature_data:
                print(f"Warning: Image {image_name} has no feature data, skipping")
                continue

            df = feature_data[image_name]
            features, coords, nucleus_ids = self._prepare_data(df, nuclei_info)

            if features is None or coords is None:
                print(f"Warning: Image {image_name} data preparation failed, skipping")
                continue

            # Build adjacency matrix
            adj_matrix = self.graph_builder.build_adjacency_matrix(coords)

            # Convert to tensor
            features_tensor = torch.FloatTensor(
                features).unsqueeze(0).to(self.device)
            adj_tensor = torch.FloatTensor(
                adj_matrix).unsqueeze(0).to(self.device)

            # Generate GAT features
            with torch.no_grad():
                gat_features = self.gat_model(features_tensor, adj_tensor)

            # Convert back to NumPy array
            gat_features_np = gat_features.squeeze(0).cpu().numpy()

            # Create feature DataFrame
            gat_df = pd.DataFrame({
                'nucleus_id': nucleus_ids,
                'image_name': image_name
            })

            # Add GAT feature columns
            for i in range(self.output_dim):
                gat_df[f'coattention_feature_{i}'] = gat_features_np[:, i]

            enhanced_features[image_name] = gat_df

        return enhanced_features

    def _prepare_data(self,
                      df: pd.DataFrame,
                      nuclei_info: Dict[int, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Prepare input data for GAT model"""
        # Check if nucleus_id column exists
        if 'nucleus_id' not in df.columns:
            return None, None, None

        # Get feature columns - exclude non-numeric and specific columns
        exclude_cols = ['image_name', 'original_image',
                        'nucleus_id', 'split', 'label', 'tile_x', 'tile_y', 'mag']
        feature_cols = [col for col in df.columns
                        if col not in exclude_cols
                        and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]

        if not feature_cols:
            return None, None, None

        # Extract features and IDs
        nucleus_ids = df['nucleus_id'].tolist()
        features = df[feature_cols].values

        # Ensure features are float type
        try:
            if isinstance(features, np.ndarray) and features.dtype == np.object_:
                features = np.vstack(
                    [np.array(f, dtype=np.float32) for f in features])
            features = features.astype(np.float32)
        except (ValueError, TypeError):
            return None, None, None

        # Ensure input dimensions are correct
        if features.shape[1] != self.input_dim:
            # If dimensions mismatch, pad or truncate
            if features.shape[1] < self.input_dim:
                # Pad
                padding = np.zeros(
                    (features.shape[0], self.input_dim - features.shape[1]), dtype=np.float32)
                features = np.hstack((features, padding))
            else:
                # Truncate
                features = features[:, :self.input_dim]

        # Extract coordinates
        coords = np.array([nuclei_info.get(nuc_id, (0, 0))
                          for nuc_id in nucleus_ids])

        return features, coords, nucleus_ids


def process_batch(batch_data, gat_generator):
    """Process a batch of image data"""
    batch_results = []

    for item in batch_data:
        image_name = item['image_name']
        nucleus_ids = item['nucleus_ids']
        features = item['features']
        coords = item['coords']

        # Ensure features are float32 type
        if isinstance(features, np.ndarray) and features.dtype == np.object_:
            try:
                features = np.vstack(
                    [np.array(f, dtype=np.float32) for f in features])
            except ValueError:
                print(f"Warning: Features for {image_name} cannot be converted to tensor, skipping")
                continue

        # Build adjacency matrix
        adj_matrix = gat_generator.graph_builder.build_adjacency_matrix(coords)

        # Convert to tensor
        try:
            features_tensor = torch.FloatTensor(
                features).unsqueeze(0).to(gat_generator.device)
            adj_tensor = torch.FloatTensor(
                adj_matrix).unsqueeze(0).to(gat_generator.device)
        except TypeError:
            print(f"Warning: Feature conversion failed for {image_name}, skipping")
            continue

        # Generate GAT features
        with torch.no_grad():
            gat_features = gat_generator.gat_model(features_tensor, adj_tensor)

        # Convert back to NumPy array
        gat_features_np = gat_features.squeeze(0).cpu().numpy()

        # Create result dictionary
        result = {
            'image_name': image_name,
            'nucleus_ids': nucleus_ids,
            'gat_features': gat_features_np
        }

        batch_results.append(result)

    return batch_results


def setup_arg_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(description='Integrated GAT nucleus classification processing')
    parser.add_argument('--mode', type=str, default='batch',
                        choices=['batch', 'single', 'demo'],
                        help='Run mode: batch, single, demo')
    parser.add_argument('--hovernet_dir', type=str, default='/path/to/step1_hovernet_results')
    parser.add_argument('--mobilevit_features', type=str,
                        default='/path/to/step3_mobilevit/pannuke_mobilevit_features.csv')
    parser.add_argument('--output_dir', type=str, default='./output/step7_gat')
    parser.add_argument('--input_dim', type=int, default=100,
                        help='Input feature dimension')
    parser.add_argument('--output_dim', type=int, default=256,
                        help='Output feature dimension')
    parser.add_argument('--distance_threshold', type=float, default=40.0,
                        help='Nucleus distance threshold')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Computing device')
    parser.add_argument('--single_file', type=str,
                        help='JSON file path for single image mode')

    return parser


def batch_mode(args):
    """Batch processing mode"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process HoVerNet segmentation results
    hovernet_processor = HoVerNetDataProcessor(args.hovernet_dir)
    nuclei_data = hovernet_processor.process_all_files()

    # Load features
    feature_files = {
        'mobilevit': args.mobilevit_features
    }

    feature_processor = FeatureProcessor(feature_files)

    # Prepare features for each image
    image_features = {}
    for image_name in tqdm(nuclei_data.keys(), desc="Merging features"):
        merged_features = feature_processor.merge_features(image_name)
        if not merged_features.empty:
            image_features[image_name] = merged_features

    # Initialize GAT feature generator
    gat_generator = GATFeatureGenerator(
        input_dim=args.input_dim,
        hidden_dim=args.input_dim // 2,
        output_dim=args.output_dim,
        n_heads=8,
        dropout=0.4,
        distance_threshold=args.distance_threshold,
        device=args.device
    )

    # Prepare batch data
    batch_data = []

    for image_name, nuclei_info in tqdm(nuclei_data.items(), desc="Preparing batch data"):
        if image_name not in image_features:
            continue

        df = image_features[image_name]

        # Check if nucleus_id column exists
        if 'nucleus_id' not in df.columns:
            continue

        # Get feature columns - exclude non-numeric and specific columns
        exclude_cols = ['image_name', 'original_image',
                        'nucleus_id', 'split', 'label', 'tile_x', 'tile_y', 'mag']
        feature_cols = [col for col in df.columns
                        if col not in exclude_cols
                        and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]

        if not feature_cols:
            continue

        # Extract features and IDs
        nucleus_ids = df['nucleus_id'].tolist()
        features = df[feature_cols].values

        # Ensure features are float type
        try:
            if isinstance(features, np.ndarray) and features.dtype == np.object_:
                features = np.vstack(
                    [np.array(f, dtype=np.float32) for f in features])
            features = features.astype(np.float32)
        except (ValueError, TypeError):
            continue

        # Ensure input dimensions are correct
        if features.shape[1] != args.input_dim:
            if features.shape[1] < args.input_dim:
                # Pad
                padding = np.zeros(
                    (features.shape[0], args.input_dim - features.shape[1]), dtype=np.float32)
                features = np.hstack((features, padding))
            else:
                # Truncate
                features = features[:, :args.input_dim]

        # Extract coordinates
        coords = np.array([nuclei_info.get(int(nuc_id), (0, 0))
                          for nuc_id in nucleus_ids])

        # Add to batch data
        batch_data.append({
            'image_name': image_name,
            'nucleus_ids': nucleus_ids,
            'features': features,
            'coords': coords
        })

    # Batch processing
    all_results = []
    for i in tqdm(range(0, len(batch_data), args.batch_size), desc="Batch processing"):
        batch = batch_data[i:i+args.batch_size]
        batch_results = process_batch(batch, gat_generator)
        all_results.extend(batch_results)

    # Merge results - Generate format compatible with other feature files
    all_gat_features = []
    for result in all_results:
        image_name = result['image_name']
        nucleus_ids = result['nucleus_ids']
        gat_features = result['gat_features']

        # Extract fold info from image_name
        fold_num = 1  # Default value
        if "_fold" in image_name:
            import re
            match = re.search(r'fold(\d+)', image_name)
            if match:
                fold_num = int(match.group(1))
        elif "fold1_" in image_name or "Fold 1" in image_name:
            fold_num = 1
        elif "fold2_" in image_name or "Fold 2" in image_name:
            fold_num = 2
        elif "fold3_" in image_name or "Fold 3" in image_name:
            fold_num = 3

        for i, nuc_id in enumerate(nucleus_ids):
            # Basic info row - Consistent with other feature file formats
            row = {
                'image_name': f"fold{fold_num}_image_{image_name.split('_')[-1] if '_' in image_name else image_name}_nucleus_{nuc_id}.png",
                'nucleus_id': nuc_id
            }

            # Add GAT feature columns
            for j in range(args.output_dim):
                row[f'coattention_feature_{j}'] = gat_features[i, j]
            all_gat_features.append(row)

    # Convert to DataFrame
    gat_df = pd.DataFrame(all_gat_features)

    # Sort by image_name, nucleus_id
    gat_df.sort_values(by=['image_name', 'nucleus_id'], inplace=True)

    # Fill null values with zero
    gat_df.fillna(0, inplace=True)

    # Remove columns that are all zeros
    gat_df = gat_df.loc[:, (gat_df != 0).any(axis=0)]

    # Extract feature columns
    feature_cols = [col for col in gat_df.columns if col.startswith(
        'coattention_feature_')]

    # Reorder columns - Consistent with other feature file formats
    id_columns = ['image_name', 'nucleus_id']
    gat_df = gat_df[id_columns + feature_cols]

    # Save GAT feature file - Format compatible with other feature files
    output_path = os.path.join(args.output_dir, 'pannuke_gat_features.csv')
    gat_df.to_csv(output_path, index=False)
    print(f"GAT features saved: {output_path}, Dimension: {gat_df.shape}")
    # print(f"Feature format: image_name + nucleus_id + {len(feature_cols)} GAT features")


    # Save extra detailed version (optional)
    detailed_df = pd.DataFrame(all_gat_features)
    detailed_df['original_image'] = detailed_df['image_name'].apply(
        lambda x: '_'.join(x.split('_')[:-2]).replace('.png', '') if '_nucleus_' in x else x.replace('.png', '')
    )
    detailed_df['split'] = detailed_df['image_name'].apply(
        lambda x: 'train' if 'train_' in x else 'test'
    )
    
    detailed_output_path = os.path.join(args.output_dir, 'pannuke_gat_features_detailed.csv')
    detailed_df.to_csv(detailed_output_path, index=False)
    print(f"Detailed feature file: {detailed_output_path}, Dimension: {detailed_df.shape}")


def single_mode(args):
    """Single image processing mode"""
    if not args.single_file:
        print("Error: Single image mode requires --single_file argument")
        return

    # Extract image name
    image_name = os.path.basename(
        args.single_file).replace("_segmentation.json", "")

    # Extract nucleus position data
    with open(args.single_file, 'r') as f:
        data = json.load(f)

    nuclei_data = {}
    for tile in data.get('tiles', []):
        for nuc_id, nuc_info in tile.get('nuc', {}).items():
            if 'centroid' in nuc_info:
                nuclei_data[int(nuc_id)] = tuple(nuc_info['centroid'])

    # Extract coordinates
    nucleus_ids = list(nuclei_data.keys())
    coords = np.array([nuclei_data[nuc_id] for nuc_id in nucleus_ids])

    # Build spatial relationship graph
    graph_builder = SpatialGraphBuilder(distance_threshold=40.0)
    adj_matrix = graph_builder.build_adjacency_matrix(coords)

    # If feature file provided
    if args.mobilevit_features and os.path.exists(args.mobilevit_features):
        # Load features
        features_df = pd.read_csv(args.mobilevit_features)

        # Filter features for current image
        if 'original_image' in features_df.columns:
            image_df = features_df[features_df['original_image'] == image_name]
        elif 'image_name' in features_df.columns:
            image_df = features_df[features_df['image_name'] == image_name]
        else:
            print("Warning: Feature file missing image_name and original_image columns")
            return

        if image_df.empty:
            print(f"Warning: Features for image {image_name} not found in feature file")
            return

        # Get feature columns
        exclude_cols = ['image_name', 'original_image',
                        'nucleus_id', 'split', 'label', 'tile_x', 'tile_y', 'mag']
        feature_cols = [col for col in image_df.columns
                        if col not in exclude_cols
                        and image_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]

        if not feature_cols:
            print("Warning: No numeric feature columns found")
            return

        # Extract features
        if 'nucleus_id' in image_df.columns:
            image_df = image_df.sort_values('nucleus_id')
            features = image_df[feature_cols].values
        else:
            # Assume feature order matches nuclei_data
            features = image_df[feature_cols].values[:len(nuclei_data)]

        # Initialize GAT feature generator
        input_dim = features.shape[1]
        gat_generator = GATFeatureGenerator(
            input_dim=input_dim,
            hidden_dim=input_dim // 2,
            output_dim=256,
            n_heads=8,
            dropout=0.4,
            distance_threshold=40.0
        )

        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        adj_tensor = torch.FloatTensor(adj_matrix).unsqueeze(0)

        # Generate GAT features
        with torch.no_grad():
            gat_features = gat_generator.gat_model(features_tensor, adj_tensor)

        # Convert back to NumPy array
        gat_features_np = gat_features.squeeze(0).cpu().numpy()

        # Save GAT features
        gat_df = pd.DataFrame(
            {'nucleus_id': nucleus_ids, 'image_name': image_name})
        for i in range(gat_features_np.shape[1]):
            gat_df[f'coattention_feature_{i}'] = gat_features_np[:, i]

        output_dir = os.path.join(args.output_dir, "single_output")
        os.makedirs(output_dir, exist_ok=True)
        gat_df.to_csv(os.path.join(
            output_dir, f"{image_name}_gat_features.csv"), index=False)


def demo_mode(args):
    """Demo mode"""
    output_dir = os.path.join(args.output_dir, "demo_output")
    os.makedirs(output_dir, exist_ok=True)

    # Get a JSON file for demo
    json_files = [f for f in os.listdir(
        args.hovernet_dir) if f.endswith("_segmentation.json")]
    if not json_files:
        print(f"Error: No segmentation result files found in {args.hovernet_dir}")
        return

    # Select first test file for demo
    test_files = [f for f in json_files if f.startswith("test_")]
    if test_files:
        demo_file = os.path.join(args.hovernet_dir, test_files[0])
    else:
        demo_file = os.path.join(args.hovernet_dir, json_files[0])

    # Call single image mode to process demo file
    args.single_file = demo_file
    single_mode(args)


def main():
    # Parse command line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Execute different processing based on mode
    if args.mode == 'batch':
        batch_mode(args)
    elif args.mode == 'single':
        single_mode(args)
    elif args.mode == 'demo':
        demo_mode(args)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

