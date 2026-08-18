import os
import re
import cv2
import json
import shutil
import numpy as np
import torch
from tile import InferManager
import sys
import glob
from tqdm import tqdm
from scipy.spatial import cKDTree
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import time
import psutil
import gc
from typing import List, Tuple, Dict, Optional
import multiprocessing as mp


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptimizedHoverNetInference:
    """Optimized HoverNet Inference Class"""

    def __init__(self, config: Dict):
        """
        Initialize the inferencer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.setup_environment()
        self.setup_inference_params()

        # Performance monitoring
        self.total_tiles = 0
        self.processed_tiles = 0
        self.start_time = None

    def setup_environment(self):
        """Set up environment variables and GPU"""
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['gpu']
        self.nr_gpus = torch.cuda.device_count()
        logger.info(f'Detected GPU count: {self.nr_gpus}')

        # Set OpenCV thread count
        cv2.setNumThreads(min(4, mp.cpu_count()))

    def setup_inference_params(self):
        """Set up inference parameters"""
        self.method_args = {
            'method': {
                'model_args': {
                    'nr_types': int(self.config['nr_types']) if int(self.config['nr_types']) > 0 else None,
                    'mode': self.config['model_mode'],
                },
                'model_path': self.config['model_path'],
            },
            'type_info_path': self.config.get('type_info_path'),
        }

        # Set type information dictionary
        self.type_info_dict = self.config.get('type_info_dict')

        self.run_args = {
            'batch_size': int(self.config['batch_size']) * self.nr_gpus,
            'nr_inference_workers': int(self.config['nr_inference_workers']),
            'nr_post_proc_workers': int(self.config['nr_post_proc_workers']),
            'patch_input_shape': 270 if self.config['model_mode'] == 'original' else 256,
            'patch_output_shape': 80 if self.config['model_mode'] == 'original' else 164,
            'mem_usage': float(self.config['mem_usage']),
            'draw_dot': self.config['draw_dot'],
            'save_qupath': self.config['save_qupath'],
            'save_raw_map': self.config['save_raw_map'],
        }

    def preprocess_image_fast(self, image_path: str, threshold: float = 0.1) -> bool:
        """
        Fast image preprocessing check

        Args:
            image_path: Image path
            threshold: Threshold

        Returns:
            bool: Whether processing is needed
        """
        try:
            # Use faster image reading method
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.error(f"Cannot read image: {image_path}")
                return False

            # Fast sampling check (check only partial pixels)
            h, w = image.shape
            step = max(1, min(h, w) // 100)  # Sampling step
            sample = image[::step, ::step]

            # Calculate non-white pixel ratio
            non_white_ratio = np.sum(sample < 240) / sample.size

            return non_white_ratio > threshold

        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            return False

    def split_image_optimized(self, image_path: str, tile_size: int,
                              overlap_ratio: float, non_white_threshold: float) -> Tuple[List, Tuple]:
        """
        Optimized image splitting method

        Args:
            image_path: Image path
            tile_size: Tile size
            overlap_ratio: Overlap ratio
            non_white_threshold: Non-white threshold

        Returns:
            Tuple[List, Tuple]: (List of tiles, original size)
        """
        logger.info("Starting tile processing...")
        start_time = time.time()

        # Use memory mapping to read large images
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        h, w, _ = image.shape
        step_size = tile_size - int(tile_size * overlap_ratio)
        tiles = []
        processed = set()

        logger.info(f"Image size: {w}x{h}, Tile size: {tile_size}, Step size: {step_size}")

        # Pre-calculate all possible tile positions
        y_positions = list(range(0, h, step_size))
        x_positions = list(range(0, w, step_size))

        total_positions = len(y_positions) * len(x_positions)
        logger.info(f"Pre-calculated tile positions: {total_positions}")

        with tqdm(total=total_positions, desc="Tile processing") as pbar:
            for y in y_positions:
                for x in x_positions:
                    # Calculate actual start coordinates
                    y_start = y if y + \
                        tile_size <= h else max(0, h - tile_size)
                    x_start = x if x + \
                        tile_size <= w else max(0, w - tile_size)

                    if (x_start, y_start) in processed:
                        pbar.update(1)
                        continue
                    processed.add((x_start, y_start))

                    # Extract tile
                    tile = image[y_start:y_start+tile_size,
                                 x_start:x_start+tile_size]

                    # Boundary padding
                    pad_vert = tile_size - tile.shape[0]
                    pad_hori = tile_size - tile.shape[1]
                    if pad_vert > 0 or pad_hori > 0:
                        tile = cv2.copyMakeBorder(tile, 0, pad_vert, 0, pad_hori,
                                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))

                    # Fast quality check
                    if self._is_tile_worth_processing(tile, non_white_threshold):
                        tiles.append((x_start, y_start, tile))

                    pbar.update(1)

        split_time = time.time() - start_time
        logger.info(f"Splitting completed, time elapsed: {split_time:.2f}s, valid tiles: {len(tiles)}")

        return tiles, (h, w)

    def _is_tile_worth_processing(self, tile: np.ndarray, non_white_threshold: float) -> bool:
        """
        Fast check if tile is worth processing

        Args:
            tile: Image tile
            non_white_threshold: Non-white threshold

        Returns:
            bool: Whether it is worth processing
        """
        # Fast non-white area detection (sampling check)
        sample_step = max(1, tile.shape[0] // 20)
        sample = tile[::sample_step, ::sample_step]
        non_white_area = np.sum(sample < 240) / sample.size

        if non_white_area <= non_white_threshold:
            return False

        # Fast edge detection
        gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_sample, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        return edge_ratio > 0.01

    def run_inference_batch(self, tiles_batch: List, save_dir: str) -> List[bool]:
        """
        Batch inference processing

        Args:
            tiles_batch: Tile batch
            save_dir: Save directory

        Returns:
            List[bool]: List of processing results
        """
        results = []

        for x, y, tile_array in tiles_batch:
            try:
                infer = InferManager(**self.method_args)
                tmp_dir = os.path.join(save_dir, f"tmp_{x}_{y}")
                os.makedirs(tmp_dir, exist_ok=True)

                # Save as uncompressed PNG
                tile_path = os.path.join(tmp_dir, f"tile_{x}_{y}.png")
                # Use no compression parameters
                cv2.imwrite(tile_path, tile_array, [
                            cv2.IMWRITE_PNG_COMPRESSION, 0])

                # Create inference parameters
                infer_args = {
                    'input_dir': tmp_dir,
                    'output_dir': os.path.join(save_dir, f"output_{x}_{y}"),
                    'type_info_dict': self.type_info_dict,  # Add type color dictionary
                    **self.run_args
                }

                # Run inference
                infer.process_multiple_tiles(infer_args)

                results.append(True)
                self.processed_tiles += 1

                # Clean up temporary files
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)

            except Exception as e:
                logger.error(f"Failed to process tile ({x}, {y}): {e}")
                results.append(False)
            finally:
                # Force garbage collection
                gc.collect()

        return results

    def process_tiles_parallel(self, tiles: List, save_dir: str, max_workers: int = None):
        """
        Process tiles in parallel

        Args:
            tiles: List of tiles
            save_dir: Save directory
            max_workers: Maximum number of worker threads
        """
        if max_workers is None:
            max_workers = min(4, mp.cpu_count())

        self.total_tiles = len(tiles)
        self.processed_tiles = 0
        self.start_time = time.time()

        logger.info(f"Starting parallel processing of {len(tiles)} tiles, using {max_workers} worker threads")

        # Batch processing to control memory usage
        batch_size = max(1, len(tiles) // (max_workers * 2))
        batches = [tiles[i:i+batch_size]
                   for i in range(0, len(tiles), batch_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(self.run_inference_batch, batch, save_dir): batch
                for batch in batches
            }

            with tqdm(total=len(tiles), desc="Inference progress") as pbar:
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        results = future.result()
                        pbar.update(len(batch))

                        # Update progress information
                        if self.processed_tiles > 0:
                            elapsed = time.time() - self.start_time
                            rate = self.processed_tiles / elapsed
                            eta = (self.total_tiles - self.processed_tiles) / \
                                rate if rate > 0 else 0
                            pbar.set_postfix({
                                'rate': f'{rate:.1f} tiles/s',
                                'eta': f'{eta:.0f}s'
                            })

                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
                        pbar.update(len(batch))

    def merge_results_optimized(self, output_dir: str, save_path: str,
                                tile_size: int, orig_shape: Tuple) -> None:
        """
        Optimized result merging method

        Args:
            output_dir: Output directory
            save_path: Save path
            tile_size: Tile size
            orig_shape: Original size
        """
        logger.info("Starting result merging...")
        start_time = time.time()

        deduplication_threshold = 5  # For deduplication
        boundary_threshold = 5  # For filtering out nuclei on tile boundaries

        json_files = glob.glob(os.path.join(
            output_dir, 'output_*', 'json', '*.json'))
        logger.info(f"Found {len(json_files)} result files")

        if not json_files:
            logger.warning("No result files found")
            return

        # Process JSON files in parallel
        def process_json_file(json_file):
            try:
                dir_path = os.path.dirname(os.path.dirname(json_file))
                coords = re.findall(r'output_(\d+)_(\d+)', dir_path)
                if not coords:
                    return []

                x_offset, y_offset = map(int, coords[0])
                nuclei = []

                with open(json_file, 'r') as f:
                    data = json.load(f)
                    for nuc in data.get('nuc', {}).values():
                        contour = np.array(nuc['contour'])
                        if (contour.size == 0 or
                                self._is_nucleus_on_boundary(contour, tile_size, boundary_threshold)):
                            continue

                        centroid = [
                            nuc['centroid'][0] + x_offset,
                            nuc['centroid'][1] + y_offset
                        ]
                        nuclei.append({
                            "mag": f"tile_{x_offset}_{y_offset}",
                            "contour": [[p[0]+x_offset, p[1]+y_offset] for p in contour],
                            "centroid": centroid,
                            "type": nuc.get('type'),
                            "type_prob": nuc.get('type_prob')
                        })
                return nuclei
            except Exception as e:
                logger.error(f"Error processing file {json_file}: {e}")
                return []

        # Use more threads for parallel processing
        max_workers = min(32, len(json_files))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(process_json_file, json_files),
                total=len(json_files),
                desc="Processing JSON files"
            ))

        # Merge all results
        all_nuclei = [nuc for sublist in results for nuc in sublist]
        logger.info(f"Total nuclei before deduplication: {len(all_nuclei)}")

        if len(all_nuclei) == 0:
            logger.warning("No nuclei detected")
            return

        # Use optimized deduplication algorithm
        unique_nuclei = self._deduplicate_nuclei_optimized(
            all_nuclei, deduplication_threshold)
        logger.info(
            f"Total nuclei after deduplication: {len(unique_nuclei)} (Removed {len(all_nuclei)-len(unique_nuclei)} duplicates)")

        # Reorganize data
        merged_data = self._reorganize_data(unique_nuclei)

        # Save results
        with open(save_path, 'w') as f:
            json.dump(merged_data, f, indent=4,
                      default=self._convert_numpy_types)

        merge_time = time.time() - start_time
        logger.info(f"Result merging completed, time elapsed: {merge_time:.2f}s, saved to: {save_path}")

    def _is_nucleus_on_boundary(self, contour: np.ndarray, tile_size: int,
                                boundary_threshold: int = 10) -> bool:
        """Check if nucleus is on the boundary"""
        if contour.size == 0:
            return True

        x_coords = contour[:, 0]
        y_coords = contour[:, 1]

        return (np.any(x_coords < boundary_threshold) or
                np.any(x_coords > tile_size - boundary_threshold) or
                np.any(y_coords < boundary_threshold) or
                np.any(y_coords > tile_size - boundary_threshold))

    def _deduplicate_nuclei_optimized(self, all_nuclei: List, threshold: float) -> List:
        """Optimized deduplication algorithm"""
        if len(all_nuclei) == 0:
            return []

        # Build coordinate array
        points = np.array([nuc['centroid'] for nuc in all_nuclei])

        # Use KDTree for fast neighbor search
        tree = cKDTree(points)
        pairs = tree.query_ball_tree(tree, r=threshold)

        # Optimized deduplication logic
        keep_mask = np.ones(len(points), dtype=bool)
        visited = set()

        for i in range(len(points)):
            if i in visited:
                continue

            # Find all duplicates
            duplicates = [j for j in pairs[i] if j > i]
            for j in duplicates:
                keep_mask[j] = False
                visited.add(j)
            visited.add(i)

        return [all_nuclei[i] for i in np.where(keep_mask)[0]]

    def _reorganize_data(self, unique_nuclei: List) -> Dict:
        """Reorganize data structure"""
        tile_dict = defaultdict(lambda: {"mag": "", "nuc": {}})

        for idx, nuc in enumerate(unique_nuclei):
            mag = nuc["mag"]
            tile_dict[mag]["mag"] = mag
            tile_dict[mag]["nuc"][f"{idx}"] = {
                "contour": nuc["contour"],
                "centroid": nuc["centroid"],
                "type": nuc["type"],
                "type_prob": nuc["type_prob"]
            }

        return {
            "tiles": sorted(
                tile_dict.values(),
                key=lambda x: tuple(map(int, re.findall(r'\d+', x["mag"])))
            )
        }

    def _convert_numpy_types(self, obj):
        """Convert numpy types"""
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def visualize_results_fast(self, merge_json_path: str, orig_image_path: str,
                               output_path: str) -> None:
        """Fast result visualization"""
        logger.info("Starting visualization generation...")
        start_time = time.time()

        # Read original image
        image = cv2.imread(orig_image_path)
        if image is None:
            logger.error(f"Cannot read original image: {orig_image_path}")
            return

        overlay = image.copy()
        nucleus_count = 0

        # Read detection results
        if os.path.exists(merge_json_path):
            try:
                with open(merge_json_path) as f:
                    data = json.load(f)

                # Draw contours
                for tile in data['tiles']:
                    for nucleus in tile['nuc'].values():
                        contour = np.array(nucleus['contour'], dtype=np.int32)
                        # Set color (0,255,0) and thickness 2 here, now changed to 1
                        cv2.drawContours(
                            overlay, [contour], -1, (0, 255, 0), 2)
                        nucleus_count += 1

                # Blend images
                cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

                # Save as uncompressed PNG
                cv2.imwrite(output_path, image, [
                            cv2.IMWRITE_PNG_COMPRESSION, 0])

                vis_time = time.time() - start_time
                logger.info(
                    f"Visualization completed, detected {nucleus_count} nuclei, time elapsed: {vis_time:.2f}s")
                logger.info(f"Visualization result saved to: {output_path}")
            except Exception as e:
                logger.error(f"Visualization generation failed: {e}")
                # At least save the original image
                cv2.imwrite(output_path, image)
                logger.info(f"Original image saved to: {output_path}")
        else:
            logger.warning(f"Merged result file not found: {merge_json_path}, saving original image directly.")
            cv2.imwrite(output_path, image)
            logger.info(f"Original image saved to: {output_path}")

    def process_large_image(self, image_path: str, save_dir: str) -> None:
        """
        Main processing flow

        Args:
            image_path: Input image path
            save_dir: Output directory
        """
        logger.info(f"Starting processing of large image: {image_path}")
        total_start_time = time.time()

        # Create output directory
        os.makedirs(save_dir, exist_ok=True)

        # Preprocessing check
        if not self.preprocess_image_fast(image_path):
            logger.info(f"Image {image_path} does not need processing")
            return

        # Tile processing
        tiles, orig_shape = self.split_image_optimized(
            image_path,
            self.config['tile_size'],
            self.config['overlap_ratio'],
            self.config['non_white_threshold']
        )

        if not tiles:
            logger.warning("No tiles found for processing")
            return

        # Parallel inference
        self.process_tiles_parallel(tiles, save_dir)

        # Merge results
        merge_path = os.path.join(save_dir, "merged.json")
        self.merge_results_optimized(
            save_dir, merge_path, self.config['tile_size'], orig_shape)

        # Generate visualization
        input_basename = os.path.basename(image_path)
        output_filename = os.path.splitext(input_basename)[0] + ".png"
        final_output = os.path.join(save_dir, output_filename)
        self.visualize_results_fast(merge_path, image_path, final_output)

        # Performance statistics
        total_time = time.time() - total_start_time
        logger.info(f"Processing completed! Total time: {total_time:.2f}s")
        logger.info(f"Average processing speed: {len(tiles)/total_time:.2f} tiles/s")


def main():
    """Main function"""
    # Optimized configuration
    config = {
        # Model configuration
        'model_path': '/path/to/segmentation/weight/hovernet_fast_pannuke_type_tf2pytorch.pth',
        'model_mode': 'fast',
        'gpu': '0',
        'nr_types': '6',

        # Add type information dictionary for color mapping
        'type_info_dict': {
            0: ('Neoplastic', (255, 0, 0)),     # Red
            1: ('Inflammatory', (0, 255, 0)),   # Green
            2: ('Connective', (0, 0, 255)),     # Blue
            3: ('Dead', (255, 255, 0)),         # Yellow
            4: ('Epithelial', (255, 0, 255)),   # Magenta
            5: ('Background', (0, 255, 255)),   # Cyan
        },

        # Inference configuration
        'nr_inference_workers': '0',
        'nr_post_proc_workers': str(min(25, mp.cpu_count())),  # Dynamic adjustment
        'batch_size': '16',  # Increase batch size. If GPU memory is sufficient, prioritize increasing batch size

        # Image processing configuration
        'mem_usage': '0.75',  # Increase memory usage
        'draw_dot': False,
        'save_qupath': False,
        'save_raw_map': False,
        'tile_size': 1000,
        'overlap_ratio': 0.1,
        'non_white_threshold': 0.1,
    }

    # Create inferencer
    inference = OptimizedHoverNetInference(config)

    # Process image
    input_image = '/path/to/XX.png'
    output_dir = './output_dir'

    inference.process_large_image(input_image, output_dir)
    logger.info("All processing flows completed")


if __name__ == '__main__':
    main()
