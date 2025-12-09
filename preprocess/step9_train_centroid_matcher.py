
import argparse
import json
import csv
import os
import numpy as np
from scipy.spatial import cKDTree
import glob
import re
from tqdm import tqdm


def extract_centroids_from_mask(mask, class_idx):
    """Extract centroid coordinates of specified class from mask"""
    centroids = []
    # The 6th dimension of mask is type info, 5th dimension is instance segmentation
    type_mask = mask[:, :, 5]  # Type mask
    instance_mask = mask[:, :, class_idx] if class_idx < 5 else None

    if instance_mask is not None:
        # Get all instances of this class
        unique_instances = np.unique(instance_mask[instance_mask > 0])
        for inst_id in unique_instances:
            # Find all pixels of this instance
            inst_pixels = np.where(instance_mask == inst_id)
            if len(inst_pixels[0]) > 0:
                # Calculate centroid
                y_center = np.mean(inst_pixels[0])
                x_center = np.mean(inst_pixels[1])
                centroids.append([x_center, y_center, inst_id])

    return centroids


def load_pannuke_data(fold_path, image_idx):
    """Load data for corresponding image in PanNuke dataset"""
    mask_path = os.path.join(
        fold_path, 'masks', f'fold{fold_path[-1]}', 'masks.npy')

    if not os.path.exists(mask_path):
        return None, None

    masks = np.load(mask_path)

    if image_idx >= masks.shape[0]:
        return None, None

    mask = masks[image_idx]  # shape: (256, 256, 6)

    # Extract centroids and type info for all nuclei
    centroids_with_types = []

    # PanNuke dataset mask structure (according to README.md):
    # Channel 0: Neoplastic cells -> Class 1
    # Channel 1: Inflammatory cells -> Class 2
    # Channel 2: Connective/Soft tissue cells -> Class 3
    # Channel 3: Dead Cells -> Class 4
    # Channel 4: Epithelial cells -> Class 5
    # Channel 5: Background -> Ignore

    for class_idx in range(5):  # Process first 5 channels (0-4)
        class_centroids = extract_centroids_from_mask(mask, class_idx)
        for x, y, inst_id in class_centroids:
            # class_idx corresponds to mask channel, label is class_idx + 1
            centroids_with_types.append([x, y, class_idx + 1])

    return np.array(centroids_with_types), mask


def load_hovernet_json(json_path):
    """Load HoverNet segmentation result JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    nuclei_data = []
    if 'tiles' in data and len(data['tiles']) > 0:
        nuc_dict = data['tiles'][0].get('nuc', {})
        for nuc_id, nuc_info in nuc_dict.items():
            if 'centroid' in nuc_info:
                x, y = nuc_info['centroid']
                current_type = nuc_info.get('type', 0)
                nuclei_data.append({
                    'id': nuc_id,
                    'centroid': [x, y],
                    'original_type': current_type,
                    'contour': nuc_info.get('contour', []),
                    'type_prob': nuc_info.get('type_prob', 0.0)
                })

    return nuclei_data


def match_nuclei(hovernet_nuclei, pannuke_centroids, dist_thresh=12):
    """Match HoverNet detected nuclei with PanNuke ground truth nuclei"""
    if len(pannuke_centroids) == 0 or len(hovernet_nuclei) == 0:
        return []

    # Only use coordinates for matching
    pannuke_coords = pannuke_centroids[:, :2]  # x, y coordinates
    hovernet_coords = np.array([nuc['centroid'] for nuc in hovernet_nuclei])

    tree = cKDTree(pannuke_coords)
    matches = []

    for i, nuc in enumerate(hovernet_nuclei):
        dist, idx = tree.query(nuc['centroid'])
        if dist <= dist_thresh:
            pannuke_type = int(pannuke_centroids[idx, 2])  # Get PanNuke type
            matches.append({
                'hovernet_idx': i,
                'hovernet_id': nuc['id'],
                'pannuke_idx': idx,
                'distance': dist,
                'original_type': nuc['original_type'],
                'new_type': pannuke_type,
                'centroid': nuc['centroid']
            })

    return matches


def update_hovernet_json(json_path, matches, hovernet_nuclei, output_dir=None):
    """Update type labels in HoverNet JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create match dictionary for easy lookup
    match_dict = {match['hovernet_id']: match['new_type'] for match in matches}

    # Update JSON data
    if 'tiles' in data and len(data['tiles']) > 0:
        nuc_dict = data['tiles'][0].get('nuc', {})
        updated_count = 0
        for nuc_id, nuc_info in nuc_dict.items():
            if nuc_id in match_dict:
                nuc_info['type'] = match_dict[nuc_id]
                updated_count += 1

    # Save updated JSON file
    filename = os.path.basename(json_path)
    updated_filename = filename.replace('.json', '_updated.json')

    if output_dir:
        output_path = os.path.join(output_dir, updated_filename)
    else:
        output_path = json_path.replace('.json', '_updated.json')

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return output_path, updated_count


def save_matching_results(matches, output_path):
    """Save matching results to CSV file"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hovernet_id', 'hovernet_idx', 'pannuke_idx', 'distance',
                        'original_type', 'new_type', 'centroid_x', 'centroid_y'])
        for match in matches:
            writer.writerow([
                match['hovernet_id'], match['hovernet_idx'], match['pannuke_idx'],
                match['distance'], match['original_type'], match['new_type'],
                match['centroid'][0], match['centroid'][1]
            ])


def parse_filename(filename):
    """Parse filename to get fold and image info"""
    # Filename format: fold2_image_1_segmentation.json
    match = re.match(r'fold(\d+)_image_(\d+)_segmentation\.json', filename)
    if match:
        fold_num = int(match.group(1))
        image_num = int(match.group(2))
        return fold_num, image_num
    return None, None


def main():
    # Set parameters directly, no command line input needed
    class Args:
        def __init__(self):
            self.hovernet_dir = "/path/to/step1_hovernet_results"
            self.pannuke_dir = "/path/to/PanNuke_dataset"
            self.output_dir = "./output/step8_centroid_revised"
            self.dist_thresh = 6.0  # Max matching distance threshold
            self.update_json = False  # Set to True to update JSON file

    args = Args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # If updating JSON file, create directory for updated JSON files
    updated_json_dir = None
    if args.update_json:
        updated_json_dir = "./output/step1_hovernet_result_update"
        os.makedirs(updated_json_dir, exist_ok=True)
        print(f"Updated JSON files will be saved to: {updated_json_dir}")

    # Get all JSON files
    json_files = glob.glob(os.path.join(args.hovernet_dir, "*.json"))

    total_matches = 0
    total_files = len(json_files)
    processed_files = 0
    failed_files = []

    print(f"Start processing {total_files} JSON files...")

    for json_file in tqdm(json_files):
        try:
            filename = os.path.basename(json_file)
            fold_num, image_num = parse_filename(filename)

            if fold_num is None or image_num is None:
                print(f"Cannot parse filename: {filename}")
                failed_files.append(filename)
                continue

            # Build PanNuke fold path
            fold_path = os.path.join(args.pannuke_dir, f"Fold {fold_num}")

            if not os.path.exists(fold_path):
                print(f"PanNuke fold directory does not exist: {fold_path}")
                failed_files.append(filename)
                continue

            # Load HoverNet data
            hovernet_nuclei = load_hovernet_json(json_file)
            if len(hovernet_nuclei) == 0:
                print(f"No nucleus data in HoverNet file: {filename}")
                failed_files.append(filename)
                continue

            # Load PanNuke data
            pannuke_centroids, mask = load_pannuke_data(fold_path, image_num)
            if pannuke_centroids is None or len(pannuke_centroids) == 0:
                print(
                    f"PanNuke data load failed or empty: fold {fold_num}, image {image_num}")
                failed_files.append(filename)
                continue

            # Perform matching
            matches = match_nuclei(
                hovernet_nuclei, pannuke_centroids, args.dist_thresh)

            if len(matches) > 0:
                # Save matching results
                match_output_path = os.path.join(args.output_dir,
                                                 f"{os.path.splitext(filename)[0]}_matches.csv")
                save_matching_results(matches, match_output_path)

                # Update JSON file (if specified)
                if args.update_json:
                    updated_json_path, updated_count = update_hovernet_json(
                        json_file, matches, hovernet_nuclei, updated_json_dir)
                    print(
                        f"Updated type labels for {updated_count} nuclei: {updated_json_path}")

                total_matches += len(matches)
                processed_files += 1

                print(
                    f"Processing complete: {filename} - Found {len(matches)} matches")
            else:
                print(f"No matches found: {filename}")
                failed_files.append(filename)

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            failed_files.append(filename)

    # Save summary report
    summary_path = os.path.join(args.output_dir, "matching_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Matching Summary Report\n")
        f.write(f"================\n")
        f.write(f"Total files: {total_files}\n")
        f.write(f"Successfully processed: {processed_files}\n")
        f.write(f"Failed files: {len(failed_files)}\n")
        f.write(f"Total matches: {total_matches}\n")
        f.write(f"Distance threshold: {args.dist_thresh}\n")
        f.write(f"\nFailed files list:\n")
        for failed_file in failed_files:
            f.write(f"  - {failed_file}\n")

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_files}/{total_files} files")
    print(f"Total matches: {total_matches}")
    print(f"Failed files: {len(failed_files)}")
    print(f"Results saved in: {args.output_dir}")
    print(f"Summary report: {summary_path}")


if __name__ == '__main__':
    main()
