
import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob


def extract_nuclei_from_pannuke():
    """Extract nucleus images from PanNuke dataset"""

    # Path settings
    json_dir = Path(
        "./output/step1_hovernet_results")
    pannuke_root = Path(
        "./PanNuke_dataset")
    output_dir = Path(
        "./output/step2_nuclei_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files
    json_files = list(json_dir.glob("*_segmentation.json"))
    print(f"Found {len(json_files)} segmentation result files")

    # Get all original images
    image_files = list(pannuke_root.glob("**/*.png"))
    image_map = {img.stem: img for img in image_files}

    total_nuclei = 0
    processed_images = 0

    for json_file in tqdm(json_files, desc="Extracting Nuclei"):
        image_name = json_file.stem.replace("_segmentation", "")

        if image_name not in image_map:
            print(f"Original image not found: {image_name}")
            continue

        image_path = image_map[image_name]

        # Load image and JSON
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Cannot read image: {image_path}")
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract each nucleus
        nucleus_count = 0
        for tile in data.get("tiles", []):
            nuclei_data = tile.get("nuc", {})

            for nucleus_id, nucleus_info in nuclei_data.items():
                contour = nucleus_info.get("contour", [])
                if not contour:
                    continue

                try:
                    # Extract nucleus region
                    contour_array = np.array(contour, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(contour_array)

                    # Expand boundary
                    padding = 10
                    x_start = max(0, x - padding)
                    y_start = max(0, y - padding)
                    x_end = min(image.shape[1], x + w + padding)
                    y_end = min(image.shape[0], y + h + padding)

                    # Extract region
                    nucleus_region = image[y_start:y_end, x_start:x_end]

                    if nucleus_region.size == 0:
                        continue

                    # Create mask
                    mask = np.zeros(
                        (y_end - y_start, x_end - x_start), dtype=np.uint8)
                    shifted_contour = contour_array - [x_start, y_start]
                    cv2.fillPoly(mask, [shifted_contour], 255)

                    # Apply mask (background set to white)
                    nucleus_image = nucleus_region.copy()
                    nucleus_image[mask == 0] = [255, 255, 255]

                    # Save image, use unified naming format: image_name_nucleus_Y.png
                    output_filename = f"{image_name}_nucleus_{nucleus_id}.png"
                    output_path = output_dir / output_filename
                    cv2.imwrite(str(output_path), nucleus_image)

                    total_nuclei += 1
                    nucleus_count += 1

                except Exception as e:
                    print(f"Nucleus {image_name}_{nucleus_id} extraction failed: {e}")
        
        processed_images += 1

    print(f"Nucleus extraction complete:")
    print(f"Processed images: {processed_images}")
    print(f"Total nuclei: {total_nuclei}")
    print(f"Output directory: {output_dir}")
    return total_nuclei > 0


if __name__ == "__main__":
    success = extract_nuclei_from_pannuke()
    import sys
    sys.exit(0 if success else 1)
