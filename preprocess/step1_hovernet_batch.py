import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import glob


# Add HoverNet segmentation module path
workspace_root = Path(__file__).parent.parent.absolute()
hovernet_path = workspace_root / "segmentation"
if str(hovernet_path) not in sys.path:
    sys.path.insert(0, str(hovernet_path))


def check_hovernet_setup():
    """Check HoverNet environment setup"""
    print("Checking HoverNet environment...")

    # Check model file
    model_path = hovernet_path / "weight" / 'hovernet_model_weights.pth'  ###hovernet_model_weights
    if model_path.exists():
        model_size = model_path.stat().st_size / (1024 * 1024)  # MB
        print(f" Model file exists: {model_path.name} ({model_size:.1f} MB)")
    else:
        print(f" Model file does not exist: {model_path}")
        return False

    # Check input images
    pannuke_root = Path(
        "/path/to/PanNuke_dataset")
    
    # Assuming PanNuke images are organized in Folds or a common directory
    # Adjust this pattern based on actual data structure
    images = list(pannuke_root.glob("**/*.png"))

    print(f"Input images: {len(images)} images found")

    if images:
        sample_image = images[0]
        image_size = sample_image.stat().st_size / (1024 * 1024)  # MB
        print(f"Sample image: {sample_image.name} ({image_size:.1f} MB)")

    return True


def process_pannuke_images():
    """Batch process PanNuke images"""

    # Input and output paths
    pannuke_root = Path(
        "/path/to/PanNuke_dataset")
    
    output_dir = Path(
        "/path/to/hovernet_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create segmentation image output directory
    segmentation_images_dir = output_dir / "segmentation_images"
    segmentation_images_dir.mkdir(exist_ok=True)

    # Get all images
    all_images = list(pannuke_root.glob("**/*.png"))

    print(f"Found images: {len(all_images)} total")

    # Import HoverNet segmentation function

    sys.path.insert(0, str(hovernet_path))
    from run_segmentation1_true import OptimizedHoverNetInference  # type: ignore ##

    # Set segmentation parameters - Adjust parameters to improve detection
    config = {
        'model_path': str(hovernet_path / "weight" / 'hovernet_model_weights.pth'),
        'model_mode': 'original',
        'gpu': '0',
        'nr_types': '0',
        'nr_inference_workers': '0',
        'nr_post_proc_workers': str(min(8, mp.cpu_count())),  
        'batch_size': '12',  
        'mem_usage': '0.7',  
        'draw_dot': True,  
        'save_qupath': False,
        'save_raw_map': False,
        'tile_size': 1000,  
        'overlap_ratio': 0.2,  
        'non_white_threshold': 0.1,  
    }


    # Create inference instance
    try:
        inference = OptimizedHoverNetInference(config)
        print(" Successfully created HoverNet inference instance")
    except Exception as e:
        print(f" Failed to create HoverNet inference instance: {e}")
        return False

    successful_count = 0
    failed_count = 0

    for image_file in tqdm(all_images, desc="HoverNet Segmentation"):
        image_name = image_file.stem
        output_json = output_dir / f"{image_name}_segmentation.json"
        output_image = segmentation_images_dir / \
            f"{image_name}_segmentation.png"

        # Check if already processed (JSON and image both exist)
        if output_json.exists() and output_image.exists():
            print(f" {image_name} already processed, skipping")
            successful_count += 1
            continue

        print(f"Processing {image_name}...")

        try:
            # Create temporary output directory
            temp_output_dir = output_dir / f"temp_{image_name}"
            temp_output_dir.mkdir(exist_ok=True)

            # Call HoverNet segmentation
            inference.process_large_image(
                str(image_file), str(temp_output_dir))

            # Detailed check of output files
            json_files = list(temp_output_dir.glob("**/*.json"))
            image_files = list(temp_output_dir.glob("**/*.png"))

            print(f"      Found JSON files: {len(json_files)}")
            print(f"      Found image files: {len(image_files)}")

            # Show found files
            for json_file in json_files:
                print(f"      JSON: {json_file.relative_to(temp_output_dir)}")
            for img_file in image_files:
                print(f"      Image: {img_file.relative_to(temp_output_dir)}")

            if json_files:
                # Read JSON content to check for nuclei
                source_json = json_files[0]
                with open(source_json, 'r') as f:
                    json_content = json.load(f)

                # Count nuclei
                total_nuclei = 0
                for tile in json_content.get('tiles', []):
                    nuclei = tile.get('nuc', {})
                    total_nuclei += len(nuclei)

                print(f"      📊 Detected nuclei: {total_nuclei}")

                # Move result files
                source_json.rename(output_json)

                # Move segmentation image file
                if image_files:
                    source_image = image_files[0]
                    source_image.rename(output_image)
                    print(f"{image_name} segmentation complete (JSON + Image)")
                else:
                    print(f"{image_name} segmentation complete (JSON only)")

                successful_count += 1
            else:
                print(f"{image_name} did not generate segmentation results")
                failed_count += 1
                # Create empty result
                empty_result = {
                    "tiles": [{
                        "mag": image_name,
                        "x": 0,
                        "y": 0,
                        "nuc": {}
                    }]
                }
                with open(output_json, 'w') as f:
                    json.dump(empty_result, f, indent=2)

            # Clean up temporary directory
            import shutil
            if temp_output_dir.exists():
                shutil.rmtree(temp_output_dir)

        except Exception as e:
            print(f" {image_name} processing failed: {e}")
            failed_count += 1
            # Create empty result file to avoid interrupting flow
            empty_result = {
                "tiles": [{
                    "mag": image_name,
                    "x": 0,
                    "y": 0,
                    "nuc": {}
                }]
            }
            with open(output_json, 'w') as f:
                json.dump(empty_result, f, indent=2)

    # Generated file statistics
    json_count = len(list(output_dir.glob("*.json")))
    image_count = len(list(segmentation_images_dir.glob("*.png")))
    print(f"Generated file statistics:")
    print(f"JSON files: {json_count}")
    print(f"Segmentation images: {image_count}")

    return successful_count > 0


def create_visualization_summary():
    """Create segmentation result visualization summary"""
    output_dir = Path(
        "./output/step1_hovernet_results")
    segmentation_images_dir = output_dir / "segmentation_images"

    # Count segmentation images
    images = list(segmentation_images_dir.glob("*.png"))

    # Show first few files as example
    print(f"\ Example files:")
    for i, img_path in enumerate(images[:5]):
        print(f"  {img_path.name}")


if __name__ == "__main__":
    success = process_pannuke_images()

    if success:
        print("\n" + "="*50)
        create_visualization_summary()

    sys.exit(0 if success else 1)
