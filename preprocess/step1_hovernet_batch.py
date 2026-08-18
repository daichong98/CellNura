import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import glob


workspace_root = Path(__file__).parent.parent.absolute()
hovernet_path = workspace_root / "segmentation"
if str(hovernet_path) not in sys.path:
    sys.path.insert(0, str(hovernet_path))


def check_hovernet_setup():
    print("Checking HoverNet environment...")


    model_path = hovernet_path / "weight" / 'hovernet_model_weights.pth'
    if model_path.exists():
        model_size = model_path.stat().st_size / (1024 * 1024)
        print(f" Model file exists: {model_path.name} ({model_size:.1f} MB)")
    else:
        print(f" Model file does not exist: {model_path}")
        return False


    pannuke_root = Path(
        "/path/to/PanNuke_dataset")
    


    images = list(pannuke_root.glob("**/*.png"))

    print(f"Input images: {len(images)} images found")

    if images:
        sample_image = images[0]
        image_size = sample_image.stat().st_size / (1024 * 1024)
        print(f"Sample image: {sample_image.name} ({image_size:.1f} MB)")

    return True


def process_pannuke_images():


    pannuke_root = Path(
        "/path/to/PanNuke_dataset")
    
    output_dir = Path(
        "/path/to/hovernet_results")
    output_dir.mkdir(parents=True, exist_ok=True)


    segmentation_images_dir = output_dir / "segmentation_images"
    segmentation_images_dir.mkdir(exist_ok=True)


    all_images = list(pannuke_root.glob("**/*.png"))

    print(f"Found images: {len(all_images)} total")


    sys.path.insert(0, str(hovernet_path))
    from run_segmentation1_true import OptimizedHoverNetInference


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


        if output_json.exists() and output_image.exists():
            print(f" {image_name} already processed, skipping")
            successful_count += 1
            continue

        print(f"Processing {image_name}...")

        try:

            temp_output_dir = output_dir / f"temp_{image_name}"
            temp_output_dir.mkdir(exist_ok=True)


            inference.process_large_image(
                str(image_file), str(temp_output_dir))


            json_files = list(temp_output_dir.glob("**/*.json"))
            image_files = list(temp_output_dir.glob("**/*.png"))

            print(f"      Found JSON files: {len(json_files)}")
            print(f"      Found image files: {len(image_files)}")


            for json_file in json_files:
                print(f"      JSON: {json_file.relative_to(temp_output_dir)}")
            for img_file in image_files:
                print(f"      Image: {img_file.relative_to(temp_output_dir)}")

            if json_files:

                source_json = json_files[0]
                with open(source_json, 'r') as f:
                    json_content = json.load(f)


                total_nuclei = 0
                for tile in json_content.get('tiles', []):
                    nuclei = tile.get('nuc', {})
                    total_nuclei += len(nuclei)

                print(f"      Detected nuclei: {total_nuclei}")


                source_json.rename(output_json)


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


            import shutil
            if temp_output_dir.exists():
                shutil.rmtree(temp_output_dir)

        except Exception as e:
            print(f" {image_name} processing failed: {e}")
            failed_count += 1

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


    json_count = len(list(output_dir.glob("*.json")))
    image_count = len(list(segmentation_images_dir.glob("*.png")))
    print(f"Generated file statistics:")
    print(f"JSON files: {json_count}")
    print(f"Segmentation images: {image_count}")

    return successful_count > 0


def create_visualization_summary():
    output_dir = Path(
        "./output/step1_hovernet_results")
    segmentation_images_dir = output_dir / "segmentation_images"


    images = list(segmentation_images_dir.glob("*.png"))


    print(f"\ Example files:")
    for i, img_path in enumerate(images[:5]):
        print(f"  {img_path.name}")


if __name__ == "__main__":
    success = process_pannuke_images()

    if success:
        print("\n" + "="*50)
        create_visualization_summary()

    sys.exit(0 if success else 1)
