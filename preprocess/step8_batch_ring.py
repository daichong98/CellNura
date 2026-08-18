
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import math
import re
import glob
from scipy.ndimage import distance_transform_edt
from sklearn.preprocessing import normalize


class PanNukeRingExtractor:
    def __init__(self, output_dir, thickness=1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.thickness = thickness

    def process_single_nucleus_image(self, image_path):
        try:

            image = cv2.imread(image_path)
            if image is None:
                print(f"Cannot read image: {image_path}")
                return {}


            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print(f"No contour found in {image_path}")
                return {}

            largest_contour = max(contours, key=cv2.contourArea)


            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=-1)


            dist_transform = distance_transform_edt(mask)
            max_distance = int(math.ceil(dist_transform.max()))

            if max_distance == 0:
                print(f"Distance transform result is 0 for {image_path}")
                return {}


            red_channel = image[:, :, 2]

            fr_values = {}


            ring_label = 'FR1_2'
            ring_mask = (dist_transform > 0) & (dist_transform < 2)

            ring_area = np.sum(ring_mask)
            if ring_area > 0:
                ring_pixels = red_channel[ring_mask]
                mean = ring_pixels.mean()
                std = ring_pixels.std()
                threshold = mean + 0.5 * std
                fr_value = np.sum(ring_pixels > threshold) / ring_area
            else:
                fr_value = 0.0
            fr_values[ring_label] = fr_value


            ring_num = 3
            current_dist = 2
            while current_dist <= max_distance:
                start_dist = current_dist
                end_dist = current_dist + self.thickness


                if end_dist > max_distance:
                    end_dist = max_distance + 1

                ring_label = f'FR{ring_num}'
                ring_mask = (dist_transform >= start_dist) & (
                    dist_transform < end_dist)

                ring_area = np.sum(ring_mask)
                if ring_area > 0:
                    ring_pixels = red_channel[ring_mask]
                    mean = ring_pixels.mean()
                    std = ring_pixels.std()
                    threshold = mean + 0.5 * std
                    fr_value = np.sum(ring_pixels > threshold) / ring_area
                else:
                    fr_value = 0.0
                fr_values[ring_label] = fr_value

                ring_num += 1
                current_dist += self.thickness

            return {'image_name': os.path.basename(image_path), **fr_values}

        except Exception as e:
            print(f" Error processing image {image_path}: {e}")
            return {}

    def extract_nucleus_info(self, file_name):

        match = re.match(
            r"fold(\d+)_image_(\d+)_nucleus_(\d+)\.png", file_name)

        if match:
            fold_idx = int(match.group(1))
            image_idx = int(match.group(2))
            nucleus_idx = int(match.group(3))
            return fold_idx, image_idx, nucleus_idx

        return None, None, None

    def process_image_nuclei(self, image_name, nuclei_folder):
        print(f" Processing image: {image_name}")


        nucleus_pattern = os.path.join(
            nuclei_folder, f"{image_name}_nucleus_*.png")
        nucleus_files = glob.glob(nucleus_pattern)

        if not nucleus_files:
            print(f" No nucleus files found for image {image_name}")
            return []

        features_list = []

        for nucleus_file in nucleus_files:
            nucleus_filename = os.path.basename(nucleus_file)
            fold_idx, image_idx, nucleus_idx = self.extract_nucleus_info(
                nucleus_filename)

            if fold_idx is None or nucleus_idx is None:
                print(f" Cannot parse filename: {nucleus_filename}")
                continue


            ring_features = self.process_single_nucleus_image(nucleus_file)

            if ring_features and len(ring_features) > 1:

                ring_features['original_image'] = image_name
                ring_features['nucleus_id'] = nucleus_idx
                ring_features['fold'] = fold_idx
                features_list.append(ring_features)

        print(f" Image {image_name}: Processed {len(features_list)} nuclei")
        return features_list

    def process_pannuke_dataset(self, nuclei_folder):
        print(" Start processing PanNuke dataset - Ring feature extraction")

        nuclei_folder = Path(nuclei_folder)
        if not nuclei_folder.exists():
            print(f" Nucleus folder does not exist: {nuclei_folder}")
            return False


        fold1_nucleus_files = list(nuclei_folder.glob("fold1_*_nucleus_*.png"))
        fold2_nucleus_files = list(nuclei_folder.glob("fold2_*_nucleus_*.png"))
        fold3_nucleus_files = list(nuclei_folder.glob("fold3_*_nucleus_*.png"))
        all_nucleus_files = fold1_nucleus_files + \
            fold2_nucleus_files + fold3_nucleus_files


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


        for i, image_name in enumerate(image_names, 1):
            fold_idx = int(image_name[4:5])
            print(f"\n[{i}/{len(image_names)}] Processing Fold {fold_idx} - {image_name}")

            image_features = self.process_image_nuclei(
                image_name, nuclei_folder)
            all_features.extend(image_features)


        if all_features:
            self.save_features(all_features)
            return True
        else:
            print(" No features extracted")
            return False

    def save_features(self, features_list):
        df = pd.DataFrame(features_list)


        df.sort_values(by=['fold', 'original_image',
                       'nucleus_id'], inplace=True)


        df.fillna(0, inplace=True)


        df = df.loc[:, (df != 0).any(axis=0)]


        id_columns = ['image_name', 'original_image', 'nucleus_id', 'fold']
        feature_columns = [
            col for col in df.columns if col not in id_columns and col.startswith('FR')]
        df = df[id_columns + feature_columns]


        output_csv = self.output_dir / "pannuke_ring_features.csv"
        df.to_csv(output_csv, index=False)

        print(f"\n Ring features saved: {output_csv}")


        training_df = df[['nucleus_id', 'fold'] + feature_columns].copy()
        training_csv = self.output_dir / "pannuke_ring_features_training.csv"
        training_df.to_csv(training_csv, index=False)

        print(f" Training feature file: {training_csv}")


        print(f"\n Ring feature statistics:")
        for col in feature_columns[:5]:
            if col in df.columns:
                print(
                    f"  {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")


def main():

    NUCLEI_FOLDER = "/path/to/step2_nuclei_images"
    OUTPUT_DIR = "./output/step7_ring"
    RING_THICKNESS = 1


    extractor = PanNukeRingExtractor(OUTPUT_DIR, RING_THICKNESS)


    success = extractor.process_pannuke_dataset(NUCLEI_FOLDER)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
