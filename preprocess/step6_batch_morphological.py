import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from skimage.measure import regionprops
import glob
import re
from sklearn.preprocessing import StandardScaler


class PanNukeMorphologicalExtractor:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_nucleus_features(self, contour_points):
        if not contour_points:
            return None

        try:
            contour_points = np.array(contour_points, dtype=np.int32)


            x_coords = contour_points[:, 0]
            y_coords = contour_points[:, 1]

            min_x, max_x = x_coords.min(), x_coords.max()
            min_y, max_y = y_coords.min(), y_coords.max()


            width = max_x - min_x + 10
            height = max_y - min_y + 10


            adjusted_contour = contour_points.copy()
            adjusted_contour[:, 0] -= (min_x - 5)
            adjusted_contour[:, 1] -= (min_y - 5)

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [adjusted_contour], -
                             1, 1, thickness=cv2.FILLED)


            props = regionprops(mask)[0]


            area = props.area
            perimeter = props.perimeter
            convex_area = props.convex_area
            bbox_area = props.bbox_area
            equivalent_diameter = props.equivalent_diameter
            extent = props.extent
            major_axis_length = props.major_axis_length
            minor_axis_length = props.minor_axis_length
            orientation = props.orientation
            solidity = props.solidity
            inertia_tensor_eigvals = props.inertia_tensor_eigvals
            eccentricity = props.eccentricity


            circularity = (4 * np.pi * area / (perimeter ** 2)
                           ) if perimeter > 0 else 0


            aspect_ratio = major_axis_length / \
                minor_axis_length if minor_axis_length > 0 else 0


            convexity = convex_area / area if area > 0 else 0


            convex_perimeter = 0
            try:
                hull = cv2.convexHull(adjusted_contour)
                if len(hull) >= 2:
                    convex_perimeter = cv2.arcLength(hull, True)
            except Exception:
                convex_perimeter = 0


            irregularity_index = convex_perimeter / \
                (2 * np.sqrt(np.pi * area)) if area > 0 and convex_perimeter > 0 else 0

            return {
                'Area': area,
                'Perimeter': perimeter,
                'Circularity': circularity,
                'Aspect_Ratio': aspect_ratio,
                'Convex_Area': convex_area,
                'Solidity': solidity,
                'Equivalent_Diameter': equivalent_diameter,
                'Eccentricity': eccentricity,
                'Bbox_Area': bbox_area,
                'Extent': extent,
                'Major_Axis_Length': major_axis_length,
                'Minor_Axis_Length': minor_axis_length,
                'Orientation': orientation,
                'Inertia_Tensor_Eigvals_X': inertia_tensor_eigvals[0],
                'Inertia_Tensor_Eigvals_Y': inertia_tensor_eigvals[1],
                'Convexity': convexity,
                'Irregularity_Index': irregularity_index,
            }

        except Exception as e:
            print(f" Error calculating morphological features: {e}")
            return None

    def normalize_features_by_column(self, df):
        print(" Executing feature column normalization...")


        id_columns = ['image_name', 'original_image',
                      'nucleus_id', 'tile_x', 'tile_y', 'mag', 'fold']
        feature_columns = [col for col in df.columns if col not in id_columns]

        if not feature_columns:
            print(" No feature columns found")
            return df

        print(f"  Found {len(feature_columns)} feature columns to normalize")


        for col in feature_columns:

            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:

                valid_mask = np.isfinite(df[col])
                if valid_mask.sum() > 0:
                    scaler = StandardScaler()
                    df.loc[valid_mask, col] = scaler.fit_transform(
                        df.loc[valid_mask, col].values.reshape(-1, 1))

        return df

    def process_single_json(self, json_file_path, image_name):

        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f" Failed to read JSON file: {e}")
            return []

        features_list = []


        fold_match = re.match(r"fold(\d+)_image_\d+", image_name)
        fold = int(fold_match.group(1)) if fold_match else None


        for tile in data.get("tiles", []):
            tile_x = tile.get("x", 0)
            tile_y = tile.get("y", 0)
            mag = tile.get("mag", "")
            nuclei_data = tile.get("nuc", {})


            for label, nucleus in nuclei_data.items():
                if nucleus is None:
                    continue

                contour_points = nucleus.get('contour', [])
                if not contour_points:
                    continue


                features = self.calculate_nucleus_features(contour_points)

                if features is not None:

                    features['image_name'] = f"{image_name}_nucleus_{label}.png"
                    features['original_image'] = image_name
                    features['nucleus_id'] = int(label)
                    features['tile_x'] = tile_x
                    features['tile_y'] = tile_y
                    features['mag'] = mag
                    features['fold'] = fold

                    features_list.append(features)
        return features_list

    def process_pannuke_dataset(self, segmentation_folder):

        segmentation_folder = Path(segmentation_folder)


        fold1_json_files = list(
            segmentation_folder.glob("fold1_*_segmentation.json"))
        fold2_json_files = list(
            segmentation_folder.glob("fold2_*_segmentation.json"))
        fold3_json_files = list(
            segmentation_folder.glob("fold3_*_segmentation.json"))
        all_json_files = fold1_json_files + fold2_json_files + fold3_json_files
        all_json_files.sort()


        all_features = []


        for i, json_file in enumerate(all_json_files, 1):


            image_name_match = re.match(
                r"(fold\d+_image_\d+)_segmentation\.json", json_file.name)

            if image_name_match:
                image_name = image_name_match.group(1)
                fold_match = re.match(r"fold(\d+)", image_name)
                fold = int(fold_match.group(1)) if fold_match else None
            else:
                print(f" Cannot parse filename: {json_file.name}")
                continue

            print(f"\n[{i}/{len(all_json_files)}] Processing Fold {fold} - {image_name}")


            image_features = self.process_single_json(json_file, image_name)
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


        id_columns = ['image_name', 'original_image',
                      'nucleus_id', 'tile_x', 'tile_y', 'mag', 'fold']
        feature_columns = [col for col in df.columns if col not in id_columns]
        df = df[id_columns + feature_columns]


        df = self.normalize_features_by_column(df)


        output_csv = self.output_dir / "pannuke_morphological_features.csv"
        df.to_csv(output_csv, index=False)

        print(f"\n Morphological features saved: {output_csv}")


        training_df = df[['nucleus_id', 'fold'] + feature_columns].copy()
        training_csv = self.output_dir / "pannuke_morphological_features_training.csv"
        training_df.to_csv(training_csv, index=False)

        print(f" Training feature file: {training_csv}")


        print(f"\n Feature statistics:")
        for col in feature_columns[:5]:
            print(
                f"  {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")


def main():

    SEGMENTATION_FOLDER = "/path/to/PanNuke_classification/step1_hovernet_results"
    OUTPUT_DIR = "./output/step6_morphological"


    extractor = PanNukeMorphologicalExtractor(OUTPUT_DIR)


    success = extractor.process_pannuke_dataset(SEGMENTATION_FOLDER)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
