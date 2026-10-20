#!/usr/bin/env python3
"""Rebuild the five registered evaluation GT files from original datasets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

SCRIPT_DIR = Path(__file__).resolve().parent
# In the detached prediction package, image inventories live beside this file.
DEFAULT_ARTIFACT_ROOT = SCRIPT_DIR
EXPECTED_PATH = SCRIPT_DIR / "expected_ground_truth.json"
HEADER = ["image_id", "instance_id", "cx", "cy", "class_id"]
CONSEP_MAP = {1: 3, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}
CRCHISTO_ORDER = (("epithelial", 1), ("fibroblast", 2),
                   ("inflammatory", 0), ("others", 3))


class ReconstructionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_ids(artifact_root: Path, filename: str) -> list[str]:
    path = artifact_root / "images" / filename
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["image_id"]:
            raise ReconstructionError(f"Invalid image-list schema: {path}")
        values = [row["image_id"] for row in reader]
    if not values or len(values) != len(set(values)):
        raise ReconstructionError(f"Empty or duplicate image IDs: {path}")
    return values


def locate_one(root: Path, direct: Path, pattern: str) -> Path:
    path = root / direct
    if path.is_file():
        return path
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ReconstructionError(
            f"Expected one source file at {path} or matching {pattern}; found {len(matches)}"
        )
    return matches[0]


def write_registered(target: Path, expected: dict, row_writer, overwrite: bool) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        actual = {"rows": sum(1 for _ in target.open("rb")) - 1,
                  "bytes": target.stat().st_size, "sha256": sha256(target)}
        if actual == {k: expected[k] for k in ("rows", "bytes", "sha256")}:
            return {"path": str(target), "status": "already_valid", **actual}
        raise ReconstructionError(f"Existing file is not registered: {target}; use --overwrite")
    temporary = target.with_name(target.name + ".tmp")
    temporary.unlink(missing_ok=True)
    rows = 0
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            rows = row_writer(writer)
        actual = {"rows": rows, "bytes": temporary.stat().st_size,
                  "sha256": sha256(temporary)}
        wanted = {k: expected[k] for k in ("rows", "bytes", "sha256")}
        if actual != wanted:
            raise ReconstructionError(
                f"Registered identity mismatch for {target.name}: actual={actual}, expected={wanted}"
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(target), "status": "rebuilt_and_verified", **actual}


def rebuild_pannuke(root: Path, artifact_root: Path, output: Path,
                    expected: dict, overwrite: bool) -> list[dict]:
    results = []
    for fold in (1, 2, 3):
        source = locate_one(root, Path(f"Fold {fold}/masks/fold{fold}/masks.npy"),
                            f"**/Fold {fold}/masks/fold{fold}/masks.npy")
        masks = np.load(source, mmap_mode="r")
        if masks.ndim != 4 or masks.shape[-1] < 5:
            raise ReconstructionError(f"Unexpected PanNuke mask shape: {source}: {masks.shape}")
        ids = image_ids(artifact_root, f"pannuke_fold{fold}.csv")
        indices = []
        pattern = re.compile(rf"^fold{fold}_image_(\d+)$")
        for image_id in ids:
            match = pattern.match(image_id)
            if not match:
                raise ReconstructionError(f"Invalid PanNuke image ID: {image_id}")
            index = int(match.group(1))
            if index >= masks.shape[0]:
                raise ReconstructionError(f"PanNuke index outside mask array: {image_id}")
            indices.append((image_id, index))
        if len(indices) != masks.shape[0]:
            raise ReconstructionError(
                f"PanNuke fold {fold} image-list length {len(indices)} != mask count {masks.shape[0]}"
            )

        def rows(writer, masks=masks, indices=indices):
            count = 0
            for image_id, index in indices:
                sequence = 0
                for class_id in range(5):
                    channel = masks[index, :, :, class_id]
                    for instance_id in np.unique(channel):
                        if instance_id <= 0:
                            continue
                        ys, xs = np.nonzero(channel == instance_id)
                        if not len(xs):
                            raise ReconstructionError(
                                f"Empty PanNuke instance {instance_id} in {image_id}/class {class_id}"
                            )
                        writer.writerow([image_id, f"g{sequence}", float(xs.mean()),
                                         float(ys.mean()), class_id])
                        sequence += 1
                        count += 1
            return count

        name = f"pannuke_fold{fold}.csv"
        results.append(write_registered(output / name, expected[name], rows, overwrite))
    return results


def consep_label_dir(root: Path) -> Path:
    for candidate in (root / "Test" / "Labels", root / "test" / "Labels",
                      root / "Labels", root / "label"):
        if candidate.is_dir():
            return candidate
    raise ReconstructionError(f"Cannot find the CoNSeP test-label directory below {root}")


def rebuild_consep(root: Path, artifact_root: Path, output: Path,
                   expected: dict, overwrite: bool) -> dict:
    labels = consep_label_dir(root)
    ids = image_ids(artifact_root, "consep.csv")

    def rows(writer):
        count = 0
        for image_id in ids:
            path = labels / f"{image_id}.mat"
            if not path.is_file():
                raise ReconstructionError(f"Missing CoNSeP label file: {path}")
            data = loadmat(path)
            centroids = np.asarray(data.get("inst_centroid"))
            types = np.asarray(data.get("inst_type")).reshape(-1)
            if centroids.ndim != 2 or centroids.shape[1] != 2 or len(centroids) != len(types):
                raise ReconstructionError(f"Unexpected CoNSeP arrays: {path}")
            sequence = 0
            for centroid, source_type in zip(centroids, types):
                class_id = CONSEP_MAP.get(int(source_type))
                if class_id is None:
                    continue
                writer.writerow([image_id, f"g{sequence}", float(centroid[0]),
                                 float(centroid[1]), class_id])
                sequence += 1
                count += 1
        return count

    name = "consep.csv"
    return write_registered(output / name, expected[name], rows, overwrite)


def crchisto_classification_dir(root: Path) -> Path:
    candidate = root / "Classification"
    return candidate if candidate.is_dir() else root


def rebuild_crchisto(root: Path, artifact_root: Path, output: Path,
                     expected: dict, overwrite: bool) -> dict:
    classification = crchisto_classification_dir(root)
    ids = image_ids(artifact_root, "crchisto.csv")

    def rows(writer):
        count = 0
        for image_id in ids:
            sequence = 0
            folder = classification / image_id
            for label, class_id in CRCHISTO_ORDER:
                path = folder / f"{image_id}_{label}.mat"
                if not path.is_file():
                    raise ReconstructionError(f"Missing CRCHistoPhenotypes file: {path}")
                data = loadmat(path)
                if "detection" not in data:
                    raise ReconstructionError(f"Missing 'detection' array: {path}")
                detections = np.asarray(data["detection"])
                if detections.size == 0:
                    continue
                if detections.ndim != 2 or detections.shape[1] != 2:
                    raise ReconstructionError(f"Unexpected detection shape: {path}: {detections.shape}")
                for x, y in detections:
                    writer.writerow([image_id, f"g{sequence}", float(x), float(y), class_id])
                    sequence += 1
                    count += 1
        return count

    name = "crchisto.csv"
    return write_registered(output / name, expected[name], rows, overwrite)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path,
                        help="Output directory; default: ARTIFACT_ROOT/ground_truth")
    parser.add_argument("--pannuke-root", type=Path)
    parser.add_argument("--consep-root", type=Path)
    parser.add_argument("--crchisto-root", type=Path)
    parser.add_argument("--only", action="append",
                        choices=("pannuke", "consep", "crchisto"),
                        help="Rebuild only selected dataset(s); repeat as needed")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    output = (args.output or artifact_root / "ground_truth").resolve()
    if output == artifact_root or artifact_root in output.parents:
        raise ReconstructionError(
            "Reconstructed GT must be written outside the prediction-only package; "
            "set --output /path/outside/package"
        )
    registered = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))["files"]
    selected = set(args.only or ("pannuke", "consep", "crchisto"))
    roots = {"pannuke": args.pannuke_root, "consep": args.consep_root,
             "crchisto": args.crchisto_root}
    missing = sorted(name for name in selected if roots[name] is None)
    if missing:
        parser.error("source root required for: " + ", ".join(missing))

    results = []
    if "pannuke" in selected:
        results.extend(rebuild_pannuke(roots["pannuke"].resolve(), artifact_root,
                                       output, registered, args.overwrite))
    if "consep" in selected:
        results.append(rebuild_consep(roots["consep"].resolve(), artifact_root,
                                      output, registered, args.overwrite))
    if "crchisto" in selected:
        results.append(rebuild_crchisto(roots["crchisto"].resolve(), artifact_root,
                                        output, registered, args.overwrite))
    print(json.dumps({"verified": True, "files": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReconstructionError as error:
        raise SystemExit(f"Ground-truth reconstruction refused: {error}") from error
