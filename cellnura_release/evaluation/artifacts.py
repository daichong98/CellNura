"""Strict, portable readers for frozen centroid predictions and annotations."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import math

from .unified_matcher import EvalInput, Instance

SCHEMA = ["image_id", "instance_id", "cx", "cy", "class_id"]


class ArtifactError(ValueError):
    """An absent, changed, or invalid frozen artifact cannot be evaluated."""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    if Path(relative).is_absolute():
        raise ArtifactError(f"Artifact path must be relative: {relative}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ArtifactError(f"Artifact escapes its directory: {relative}")
    if not result.is_file():
        raise ArtifactError(f"Missing artifact: {relative}")
    return result


def checked_file(root, entry):
    path = inside(root, entry["path"])
    expected = entry.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ArtifactError(f"Missing SHA-256 for {entry['path']}")
    if sha256(path) != expected:
        raise ArtifactError(f"SHA-256 mismatch: {entry['path']}")
    return path


def load_catalog(root, verify_all=True):
    root = Path(root)
    if verify_all:
        manifest_path = inside(root, "artifact_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            checked_file(root, entry)
    catalog = json.loads(inside(root, "jobs.json").read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 1:
        raise ArtifactError("Unsupported job registry schema")
    jobs = catalog.get("jobs", [])
    if not jobs or len({j["job_id"] for j in jobs}) != len(jobs):
        raise ArtifactError("Empty registry or duplicate job IDs")
    expected = catalog.get("evaluator_sha256")
    if expected and sha256(Path(__file__).parent / "unified_matcher.py") != expected:
        raise ArtifactError("The packaged matcher differs from the frozen matcher")
    catalog["_catalog_sha256"] = sha256(inside(root, "jobs.json"))
    return catalog


def read_image_ids(root, entry):
    path = checked_file(root, entry)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["image_id"]:
            raise ArtifactError(f"Invalid image-list schema: {entry['path']}")
        ids = [r["image_id"] for r in reader]
    if not ids or any(not v for v in ids) or len(ids) != len(set(ids)):
        raise ArtifactError(f"Empty or duplicate image IDs: {entry['path']}")
    if len(ids) != entry["rows"]:
        raise ArtifactError(f"Image count mismatch: {entry['path']}")
    return tuple(ids)


def read_instances(root, entry, image_ids, allowed_classes, *, ground_truth=False):
    path = checked_file(root, entry)
    allowed_images = set(image_ids)
    allowed_classes = set(allowed_classes)
    instances, seen = [], set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SCHEMA:
            raise ArtifactError(f"Invalid instance schema: {entry['path']}")
        for line, row in enumerate(reader, 2):
            try:
                im, iid = row["image_id"], row["instance_id"]
                x, y = float(row["cx"]), float(row["cy"])
                c = int(row["class_id"])
            except (TypeError, ValueError, KeyError) as error:
                raise ArtifactError(f"Malformed row {line}: {entry['path']}") from error
            if not im or not iid or im not in allowed_images:
                raise ArtifactError(f"Unknown image or empty instance ID at row {line}")
            if (im, iid) in seen:
                raise ArtifactError(f"Duplicate instance ID: {(im, iid)}")
            if not math.isfinite(x) or not math.isfinite(y):
                raise ArtifactError(f"Nonfinite coordinates: {(im, iid)}")
            if c not in allowed_classes:
                side = "GT" if ground_truth else "prediction"
                raise ArtifactError(f"Undeclared {side} class {c}: {(im, iid)}")
            seen.add((im, iid))
            instances.append(Instance(im, iid, x, y, c))
    if len(instances) != entry["rows"]:
        raise ArtifactError(f"Instance count mismatch: {entry['path']}")
    return instances


def load_input(root, job, tau=None):
    ids = read_image_ids(root, job["images"])
    real = tuple(job["real_class_ids"])
    allowed = tuple(job["class_ids"])
    if not real or not set(real).issubset(allowed):
        raise ArtifactError(f"Invalid fixed class vocabulary: {job['job_id']}")
    gt = read_instances(root, job["ground_truth"], ids, real, ground_truth=True)
    pred = read_instances(root, job["prediction"], ids, allowed)
    return EvalInput(dataset=job["dataset"], tau=float(job["tau"] if tau is None else tau),
                     gt=gt, pred=pred, image_ids=ids,
                     class_ids=allowed, real_class_ids=real)


def verify_catalog_inputs(root, catalog):
    """Validate every registered image list, GT file, and prediction file.

    ``load_catalog`` verifies the registry and files listed by its outer
    manifest.  A code-only manifest may intentionally omit the large files
    referenced by individual jobs, so it must not by itself be interpreted as
    proof that evaluation data are present.  This second stage follows every
    job reference and applies the same schema, row-count, class, and SHA checks
    used by evaluation.
    """
    jobs = catalog.get("jobs", [])
    for job in jobs:
        load_input(root, job)
    return len(jobs)
