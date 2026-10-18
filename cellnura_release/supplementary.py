from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil

from .paths import project_root
from .verification import file_sha256


def _inside(root, relative, *, existing=False):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"Invalid artifact path: {relative!r}")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or rel.as_posix() != relative:
        raise ValueError(f"Artifact path must be a canonical relative path: {relative}")
    path = root.joinpath(*rel.parts)
    if root not in path.resolve().parents:
        raise ValueError(f"Artifact path escapes its directory: {relative}")
    for current in (path, *path.parents):
        if current == root:
            break
        if current.is_symlink():
            raise ValueError(f"Artifact symlink refused: {relative}")
    if existing and not path.is_file():
        raise FileNotFoundError(f"Missing supplementary artifact: {relative}")
    return path


def attach_artifacts(source, destination, expected_manifest, *, check_only=False):
    source = Path(source).expanduser().resolve()
    if not (source / "artifact_manifest.json").is_file():
        source = source / "artifacts"
    expected_manifest = Path(expected_manifest)
    manifest_path = _inside(source, "artifact_manifest.json", existing=True)
    if file_sha256(manifest_path) != file_sha256(expected_manifest):
        raise ValueError("Supplementary manifest differs from the manifest pinned in this code release")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Empty or invalid supplementary manifest")
    entries = entries + [{"path": "artifact_manifest.json", "sha256": file_sha256(manifest_path)}]
    destination = Path(destination).expanduser()
    if destination.is_symlink():
        raise ValueError("Destination directory cannot be a symlink")
    destination = destination.resolve()
    if destination.exists() and not destination.is_dir():
        raise ValueError("Destination must be a directory")
    seen, pending, reused, total_bytes = set(), [], 0, 0
    # Verify every source and every existing destination before making any changes.
    for entry in entries:
        relative, digest = entry["path"], entry["sha256"]
        if relative in seen:
            raise ValueError(f"Duplicate artifact path: {relative}")
        seen.add(relative)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid SHA256: {relative}")
        original = _inside(source, relative, existing=True)
        if file_sha256(original) != digest:
            raise ValueError(f"Source SHA256 mismatch: {relative}")
        target = _inside(destination, relative)
        total_bytes += original.stat().st_size
        if target.exists():
            if not target.is_file() or file_sha256(target) != digest:
                raise ValueError(f"Existing destination differs; no files replaced: {relative}")
            reused += 1
        else:
            pending.append((original, target, digest))
    if not check_only:
        for original, target, digest in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also refuses a conflicting file created after preflight.
            with target.open("xb") as output:
                try:
                    with original.open("rb") as input_stream:
                        shutil.copyfileobj(input_stream, output, length=8 * 1024 * 1024)
                except BaseException:
                    output.close()
                    target.unlink()
                    raise
            if file_sha256(target) != digest:
                target.unlink()
                raise ValueError(f"Source changed while copying: {original.name}")
    return {"supplementary_manifest_sha256": file_sha256(manifest_path),
            "source_files_verified": len(entries), "source_bytes": total_bytes,
            "existing_identical_files": reused,
            "files_to_copy": len(pending), "copied_files": 0 if check_only else len(pending),
            "check_only": check_only}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Extracted supplementary artifacts directory, or the complete release containing artifacts/")
    parser.add_argument("--destination", type=Path, default=project_root() / "artifacts")
    parser.add_argument("--check-only", action="store_true", help="Verify without copying any files")
    args = parser.parse_args(argv)
    try:
        result = attach_artifacts(args.source, args.destination,
                                  project_root() / "artifacts/artifact_manifest.json",
                                  check_only=args.check_only)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Supplementary data refused: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
