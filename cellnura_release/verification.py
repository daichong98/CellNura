import argparse
import hashlib
import json
from pathlib import Path
from .paths import project_root


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(root, manifest):
    root = Path(root).resolve()
    failures = []
    for entry in manifest["files"]:
        path = root/entry["path"]
        if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
            failures.append(f"Missing/outside-root/symlink: {entry['path']}")
        elif path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            failures.append(f"Identity changed: {entry['path']}")
    if failures:
        raise ValueError("\n".join(failures))
    return len(manifest["files"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=project_root())
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--artifacts",type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest_path = args.manifest or root/"release_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        count = verify_files(root,manifest)
        profile = manifest.get("delivery_profile", "complete")
        catalog = None
        verified_jobs = 0
        if args.artifacts is not None or profile != "code-only":
            from .evaluation.artifacts import load_catalog, verify_catalog_inputs
            artifact_root = args.artifacts or root/"artifacts"
            if profile == "code-only" and file_sha256(artifact_root/"artifact_manifest.json") != file_sha256(root/"artifacts/artifact_manifest.json"):
                raise ValueError("Supplementary manifest differs from this code release")
            catalog = load_catalog(artifact_root)
            verified_jobs = verify_catalog_inputs(artifact_root, catalog)
    except (OSError,ValueError,KeyError) as error:
        parser.exit(2,f"Release verification refused: {error}\n")
    print(json.dumps({"release_id":manifest["release_id"],"delivery_profile":profile,
                      "verified_files":count,"code_files_verified":True,
                      "evaluation_data_verified":catalog is not None and verified_jobs == len(catalog["jobs"]),
                      "main_evaluation_jobs":verified_jobs if catalog is not None else None,
                      "verified":True},indent=2))
    return 0
