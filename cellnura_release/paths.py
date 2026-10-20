import json
import os
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path or os.environ.get("CELLNURA_CONFIG_PATH") or project_root()/"configs/release.json").expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    config["_config_dir"] = str(path.parent)
    config["_config_path"] = str(path)
    base = Path(config.get("path_base", ".."))
    config["_path_base"] = str((path.parent/base).resolve() if not base.is_absolute() else base)
    return config


def resolve_path(value, config=None):
    if value is None or value == "":
        return None
    config = load_config() if config is None else config
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else Path(config.get("_path_base", project_root()))/path


def get_path(key, config=None, required=False):
    config = load_config() if config is None else config
    value = config.get("paths", {}).get(key, config.get("common_paths", {}).get(key))
    path = resolve_path(value, config)
    if required and (path is None or not path.exists()):
        raise FileNotFoundError(f"Configure paths.{key} in {config.get('_config_path', 'config')}: {path}")
    return path

