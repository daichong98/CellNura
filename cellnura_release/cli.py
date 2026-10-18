"""Lazy command dispatch: table reproduction does not require PyTorch."""
import argparse
import importlib
import sys
from . import __version__


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="CellNura second-revision code: image-level training and reproducible evaluation with supplementary data")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", nargs="?", choices=["evaluate", "statistics", "features", "train", "infer", "baseline", "verify", "attach-data"])
    if not argv or argv[0] in ("-h", "--help", "--version"):
        parser.parse_args(argv)
        if not argv:
            parser.print_help()
        return
    command = argv.pop(0)
    if command in ("evaluate", "statistics"):
        return importlib.import_module(".evaluation.reproduce", __package__).main([command] + argv)
    modules = {"features": ".features", "train": ".training", "infer": ".inference", "baseline": ".baselines", "verify": ".verification", "attach-data": ".supplementary"}
    if command not in modules:
        parser.error(f"unknown command: {command}")
    return importlib.import_module(modules[command], __package__).main(argv)
