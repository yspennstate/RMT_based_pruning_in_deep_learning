#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
OPTUNA_ROOT = THIS_FILE.parent
SOURCE_PATH = OPTUNA_ROOT / "removed_matrix_audit" / "run_removed_matrix_audit_v5.py"
MODEL_NEEDLE = 'model_name = "vit_base_patch16_224.augreg2_in21k_ft_in1k"'
CACHE_NEEDLE = 'optuna_root / "rmt_cache",'


def patch_pillow_exiftags() -> None:
    try:
        import PIL.ExifTags
        import PIL.Image
    except Exception:
        return
    if not hasattr(PIL.Image, "ExifTags"):
        PIL.Image.ExifTags = PIL.ExifTags


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-name-override", required=True)
    parser.add_argument("--cache-dir-override", default="")
    return parser.parse_known_args()


def transformed_source_text() -> str:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    replacement = 'model_name = os.environ["RMT_MODEL_NAME_OVERRIDE"]'
    if MODEL_NEEDLE not in source:
        raise RuntimeError(f"Did not find model needle in {SOURCE_PATH}")
    source = source.replace(MODEL_NEEDLE, replacement, 1)

    cache_replacement = 'Path(os.environ.get("RMT_CACHE_OVERRIDE", str(optuna_root / "rmt_cache"))),'
    if CACHE_NEEDLE not in source:
        raise RuntimeError(f"Did not find cache needle in {SOURCE_PATH}")
    source = source.replace(CACHE_NEEDLE, cache_replacement, 1)
    return source


def main() -> None:
    args, passthrough = parse_args()
    patch_pillow_exiftags()
    os.environ["RMT_MODEL_NAME_OVERRIDE"] = args.model_name_override
    if args.cache_dir_override:
        os.environ["RMT_CACHE_OVERRIDE"] = args.cache_dir_override

    source = transformed_source_text()
    sys.argv = [str(SOURCE_PATH), *passthrough]
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(SOURCE_PATH),
        "__package__": None,
    }
    exec(compile(source, str(SOURCE_PATH), "exec"), globals_dict)


if __name__ == "__main__":
    main()
