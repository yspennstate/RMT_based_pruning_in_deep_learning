#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
OPTUNA_ROOT = THIS_FILE.parent
SOURCE_PATH = OPTUNA_ROOT / "run_finetune_magnitude.py"


def patch_pillow_exiftags() -> None:
    try:
        import PIL.ExifTags
        import PIL.Image
    except Exception:
        return
    if not hasattr(PIL.Image, "ExifTags"):
        PIL.Image.ExifTags = PIL.ExifTags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the original magnitude pipeline with a model-name override.")
    parser.add_argument("--model-name-override", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--batch-size-train", type=int, default=0)
    parser.add_argument("--batch-size-val", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_source_module():
    spec = importlib.util.spec_from_file_location("run_finetune_magnitude_source", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    os.environ.setdefault("RMT_SKIP_PRE_FT_EVAL", "1")
    patch_pillow_exiftags()
    module = load_source_module()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module.OUT_DIR = output_dir
    module.CKPT_DIR = output_dir / "checkpoints"
    module.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    module.RESULTS_FILE = output_dir / "finetune_results.json"
    module.LOG_FILE = output_dir / "finetune_log.txt"

    if args.max_cycles > 0:
        module.CYCLES = list(module.CYCLES[: args.max_cycles])
    if args.batch_size_train > 0:
        module.BATCH_SIZE_TRAIN = int(args.batch_size_train)
    if args.batch_size_val > 0:
        module.BATCH_SIZE_VAL = int(args.batch_size_val)
    if args.num_workers > 0:
        module.NUM_WORKERS = int(args.num_workers)

    original_create_model = module.timm.create_model

    def create_model_override(_model_name, *create_args, **create_kwargs):
        return original_create_model(args.model_name_override, *create_args, **create_kwargs)

    module.timm.create_model = create_model_override
    print(
        f"[run_finetune_magnitude_model_exec] model={args.model_name_override} "
        f"output_dir={output_dir} max_cycles={args.max_cycles or len(module.CYCLES)}",
        flush=True,
    )
    module.main()


if __name__ == "__main__":
    main()
