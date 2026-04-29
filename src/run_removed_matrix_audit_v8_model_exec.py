#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
OPTUNA_ROOT = THIS_FILE.parent
V5_MODEL_EXEC = OPTUNA_ROOT / "run_removed_matrix_audit_v5_model_exec.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-name-override", required=True)
    parser.add_argument("--cache-dir-override", default="")
    return parser.parse_known_args()


def inject_default(argv: list[str], flag: str, value: str) -> list[str]:
    if flag not in argv:
        return [flag, value, *argv]
    return argv


def main() -> None:
    args, passthrough = parse_args()

    wrapped_argv = list(passthrough)
    defaults = [
        ("--phase-b-schedule", "linear"),
        ("--phase-b-linear-start-multiplier", "1.0"),
        ("--phase-b-linear-end-multiplier", "2.0"),
        ("--keep-ft-epochs", "1.0"),
        ("--keep-ft-lr", "5e-5"),
        ("--prune-weight-mode", "paper_rmt"),
        ("--reinsert-budget-mode", "absolute_total_fraction"),
        ("--reinsert-total-fraction", "0.03"),
        ("--final-sparsity-accounting", "post_reinsert_exact"),
        ("--reinsert-rank-mode", "rmt_magnitude"),
        ("--phase-a-batch-budget-mode", "trainable_fraction"),
        ("--phase-a-min-lr-ratio", "0.25"),
    ]
    for flag, value in defaults:
        wrapped_argv = inject_default(wrapped_argv, flag, value)

    command = [
        sys.executable,
        "-u",
        str(V5_MODEL_EXEC),
        "--model-name-override",
        args.model_name_override,
    ]
    if args.cache_dir_override:
        command.extend(["--cache-dir-override", args.cache_dir_override])
    command.extend(wrapped_argv)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
