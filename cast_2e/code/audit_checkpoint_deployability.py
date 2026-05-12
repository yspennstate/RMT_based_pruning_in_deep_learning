"""Audit pruning-checkpoint sparsity patterns for deployment claims.

This script checks checkpoint tensors directly. It distinguishes:

* native Linear 2:4 deployability under the PyTorch/NVIDIA semi-structured path;
* flattened Conv2d 2:4 legality, which is a structured MAC-accounting property;
* TensorRT Conv2d 2:4 legality, which checks the deployable pattern over input
  channels at each kernel pixel;
* wider k:n legality inferred from checkpoint labels such as D612 or D816.

It does not benchmark throughput. A row is a native sparse-kernel deployment row
only if exact Linear 2:4 legality is present and a separate benchmark confirms
successful runtime conversion and wall-clock speedup.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def state_dict_from(raw: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model", "student", "net", "module"):
            value = raw.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in raw.keys()):
            tensor_items = {k: v for k, v in raw.items() if torch.is_tensor(v)}
            if tensor_items:
                return tensor_items
    raise TypeError("unrecognized checkpoint format")


def infer_kn(path: str, raw: Any) -> tuple[int | None, int | None]:
    if isinstance(raw, dict):
        k = raw.get("k")
        n = raw.get("n")
        if isinstance(k, int) and isinstance(n, int):
            return k, n

    name = Path(path).stem
    parent = str(Path(path).parent)
    text = f"{name} {parent}"
    match = re.search(r"[DS](\d{2,4})(?:_|\.|$)", text, flags=re.I)
    if match:
        token = match.group(1)
        if token in {"24", "48", "612", "816", "1016", "1216", "1632"}:
            if token == "612":
                return 6, 12
            return int(token[:-2]), int(token[-2:])

    if re.search(r"(?:2[_-]?4|2to4|canonical)", text, flags=re.I):
        return 2, 4
    if re.search(r"(?:8[_-]?16)", text, flags=re.I):
        return 8, 16
    return None, None


def flatten_weight(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor] | None:
    if not name.endswith(".weight"):
        return None
    if not tensor.is_floating_point():
        return None
    if tensor.ndim == 2:
        return "linear", tensor.detach()
    if tensor.ndim == 4:
        return "conv2d_flat", tensor.detach().reshape(tensor.shape[0], -1)
    return None


def group_stats(weight_2d: torch.Tensor, n: int, k: int) -> dict[str, Any] | None:
    cols = int(weight_2d.shape[1])
    if cols % n != 0:
        return None
    grouped = weight_2d.reshape(weight_2d.shape[0], cols // n, n)
    nnz = grouped.ne(0).sum(dim=-1)
    total = int(nnz.numel())
    bad = int(nnz.ne(k).sum().item())
    hist: dict[str, int] = {}
    for value in nnz.flatten().tolist():
        key = str(int(value))
        hist[key] = hist.get(key, 0) + 1
    return {
        "groups": total,
        "bad_groups": bad,
        "exact": bad == 0,
        "nnz_hist": hist,
    }


def tensorrt_conv2d_2_4_stats(weight_4d: torch.Tensor) -> dict[str, Any] | None:
    """Check TensorRT's Conv2d 2:4 rule.

    TensorRT checks every four input channels for each output channel and each
    kernel pixel. PyTorch Conv2d weights are stored as [out, in, kh, kw].
    """
    if weight_4d.ndim != 4:
        return None
    out_channels, in_channels, kh, kw = [int(x) for x in weight_4d.shape]
    if in_channels < 4 or in_channels % 4 != 0:
        return None
    grouped = weight_4d.detach().permute(0, 2, 3, 1).reshape(
        out_channels, kh, kw, in_channels // 4, 4
    )
    nnz = grouped.ne(0).sum(dim=-1)
    total = int(nnz.numel())
    bad = int(nnz.gt(2).sum().item())
    hist: dict[str, int] = {}
    for value in nnz.flatten().tolist():
        key = str(int(value))
        hist[key] = hist.get(key, 0) + 1
    return {
        "groups": total,
        "bad_groups": bad,
        "exact": bad == 0,
        "nnz_hist": hist,
    }


def audit_one(path: Path) -> dict[str, Any]:
    t0 = time.time()
    raw = load_checkpoint(path)
    state = state_dict_from(raw)
    expected_k, expected_n = infer_kn(str(path), raw)

    totals = {
        "params": 0,
        "nnz": 0,
        "linear_layers": 0,
        "conv2d_layers": 0,
        "linear_native_2_4_eligible_layers": 0,
        "linear_native_2_4_exact_layers": 0,
        "linear_native_2_4_convertible_layers": 0,
        "linear_native_2_4_exact_convertible_layers": 0,
        "linear_native_2_4_unsupported_shape_layers": 0,
        "linear_native_2_4_unsupported_shape_params": 0,
        "linear_native_2_4_groups": 0,
        "linear_native_2_4_bad_groups": 0,
        "conv_flat_2_4_eligible_layers": 0,
        "conv_flat_2_4_exact_layers": 0,
        "conv_flat_2_4_groups": 0,
        "conv_flat_2_4_bad_groups": 0,
        "conv_tensorrt_2_4_eligible_layers": 0,
        "conv_tensorrt_2_4_exact_layers": 0,
        "conv_tensorrt_2_4_groups": 0,
        "conv_tensorrt_2_4_bad_groups": 0,
        "expected_kn_eligible_layers": 0,
        "expected_kn_exact_layers": 0,
        "expected_kn_groups": 0,
        "expected_kn_bad_groups": 0,
    }
    failing_layers: list[dict[str, Any]] = []

    for name, tensor in state.items():
        flattened = flatten_weight(name, tensor)
        if flattened is None:
            continue
        layer_type, weight = flattened
        params = int(weight.numel())
        nnz = int(weight.ne(0).sum().item())
        totals["params"] += params
        totals["nnz"] += nnz
        if layer_type == "linear":
            totals["linear_layers"] += 1
        elif layer_type == "conv2d_flat":
            totals["conv2d_layers"] += 1
            stats_trt = tensorrt_conv2d_2_4_stats(tensor)
            if stats_trt is not None:
                totals["conv_tensorrt_2_4_eligible_layers"] += 1
                totals["conv_tensorrt_2_4_groups"] += stats_trt["groups"]
                totals["conv_tensorrt_2_4_bad_groups"] += stats_trt["bad_groups"]
                if stats_trt["exact"]:
                    totals["conv_tensorrt_2_4_exact_layers"] += 1
                elif len(failing_layers) < 25:
                    failing_layers.append({
                        "name": name,
                        "layer_type": "conv2d",
                        "check": "TensorRT Conv2d 2:4",
                        "groups": stats_trt["groups"],
                        "bad_groups": stats_trt["bad_groups"],
                        "nnz_hist": stats_trt["nnz_hist"],
                    })

        stats_24 = group_stats(weight, 4, 2)
        if stats_24 is not None:
            prefix = "linear_native_2_4" if layer_type == "linear" else "conv_flat_2_4"
            totals[f"{prefix}_eligible_layers"] += 1
            totals[f"{prefix}_groups"] += stats_24["groups"]
            totals[f"{prefix}_bad_groups"] += stats_24["bad_groups"]
            if stats_24["exact"]:
                totals[f"{prefix}_exact_layers"] += 1
            if layer_type == "linear":
                rows, cols = int(weight.shape[0]), int(weight.shape[1])
                convertible = rows >= 16 and cols >= 16 and rows % 16 == 0 and cols % 16 == 0
                if convertible:
                    totals["linear_native_2_4_convertible_layers"] += 1
                    if stats_24["exact"]:
                        totals["linear_native_2_4_exact_convertible_layers"] += 1
                else:
                    totals["linear_native_2_4_unsupported_shape_layers"] += 1
                    totals["linear_native_2_4_unsupported_shape_params"] += params
            if not stats_24["exact"] and len(failing_layers) < 25:
                failing_layers.append({
                    "name": name,
                    "layer_type": layer_type,
                    "check": "2:4",
                    "groups": stats_24["groups"],
                    "bad_groups": stats_24["bad_groups"],
                    "nnz_hist": stats_24["nnz_hist"],
                })

        if expected_k is not None and expected_n is not None:
            stats_kn = group_stats(weight, expected_n, expected_k)
            if stats_kn is not None:
                totals["expected_kn_eligible_layers"] += 1
                totals["expected_kn_groups"] += stats_kn["groups"]
                totals["expected_kn_bad_groups"] += stats_kn["bad_groups"]
                if stats_kn["exact"]:
                    totals["expected_kn_exact_layers"] += 1
                elif len(failing_layers) < 25:
                    failing_layers.append({
                        "name": name,
                        "layer_type": layer_type,
                        "check": f"{expected_k}:{expected_n}",
                        "groups": stats_kn["groups"],
                        "bad_groups": stats_kn["bad_groups"],
                        "nnz_hist": stats_kn["nnz_hist"],
                    })

    density = totals["nnz"] / totals["params"] if totals["params"] else None
    linear_native_exact = (
        totals["linear_native_2_4_eligible_layers"] > 0
        and totals["linear_native_2_4_bad_groups"] == 0
    )
    linear_native_convertible_exact = (
        totals["linear_native_2_4_convertible_layers"] > 0
        and totals["linear_native_2_4_exact_convertible_layers"]
        == totals["linear_native_2_4_convertible_layers"]
    )
    conv_flat_exact = (
        totals["conv_flat_2_4_eligible_layers"] > 0
        and totals["conv_flat_2_4_bad_groups"] == 0
    )
    conv_tensorrt_exact = (
        totals["conv_tensorrt_2_4_eligible_layers"] > 0
        and totals["conv_tensorrt_2_4_bad_groups"] == 0
    )
    expected_kn_exact = (
        expected_k is not None
        and totals["expected_kn_eligible_layers"] > 0
        and totals["expected_kn_bad_groups"] == 0
    )

    return {
        "path": str(path),
        "file_size": path.stat().st_size,
        "expected_k": expected_k,
        "expected_n": expected_n,
        "sparsity_over_weight_tensors": None if density is None else 1.0 - density,
        "linear_native_2_4_exact": linear_native_exact,
        "linear_native_2_4_convertible_exact": linear_native_convertible_exact,
        "conv_flat_2_4_exact": conv_flat_exact,
        "conv_tensorrt_2_4_exact": conv_tensorrt_exact,
        "expected_kn_exact": expected_kn_exact,
        "native_deployability_interpretation": (
            "native_linear_2_4_candidate"
            if linear_native_convertible_exact
            and totals["linear_native_2_4_unsupported_shape_layers"] == 0
            else "native_linear_2_4_candidate_with_skipped_shapes"
            if linear_native_convertible_exact
            and totals["linear_native_2_4_unsupported_shape_layers"] > 0
            else "not_native_linear_2_4"
        ),
        "totals": totals,
        "sample_failing_layers": failing_layers,
        "seconds": time.time() - t0,
    }


def read_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.paths:
        paths.extend(Path(p) for p in args.paths)
    if args.paths_file:
        data = json.loads(args.paths_file.read_text(encoding="utf-8"))
        for item in data:
            if isinstance(item, str):
                paths.append(Path(item))
            elif isinstance(item, dict):
                paths.append(Path(item["FullName"] if "FullName" in item else item["path"]))
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve()).lower()
        if key not in seen and path.exists():
            seen.add(key)
            out.append(path)
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "path",
        "expected_k",
        "expected_n",
        "sparsity_over_weight_tensors",
        "linear_native_2_4_exact",
        "linear_native_2_4_convertible_exact",
        "conv_flat_2_4_exact",
        "conv_tensorrt_2_4_exact",
        "expected_kn_exact",
        "native_deployability_interpretation",
        "linear_native_2_4_eligible_layers",
        "linear_native_2_4_convertible_layers",
        "linear_native_2_4_unsupported_shape_layers",
        "linear_native_2_4_unsupported_shape_params",
        "linear_native_2_4_bad_groups",
        "conv_flat_2_4_eligible_layers",
        "conv_flat_2_4_bad_groups",
        "conv_tensorrt_2_4_eligible_layers",
        "conv_tensorrt_2_4_bad_groups",
        "expected_kn_eligible_layers",
        "expected_kn_bad_groups",
        "seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {k: row.get(k) for k in fields}
            totals = row.get("totals", {})
            for key in fields:
                if key in totals:
                    flat[key] = totals[key]
            writer.writerow(flat)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="*")
    parser.add_argument("--paths-file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    paths = read_paths(args)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for idx, path in enumerate(paths, start=1):
        print(f"[{idx}/{len(paths)}] {path}", flush=True)
        try:
            rows.append(audit_one(path))
        except Exception as exc:
            errors.append({"path": str(path), "error": repr(exc)})
            print(f"  ERROR: {exc}", flush=True)
            if not args.keep_going:
                break

    result = {
        "schema": "checkpoint_deployability_audit.v1",
        "n_paths": len(paths),
        "n_audited": len(rows),
        "n_errors": len(errors),
        "rows": rows,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, args.csv_output)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
