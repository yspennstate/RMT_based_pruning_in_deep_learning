#!/usr/bin/env python
"""Build a table-ready deployability ledger for the paper checkpoints.

This is a read-only audit. It joins the checkpoint tensor audit with archived
throughput logs and paper-row metadata. It does not rewrite checkpoints.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
AUDIT_CSV = DATA / "main_structured_deployability_audit_20260512.csv"
CONV_TRT_AUDIT_CSV = DATA / "conv_tensorrt_deployability_audit_20260512.csv"
A40_BACKEND_SUMMARY_CSV = (
    DATA
    / "runpod_a40_deploy_audit_20260512"
    / "deployable_backend_summary_a40_20260512.csv"
)
OUT_JSON = DATA / "paper_checkpoint_deployable_speedup_ledger_20260512.json"
OUT_CSV = DATA / "paper_checkpoint_deployable_speedup_ledger_20260512.csv"


def p(path: str | None) -> str | None:
    return str(Path(path)) if path else None


def load_audit(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {str(Path(row["path"])).lower(): row for row in rows}


def load_rows_by_id(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["row_id"]: row for row in rows if row.get("row_id")}


def load_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    candidate = Path(path)
    if not candidate.exists():
        return {}
    return json.loads(candidate.read_text(encoding="utf-8"))


def throughput_from_results(path: str | None) -> dict[str, Any]:
    data = load_json(path)
    throughput = data.get("throughput", {})
    dense = throughput.get("pytorch_dense_tensor") or {}
    sparse = throughput.get("pytorch_semi_structured") or {}
    conversion = throughput.get("semi_structured_conversion") or {}
    dense_ips = dense.get("images_per_second")
    sparse_ips = sparse.get("images_per_second")
    ratio = None
    if isinstance(dense_ips, (int, float)) and isinstance(sparse_ips, (int, float)) and dense_ips > 0:
        ratio = sparse_ips / dense_ips
    acc = data.get("accuracy_top1", {})
    return {
        "result_json_exists": bool(data),
        "dense_ips": dense_ips,
        "sparse_ips": sparse_ips,
        "measured_native_incremental_speedup_x": ratio,
        "converted_layers": conversion.get("converted"),
        "skipped_layers": len(conversion.get("skipped") or []),
        "result_top1_post_ft_pct": (
            acc.get("tome_post_ft")
            or acc.get("post_ft")
            or acc.get("teacher_sparse_cast2e_no_tome")
        ),
    }


def round_or_none(value: Any, ndigits: int = 3) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value), ndigits)
    return None


def theoretical_speedup(mac_red_pct: float | None) -> float | None:
    if mac_red_pct is None:
        return None
    kept = 1.0 - mac_red_pct / 100.0
    if kept <= 0:
        return None
    return 1.0 / kept


ROWS: list[dict[str, Any]] = [
    {
        "row_id": "vitb_magnitude_24_tome",
        "architecture": "ViT-B/16",
        "source_s": "0.35",
        "method": "Magnitude 2:4 + ToMe",
        "dense_macs": "17.56G",
        "mac_reduction_pct": 59.81,
        "paper_speedup_current": "1.84x meas.",
        "top1_pct": 82.92,
        "delta_pp": -2.19,
        "checkpoint_path": p(r"C:\Users\owner\Downloads\cast_2e_resnet_review\github_archive\pod1_complete\best_pod1_ckpts\ckpts\vitb\D24_dense_perm.pt"),
        "checkpoint_status": "stage-1/pre-FT 2:4 artifact located; final post-FT magnitude checkpoint not located locally",
        "result_json": None,
        "deployability_class": "native_2_4_pattern_audited_but_final_checkpoint_missing",
        "recommended_table_speedup": "do not assign row-specific measured speed unless final checkpoint is found; same backend path as ViT-B CAST",
        "notes": "The located artifact has exact native Linear 2:4 on convertible layers, but it is not the final 82.92% post-FT checkpoint.",
    },
    {
        "row_id": "vitb_cast_24_tome",
        "architecture": "ViT-B/16",
        "source_s": "0.35",
        "method": "CAST 2:4 + ToMe",
        "dense_macs": "17.56G",
        "mac_reduction_pct": 59.81,
        "paper_speedup_current": "1.84x meas.",
        "top1_pct": 83.41,
        "delta_pp": -1.70,
        "checkpoint_path": p(r"C:\Users\owner\v11_pod_debug\cast_canonical_local\vit_base_canonical_cast_20260504T170942Z\ft_phase\checkpoints\vit_base_patch16_224.augreg2_in21k_ft_in1k_cast2e_2to4_tome_r8_post_ft.pt"),
        "checkpoint_status": "final post-FT checkpoint located",
        "result_json": p(r"C:\Users\owner\v11_pod_debug\cast_canonical_local\vit_base_canonical_cast_20260504T170942Z\ft_phase\results.json"),
        "deployability_class": "native_linear_2_4_measured",
        "recommended_table_speedup": "1.419x native incr.; 1.84x composed with ToMe",
        "notes": "A100 dense-to-2:4 no-ToMe benchmark also reports 2.705x for the same model family and native sparse path.",
    },
    {
        "row_id": "vitb_cast_612",
        "architecture": "ViT-B/16",
        "source_s": "0.35",
        "method": "CAST 6:12 SER+alpha=0.5 (no ToMe)",
        "dense_macs": "17.56G",
        "mac_reduction_pct": 50.0,
        "paper_speedup_current": "2.00x theor.",
        "top1_pct": 83.74,
        "delta_pp": -1.37,
        "checkpoint_path": p(r"C:\Users\owner\Downloads\cast_results_2026_05_10\pod1_vitb_ft_D612_ser_a05\student_final.pt"),
        "checkpoint_status": "final post-FT checkpoint located",
        "result_json": None,
        "deployability_class": "structured_mac_accounting_only_not_native_2_4",
        "recommended_table_speedup": "2.00x theoretical MAC only; no native measured sparse-kernel speedup",
        "notes": "6:12 is not accepted by the tested PyTorch/NVIDIA 2:4 sparse Tensor Core path.",
    },
    {
        "row_id": "vitl_cast_24_tome",
        "architecture": "ViT-L/16",
        "source_s": "0.35",
        "method": "CAST 2:4 + ToMe",
        "dense_macs": "61.55G",
        "mac_reduction_pct": 60.0,
        "paper_speedup_current": "1.55x meas.",
        "top1_pct": 84.37,
        "delta_pp": -1.47,
        "checkpoint_path": p(r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\vitl_canonical\vit_large_patch16_224.augreg_in21k_ft_in1k_cast2e_2to4_tome_r8_post_ft.pt"),
        "checkpoint_status": "final post-FT checkpoint located",
        "result_json": p(r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\vitl_canonical\results.json"),
        "deployability_class": "native_linear_2_4_measured",
        "recommended_table_speedup": "1.193x native incr.; 1.55x composed with ToMe",
        "notes": "Native sparse conversion reports 96 converted Linear layers and one skipped classifier head.",
    },
    {
        "row_id": "vitl_cast_816",
        "architecture": "ViT-L/16",
        "source_s": "dense",
        "method": "CAST 8:16 dense+perm (no ToMe)",
        "dense_macs": "61.55G",
        "mac_reduction_pct": 50.0,
        "paper_speedup_current": "2.00x theor.",
        "top1_pct": 85.33,
        "delta_pp": -0.51,
        "checkpoint_path": p(r"C:\Users\owner\Downloads\cast_results_2026_05_10\pod1_vitl_d816_dense\student_ep2.pt"),
        "checkpoint_status": "epoch-2 checkpoint located; final_eval reports epoch-3 top-1 but final post-FT weights are unavailable locally",
        "result_json": None,
        "deployability_class": "structured_mac_accounting_only_final_checkpoint_missing",
        "recommended_table_speedup": "2.00x theoretical MAC only; no native measured sparse-kernel speedup",
        "notes": "8:16 is not accepted by the tested PyTorch/NVIDIA 2:4 path.",
    },
    {
        "row_id": "deitb_cast_24_tome",
        "architecture": "DeiT-B",
        "source_s": "0.35",
        "method": "CAST 2:4 + ToMe",
        "dense_macs": "17.56G",
        "mac_reduction_pct": 59.81,
        "paper_speedup_current": "2.49x theor.",
        "top1_pct": 80.48,
        "delta_pp": -1.32,
        "checkpoint_path": p(r"C:\Users\owner\v11_pod_debug\cast_canonical_local\deit_base_cast_20260505T065522Z\ft_phase\checkpoints\deit_base_patch16_224.fb_in1k_cast2e_2to4_tome_r8_post_ft.pt"),
        "checkpoint_status": "final post-FT checkpoint located",
        "result_json": p(r"C:\Users\owner\v11_pod_debug\cast_canonical_local\deit_base_cast_20260505T065522Z\ft_phase\results.json"),
        "deployability_class": "native_linear_2_4_measured",
        "recommended_table_speedup": "1.425x native incr.; avoid replacing with 2.49x as measured",
        "notes": "Measured speedup holds ToMe fixed and changes dense Linear weights to native 2:4 sparse tensors.",
    },
    {
        "row_id": "deits_cast_24_tome",
        "architecture": "DeiT-S",
        "source_s": "0.35",
        "method": "CAST 2:4 + ToMe",
        "dense_macs": "4.61G",
        "mac_reduction_pct": 59.81,
        "paper_speedup_current": "2.49x theor.",
        "top1_pct": 76.96,
        "delta_pp": -2.89,
        "checkpoint_path": p(r"C:\Users\owner\v11_pod_debug\cast_canonical_local\deit_small_cast_20260505T032704Z\ft_phase\checkpoints\deit_small_patch16_224.fb_in1k_cast2e_2to4_tome_r8_post_ft.pt"),
        "checkpoint_status": "final post-FT checkpoint located",
        "result_json": p(r"C:\Users\owner\v11_pod_debug\cast_canonical_local\deit_small_cast_20260505T032704Z\ft_phase\results.json"),
        "deployability_class": "native_linear_2_4_measured",
        "recommended_table_speedup": "1.280x native incr.; avoid replacing with 2.49x as measured",
        "notes": "Measured speedup holds ToMe fixed and changes dense Linear weights to native 2:4 sparse tensors.",
    },
    {
        "row_id": "deitt_cast_24_tome",
        "architecture": "DeiT-T",
        "source_s": "0.35",
        "method": "CAST 2:4 + ToMe",
        "dense_macs": "1.26G",
        "mac_reduction_pct": 59.81,
        "paper_speedup_current": "2.49x theor.",
        "top1_pct": 65.93,
        "delta_pp": -6.28,
        "checkpoint_path": p(r"C:\Users\owner\v11_pod_debug\cast_canonical_local\deit_tiny_cast_20260505T003151Z\ft_phase\checkpoints\deit_tiny_patch16_224_cast2e_2to4_tome_r8_post_ft.pt"),
        "checkpoint_status": "final post-FT checkpoint located",
        "result_json": p(r"C:\Users\owner\v11_pod_debug\cast_canonical_local\deit_tiny_cast_20260505T003151Z\ft_phase\results.json"),
        "deployability_class": "native_linear_2_4_measured_but_slower",
        "recommended_table_speedup": "0.566x native incr. (slowdown); do not call this a measured speedup",
        "notes": "Sparse backend overhead dominates this small model in the archived benchmark.",
    },
    {
        "row_id": "resnet50_cast_conv",
        "architecture": "ResNet50",
        "source_s": "0.35",
        "method": "CAST-conv",
        "dense_macs": "4.09G",
        "mac_reduction_pct": 48.5,
        "paper_speedup_current": "1.94x theor.",
        "top1_pct": 73.14,
        "delta_pp": -2.99,
        "checkpoint_path": None,
        "checkpoint_status": "paper validation number located in logs; final checkpoint not located locally",
        "result_json": None,
        "deployability_class": "conv_structured_mac_accounting_only_checkpoint_missing",
        "recommended_table_speedup": "1.94x theoretical MAC only; no native sparse Conv2d speedup",
        "notes": "No standard PyTorch/NVIDIA 2:4 sparse Conv2d path was benchmarked for this row.",
    },
    {
        "row_id": "resnet50_cast_conv_perm",
        "architecture": "ResNet50",
        "source_s": "0.35",
        "method": "CAST-conv+perm",
        "dense_macs": "4.09G",
        "mac_reduction_pct": 48.5,
        "paper_speedup_current": "1.94x theor.",
        "top1_pct": 75.67,
        "delta_pp": -0.46,
        "checkpoint_path": p(r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\4resnet_2_4_cert_1_resnet50.tv_in1k\epoch3.pt"),
        "checkpoint_status": "final post-FT checkpoint located",
        "result_json": None,
        "deployability_class": "conv_structured_mac_accounting_only_not_native_sparse_kernel",
        "recommended_table_speedup": "1.94x theoretical MAC only; no native sparse Conv2d speedup",
        "notes": "Checkpoint has exact flattened Conv2d 2:4, but the tested native sparse path converts Linear layers, not Conv2d kernels.",
    },
    {
        "row_id": "resnet50_cast_816",
        "architecture": "ResNet50",
        "source_s": "dense",
        "method": "CAST 8:16 dense+perm",
        "dense_macs": "4.09G",
        "mac_reduction_pct": 50.0,
        "paper_speedup_current": "2.00x theor.",
        "top1_pct": 75.87,
        "delta_pp": -0.26,
        "checkpoint_path": p(r"C:\Users\owner\Downloads\cast_results_2026_05_10\pod3_4resnet_8_16_chain\resnet50.tv_in1k\student_final.pt"),
        "checkpoint_status": "final post-FT checkpoint located",
        "result_json": None,
        "deployability_class": "conv_structured_mac_accounting_only_not_native_2_4",
        "recommended_table_speedup": "2.00x theoretical MAC only; no native measured sparse-kernel speedup",
        "notes": "8:16 Conv2d structure requires a separate backend to become a wall-clock speed claim.",
    },
    {
        "row_id": "resnet50d_cast_conv",
        "architecture": "ResNet50d",
        "source_s": "0.35",
        "method": "CAST-conv",
        "dense_macs": "4.33G",
        "mac_reduction_pct": 49.85,
        "paper_speedup_current": "1.99x theor.",
        "top1_pct": 78.08,
        "delta_pp": -2.47,
        "checkpoint_path": None,
        "checkpoint_status": "paper validation number located in final_eval; checkpoint not present in local artifact directory",
        "result_json": None,
        "deployability_class": "conv_structured_mac_accounting_only_checkpoint_missing",
        "recommended_table_speedup": "1.99x theoretical MAC only; no native sparse Conv2d speedup",
        "notes": "The local M2_resnet50d artifact includes eval/stat files but no checkpoint file.",
    },
    {
        "row_id": "resnet50d_cast_conv_perm",
        "architecture": "ResNet50d",
        "source_s": "0.35",
        "method": "CAST-conv+perm",
        "dense_macs": "4.33G",
        "mac_reduction_pct": 49.85,
        "paper_speedup_current": "1.99x theor.",
        "top1_pct": 78.00,
        "delta_pp": -2.55,
        "checkpoint_path": p(r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\4resnet_2_4_cert_2_resnet50d.ra2_in1k\epoch3.pt"),
        "checkpoint_status": "final post-FT checkpoint located",
        "result_json": None,
        "deployability_class": "conv_structured_mac_accounting_only_not_native_sparse_kernel",
        "recommended_table_speedup": "1.99x theoretical MAC only; no native sparse Conv2d speedup",
        "notes": "Checkpoint has exact flattened Conv2d 2:4, but no native sparse Conv2d backend was benchmarked.",
    },
    {
        "row_id": "resnet50d_cast_816",
        "architecture": "ResNet50d",
        "source_s": "dense",
        "method": "CAST 8:16 dense+perm",
        "dense_macs": "4.33G",
        "mac_reduction_pct": 50.0,
        "paper_speedup_current": "2.00x theor.",
        "top1_pct": 78.57,
        "delta_pp": -1.98,
        "checkpoint_path": p(r"C:\Users\owner\Downloads\cast_results_2026_05_10\pod3_4resnet_8_16_chain\resnet50d.ra2_in1k\student_final.pt"),
        "checkpoint_status": "final post-FT checkpoint located",
        "result_json": None,
        "deployability_class": "conv_structured_mac_accounting_only_not_native_2_4",
        "recommended_table_speedup": "2.00x theoretical MAC only; no native measured sparse-kernel speedup",
        "notes": "8:16 Conv2d structure requires a separate backend to become a wall-clock speed claim.",
    },
    {
        "row_id": "resnet101d_cast_conv",
        "architecture": "ResNet101d",
        "source_s": "0.35",
        "method": "CAST-conv",
        "dense_macs": "8.0G",
        "mac_reduction_pct": 50.0,
        "paper_speedup_current": "~2.00x theor.",
        "top1_pct": 80.13,
        "delta_pp": -2.13,
        "checkpoint_path": None,
        "checkpoint_status": "paper validation number located in logs; final checkpoint not located locally",
        "result_json": None,
        "deployability_class": "conv_structured_mac_accounting_only_checkpoint_missing",
        "recommended_table_speedup": "2.00x theoretical MAC only; no native sparse Conv2d speedup",
        "notes": "No standard PyTorch/NVIDIA 2:4 sparse Conv2d path was benchmarked for this row.",
    },
    {
        "row_id": "resnet101d_cast_conv_perm",
        "architecture": "ResNet101d",
        "source_s": "0.35",
        "method": "CAST-conv+perm",
        "dense_macs": "8.0G",
        "mac_reduction_pct": 50.0,
        "paper_speedup_current": "~2.00x theor.",
        "top1_pct": 80.59,
        "delta_pp": -1.67,
        "checkpoint_path": p(r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\4resnet_2_4_cert_3_resnet101d.ra2_in1k\epoch3.pt"),
        "checkpoint_status": "final post-FT checkpoint located",
        "result_json": None,
        "deployability_class": "conv_structured_mac_accounting_only_not_native_sparse_kernel",
        "recommended_table_speedup": "2.00x theoretical MAC only; no native sparse Conv2d speedup",
        "notes": "Checkpoint has exact flattened Conv2d 2:4, but no native sparse Conv2d backend was benchmarked.",
    },
    {
        "row_id": "resnet101d_cast_816",
        "architecture": "ResNet101d",
        "source_s": "dense",
        "method": "CAST 8:16 dense+perm",
        "dense_macs": "8.0G",
        "mac_reduction_pct": 50.0,
        "paper_speedup_current": "2.00x theor.",
        "top1_pct": 80.92,
        "delta_pp": -1.34,
        "checkpoint_path": p(r"C:\Users\owner\Downloads\cast_results_2026_05_10\pod3_4resnet_8_16_chain\resnet101d.ra2_in1k\student_final.pt"),
        "checkpoint_status": "final post-FT checkpoint located",
        "result_json": None,
        "deployability_class": "conv_structured_mac_accounting_only_not_native_2_4",
        "recommended_table_speedup": "2.00x theoretical MAC only; no native measured sparse-kernel speedup",
        "notes": "8:16 Conv2d structure requires a separate backend to become a wall-clock speed claim.",
    },
    {
        "row_id": "resnet152d_cast_conv_perm",
        "architecture": "ResNet152d",
        "source_s": "0.35",
        "method": "CAST-conv+perm",
        "dense_macs": "11.8G",
        "mac_reduction_pct": 50.0,
        "paper_speedup_current": "~2.00x theor.",
        "top1_pct": 81.33,
        "delta_pp": -1.53,
        "checkpoint_path": p(r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\4resnet_2_4_cert_4_resnet152d.ra2_in1k\epoch3.pt"),
        "checkpoint_status": "final post-FT checkpoint located",
        "result_json": None,
        "deployability_class": "conv_structured_mac_accounting_only_not_native_sparse_kernel",
        "recommended_table_speedup": "2.00x theoretical MAC only; no native sparse Conv2d speedup",
        "notes": "Checkpoint has exact flattened Conv2d 2:4, but no native sparse Conv2d backend was benchmarked.",
    },
    {
        "row_id": "convnextv2_cast_24",
        "architecture": "ConvNeXtV2-Base",
        "source_s": "0.35",
        "method": "CAST 2:4 cert + free-restore",
        "dense_macs": "15.4G",
        "mac_reduction_pct": 50.0,
        "paper_speedup_current": "1.086x meas.",
        "top1_pct": 85.47,
        "delta_pp": -1.25,
        "checkpoint_path": p(r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\convnextv2_canonical\convnextv2_base.fcmae_ft_in22k_in1k_cast2e_2to4_tome_r0_post_ft.pt"),
        "checkpoint_status": "final post-FT checkpoint located",
        "result_json": p(r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\convnextv2_canonical\results.json"),
        "deployability_class": "native_linear_2_4_measured",
        "recommended_table_speedup": "1.086x measured native",
        "notes": "Native sparse conversion reports 72 converted Linear pointwise layers and one skipped classifier head.",
    },
    {
        "row_id": "convnextv2_cast_1216",
        "architecture": "ConvNeXtV2-Base",
        "source_s": "dense",
        "method": "CAST 12:16 dense+perm (25% sparse)",
        "dense_macs": "15.4G",
        "mac_reduction_pct": 25.0,
        "paper_speedup_current": "1.33x theor.",
        "top1_pct": 86.35,
        "delta_pp": -0.37,
        "checkpoint_path": p(r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\convnextv2_d1216\convnextv2_base.fcmae_ft_in22k_in1k_cast2e_d1216_tome_r0_post_ft.pt"),
        "checkpoint_status": "final post-FT checkpoint located, but tensor audit shows no zeros in saved weights",
        "result_json": p(r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\convnextv2_d1216\results.json"),
        "deployability_class": "accuracy_row_checkpoint_saved_dense_no_sparse_weights",
        "recommended_table_speedup": "1.33x theoretical MAC only; no native measured sparse-kernel speedup",
        "notes": "The saved final checkpoint audited as dense, so this row should not be presented as a deployable sparse checkpoint without the mask/runtime artifact.",
    },
    {
        "row_id": "convnextv2_cast_816",
        "architecture": "ConvNeXtV2-Base",
        "source_s": "dense",
        "method": "CAST 8:16 dense+perm (50% sparse)",
        "dense_macs": "15.4G",
        "mac_reduction_pct": 50.0,
        "paper_speedup_current": "2.00x theor.",
        "top1_pct": 85.85,
        "delta_pp": -0.87,
        "checkpoint_path": p(r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\convnextv2_d816\convnextv2_base.fcmae_ft_in22k_in1k_cast2e_d816_tome_r0_post_ft.pt"),
        "checkpoint_status": "final post-FT checkpoint located, but tensor audit shows no zeros in saved weights",
        "result_json": p(r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\convnextv2_d816\results.json"),
        "deployability_class": "accuracy_row_checkpoint_saved_dense_no_sparse_weights",
        "recommended_table_speedup": "2.00x theoretical MAC only; no native measured sparse-kernel speedup",
        "notes": "The saved final checkpoint audited as dense, so this row should not be presented as a deployable sparse checkpoint without the mask/runtime artifact.",
    },
]


def main() -> None:
    audit = load_audit(AUDIT_CSV)
    conv_trt_audit = load_audit(CONV_TRT_AUDIT_CSV)
    a40_backend = load_rows_by_id(A40_BACKEND_SUMMARY_CSV)
    ledger: list[dict[str, Any]] = []
    for row in ROWS:
        out = dict(row)
        out["theoretical_mac_speedup_x"] = round_or_none(
            theoretical_speedup(row.get("mac_reduction_pct")), 3
        )

        checkpoint_path = row.get("checkpoint_path")
        audit_row = audit.get(str(Path(checkpoint_path)).lower()) if checkpoint_path else None
        out["checkpoint_exists"] = bool(checkpoint_path and Path(checkpoint_path).exists())
        out["tensor_audit_available"] = bool(audit_row)
        if audit_row:
            out["weight_sparsity"] = round_or_none(
                float(audit_row["sparsity_over_weight_tensors"]), 4
            )
            out["linear_native_2_4_exact"] = audit_row["linear_native_2_4_exact"]
            out["linear_native_2_4_convertible_exact"] = audit_row[
                "linear_native_2_4_convertible_exact"
            ]
            out["conv_flat_2_4_exact"] = audit_row["conv_flat_2_4_exact"]
            out["native_deployability_interpretation"] = audit_row[
                "native_deployability_interpretation"
            ]
            out["linear_convertible_layers"] = audit_row[
                "linear_native_2_4_convertible_layers"
            ]
            out["conv_eligible_layers"] = audit_row["conv_flat_2_4_eligible_layers"]
        else:
            out["weight_sparsity"] = None
            out["linear_native_2_4_exact"] = None
            out["linear_native_2_4_convertible_exact"] = None
            out["conv_flat_2_4_exact"] = None
            out["native_deployability_interpretation"] = None
            out["linear_convertible_layers"] = None
            out["conv_eligible_layers"] = None

        conv_trt_row = (
            conv_trt_audit.get(str(Path(checkpoint_path)).lower())
            if checkpoint_path
            else None
        )
        if conv_trt_row:
            out["conv_tensorrt_2_4_exact"] = conv_trt_row.get(
                "conv_tensorrt_2_4_exact"
            )
            out["conv_tensorrt_2_4_eligible_layers"] = conv_trt_row.get(
                "conv_tensorrt_2_4_eligible_layers"
            )
            out["conv_tensorrt_2_4_bad_groups"] = conv_trt_row.get(
                "conv_tensorrt_2_4_bad_groups"
            )
        else:
            out["conv_tensorrt_2_4_exact"] = None
            out["conv_tensorrt_2_4_eligible_layers"] = None
            out["conv_tensorrt_2_4_bad_groups"] = None

        throughput = throughput_from_results(row.get("result_json"))
        out.update(throughput)
        out["measured_native_incremental_speedup_x"] = round_or_none(
            out.get("measured_native_incremental_speedup_x"), 3
        )
        out["dense_ips"] = round_or_none(out.get("dense_ips"), 1)
        out["sparse_ips"] = round_or_none(out.get("sparse_ips"), 1)

        a40_row = a40_backend.get(row["row_id"])
        out["a40_backend_audit_available"] = bool(a40_row and not a40_row.get("error"))
        if a40_row and not a40_row.get("error"):
            out["a40_backend_kind"] = a40_row.get("kind")
            out["a40_dense_endpoint"] = a40_row.get("dense_endpoint")
            out["a40_sparse_endpoint"] = a40_row.get("sparse_endpoint")
            out["a40_dense_ips"] = round_or_none(float(a40_row["dense_ips"]), 1)
            out["a40_sparse_ips"] = round_or_none(float(a40_row["sparse_ips"]), 1)
            out["a40_deploy_speedup_x"] = round_or_none(
                float(a40_row["deploy_speedup_x"]), 3
            )
            native_conv_ips = a40_row.get("native_conv_ips")
            out["a40_native_conv_ips"] = (
                round_or_none(float(native_conv_ips), 1)
                if native_conv_ips
                else None
            )
            end_to_end = a40_row.get("end_to_end_vs_native_conv_x")
            out["a40_end_to_end_vs_native_conv_x"] = (
                round_or_none(float(end_to_end), 3)
                if end_to_end
                else None
            )
            out["a40_converted_layers"] = a40_row.get("converted_layers")
            out["a40_equiv_max_abs_diff"] = (
                round_or_none(float(a40_row["equiv_max_abs_diff"]), 6)
                if a40_row.get("equiv_max_abs_diff")
                else None
            )
            out["a40_equiv_mean_abs_diff"] = (
                round_or_none(float(a40_row["equiv_mean_abs_diff"]), 6)
                if a40_row.get("equiv_mean_abs_diff")
                else None
            )
        else:
            out["a40_backend_kind"] = None
            out["a40_dense_endpoint"] = None
            out["a40_sparse_endpoint"] = None
            out["a40_dense_ips"] = None
            out["a40_sparse_ips"] = None
            out["a40_deploy_speedup_x"] = None
            out["a40_native_conv_ips"] = None
            out["a40_end_to_end_vs_native_conv_x"] = None
            out["a40_converted_layers"] = None
            out["a40_equiv_max_abs_diff"] = None
            out["a40_equiv_mean_abs_diff"] = None

        ledger.append(out)

    OUT_JSON.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    fieldnames = [
        "row_id",
        "architecture",
        "source_s",
        "method",
        "top1_pct",
        "delta_pp",
        "dense_macs",
        "mac_reduction_pct",
        "theoretical_mac_speedup_x",
        "paper_speedup_current",
        "measured_native_incremental_speedup_x",
        "dense_ips",
        "sparse_ips",
        "converted_layers",
        "skipped_layers",
        "a40_backend_audit_available",
        "a40_backend_kind",
        "a40_dense_endpoint",
        "a40_sparse_endpoint",
        "a40_dense_ips",
        "a40_sparse_ips",
        "a40_deploy_speedup_x",
        "a40_native_conv_ips",
        "a40_end_to_end_vs_native_conv_x",
        "a40_converted_layers",
        "a40_equiv_max_abs_diff",
        "a40_equiv_mean_abs_diff",
        "deployability_class",
        "recommended_table_speedup",
        "checkpoint_exists",
        "checkpoint_status",
        "tensor_audit_available",
        "weight_sparsity",
        "linear_native_2_4_convertible_exact",
        "conv_flat_2_4_exact",
        "conv_tensorrt_2_4_exact",
        "conv_tensorrt_2_4_eligible_layers",
        "conv_tensorrt_2_4_bad_groups",
        "native_deployability_interpretation",
        "notes",
        "checkpoint_path",
        "result_json",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ledger:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
