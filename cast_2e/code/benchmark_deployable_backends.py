#!/usr/bin/env python
"""Benchmark exact deployable backends for paper checkpoints.

This is a read-only audit. It loads the paper checkpoints, builds execution
endpoints in memory, and writes benchmark JSON. It never rewrites a checkpoint.

Two deployable paths are tested where the checkpoint structure permits it:

* native PyTorch/NVIDIA semi-structured 2:4 for nn.Linear weights;
* explicit Conv2d lowering to im2col + nn.Linear, then native 2:4 for the
  lowered Linear weight. This is an exact backend transformation for flattened
  Conv2d 2:4 checkpoints, not a weight projection.

Wider k:n checkpoints are recorded as audit-only because converting them to
2:4 would change weights and therefore the already-reported validation result.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "cast_2e" / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


@dataclass(frozen=True)
class PaperRow:
    row_id: str
    architecture: str
    timm_name: str
    checkpoint: str | None
    kind: str
    tome_r: int = 0
    top1_pct: float | None = None
    note: str = ""


ROWS: dict[str, PaperRow] = {
    "vitb_magnitude_24_tome": PaperRow(
        "vitb_magnitude_24_tome",
        "ViT-B/16",
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        r"C:\Users\owner\Downloads\cast_2e_resnet_review\github_archive\pod1_complete\best_pod1_ckpts\ckpts\vitb\D24_dense_perm.pt",
        "linear_2_4",
        tome_r=8,
        top1_pct=82.92,
        note="Located checkpoint is the stage-1 artifact, not the final post-FT paper checkpoint.",
    ),
    "vitb_cast_24_tome": PaperRow(
        "vitb_cast_24_tome",
        "ViT-B/16",
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        r"C:\Users\owner\v11_pod_debug\cast_canonical_local\vit_base_canonical_cast_20260504T170942Z\ft_phase\checkpoints\vit_base_patch16_224.augreg2_in21k_ft_in1k_cast2e_2to4_tome_r8_post_ft.pt",
        "linear_2_4",
        tome_r=8,
        top1_pct=83.41,
    ),
    "vitb_cast_612": PaperRow(
        "vitb_cast_612",
        "ViT-B/16",
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        r"C:\Users\owner\Downloads\cast_results_2026_05_10\pod1_vitb_ft_D612_ser_a05\student_final.pt",
        "audit_only",
        top1_pct=83.74,
        note="6:12 is not accepted by the tested native 2:4 path without changing weights.",
    ),
    "vitl_cast_24_tome": PaperRow(
        "vitl_cast_24_tome",
        "ViT-L/16",
        "vit_large_patch16_224.augreg_in21k_ft_in1k",
        r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\vitl_canonical\vit_large_patch16_224.augreg_in21k_ft_in1k_cast2e_2to4_tome_r8_post_ft.pt",
        "linear_2_4",
        tome_r=8,
        top1_pct=84.37,
    ),
    "vitl_cast_816": PaperRow(
        "vitl_cast_816",
        "ViT-L/16",
        "vit_large_patch16_224.augreg_in21k_ft_in1k",
        r"C:\Users\owner\Downloads\cast_results_2026_05_10\pod1_vitl_d816_dense\student_ep2.pt",
        "audit_only",
        top1_pct=85.33,
        note="Only an epoch-2 local checkpoint was found; 8:16 is not native 2:4.",
    ),
    "deitb_cast_24_tome": PaperRow(
        "deitb_cast_24_tome",
        "DeiT-B",
        "deit_base_patch16_224.fb_in1k",
        r"C:\Users\owner\v11_pod_debug\cast_canonical_local\deit_base_cast_20260505T065522Z\ft_phase\checkpoints\deit_base_patch16_224.fb_in1k_cast2e_2to4_tome_r8_post_ft.pt",
        "linear_2_4",
        tome_r=8,
        top1_pct=80.48,
    ),
    "deits_cast_24_tome": PaperRow(
        "deits_cast_24_tome",
        "DeiT-S",
        "deit_small_patch16_224.fb_in1k",
        r"C:\Users\owner\v11_pod_debug\cast_canonical_local\deit_small_cast_20260505T032704Z\ft_phase\checkpoints\deit_small_patch16_224.fb_in1k_cast2e_2to4_tome_r8_post_ft.pt",
        "linear_2_4",
        tome_r=8,
        top1_pct=76.96,
    ),
    "deitt_cast_24_tome": PaperRow(
        "deitt_cast_24_tome",
        "DeiT-T",
        "deit_tiny_patch16_224.fb_in1k",
        r"C:\Users\owner\v11_pod_debug\cast_canonical_local\deit_tiny_cast_20260505T003151Z\ft_phase\checkpoints\deit_tiny_patch16_224_cast2e_2to4_tome_r8_post_ft.pt",
        "linear_2_4",
        tome_r=8,
        top1_pct=65.93,
    ),
    "resnet50_cast_conv_perm": PaperRow(
        "resnet50_cast_conv_perm",
        "ResNet50",
        "resnet50.tv_in1k",
        r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\4resnet_2_4_cert_1_resnet50.tv_in1k\epoch3.pt",
        "conv_flat_2_4",
        top1_pct=75.67,
    ),
    "resnet50d_cast_conv_perm": PaperRow(
        "resnet50d_cast_conv_perm",
        "ResNet50d",
        "resnet50d.ra2_in1k",
        r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\4resnet_2_4_cert_2_resnet50d.ra2_in1k\epoch3.pt",
        "conv_flat_2_4",
        top1_pct=78.00,
    ),
    "resnet101d_cast_conv_perm": PaperRow(
        "resnet101d_cast_conv_perm",
        "ResNet101d",
        "resnet101d.ra2_in1k",
        r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\4resnet_2_4_cert_3_resnet101d.ra2_in1k\epoch3.pt",
        "conv_flat_2_4",
        top1_pct=80.59,
    ),
    "resnet152d_cast_conv_perm": PaperRow(
        "resnet152d_cast_conv_perm",
        "ResNet152d",
        "resnet152d.ra2_in1k",
        r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\4resnet_2_4_cert_4_resnet152d.ra2_in1k\epoch3.pt",
        "conv_flat_2_4",
        top1_pct=81.33,
    ),
    "resnet50_cast_816": PaperRow(
        "resnet50_cast_816",
        "ResNet50",
        "resnet50.tv_in1k",
        r"C:\Users\owner\Downloads\cast_results_2026_05_10\pod3_4resnet_8_16_chain\resnet50.tv_in1k\student_final.pt",
        "audit_only",
        top1_pct=75.87,
        note="8:16 is not accepted by the tested native 2:4 path without changing weights.",
    ),
    "resnet50d_cast_816": PaperRow(
        "resnet50d_cast_816",
        "ResNet50d",
        "resnet50d.ra2_in1k",
        r"C:\Users\owner\Downloads\cast_results_2026_05_10\pod3_4resnet_8_16_chain\resnet50d.ra2_in1k\student_final.pt",
        "audit_only",
        top1_pct=78.57,
        note="8:16 is not accepted by the tested native 2:4 path without changing weights.",
    ),
    "resnet101d_cast_816": PaperRow(
        "resnet101d_cast_816",
        "ResNet101d",
        "resnet101d.ra2_in1k",
        r"C:\Users\owner\Downloads\cast_results_2026_05_10\pod3_4resnet_8_16_chain\resnet101d.ra2_in1k\student_final.pt",
        "audit_only",
        top1_pct=80.92,
        note="8:16 is not accepted by the tested native 2:4 path without changing weights.",
    ),
    "convnextv2_cast_24": PaperRow(
        "convnextv2_cast_24",
        "ConvNeXtV2-Base",
        "convnextv2_base.fcmae_ft_in22k_in1k",
        r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\convnextv2_canonical\convnextv2_base.fcmae_ft_in22k_in1k_cast2e_2to4_tome_r0_post_ft.pt",
        "linear_2_4",
        top1_pct=85.47,
    ),
    "convnextv2_cast_1216": PaperRow(
        "convnextv2_cast_1216",
        "ConvNeXtV2-Base",
        "convnextv2_base.fcmae_ft_in22k_in1k",
        r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\convnextv2_d1216\convnextv2_base.fcmae_ft_in22k_in1k_cast2e_d1216_tome_r0_post_ft.pt",
        "audit_only",
        top1_pct=86.35,
        note="Saved final checkpoint audits as dense/no zeros; no exact sparse backend claim is possible from this file.",
    ),
    "convnextv2_cast_816": PaperRow(
        "convnextv2_cast_816",
        "ConvNeXtV2-Base",
        "convnextv2_base.fcmae_ft_in22k_in1k",
        r"C:\Users\owner\Downloads\cast_2e_resnet_review\checkpoints_for_drive\convnextv2_d816\convnextv2_base.fcmae_ft_in22k_in1k_cast2e_d816_tome_r0_post_ft.pt",
        "audit_only",
        top1_pct=85.85,
        note="Saved final checkpoint audits as dense/no zeros; no exact sparse backend claim is possible from this file.",
    ),
}


def log(msg: str) -> None:
    print(msg, flush=True)


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
                return {str(k): v for k, v in value.items() if torch.is_tensor(v)}
        tensor_items = {str(k): v for k, v in raw.items() if torch.is_tensor(v)}
        if tensor_items:
            return tensor_items
    raise TypeError("unrecognized checkpoint format")


def strip_prefix_if_present(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not state or not all(k.startswith(prefix) for k in state):
        return state
    return {k[len(prefix):]: v for k, v in state.items()}


def normalize_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for prefix in ("module.", "model.", "student."):
        state = strip_prefix_if_present(state, prefix)
    return state


def checkpoint_sparsity_summary(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    total = 0
    nnz = 0
    linear_24_bad = 0
    linear_24_groups = 0
    conv_flat_24_bad = 0
    conv_flat_24_groups = 0
    for name, tensor in state.items():
        if not name.endswith(".weight") or not tensor.is_floating_point():
            continue
        if tensor.ndim == 2:
            flat = tensor.detach()
            layer_type = "linear"
        elif tensor.ndim == 4:
            flat = tensor.detach().reshape(tensor.shape[0], -1)
            layer_type = "conv"
        else:
            continue
        total += int(flat.numel())
        nnz += int(flat.ne(0).sum().item())
        if flat.shape[1] % 4 == 0:
            grouped = flat.reshape(flat.shape[0], flat.shape[1] // 4, 4)
            bad = int(grouped.ne(0).sum(dim=-1).ne(2).sum().item())
            groups = int(grouped.shape[0] * grouped.shape[1])
            if layer_type == "linear":
                linear_24_bad += bad
                linear_24_groups += groups
            else:
                conv_flat_24_bad += bad
                conv_flat_24_groups += groups
    return {
        "weight_tensors_params": total,
        "weight_tensors_nnz": nnz,
        "weight_sparsity": (1.0 - nnz / total) if total else None,
        "linear_2_4_groups": linear_24_groups,
        "linear_2_4_bad_groups": linear_24_bad,
        "linear_2_4_exact": linear_24_groups > 0 and linear_24_bad == 0,
        "conv_flat_2_4_groups": conv_flat_24_groups,
        "conv_flat_2_4_bad_groups": conv_flat_24_bad,
        "conv_flat_2_4_exact": conv_flat_24_groups > 0 and conv_flat_24_bad == 0,
    }


# Minimal ToMe patch adapted from the run_vit_tome_flop_reduction.py runner.
def do_nothing(x: torch.Tensor, mode: str | None = None) -> torch.Tensor:
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> tuple[Callable, Callable]:
    protected = int(class_token) + int(distill_token)
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        return out

    return merge, unmerge


def merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    return x / size, size


def parse_r(num_layers: int, r_value: int | list[int] | tuple[int, float]) -> list[int]:
    if isinstance(r_value, list):
        return (r_value + [0] * num_layers)[:num_layers]
    if isinstance(r_value, tuple):
        r, inflect = r_value
        min_val = int(r * (1.0 - inflect))
        max_val = 2 * r - min_val
        step = (max_val - min_val) / max(1, num_layers - 1)
        return [int(min_val + step * i) for i in range(num_layers)]
    return [int(r_value)] * num_layers


def apply_tome_patch(model: nn.Module, prop_attn: bool = True) -> None:
    try:
        from timm.models.vision_transformer import Attention, Block
    except Exception:
        return

    if not hasattr(model, "blocks") or not isinstance(model.blocks, (list, nn.ModuleList, nn.Sequential)):
        return
    if not any(isinstance(m, Block) for m in model.modules()):
        return

    class ToMeAttention(Attention):
        def forward(
            self,
            x: torch.Tensor,
            size: torch.Tensor | None = None,
            attn_mask: torch.Tensor | None = None,
            is_causal: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if attn_mask is not None or is_causal:
                raise NotImplementedError("ToMe audit expects plain ViT self-attention without masks.")
            bsz, num_tokens, channels = x.shape
            qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if size is not None:
                attn = attn + size.log()[:, None, None, :, 0]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            attn_dim = getattr(self, "attn_dim", self.num_heads * self.head_dim)
            x = x.transpose(1, 2).reshape(bsz, num_tokens, attn_dim)
            if hasattr(self, "norm"):
                x = self.norm(x)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x, k.mean(1)

    class ToMeBlock(Block):
        def forward(
            self,
            x: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            is_causal: bool = False,
        ) -> torch.Tensor:
            attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
            x_attn, metric = self.attn(self.norm1(x), size=attn_size, attn_mask=attn_mask, is_causal=is_causal)
            x = x + self.drop_path1(self.ls1(x_attn))
            r = self._tome_info["r"].pop(0)
            if r > 0:
                merge, _ = bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                )
                x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x

    base_class = model.__class__

    class ToMeVisionTransformer(base_class):
        def forward(self, *args, **kwargs):
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            return super().forward(*args, **kwargs)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "prop_attn": prop_attn,
        "class_token": getattr(model, "cls_token", None) is not None,
        "distill_token": getattr(model, "dist_token", None) is not None,
    }
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention


def attach_conv_permutations(model: nn.Module, state: dict[str, torch.Tensor]) -> int:
    try:
        from project_conv_2_4 import attach_permutations_from_state_dict
    except Exception:
        return 0
    return int(attach_permutations_from_state_dict(model, state))


def load_model(row: PaperRow, device: torch.device, *, attach_perms: bool = False) -> tuple[nn.Module, dict[str, Any]]:
    import timm

    if not row.checkpoint:
        raise FileNotFoundError(f"{row.row_id} has no checkpoint path")
    ckpt_path = resolve_checkpoint_path(row)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))
    raw = load_checkpoint(ckpt_path)
    state = normalize_state_dict(state_dict_from(raw))
    model = timm.create_model(row.timm_name, pretrained=False)
    if row.tome_r:
        apply_tome_patch(model)
        if hasattr(model, "r"):
            model.r = row.tome_r
    perm_count = attach_conv_permutations(model, state) if attach_perms else 0
    incompatible = model.load_state_dict(state, strict=False)
    model.eval().to(device)
    meta = {
        "checkpoint": str(ckpt_path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "attached_conv_permutations": perm_count,
        "sparsity": checkpoint_sparsity_summary(state),
    }
    return model, meta


def resolve_checkpoint_path(row: PaperRow) -> Path:
    """Resolve a row checkpoint locally or under DEPLOY_AUDIT_CKPT_DIR.

    The hard-coded row paths are Windows-local provenance paths. On RunPod we
    copy each checkpoint to ``$DEPLOY_AUDIT_CKPT_DIR/<row_id>.pt`` and use this
    fallback without changing row metadata.
    """
    if row.checkpoint:
        original = Path(row.checkpoint)
        if original.exists():
            return original
    ckpt_dir = os.environ.get("DEPLOY_AUDIT_CKPT_DIR")
    if ckpt_dir:
        candidate = Path(ckpt_dir) / f"{row.row_id}.pt"
        if candidate.exists():
            return candidate
    return Path(row.checkpoint or f"{row.row_id}.pt")


def input_size_for_model(model: nn.Module) -> tuple[int, int, int]:
    import timm

    cfg = timm.data.resolve_model_data_config(model)
    size = cfg.get("input_size", (3, 224, 224))
    return int(size[0]), int(size[1]), int(size[2])


def device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "cuda": torch.version.cuda,
                "gpu_name": props.name,
                "gpu_capability": [props.major, props.minor],
                "gpu_total_memory_gb": round(props.total_memory / (1024**3), 3),
                "cudnn": torch.backends.cudnn.version(),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return info


@torch.no_grad()
def benchmark(
    model: nn.Module,
    *,
    device: torch.device,
    batch: int,
    input_size: tuple[int, int, int],
    warmup: int,
    iters: int,
    label: str,
    input_dtype: torch.dtype,
    autocast_bf16: bool,
) -> dict[str, Any]:
    model.eval()
    x = torch.randn((batch, *input_size), device=device, dtype=input_dtype)
    enabled = bool(autocast_bf16 and device.type == "cuda")
    for _ in range(warmup):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
                _ = model(x)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0
        else:
            t0 = time.perf_counter()
            _ = model(x)
            elapsed = time.perf_counter() - t0
        times.append(float(elapsed))
    median_s = statistics.median(times)
    total_s = sum(times)
    return {
        "label": label,
        "batch_size": batch,
        "input_size": list(input_size),
        "warmup_iters": warmup,
        "timed_iters": iters,
        "input_dtype": str(input_dtype).replace("torch.", ""),
        "autocast_bfloat16": enabled,
        "median_batch_s": median_s,
        "median_images_per_second": batch / median_s if median_s > 0 else None,
        "mean_batch_s": total_s / len(times),
        "mean_images_per_second": (batch * len(times)) / total_s if total_s > 0 else None,
        "all_batch_s": times,
    }


def convert_linears_to_sparse(model: nn.Module, *, only_names: set[str] | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available() or not hasattr(torch.sparse, "to_sparse_semi_structured"):
        return {"converted": 0, "skipped": [], "error": "torch sparse semi-structured unavailable"}
    from torch.sparse import to_sparse_semi_structured

    try:
        from torch.sparse import SparseSemiStructuredTensor

        SparseSemiStructuredTensor._FORCE_CUTLASS = True
    except Exception:
        pass

    model.to(dtype=torch.bfloat16)
    converted: list[str] = []
    skipped: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if only_names is not None and name not in only_names:
            continue
        weight = module.weight.detach()
        if weight.ndim != 2:
            skipped.append({"name": name, "shape": list(weight.shape), "reason": "weight is not 2D"})
            continue
        try:
            module.weight = nn.Parameter(to_sparse_semi_structured(weight.contiguous()), requires_grad=False)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(dtype=torch.bfloat16)
            converted.append(name)
        except Exception as exc:
            skipped.append({"name": name, "shape": list(weight.shape), "reason": str(exc)})
    return {"converted": len(converted), "converted_layers": converted, "skipped": skipped}


def clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class Im2ColConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, *, name: str):
        super().__init__()
        if conv.groups != 1:
            raise ValueError(f"{name}: grouped conv is not supported")
        if conv.padding_mode != "zeros":
            raise ValueError(f"{name}: padding_mode={conv.padding_mode!r} is not supported")
        self.name = name
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        flat = conv.weight.detach().reshape(conv.out_channels, -1).contiguous()
        self.linear = nn.Linear(flat.shape[1], conv.out_channels, bias=conv.bias is not None)
        with torch.no_grad():
            self.linear.weight.copy_(flat)
            if conv.bias is not None and self.linear.bias is not None:
                self.linear.bias.copy_(conv.bias.detach())
        if hasattr(conv, "_cin_perm"):
            self.register_buffer("_cin_perm", conv._cin_perm.detach().clone().long())
        else:
            self._cin_perm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cin_perm is not None:
            x = x.index_select(1, self._cin_perm.to(x.device))
        bsz = x.shape[0]
        h_in, w_in = int(x.shape[-2]), int(x.shape[-1])
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        h_out = math.floor((h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
        w_out = math.floor((w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1)
        cols = F.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        cols = cols.transpose(1, 2).reshape(bsz * h_out * w_out, -1)
        out = self.linear(cols)
        return out.reshape(bsz, h_out * w_out, self.out_channels).transpose(1, 2).reshape(
            bsz, self.out_channels, h_out, w_out
        )


class Conv1x1AsLinear(nn.Module):
    """Exact 1x1 Conv2d execution through Linear.

    This avoids the full im2col materialization used for 3x3 convolutions and
    lets the audit test a hybrid ResNet endpoint: native cuDNN Conv2d for
    non-1x1 layers, native 2:4 Linear for exact 1x1 sparse weights.
    """

    def __init__(self, conv: nn.Conv2d, name: str = "") -> None:
        super().__init__()
        if conv.kernel_size != (1, 1) or conv.groups != 1 or conv.padding != (0, 0) or conv.dilation != (1, 1):
            raise ValueError(f"{name} is not a supported exact 1x1 Conv2d")
        self.name = name
        self.out_channels = conv.out_channels
        self.stride = conv.stride
        flat = conv.weight.detach().reshape(conv.out_channels, conv.in_channels).contiguous()
        self.linear = nn.Linear(conv.in_channels, conv.out_channels, bias=conv.bias is not None)
        with torch.no_grad():
            self.linear.weight.copy_(flat)
            if conv.bias is not None and self.linear.bias is not None:
                self.linear.bias.copy_(conv.bias.detach())
        if hasattr(conv, "_cin_perm"):
            self.register_buffer("_cin_perm", conv._cin_perm.detach().clone().long())
        else:
            self._cin_perm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cin_perm is not None:
            x = x.index_select(1, self._cin_perm.to(x.device))
        sh, sw = self.stride
        if sh != 1 or sw != 1:
            x = x[:, :, ::sh, ::sw]
        bsz, channels, h_out, w_out = x.shape
        values = x.permute(0, 2, 3, 1).reshape(bsz * h_out * w_out, channels)
        out = self.linear(values)
        return out.reshape(bsz, h_out, w_out, self.out_channels).permute(0, 3, 1, 2).contiguous()


def flattened_2_4_exact(weight: torch.Tensor) -> bool:
    flat = weight.detach().reshape(weight.shape[0], -1)
    if flat.shape[1] % 4 != 0:
        return False
    grouped = flat.reshape(flat.shape[0], flat.shape[1] // 4, 4)
    return bool(grouped.ne(0).sum(dim=-1).eq(2).all().item())


def sparse_convertible_shape(weight_2d: torch.Tensor) -> bool:
    rows, cols = int(weight_2d.shape[0]), int(weight_2d.shape[1])
    return rows >= 16 and cols >= 16 and rows % 16 == 0 and cols % 16 == 0


def lower_convs_to_im2col(module: nn.Module, prefix: str = "") -> dict[str, Any]:
    lowered: list[str] = []
    skipped: list[dict[str, Any]] = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Conv2d):
            flat = child.weight.detach().reshape(child.out_channels, -1)
            if (
                child.groups == 1
                and child.in_channels >= 4
                and child.padding_mode == "zeros"
                and flattened_2_4_exact(child.weight)
                and sparse_convertible_shape(flat)
            ):
                setattr(module, child_name, Im2ColConv2d(child, name=full_name))
                lowered.append(full_name)
            else:
                skipped.append(
                    {
                        "name": full_name,
                        "shape": list(child.weight.shape),
                        "reason": "not flattened-2:4 exact or unsupported shape",
                    }
                )
        else:
            sub = lower_convs_to_im2col(child, full_name)
            lowered.extend(sub["lowered"])
            skipped.extend(sub["skipped"])
    return {"lowered": lowered, "skipped": skipped}


def lower_1x1_convs_to_linear(module: nn.Module, prefix: str = "") -> dict[str, Any]:
    lowered: list[str] = []
    skipped: list[dict[str, Any]] = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Conv2d):
            flat = child.weight.detach().reshape(child.out_channels, -1)
            if (
                child.groups == 1
                and child.kernel_size == (1, 1)
                and child.padding == (0, 0)
                and child.dilation == (1, 1)
                and child.padding_mode == "zeros"
                and flattened_2_4_exact(child.weight)
                and sparse_convertible_shape(flat)
            ):
                setattr(module, child_name, Conv1x1AsLinear(child, name=full_name))
                lowered.append(full_name)
            else:
                skipped.append(
                    {
                        "name": full_name,
                        "shape": list(child.weight.shape),
                        "reason": "not exact convertible 1x1 2:4 Conv2d",
                    }
                )
        else:
            sub = lower_1x1_convs_to_linear(child, full_name)
            lowered.extend(sub["lowered"])
            skipped.extend(sub["skipped"])
    return {"lowered": lowered, "skipped": skipped}


@torch.no_grad()
def compare_native_vs_im2col(
    row: PaperRow,
    *,
    device: torch.device,
    batch: int,
    input_size: tuple[int, int, int],
) -> dict[str, Any]:
    native, _ = load_model(row, device, attach_perms=True)
    lowered, _ = load_model(row, device, attach_perms=True)
    lowered_report = lower_convs_to_im2col(lowered)
    lowered.to(device)
    native.eval()
    lowered.eval()
    x = torch.randn((batch, *input_size), device=device)
    y_native = native(x)
    y_lowered = lowered(x)
    diff = (y_native - y_lowered).detach().float()
    result = {
        "check_batch": batch,
        "lowered_layers": len(lowered_report["lowered"]),
        "skipped_layers": lowered_report["skipped"],
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "native_output_norm": float(y_native.detach().float().norm().item()),
    }
    del native, lowered, x, y_native, y_lowered
    clear_cuda()
    return result


@torch.no_grad()
def compare_native_vs_1x1_linear(
    row: PaperRow,
    *,
    device: torch.device,
    batch: int,
    input_size: tuple[int, int, int],
) -> dict[str, Any]:
    native, _ = load_model(row, device, attach_perms=True)
    lowered, _ = load_model(row, device, attach_perms=True)
    lowered_report = lower_1x1_convs_to_linear(lowered)
    lowered.to(device)
    native.eval()
    lowered.eval()
    x = torch.randn((batch, *input_size), device=device)
    y_native = native(x)
    y_lowered = lowered(x)
    diff = (y_native - y_lowered).detach().float()
    result = {
        "check_batch": batch,
        "lowered_layers": len(lowered_report["lowered"]),
        "skipped_layers": lowered_report["skipped"],
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "native_output_norm": float(y_native.detach().float().norm().item()),
    }
    del native, lowered, x, y_native, y_lowered
    clear_cuda()
    return result


def benchmark_linear_row(row: PaperRow, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    log(f"[{row.row_id}] loading dense endpoint")
    dense, meta = load_model(row, device)
    input_size = input_size_for_model(dense)
    dense_result = benchmark(
        dense,
        device=device,
        batch=args.batch,
        input_size=input_size,
        warmup=args.warmup,
        iters=args.iters,
        label="dense_tensor_autocast_bf16",
        input_dtype=torch.float32,
        autocast_bf16=True,
    )
    log(f"[{row.row_id}] dense {dense_result['median_images_per_second']:.2f} img/s")
    del dense
    clear_cuda()

    dense_bf16_result = None
    if args.include_dense_bf16:
        dense_bf16, _ = load_model(row, device)
        dense_bf16.to(dtype=torch.bfloat16)
        dense_bf16_result = benchmark(
            dense_bf16,
            device=device,
            batch=args.batch,
            input_size=input_size,
            warmup=args.warmup,
            iters=args.iters,
            label="dense_tensor_bfloat16",
            input_dtype=torch.bfloat16,
            autocast_bf16=False,
        )
        log(f"[{row.row_id}] dense bf16 {dense_bf16_result['median_images_per_second']:.2f} img/s")
        del dense_bf16
        clear_cuda()

    log(f"[{row.row_id}] loading native sparse Linear 2:4 endpoint")
    sparse, _ = load_model(row, device)
    conversion = convert_linears_to_sparse(sparse)
    sparse_result = None
    if conversion.get("converted", 0) > 0:
        sparse_result = benchmark(
            sparse,
            device=device,
            batch=args.batch,
            input_size=input_size,
            warmup=args.warmup,
            iters=args.iters,
            label="native_linear_2_4_sparse_bfloat16",
            input_dtype=torch.bfloat16,
            autocast_bf16=False,
        )
        log(f"[{row.row_id}] sparse {sparse_result['median_images_per_second']:.2f} img/s")
    else:
        log(f"[{row.row_id}] sparse conversion produced no converted layers")
    del sparse
    clear_cuda()

    speedup = None
    if sparse_result is not None and dense_result["median_images_per_second"]:
        speedup = sparse_result["median_images_per_second"] / dense_result["median_images_per_second"]
    speedup_vs_dense_bf16 = None
    if sparse_result is not None and dense_bf16_result and dense_bf16_result["median_images_per_second"]:
        speedup_vs_dense_bf16 = sparse_result["median_images_per_second"] / dense_bf16_result["median_images_per_second"]
    return {
        "row_id": row.row_id,
        "architecture": row.architecture,
        "timm_name": row.timm_name,
        "kind": row.kind,
        "top1_pct": row.top1_pct,
        "note": row.note,
        "checkpoint_meta": meta,
        "input_size": list(input_size),
        "dense_tensor_autocast_bf16": dense_result,
        "dense_tensor_bfloat16": dense_bf16_result,
        "semi_structured_conversion": conversion,
        "native_linear_2_4_sparse_bfloat16": sparse_result,
        "deployable_speedup_sparse_vs_dense_autocast": speedup,
        "deployable_speedup_sparse_vs_dense_bfloat16": speedup_vs_dense_bf16,
    }


def benchmark_conv_row(row: PaperRow, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    log(f"[{row.row_id}] loading native Conv2d endpoint")
    native, meta = load_model(row, device, attach_perms=True)
    input_size = input_size_for_model(native)
    native_result = benchmark(
        native,
        device=device,
        batch=args.batch,
        input_size=input_size,
        warmup=args.warmup,
        iters=args.iters,
        label="native_dense_conv2d_autocast_bf16",
        input_dtype=torch.float32,
        autocast_bf16=True,
    )
    log(f"[{row.row_id}] native conv {native_result['median_images_per_second']:.2f} img/s")
    del native
    clear_cuda()

    equivalence = None
    if args.check_outputs:
        log(f"[{row.row_id}] checking im2col equivalence")
        equivalence = compare_native_vs_im2col(
            row,
            device=device,
            batch=args.check_batch,
            input_size=input_size,
        )

    one_by_one_equivalence = None
    if args.check_outputs:
        log(f"[{row.row_id}] checking 1x1-linear equivalence")
        one_by_one_equivalence = compare_native_vs_1x1_linear(
            row,
            device=device,
            batch=args.check_batch,
            input_size=input_size,
        )

    log(f"[{row.row_id}] loading dense 1x1-linear hybrid endpoint")
    one_by_one_dense, _ = load_model(row, device, attach_perms=True)
    one_by_one_dense_lowering = lower_1x1_convs_to_linear(one_by_one_dense)
    one_by_one_dense.to(device=device, dtype=torch.bfloat16)
    one_by_one_dense_result = benchmark(
        one_by_one_dense,
        device=device,
        batch=args.batch,
        input_size=input_size,
        warmup=args.warmup,
        iters=args.iters,
        label="conv1x1_as_dense_linear_bfloat16",
        input_dtype=torch.bfloat16,
        autocast_bf16=False,
    )
    log(f"[{row.row_id}] 1x1 dense-linear hybrid {one_by_one_dense_result['median_images_per_second']:.2f} img/s")
    del one_by_one_dense
    clear_cuda()

    log(f"[{row.row_id}] loading sparse 1x1-linear hybrid endpoint")
    one_by_one_sparse, _ = load_model(row, device, attach_perms=True)
    one_by_one_sparse_lowering = lower_1x1_convs_to_linear(one_by_one_sparse)
    one_by_one_sparse.to(device=device, dtype=torch.bfloat16)
    one_by_one_linear_names = {f"{name}.linear" for name in one_by_one_sparse_lowering["lowered"]}
    one_by_one_conversion = convert_linears_to_sparse(one_by_one_sparse, only_names=one_by_one_linear_names)
    one_by_one_sparse_result = None
    if one_by_one_conversion.get("converted", 0) > 0:
        one_by_one_sparse_result = benchmark(
            one_by_one_sparse,
            device=device,
            batch=args.batch,
            input_size=input_size,
            warmup=args.warmup,
            iters=args.iters,
            label="conv1x1_as_native_2_4_sparse_linear_bfloat16",
            input_dtype=torch.bfloat16,
            autocast_bf16=False,
        )
        log(f"[{row.row_id}] 1x1 sparse-linear hybrid {one_by_one_sparse_result['median_images_per_second']:.2f} img/s")
    else:
        log(f"[{row.row_id}] sparse 1x1-linear conversion produced no converted layers")
    del one_by_one_sparse
    clear_cuda()

    lowered_dense = None
    im2col_dense_result = None
    lowered_sparse = None
    conversion = None
    im2col_sparse_result = None

    if not args.skip_full_conv_im2col:
        log(f"[{row.row_id}] loading dense im2col endpoint")
        im2col_dense, _ = load_model(row, device, attach_perms=True)
        lowered_dense = lower_convs_to_im2col(im2col_dense)
        im2col_dense.to(device=device, dtype=torch.bfloat16)
        im2col_dense_result = benchmark(
            im2col_dense,
            device=device,
            batch=args.batch,
            input_size=input_size,
            warmup=args.warmup,
            iters=args.iters,
            label="im2col_dense_linear_bfloat16",
            input_dtype=torch.bfloat16,
            autocast_bf16=False,
        )
        log(f"[{row.row_id}] im2col dense {im2col_dense_result['median_images_per_second']:.2f} img/s")
        del im2col_dense
        clear_cuda()

        log(f"[{row.row_id}] loading sparse im2col endpoint")
        im2col_sparse, _ = load_model(row, device, attach_perms=True)
        lowered_sparse = lower_convs_to_im2col(im2col_sparse)
        im2col_sparse.to(device=device, dtype=torch.bfloat16)
        im2col_linear_names = {f"{name}.linear" for name in lowered_sparse["lowered"]}
        conversion = convert_linears_to_sparse(im2col_sparse, only_names=im2col_linear_names)
        if conversion.get("converted", 0) > 0:
            im2col_sparse_result = benchmark(
                im2col_sparse,
                device=device,
                batch=args.batch,
                input_size=input_size,
                warmup=args.warmup,
                iters=args.iters,
                label="im2col_native_2_4_sparse_bfloat16",
                input_dtype=torch.bfloat16,
                autocast_bf16=False,
            )
            log(f"[{row.row_id}] im2col sparse {im2col_sparse_result['median_images_per_second']:.2f} img/s")
        else:
            log(f"[{row.row_id}] sparse im2col conversion produced no converted layers")
        del im2col_sparse
        clear_cuda()

    sparse_vs_native = None
    sparse_vs_im2col_dense = None
    one_by_one_sparse_vs_native = None
    one_by_one_sparse_vs_dense_1x1 = None
    if im2col_sparse_result is not None and native_result["median_images_per_second"]:
        sparse_vs_native = im2col_sparse_result["median_images_per_second"] / native_result["median_images_per_second"]
    if (
        im2col_sparse_result is not None
        and im2col_dense_result is not None
        and im2col_dense_result["median_images_per_second"]
    ):
        sparse_vs_im2col_dense = im2col_sparse_result["median_images_per_second"] / im2col_dense_result["median_images_per_second"]
    if one_by_one_sparse_result is not None and native_result["median_images_per_second"]:
        one_by_one_sparse_vs_native = (
            one_by_one_sparse_result["median_images_per_second"] / native_result["median_images_per_second"]
        )
    if one_by_one_sparse_result is not None and one_by_one_dense_result["median_images_per_second"]:
        one_by_one_sparse_vs_dense_1x1 = (
            one_by_one_sparse_result["median_images_per_second"]
            / one_by_one_dense_result["median_images_per_second"]
        )
    return {
        "row_id": row.row_id,
        "architecture": row.architecture,
        "timm_name": row.timm_name,
        "kind": row.kind,
        "top1_pct": row.top1_pct,
        "note": row.note,
        "checkpoint_meta": meta,
        "input_size": list(input_size),
        "native_dense_conv2d_autocast_bf16": native_result,
        "conv1x1_linear_equivalence": one_by_one_equivalence,
        "conv1x1_dense_lowering": one_by_one_dense_lowering,
        "conv1x1_as_dense_linear_bfloat16": one_by_one_dense_result,
        "conv1x1_sparse_lowering": one_by_one_sparse_lowering,
        "conv1x1_semi_structured_conversion": one_by_one_conversion,
        "conv1x1_as_native_2_4_sparse_linear_bfloat16": one_by_one_sparse_result,
        "deployable_speedup_1x1_sparse_vs_native_conv": one_by_one_sparse_vs_native,
        "deployable_speedup_1x1_sparse_vs_dense_1x1_linear": one_by_one_sparse_vs_dense_1x1,
        "im2col_equivalence": equivalence,
        "im2col_dense_lowering": lowered_dense,
        "im2col_dense_linear_bfloat16": im2col_dense_result,
        "im2col_sparse_lowering": lowered_sparse,
        "semi_structured_conversion": conversion,
        "im2col_native_2_4_sparse_bfloat16": im2col_sparse_result,
        "deployable_speedup_im2col_sparse_vs_native_conv": sparse_vs_native,
        "deployable_speedup_im2col_sparse_vs_im2col_dense": sparse_vs_im2col_dense,
    }


def audit_only_row(row: PaperRow) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    resolved = resolve_checkpoint_path(row)
    if resolved.exists():
        raw = load_checkpoint(resolved)
        state = normalize_state_dict(state_dict_from(raw))
        meta["checkpoint"] = str(resolved)
        meta["source_checkpoint"] = row.checkpoint
        meta["sparsity"] = checkpoint_sparsity_summary(state)
    else:
        meta["checkpoint"] = str(resolved)
        meta["source_checkpoint"] = row.checkpoint
        meta["error"] = "checkpoint not found"
    return {
        "row_id": row.row_id,
        "architecture": row.architecture,
        "timm_name": row.timm_name,
        "kind": row.kind,
        "top1_pct": row.top1_pct,
        "note": row.note,
        "checkpoint_meta": meta,
        "deployable_backend_result": None,
        "deployability_interpretation": "audit-only row; exact native 2:4 backend was not applied because that would change the checkpoint or the saved file has no sparse weights",
    }


def selected_rows(spec: str) -> list[PaperRow]:
    if spec == "all":
        return list(ROWS.values())
    if spec == "linear":
        return [row for row in ROWS.values() if row.kind == "linear_2_4"]
    if spec == "conv":
        return [row for row in ROWS.values() if row.kind == "conv_flat_2_4"]
    if spec == "audit":
        return [row for row in ROWS.values() if row.kind == "audit_only"]
    rows: list[PaperRow] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in ROWS:
            raise KeyError(f"unknown row_id {token!r}; choices: {', '.join(sorted(ROWS))}")
        rows.append(ROWS[token])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="linear", help="all, linear, conv, audit, or comma-separated row ids")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-dense-bf16", action="store_true")
    parser.add_argument("--skip-conv-im2col", action="store_true")
    parser.add_argument("--skip-full-conv-im2col", action="store_true")
    parser.add_argument("--check-outputs", action="store_true")
    parser.add_argument("--check-batch", type=int, default=2)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = selected_rows(args.rows)
    result: dict[str, Any] = {
        "schema": "deployable_backend_benchmark.v1",
        "created_unix": time.time(),
        "working_directory": os.getcwd(),
        "device_info": device_info(device),
        "args": vars(args),
        "rows": [],
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for row in rows:
        log(f"\n=== {row.row_id} ({row.kind}) ===")
        try:
            if row.kind == "linear_2_4":
                row_result = benchmark_linear_row(row, args, device)
            elif row.kind == "conv_flat_2_4":
                if args.skip_conv_im2col:
                    row_result = audit_only_row(row)
                    row_result["deployability_interpretation"] = "conv im2col benchmark skipped by command line"
                else:
                    row_result = benchmark_conv_row(row, args, device)
            else:
                row_result = audit_only_row(row)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            row_result = {
                "row_id": row.row_id,
                "architecture": row.architecture,
                "timm_name": row.timm_name,
                "kind": row.kind,
                "error": repr(exc),
            }
            log(f"[{row.row_id}] ERROR: {exc!r}")
        result["rows"].append(row_result)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log(f"[{row.row_id}] partial results saved to {out_path}")

    log(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
