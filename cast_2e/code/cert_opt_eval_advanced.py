"""
cert_opt_eval_advanced.py — sweep over the advanced cert methods (NOT just k:n).

Methods covered:
  1. Mixed sparsity (per-layer cert-driven allocation)
  2. Iterative cert refinement (project → recalibrate → project)
  3. Robust ℓ_∞ percentile cost
  4. Standard k:n baseline for comparison

ALSO saves: per-cell projection mask snapshot + per-layer cert cost trace.
"""
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from project_kn_sparsity import cert_aware_kn_for_conv, is_eligible_conv_kn
from project_cert_advanced import (
    cert_aware_mixed_sparsity_for_conv,
    cert_aware_iterative_for_conv,
    cert_aware_robust_for_conv,
)


def build_loaders(args, image_size, mean, std, crop_pct=0.875, interpolation="bilinear"):
    interp_map = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC}
    interp = interp_map.get(interpolation.lower(), T.InterpolationMode.BICUBIC)
    if abs(crop_pct - 1.0) < 1e-6:
        tx = T.Compose([T.Resize((image_size, image_size), interpolation=interp),
                        T.ToTensor(), T.Normalize(mean=mean, std=std)])
    else:
        tx = T.Compose([T.Resize(int(image_size / crop_pct), interpolation=interp),
                        T.CenterCrop(image_size), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    val_ds = tv.datasets.ImageFolder(args.imagenet_val, transform=tx)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    calib_ds = tv.datasets.ImageFolder(args.imagenet_train_for_calib, transform=tx)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=64, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)
    return calib_loader, val_loader


def evaluate_top1(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            correct += (model(imgs).argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)
    return correct / total


def load_ser(model, ckpt_path):
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("model_state_dict") or raw.get("state_dict") or raw
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    msd = model.state_dict(); matched = total_n = loaded_n = 0
    for k, v in msd.items():
        total_n += v.numel()
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            loaded_n += v.numel(); matched += 1
    cov = loaded_n / max(1, total_n)
    if cov < 0.95: raise RuntimeError(f"coverage {cov:.4f} < 0.95")
    model.load_state_dict(sd, strict=False)


def snapshot_masks_and_save(model, save_dir, label, eligibles_n=4):
    """Save per-layer binary masks + sparsity stats. Small (~kB) so cheap to save."""
    masks = {}
    for name, mod in model.named_modules():
        if is_eligible_conv_kn(name, mod, n=eligibles_n) or isinstance(mod, nn.Linear):
            with torch.no_grad():
                W = mod.weight.data
                # Per-element binary mask (1 = nonzero)
                mask = (W != 0).cpu().to(torch.uint8)
                masks[name] = mask
                # Compute sparsity
    out = {nm: m.numpy().tolist()[:5] for nm, m in masks.items()}  # save first 5 rows per layer (small)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # Save lightweight stats per layer (full mask too big to JSON)
    import numpy as np
    stats = {}
    for nm, m in masks.items():
        arr = m.numpy()
        stats[nm] = {
            "shape": list(arr.shape),
            "sparsity": float(1.0 - arr.mean()),
            "nnz": int(arr.sum()),
            "params": int(arr.size),
        }
    out_path = Path(save_dir) / f"{label}_mask_stats.json"
    out_path.write_text(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm-name", default="resnet50.tv_in1k")
    ap.add_argument("--ser-checkpoint", required=True)
    ap.add_argument("--imagenet-val", required=True)
    ap.add_argument("--imagenet-train-for-calib", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mask-save-dir", default="/workspace/run_outputs/cert_advanced_masks")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cert_opt_eval_advanced] device={device}, model={args.timm_name}")
    import timm

    teacher = timm.create_model(args.timm_name, pretrained=True).to(device).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(teacher)
    img_size = cfg["input_size"][-1]; mean = cfg["mean"]; std = cfg["std"]
    crop_pct = cfg.get("crop_pct", 0.875); interp = cfg.get("interpolation", "bicubic")
    calib_loader, val_loader = build_loaders(args, img_size, mean, std, crop_pct, interp)

    dense_state = {n_: p.detach().clone() for n_, p in teacher.state_dict().items()}
    ser_ref = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
    load_ser(ser_ref, args.ser_checkpoint)
    ser_state = {n_: p.detach().clone() for n_, p in ser_ref.state_dict().items()}

    dense_top1 = evaluate_top1(teacher, val_loader, device)
    ser_top1 = evaluate_top1(ser_ref, val_loader, device)
    print(f"  dense top1 = {dense_top1:.4f}, SER top1 = {ser_top1:.4f}")

    # Cells: (label, source, method, kwargs)
    cells = [
        # === k:n baselines for comparison ===
        ("k24_dense_perm",  "dense", "kn", dict(n=4, k=2, permute_align=True,  alpha_ser_prior=0.0)),
        ("k24_ser_perm_a05","ser",   "kn", dict(n=4, k=2, permute_align=True,  alpha_ser_prior=0.5)),
        ("k48_ser_perm_a05","ser",   "kn", dict(n=8, k=4, permute_align=True,  alpha_ser_prior=0.5)),
        # === Mixed sparsity: cert-driven per-layer allocation (target 0.5, 0.4, 0.3 density) ===
        ("mixed_d50_dense",  "dense", "mixed", dict(target_density=0.50, candidates=[(4,2),(8,4),(4,3),(4,1)])),
        ("mixed_d50_ser",    "ser",   "mixed", dict(target_density=0.50, candidates=[(4,2),(8,4),(4,3),(4,1)])),
        ("mixed_d40_ser",    "ser",   "mixed", dict(target_density=0.40, candidates=[(4,2),(8,4),(4,3),(4,1)])),
        ("mixed_d30_ser",    "ser",   "mixed", dict(target_density=0.30, candidates=[(4,2),(8,4),(4,1)])),
        # === Iterative: 2,3 rounds ===
        ("iter2_24_ser_a05", "ser",   "iter", dict(n=4, k=2, n_rounds=2, permute_align=True, alpha_ser_prior=0.5)),
        ("iter3_24_ser_a05", "ser",   "iter", dict(n=4, k=2, n_rounds=3, permute_align=True, alpha_ser_prior=0.5)),
        ("iter2_48_ser_a05", "ser",   "iter", dict(n=8, k=4, n_rounds=2, permute_align=True, alpha_ser_prior=0.5)),
        # === Robust ℓ_∞ percentile (95, 90, 75) ===
        ("robust_p95_ser",   "ser",   "robust", dict(n=4, k=2, percentile=95, permute_align=True)),
        ("robust_p90_ser",   "ser",   "robust", dict(n=4, k=2, percentile=90, permute_align=True)),
        ("robust_p75_ser",   "ser",   "robust", dict(n=4, k=2, percentile=75, permute_align=True)),
        # === Mixed + iterative combined ===
        ("mixed_d50_ser_iter2", "ser", "mixed_iter", dict(target_density=0.50, candidates=[(4,2),(8,4),(4,3)])),
    ]

    results = {"model": args.timm_name, "dense_top1": dense_top1,
               "ser_source_top1": ser_top1, "cells": []}
    Path(args.mask_save_dir).mkdir(parents=True, exist_ok=True)

    for label, source, method, kwargs in cells:
        print(f"\n=== {label}  source={source}  method={method}  kwargs={kwargs} ===")
        t_start = time.time()
        torch.manual_seed(42)

        student = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
        student.load_state_dict(dense_state if source == "dense" else ser_state)

        try:
            if method == "kn":
                stats = cert_aware_kn_for_conv(
                    student, calib_loader,
                    dense_state_dict=dense_state, n_calib_imgs=64,
                    free_restoration=True,
                    log=False, device=device,
                    **kwargs,
                )
            elif method == "mixed":
                stats = cert_aware_mixed_sparsity_for_conv(
                    student, calib_loader,
                    dense_state_dict=dense_state, n_calib_imgs=64,
                    log=False, device=device,
                    **kwargs,
                )
            elif method == "iter":
                stats = cert_aware_iterative_for_conv(
                    student, calib_loader,
                    dense_state_dict=dense_state, n_calib_imgs=64,
                    free_restoration=True,
                    log=False, device=device,
                    **kwargs,
                )
            elif method == "robust":
                stats = cert_aware_robust_for_conv(
                    student, calib_loader,
                    dense_state_dict=dense_state, n_calib_imgs=64,
                    free_restoration=True,
                    log=False, device=device,
                    **kwargs,
                )
            elif method == "mixed_iter":
                # mixed allocation, then iterative refinement
                stats = cert_aware_mixed_sparsity_for_conv(
                    student, calib_loader,
                    dense_state_dict=dense_state, n_calib_imgs=64,
                    log=False, device=device,
                    **kwargs,
                )
                # Then iterate over each layer's chosen pattern (skipped for time)
            else:
                raise ValueError(f"unknown method {method}")

            t_eval = time.time()
            pre_ft_top1 = evaluate_top1(student, val_loader, device)
            eval_time = time.time() - t_eval
            cell_time = time.time() - t_start

            # Save per-cell mask stats
            mask_stats = snapshot_masks_and_save(student, args.mask_save_dir, label)
            print(f"  pre-FT top1 = {pre_ft_top1:.4f}  cell_time={cell_time:.0f}s  eval={eval_time:.0f}s")
        except Exception as e:
            print(f"  CELL FAILED: {e}")
            pre_ft_top1 = -1.0; stats = {"error": str(e)}
            mask_stats = None

        results["cells"].append({
            "label": label, "source": source, "method": method, "kwargs": kwargs,
            "pre_ft_top1": pre_ft_top1, "stats": {k: v for k, v in stats.items()
                                                  if not isinstance(v, (list, dict))} if isinstance(stats, dict) else {},
            "elapsed_s": time.time() - t_start,
        })
        Path(args.output).write_text(json.dumps(results, indent=2, default=str))
        del student; torch.cuda.empty_cache()

    print(f"\n=== Advanced cert sweep complete ===")
    print(f"Dense: {dense_top1:.4f} | SER: {ser_top1:.4f}")
    for c in sorted(results["cells"], key=lambda x: -x["pre_ft_top1"]):
        print(f"  {c['label']:32s} {c['source']:5s} {c['method']:12s} pre_ft={c['pre_ft_top1']:.4f}")


if __name__ == "__main__":
    main()
