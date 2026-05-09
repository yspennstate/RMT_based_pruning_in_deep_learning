"""
cert_opt_eval.py — sweep over CAST-conv certificate-cost knobs on resnet50,
report the **pre-FT top-1** for each cell on ImageNet val. No FT — projection
+ eval only, ~3 min per cell on A100.

Cells:
  baseline   :  calib=64, alpha_kd=0, perm=off  (matches the original M1 we ran)
  +perm      :  calib=64, alpha_kd=0, perm=on   (matches M1pa)
  +alpha     :  calib=64, alpha_kd=0.1, perm=on (proposal's #2 with Fisher approx)
  +alpha-hi  :  calib=64, alpha_kd=0.5, perm=on
  +calib     :  calib=256, alpha_kd=0, perm=on  (more activation samples)
  +calib+alpha: calib=256, alpha_kd=0.1, perm=on (combined)

Usage:
  python cert_opt_eval.py \\
    --imagenet-val /workspace/val_imagefolder \\
    --imagenet-train-for-calib /workspace/val_imagefolder \\  # reuse val for calib if no train
    --ser-checkpoint /workspace/sweep_ckpts/resnet50.tv_in1k_keep_s35.pt \\
    --output /workspace/cert_opt_eval_results.json
"""
from __future__ import annotations

import argparse
import json
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T

import sys
sys.path.insert(0, str(Path(__file__).parent))
from project_conv_2_4 import cert_aware_2_4_for_conv


def build_loaders(args, image_size, mean, std, crop_pct=0.875, interpolation="bilinear"):
    # Use crop_pct + interpolation from timm data_config — important for ViT-B/384
    # which uses crop_pct=1.0 (no resize-then-crop, just resize-to-image-size).
    interp_map = {"bilinear": T.InterpolationMode.BILINEAR,
                  "bicubic": T.InterpolationMode.BICUBIC,
                  "nearest": T.InterpolationMode.NEAREST}
    interp = interp_map.get(interpolation.lower(), T.InterpolationMode.BICUBIC)
    if abs(crop_pct - 1.0) < 1e-6:
        # No center-crop: resize directly to image_size (ViT-B/384 default)
        tx_eval = T.Compose([
            T.Resize((image_size, image_size), interpolation=interp),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        tx_eval = T.Compose([
            T.Resize(int(image_size / crop_pct), interpolation=interp),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    val_ds = tv.datasets.ImageFolder(args.imagenet_val, transform=tx_eval)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    calib_ds = tv.datasets.ImageFolder(args.imagenet_train_for_calib, transform=tx_eval)
    calib_loader = torch.utils.data.DataLoader(
        calib_ds, batch_size=64, shuffle=True, num_workers=args.num_workers,
        pin_memory=True,
    )
    return calib_loader, val_loader


def evaluate_top1(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm-name", default="resnet50.tv_in1k")
    ap.add_argument("--ser-checkpoint", required=True)
    ap.add_argument("--imagenet-val", required=True)
    ap.add_argument("--imagenet-train-for-calib", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--include-3x3-convs", action="store_true", default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cert_opt_eval] device={device}")

    import timm
    print(f"[cert_opt_eval] loading dense teacher: {args.timm_name}")
    teacher = timm.create_model(args.timm_name, pretrained=True).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    data_cfg = timm.data.resolve_model_data_config(teacher)
    image_size = data_cfg["input_size"][-1]
    mean = data_cfg["mean"]; std = data_cfg["std"]
    crop_pct = data_cfg.get("crop_pct", 0.875)
    interp = data_cfg.get("interpolation", "bicubic")
    print(f"[cert_opt_eval] data_cfg: image_size={image_size}, crop_pct={crop_pct}, interp={interp}")

    calib_loader, val_loader = build_loaders(args, image_size, mean, std, crop_pct=crop_pct, interpolation=interp)

    # Pre-eval the SER source ONCE (same for all cells).
    # Use the SAME load mechanism as run_resnet_cast_aws.py:
    # - try `model_state_dict` THEN `state_dict` THEN raw dict
    # - strip a `module.` prefix if present
    # - assert >=95% tensor-mass coverage so we error loudly on mismatched keys
    print(f"[cert_opt_eval] loading SER ckpt: {args.ser_checkpoint}")
    student_ref = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
    raw = torch.load(args.ser_checkpoint, map_location="cpu", weights_only=False)
    sd = raw.get("model_state_dict") or raw.get("state_dict") or raw
    # strip module. prefix
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model_sd = student_ref.state_dict()
    loaded_numel = 0
    total_numel = 0
    matched = 0
    for k, v in model_sd.items():
        total_numel += v.numel()
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            loaded_numel += v.numel()
            matched += 1
    coverage = loaded_numel / max(1, total_numel)
    print(f"  SER load: matched {matched}/{len(model_sd)} keys, coverage={coverage:.4f}")
    if coverage < 0.95:
        raise RuntimeError(f"SER load coverage {coverage:.4f} < 0.95 — abort")
    student_ref.load_state_dict(sd, strict=False)
    student_state = {n: p.detach().clone() for n, p in student_ref.state_dict().items()}
    dense_state = {n: p.detach().clone() for n, p in teacher.state_dict().items()}

    # Reuse the same `sd` for resetting student between cells (no re-load from disk)
    def _fresh_student():
        m = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
        m.load_state_dict(student_state, strict=True)
        return m

    print("[cert_opt_eval] eval SER source baseline...")
    t0 = time.time()
    ser_top1 = evaluate_top1(student_ref, val_loader, device)
    print(f"  SER source top1 = {ser_top1:.4f}  ({time.time()-t0:.0f}s)")

    # 5 methods (each with permute_align=True except #1 baseline + #2 perm-only).
    # Sub-knobs swept inside the heavier methods to find a good operating point.
    cells = [
        # (label, n_calib, alpha_kd, permute_align, cost_form, alpha_ser_prior)
        # === V3: focused fine-grained sweep around the M5 winner (0.4590) ===
        # Re-confirm M5 + variance check
        ("V3_M5_replicate",           64, 0.0,  True, "l2",   0.50),  # variance vs v1+v2 M5 (0.4692, 0.4590)
        # Finer α_ser around 0.50 peak
        ("V3_ser_0p45",               64, 0.0,  True, "l2",   0.45),
        ("V3_ser_0p48",               64, 0.0,  True, "l2",   0.48),
        ("V3_ser_0p52",               64, 0.0,  True, "l2",   0.52),
        ("V3_ser_0p55",               64, 0.0,  True, "l2",   0.55),
        # Tiny α_kd corrections (0.1+ hurt; try smaller)
        ("V3_ser_0p5_kd_0p01",        64, 0.01, True, "l2",   0.50),
        ("V3_ser_0p5_kd_0p02",        64, 0.02, True, "l2",   0.50),
        ("V3_ser_0p5_kd_0p05",        64, 0.05, True, "l2",   0.50),
        # Calib variants between 64 and 1024 (B7 was 0.4061 — surprisingly worse)
        ("V3_ser_0p5_calib_128",     128, 0.0,  True, "l2",   0.50),
        ("V3_ser_0p5_calib_256",     256, 0.0,  True, "l2",   0.50),
        ("V3_ser_0p5_calib_512",     512, 0.0,  True, "l2",   0.50),
        # M1 baseline re-confirm for variance reference
        ("V3_M1_baseline_replicate",  64, 0.0,  False,"l2",   0.00),
    ]

    results = {"ser_source_top1": ser_top1, "cells": []}

    for label, n_calib, alpha_kd, perm, cost_form, alpha_ser in cells:
        print(f"\n=== {label}  (calib={n_calib}, α_kd={alpha_kd}, perm={perm}, "
              f"cost={cost_form}, α_ser={alpha_ser}) ===")
        t_start = time.time()

        # Re-load fresh student from snapshot (always SER s=0.35, never dense)
        student = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
        student.load_state_dict(student_state)

        cert_stats = cert_aware_2_4_for_conv(
            student, calib_loader, dense_state_dict=dense_state,
            n_calib_imgs=n_calib, device=device,
            free_restoration=True, only_1x1=not args.include_3x3_convs,
            permute_align=perm, alpha_kd=alpha_kd,
            teacher_for_kd=teacher if alpha_kd > 0 else None,
            cost_form=cost_form, alpha_ser_prior=alpha_ser,
            log=False,
        )

        # legality + sparsity sanity
        bad = cert_stats["groups_with_more_than_2_nonzero_after"]
        n_layers = cert_stats["n_layers_modified"]

        t_eval = time.time()
        pre_ft_top1 = evaluate_top1(student, val_loader, device)
        eval_time = time.time() - t_eval
        cell_time = time.time() - t_start

        print(f"  pre-FT top1 = {pre_ft_top1:.4f}  "
              f"bad_groups={bad}  layers={n_layers}  "
              f"projected_in={cell_time-eval_time:.0f}s  eval_in={eval_time:.0f}s")

        results["cells"].append({
            "label": label,
            "n_calib_imgs": n_calib,
            "alpha_kd": alpha_kd,
            "permute_align": perm,
            "pre_ft_top1": pre_ft_top1,
            "bad_groups": bad,
            "n_layers_modified": n_layers,
            "elapsed_s": cell_time,
        })
        # Write incremental results so partial sweep is recoverable
        Path(args.output).write_text(json.dumps(results, indent=2))

    print(f"\n=== Sweep complete ===")
    print(f"SER source: {ser_top1:.4f}")
    for c in results["cells"]:
        delta = c["pre_ft_top1"] - results["cells"][0]["pre_ft_top1"]
        print(f"  {c['label']:25s} pre_ft={c['pre_ft_top1']:.4f}  "
              f"Δvs_baseline={delta:+.4f}  bad={c['bad_groups']}")

    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
