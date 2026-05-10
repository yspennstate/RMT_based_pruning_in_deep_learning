"""
cert_opt_eval_vitb.py - same 5-method pre-FT ablation as cert_opt_eval.py but
for ViT-B/16 (Linear pipeline). Uses cert_aware_2_4_for_linear.

Cells:
  M1 baseline (no perm, l², no extras)
  M2 perm
  M3/M3b alpha_kd 0.1 / 0.5
  M5/M5b ser_prior 0.5 / 0.3
  B5  combo perm + alpha_kd + ser_prior

Usage:
  python cert_opt_eval_vitb.py --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k \\
      --ser-checkpoint .../keep_s35.pt --imagenet-val ... --imagenet-train-for-calib ... \\
      --output ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T

import sys
sys.path.insert(0, str(Path(__file__).parent))
from project_conv_2_4 import cert_aware_2_4_for_linear


def build_loaders(args, image_size, mean, std, crop_pct=0.9, interpolation="bicubic"):
    interp_map = {"bilinear": T.InterpolationMode.BILINEAR,
                  "bicubic": T.InterpolationMode.BICUBIC,
                  "nearest": T.InterpolationMode.NEAREST}
    interp = interp_map.get(interpolation.lower(), T.InterpolationMode.BICUBIC)
    if abs(crop_pct - 1.0) < 1e-6:
        tx = T.Compose([T.Resize((image_size, image_size), interpolation=interp),
                        T.ToTensor(), T.Normalize(mean=mean, std=std)])
    else:
        tx = T.Compose([T.Resize(int(image_size / crop_pct), interpolation=interp),
                        T.CenterCrop(image_size), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    val_ds = tv.datasets.ImageFolder(args.imagenet_val, transform=tx)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size_val,
                                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    calib_ds = tv.datasets.ImageFolder(args.imagenet_train_for_calib, transform=tx)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=32,
                                                shuffle=True, num_workers=args.num_workers, pin_memory=True)
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
    ap.add_argument("--timm-name", required=True)
    ap.add_argument("--ser-checkpoint", required=True)
    ap.add_argument("--imagenet-val", required=True)
    ap.add_argument("--imagenet-train-for-calib", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size-val", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cert_opt_eval_vitb] device={device}")

    import timm
    print(f"[cert_opt_eval_vitb] loading dense teacher: {args.timm_name}")
    teacher = timm.create_model(args.timm_name, pretrained=True).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    data_cfg = timm.data.resolve_model_data_config(teacher)
    image_size = data_cfg["input_size"][-1]
    mean = data_cfg["mean"]; std = data_cfg["std"]
    crop_pct = data_cfg.get("crop_pct", 0.9)
    interp = data_cfg.get("interpolation", "bicubic")
    print(f"[cert_opt_eval_vitb] data_cfg: image_size={image_size}, crop_pct={crop_pct}, interp={interp}")

    calib_loader, val_loader = build_loaders(args, image_size, mean, std,
                                              crop_pct=crop_pct, interpolation=interp)

    print(f"[cert_opt_eval_vitb] loading SER ckpt: {args.ser_checkpoint}")
    student_ref = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
    raw = torch.load(args.ser_checkpoint, map_location="cpu", weights_only=False)
    sd = raw.get("model_state_dict") or raw.get("state_dict") or raw
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model_sd = student_ref.state_dict()
    loaded_numel = total_numel = 0
    matched = 0
    for k, v in model_sd.items():
        total_numel += v.numel()
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            loaded_numel += v.numel(); matched += 1
    coverage = loaded_numel / max(1, total_numel)
    print(f"  SER load: matched {matched}/{len(model_sd)} keys, coverage={coverage:.4f}")
    if coverage < 0.95:
        raise RuntimeError(f"SER load coverage {coverage:.4f} < 0.95; abort")
    student_ref.load_state_dict(sd, strict=False)
    student_state = {n: p.detach().clone() for n, p in student_ref.state_dict().items()}
    dense_state = {n: p.detach().clone() for n, p in teacher.state_dict().items()}

    print("[cert_opt_eval_vitb] eval SER source baseline...")
    t0 = time.time()
    ser_top1 = evaluate_top1(student_ref, val_loader, device)
    print(f"  SER source top1 = {ser_top1:.4f}  ({time.time()-t0:.0f}s)")

    cells = [
        # (label, n_calib, alpha_kd, perm, cost_form, alpha_ser_prior)
        # === Pass A: broad 5-method baseline sweep ===
        ("A1_baseline_l2",            64, 0.0,  False, "l2",   0.0),  # canonical CAST
        ("A2_perm",                   64, 0.0,  True,  "l2",   0.0),
        ("A3_alpha_kd_0p1",           64, 0.1,  True,  "l2",   0.0),
        ("A3b_alpha_kd_0p5",          64, 0.5,  True,  "l2",   0.0),
        ("A4_linf",                   64, 0.0,  True,  "linf", 0.0),
        ("A5_ser_0p3",                64, 0.0,  True,  "l2",   0.3),
        ("A5b_ser_0p4",               64, 0.0,  True,  "l2",   0.4),
        ("A5c_ser_0p5",               64, 0.0,  True,  "l2",   0.5),  # 0.5 candidate
        ("A5d_ser_0p6",               64, 0.0,  True,  "l2",   0.6),
        ("A5e_ser_0p7",               64, 0.0,  True,  "l2",   0.7),
        ("A6_combo_kd05_ser05",       64, 0.5,  True,  "l2",   0.5),
        ("A6b_combo_kd01_ser05",      64, 0.1,  True,  "l2",   0.5),
        # === Pass B: fine sub-sweep around alpha_ser=0.5 ===
        ("B1_ser_0p45",               64, 0.0,  True,  "l2",   0.45),
        ("B2_ser_0p48",               64, 0.0,  True,  "l2",   0.48),
        ("B3_ser_0p52",               64, 0.0,  True,  "l2",   0.52),
        ("B4_ser_0p55",               64, 0.0,  True,  "l2",   0.55),
        # === Pass C: small alpha_kd corrections to 0.5 ===
        ("C1_ser_0p5_kd_0p01",        64, 0.01, True,  "l2",   0.50),
        ("C2_ser_0p5_kd_0p02",        64, 0.02, True,  "l2",   0.50),
        ("C3_ser_0p5_kd_0p05",        64, 0.05, True,  "l2",   0.50),
        # === Pass D: calib variants ===
        ("D1_ser_0p5_calib_128",     128, 0.0,  True,  "l2",   0.50),
        ("D2_ser_0p5_calib_256",     256, 0.0,  True,  "l2",   0.50),
        ("D3_ser_0p5_calib_512",     512, 0.0,  True,  "l2",   0.50),
        # === Pass E: variance / replicate ===
        ("E1_ser_0p5_replicate",      64, 0.0,  True,  "l2",   0.50),  # check stochastic variance
    ]

    results = {"ser_source_top1": ser_top1, "model": args.timm_name, "cells": []}

    for label, n_calib, alpha_kd, perm, cost_form, alpha_ser in cells:
        print(f"\n=== {label}  (calib={n_calib}, α_kd={alpha_kd}, perm={perm}, "
              f"cost={cost_form}, α_ser={alpha_ser}) ===")
        t_start = time.time()

        student = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
        student.load_state_dict(student_state)

        cert_stats = cert_aware_2_4_for_linear(
            student, calib_loader, dense_state_dict=dense_state,
            n_calib_imgs=n_calib, device=device,
            free_restoration=True,
            permute_align=perm, alpha_kd=alpha_kd,
            teacher_for_kd=teacher if alpha_kd > 0 else None,
            cost_form=cost_form, alpha_ser_prior=alpha_ser,
            log=False,
        )

        bad = cert_stats["groups_with_more_than_2_nonzero_after"]
        n_layers = cert_stats["n_layers_modified"]
        t_eval = time.time()
        pre_ft_top1 = evaluate_top1(student, val_loader, device)
        eval_time = time.time() - t_eval
        cell_time = time.time() - t_start
        print(f"  pre-FT top1 = {pre_ft_top1:.4f}  bad_groups={bad}  layers={n_layers}  "
              f"projected_in={cell_time-eval_time:.0f}s  eval_in={eval_time:.0f}s")

        results["cells"].append({
            "label": label, "n_calib_imgs": n_calib, "alpha_kd": alpha_kd,
            "permute_align": perm, "cost_form": cost_form, "alpha_ser_prior": alpha_ser,
            "pre_ft_top1": pre_ft_top1, "bad_groups": bad, "n_layers_modified": n_layers,
            "elapsed_s": cell_time,
        })
        Path(args.output).write_text(json.dumps(results, indent=2))

    print(f"\n=== ViT-B sweep complete ===")
    print(f"SER source: {ser_top1:.4f}")
    base = results["cells"][0]["pre_ft_top1"]
    for c in results["cells"]:
        d = c["pre_ft_top1"] - base
        print(f"  {c['label']:25s} pre_ft={c['pre_ft_top1']:.4f}  Δvs_baseline={d:+.4f}")


if __name__ == "__main__":
    main()
