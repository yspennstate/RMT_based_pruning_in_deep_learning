"""
cert_opt_eval_vitb_kn.py - k:n sparsity ablation (Linear / ViT pipeline).

Same cell layout as cert_opt_eval_kn.py but for nn.Linear and ViT-B/16 224.

Usage:
  python cert_opt_eval_vitb_kn.py \\
    --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k \\
    --ser-checkpoint /workspace/sweep_ckpts/vit_base_patch16_224.augreg2_in21k_ft_in1k_keep_s35_classmag.pt \\
    --imagenet-val ... --imagenet-train-for-calib ... --output ...
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
from project_kn_sparsity import cert_aware_kn_for_linear


def build_loaders(args, image_size, mean, std, crop_pct=0.9, interpolation="bicubic"):
    interp_map = {"bilinear": T.InterpolationMode.BILINEAR,
                  "bicubic": T.InterpolationMode.BICUBIC}
    interp = interp_map.get(interpolation.lower(), T.InterpolationMode.BICUBIC)
    if abs(crop_pct - 1.0) < 1e-6:
        tx = T.Compose([T.Resize((image_size, image_size), interpolation=interp),
                        T.ToTensor(), T.Normalize(mean=mean, std=std)])
    else:
        tx = T.Compose([T.Resize(int(image_size / crop_pct), interpolation=interp),
                        T.CenterCrop(image_size), T.ToTensor(),
                        T.Normalize(mean=mean, std=std)])
    val_ds = tv.datasets.ImageFolder(args.imagenet_val, transform=tx)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size_val,
                                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    calib_ds = tv.datasets.ImageFolder(args.imagenet_train_for_calib, transform=tx)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=32, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)
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


def load_ser_with_coverage(model, ckpt_path):
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("model_state_dict") or raw.get("state_dict") or raw
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    msd = model.state_dict()
    matched, total_n, loaded_n = 0, 0, 0
    for k, v in msd.items():
        total_n += v.numel()
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            loaded_n += v.numel(); matched += 1
    coverage = loaded_n / max(1, total_n)
    print(f"  SER load: matched {matched}/{len(msd)} keys, coverage={coverage:.4f}")
    if coverage < 0.95:
        raise RuntimeError(f"SER coverage {coverage:.4f} < 0.95")
    model.load_state_dict(sd, strict=False)


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
    print(f"[cert_opt_eval_vitb_kn] device={device}, model={args.timm_name}")
    import timm

    teacher = timm.create_model(args.timm_name, pretrained=True).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    data_cfg = timm.data.resolve_model_data_config(teacher)
    image_size = data_cfg["input_size"][-1]
    mean = data_cfg["mean"]; std = data_cfg["std"]
    crop_pct = data_cfg.get("crop_pct", 0.9)
    interp = data_cfg.get("interpolation", "bicubic")
    calib_loader, val_loader = build_loaders(args, image_size, mean, std, crop_pct, interp)

    print("[cert_opt_eval_vitb_kn] capturing dense state + eval...")
    dense_state = {n_: p.detach().clone() for n_, p in teacher.state_dict().items()}
    t0 = time.time()
    dense_top1 = evaluate_top1(teacher, val_loader, device)
    print(f"  dense top1 = {dense_top1:.4f} ({time.time()-t0:.0f}s)")

    print(f"[cert_opt_eval_vitb_kn] loading SER ckpt: {args.ser_checkpoint}")
    ser_ref = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
    load_ser_with_coverage(ser_ref, args.ser_checkpoint)
    ser_state = {n_: p.detach().clone() for n_, p in ser_ref.state_dict().items()}
    t0 = time.time()
    ser_top1 = evaluate_top1(ser_ref, val_loader, device)
    print(f"  SER source top1 = {ser_top1:.4f} ({time.time()-t0:.0f}s)")

    cells = [
        # 2:4 baseline + dense vs SER
        ("D24_dense_perm",          2, 4, "dense", True,  0.0),
        ("D24_dense_no_perm",       2, 4, "dense", False, 0.0),
        ("S24_ser_perm_ser0p5",     2, 4, "ser",   True,  0.5),
        ("S24_ser_perm_ser0p6",     2, 4, "ser",   True,  0.6),  # ViT-B's pre-FT winner
        ("S24_ser_perm_no_ser",     2, 4, "ser",   True,  0.0),
        ("S24_ser_no_perm",         2, 4, "ser",   False, 0.0),
        # 4:8, same 50% rate with more flexibility
        ("D48_dense_perm",          4, 8, "dense", True,  0.0),
        ("S48_ser_perm_ser0p5",     4, 8, "ser",   True,  0.5),
        ("S48_ser_perm_ser0p6",     4, 8, "ser",   True,  0.6),
        ("S48_ser_perm_no_ser",     4, 8, "ser",   True,  0.0),
        ("S48_ser_no_perm",         4, 8, "ser",   False, 0.0),
        # 1:4, 75% sparse
        ("D14_dense_perm",          1, 4, "dense", True,  0.0),
        ("S14_ser_perm_ser0p5",     1, 4, "ser",   True,  0.5),
        # 3:4, 25% sparse ceiling
        ("D34_dense_perm",          3, 4, "dense", True,  0.0),
        ("S34_ser_perm_ser0p5",     3, 4, "ser",   True,  0.5),
    ]

    results = {"model": args.timm_name, "dense_top1": dense_top1,
               "ser_source_top1": ser_top1, "cells": []}

    for label, k, n, source, perm, alpha_ser in cells:
        print(f"\n=== {label}  k:n={k}:{n}  source={source}  perm={perm}  α_ser={alpha_ser} ===")
        t_start = time.time()

        student = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
        if source == "dense":
            student.load_state_dict(dense_state)
        else:
            student.load_state_dict(ser_state)

        cert_stats = cert_aware_kn_for_linear(
            student, calib_loader,
            n=n, k=k,
            dense_state_dict=dense_state,
            n_calib_imgs=64,
            free_restoration=True,
            permute_align=perm,
            alpha_ser_prior=alpha_ser,
            log=False,
            device=device,
        )

        bad = cert_stats["groups_with_more_than_2_nonzero_after"]
        n_layers = cert_stats["n_layers_modified"]
        t_eval = time.time()
        pre_ft_top1 = evaluate_top1(student, val_loader, device)
        eval_time = time.time() - t_eval
        cell_time = time.time() - t_start
        print(f"  pre-FT top1 = {pre_ft_top1:.4f}  bad={bad}  layers={n_layers}  "
              f"projected_in={cell_time-eval_time:.0f}s  eval_in={eval_time:.0f}s")

        results["cells"].append({
            "label": label, "k": k, "n": n, "source": source, "perm": perm,
            "alpha_ser_prior": alpha_ser,
            "pre_ft_top1": pre_ft_top1, "bad_groups": bad,
            "n_layers_modified": n_layers, "elapsed_s": cell_time,
        })
        Path(args.output).write_text(json.dumps(results, indent=2))

    print(f"\n=== ViT-B k:n sweep complete ===")
    print(f"Dense:      {dense_top1:.4f}")
    print(f"SER source: {ser_top1:.4f}")
    for c in results["cells"]:
        print(f"  {c['label']:32s} k:n={c['k']}:{c['n']}  {c['source']:5s} "
              f"pre_ft={c['pre_ft_top1']:.4f}")


if __name__ == "__main__":
    main()
