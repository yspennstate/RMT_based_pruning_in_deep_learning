"""
cert_opt_eval_vitb_kn_extended.py — same as cert_opt_eval_kn_extended.py
but Linear pipeline (ViT-S, ViT-B, etc.).
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
    interp_map = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC}
    interp = interp_map.get(interpolation.lower(), T.InterpolationMode.BICUBIC)
    if abs(crop_pct - 1.0) < 1e-6:
        tx = T.Compose([T.Resize((image_size, image_size), interpolation=interp),
                        T.ToTensor(), T.Normalize(mean=mean, std=std)])
    else:
        tx = T.Compose([T.Resize(int(image_size / crop_pct), interpolation=interp),
                        T.CenterCrop(image_size), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    val_ds = tv.datasets.ImageFolder(args.imagenet_val, transform=tx)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size_val, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    calib_ds = tv.datasets.ImageFolder(args.imagenet_train_for_calib, transform=tx)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=32, shuffle=True,
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
    print(f"  SER load: {matched}/{len(msd)} keys, coverage={cov:.4f}")
    if cov < 0.95: raise RuntimeError(f"coverage {cov:.4f} < 0.95")
    model.load_state_dict(sd, strict=False)


def build_cells():
    cells = []
    # 2:4 + dense vs SER + α_ser sweep
    for label_pre, source in [("D24", "dense"), ("S24", "ser")]:
        for alpha_ser in [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 1.0]:
            cells.append((f"{label_pre}_perm_a{int(alpha_ser*100):03d}",
                          2, 4, source, True, alpha_ser, 64, 0))
        cells.append((f"{label_pre}_noperm_a000", 2, 4, source, False, 0.0, 64, 0))
        cells.append((f"{label_pre}_noperm_a050", 2, 4, source, False, 0.5, 64, 0))
    # 4:8
    for label_pre, source in [("D48", "dense"), ("S48", "ser")]:
        for alpha_ser in [0.0, 0.3, 0.5, 0.6, 0.7]:
            cells.append((f"{label_pre}_perm_a{int(alpha_ser*100):03d}",
                          4, 8, source, True, alpha_ser, 64, 0))
        cells.append((f"{label_pre}_noperm_a000", 4, 8, source, False, 0.0, 64, 0))
    # 1:4 (75% sparse)
    for label_pre, source in [("D14", "dense"), ("S14", "ser")]:
        for alpha_ser in [0.0, 0.3, 0.5]:
            cells.append((f"{label_pre}_perm_a{int(alpha_ser*100):03d}",
                          1, 4, source, True, alpha_ser, 64, 0))
    # 3:4 (25% sparse)
    for label_pre, source in [("D34", "dense"), ("S34", "ser")]:
        cells.append((f"{label_pre}_perm_a000", 3, 4, source, True, 0.0, 64, 0))
        cells.append((f"{label_pre}_perm_a050", 3, 4, source, True, 0.5, 64, 0))
    # Calib variance probe
    for calib in [256, 512]:
        cells.append((f"S24_perm_a050_c{calib}", 2, 4, "ser", True, 0.5, calib, 0))
    # Replicates for variance bound
    for r in range(5):
        cells.append((f"S24_perm_a050_rep{r}", 2, 4, "ser", True, 0.5, 64, r))
    return cells


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
    print(f"[cert_opt_eval_vitb_kn_extended] device={device}, model={args.timm_name}")
    import timm
    teacher = timm.create_model(args.timm_name, pretrained=True).to(device).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(teacher)
    img_size = cfg["input_size"][-1]; mean = cfg["mean"]; std = cfg["std"]
    crop_pct = cfg.get("crop_pct", 0.9); interp = cfg.get("interpolation", "bicubic")
    calib_loader, val_loader = build_loaders(args, img_size, mean, std, crop_pct, interp)

    dense_state = {n_: p.detach().clone() for n_, p in teacher.state_dict().items()}
    t0 = time.time(); dense_top1 = evaluate_top1(teacher, val_loader, device)
    print(f"  dense top1 = {dense_top1:.4f} ({time.time()-t0:.0f}s)")
    ser_ref = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
    load_ser(ser_ref, args.ser_checkpoint)
    ser_state = {n_: p.detach().clone() for n_, p in ser_ref.state_dict().items()}
    t0 = time.time(); ser_top1 = evaluate_top1(ser_ref, val_loader, device)
    print(f"  SER top1 = {ser_top1:.4f} ({time.time()-t0:.0f}s)")

    cells = build_cells()
    print(f"[cert_opt_eval_vitb_kn_extended] {len(cells)} cells planned")
    results = {"model": args.timm_name, "dense_top1": dense_top1, "ser_source_top1": ser_top1, "cells": []}
    for label, k, n, source, perm, alpha_ser, calib, seed in cells:
        torch.manual_seed(42 + seed)
        print(f"\n=== {label}  k:n={k}:{n}  src={source}  perm={perm}  α_ser={alpha_ser}  calib={calib} seed={seed} ===")
        t_start = time.time()
        student = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
        student.load_state_dict(dense_state if source == "dense" else ser_state)
        try:
            cert_stats = cert_aware_kn_for_linear(
                student, calib_loader, n=n, k=k,
                dense_state_dict=dense_state, n_calib_imgs=calib,
                free_restoration=True, permute_align=perm, alpha_ser_prior=alpha_ser,
                log=False, device=device,
            )
            bad = cert_stats["groups_with_more_than_2_nonzero_after"]
            n_layers = cert_stats["n_layers_modified"]
            t_eval = time.time()
            pre_ft_top1 = evaluate_top1(student, val_loader, device)
            print(f"  pre-FT top1 = {pre_ft_top1:.4f}  bad={bad}  layers={n_layers}  eval_in={time.time()-t_eval:.0f}s")
        except Exception as e:
            print(f"  CELL FAILED: {e}")
            pre_ft_top1 = -1.0; bad = -1; n_layers = -1
        results["cells"].append({
            "label": label, "k": k, "n": n, "source": source, "perm": perm,
            "alpha_ser_prior": alpha_ser, "calib": calib, "seed": seed,
            "pre_ft_top1": pre_ft_top1, "bad_groups": bad, "n_layers_modified": n_layers,
            "elapsed_s": time.time() - t_start,
        })
        Path(args.output).write_text(json.dumps(results, indent=2))
        del student; torch.cuda.empty_cache()
    print(f"\n=== Extended ViT k:n sweep complete ===")
    print(f"Dense: {dense_top1:.4f} | SER: {ser_top1:.4f}")
    for c in sorted(results["cells"], key=lambda x: -x["pre_ft_top1"])[:15]:
        print(f"  {c['label']:32s} k:n={c['k']}:{c['n']} {c['source']:5s} pre_ft={c['pre_ft_top1']:.4f}")


if __name__ == "__main__":
    main()
