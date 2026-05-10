"""
cert_opt_eval_8_16.py - test 8:16 sparsity against 2:4 and 4:8 baselines.

C(8,4)=70 patterns -> C(16,8)=12,870 patterns. The sparsity rate remains 50%,
with a larger candidate set for certificate-based projection. Tests on ResNet50
and ViT-B.

For Conv2d: layers must have Cin divisible by 16. ResNet50 1x1 + 3x3 layers are
all multiples of 16, so eligibility is preserved.

Cells (~12 total, ~15-20 min run):
  - 2:4 dense + perm (baseline)
  - 4:8 dense + perm
  - 8:16 dense + perm
  - 2:4 SER + perm + α=0.5
  - 4:8 SER + perm + α=0.5
  - 8:16 SER + perm + α=0.5
  - dense+nopperm at each pattern (3 cells)
  - 6:12 (sanity check between 4:8 and 8:16, C(12,6)=924)
"""
from __future__ import annotations
import torch as _t; _t.backends.cudnn.enabled = False

import argparse, json, time, sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from project_kn_sparsity import cert_aware_kn_for_conv, cert_aware_kn_for_linear


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm-name", required=True)
    ap.add_argument("--ser-checkpoint", required=True)
    ap.add_argument("--imagenet-val", required=True)
    ap.add_argument("--imagenet-train-for-calib", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pipeline", choices=["conv", "linear"], default="conv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cert_opt_eval_8_16] device={device}, model={args.timm_name}, pipeline={args.pipeline}")
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

    # k:n cells progression: 2:4 → 4:8 → 6:12 → 8:16
    cells = [
        # Dense + perm baseline at increasing pattern sizes
        ("D24_dense_perm",  2,  4, "dense", True,  0.0),
        ("D48_dense_perm",  4,  8, "dense", True,  0.0),
        ("D612_dense_perm", 6, 12, "dense", True,  0.0),
        ("D816_dense_perm", 8, 16, "dense", True,  0.0),
        # SER + perm + α_ser=0.5 at each
        ("S24_ser_perm_a05",  2,  4, "ser", True, 0.5),
        ("S48_ser_perm_a05",  4,  8, "ser", True, 0.5),
        ("S612_ser_perm_a05", 6, 12, "ser", True, 0.5),
        ("S816_ser_perm_a05", 8, 16, "ser", True, 0.5),
    ]

    results = {"model": args.timm_name, "dense_top1": dense_top1, "ser_source_top1": ser_top1, "cells": []}
    fn = cert_aware_kn_for_conv if args.pipeline == "conv" else cert_aware_kn_for_linear
    for label, k, n, source, perm, alpha_ser in cells:
        print(f"\n=== {label}  k:n={k}:{n}  src={source}  perm={perm}  α_ser={alpha_ser} ===")
        t_start = time.time()
        student = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
        student.load_state_dict(dense_state if source == "dense" else ser_state)
        try:
            cert_stats = fn(
                student, calib_loader, n=n, k=k,
                dense_state_dict=dense_state, n_calib_imgs=64,
                free_restoration=True, permute_align=perm, alpha_ser_prior=alpha_ser,
                log=False, device=device,
            )
            bad = cert_stats.get("groups_with_more_than_2_nonzero_after", 0)
            n_layers = cert_stats["n_layers_modified"]
            t_eval = time.time()
            pre_ft_top1 = evaluate_top1(student, val_loader, device)
            print(f"  pre-FT top1 = {pre_ft_top1:.4f}  bad={bad}  layers={n_layers}  "
                  f"projected_in={t_eval-t_start:.0f}s  eval_in={time.time()-t_eval:.0f}s")
        except Exception as e:
            print(f"  CELL FAILED: {e}")
            pre_ft_top1 = -1.0; bad = -1; n_layers = -1
        results["cells"].append({
            "label": label, "k": k, "n": n, "source": source, "perm": perm,
            "alpha_ser_prior": alpha_ser,
            "pre_ft_top1": pre_ft_top1, "bad_groups": bad, "n_layers_modified": n_layers,
            "elapsed_s": time.time() - t_start,
        })
        Path(args.output).write_text(json.dumps(results, indent=2))
        del student; torch.cuda.empty_cache()

    print("\n=== 8:16 sweep complete ===")
    print(f"Dense {dense_top1:.4f} | SER {ser_top1:.4f}")
    for c in results["cells"]:
        print(f"  {c['label']:25s} k:n={c['k']}:{c['n']} pre_ft={c['pre_ft_top1']:.4f}")


if __name__ == "__main__":
    main()
