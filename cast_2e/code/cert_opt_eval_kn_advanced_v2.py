"""
cert_opt_eval_kn_advanced_v2.py - extended k:n, certificate, and RMT sweep.

Tests on a model:
  - Sampled 16:32 cert (n=32, k=16, 10K random patterns) for large pattern spaces
  - ConvNeXtV2 Linear pipeline (catches MLP layers Conv-pipeline missed)
  - Broader k:n at n=16: 4:16 (75% sparse), 6:16 (62.5%), 8:16, 10:16 (37.5%), 12:16 (25%)
  - 6:12 dense AND SER (for ViT-B Linear which has in_features=768=12×64)
  - Saves: per-cell JSON, mask snapshot, best-cell student ckpt

Pipeline auto-detected from --pipeline arg (conv or linear).
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
from project_kn_sparsity import cert_aware_kn_for_conv, cert_aware_kn_for_linear, is_eligible_conv_kn
from project_kn_sampled import cert_aware_kn_sampled_for_conv


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


def save_mask_snapshot(model, save_dir, label):
    """Save per-layer mask + sparsity stats."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                W = mod.weight.data
                mask = (W != 0).cpu().to(torch.uint8)
                stats[name] = {
                    "shape": list(mask.shape),
                    "sparsity": float(1.0 - mask.float().mean()),
                    "nnz": int(mask.sum()),
                    "params": int(mask.numel()),
                }
    out = Path(save_dir) / f"{label}_mask_stats.json"
    out.write_text(json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm-name", required=True)
    ap.add_argument("--ser-checkpoint", required=True)
    ap.add_argument("--imagenet-val", required=True)
    ap.add_argument("--imagenet-train-for-calib", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mask-save-dir", default=None)
    ap.add_argument("--ckpt-save-best", default=None,
                    help="Save state_dict of best-cell student to this path")
    ap.add_argument("--ckpt-save-all-dir", default=None,
                    help="If set, save every cell's projected student state_dict to this dir")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pipeline", choices=["conv", "linear"], default="conv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[advanced_v2] device={device}, model={args.timm_name}, pipeline={args.pipeline}")
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

    # Cells: (label, k, n, source, perm, alpha_ser, method, extra_kwargs)
    # method: "kn", "kn_sampled"
    cells = [
        # === Baselines for cross-pod confirmation ===
        ("D24_dense_perm",      2,  4, "dense", True,  0.0, "kn", {}),
        ("D48_dense_perm",      4,  8, "dense", True,  0.0, "kn", {}),
        ("D816_dense_perm",     8, 16, "dense", True,  0.0, "kn", {}),
        # === Different k at n=16 (vary density at fixed pattern size) ===
        ("D416_dense_perm",     4, 16, "dense", True,  0.0, "kn", {}),  # 75% sparse
        ("D616_dense_perm",     6, 16, "dense", True,  0.0, "kn", {}),  # 62.5% sparse
        ("D1016_dense_perm",   10, 16, "dense", True,  0.0, "kn", {}),  # 37.5% sparse
        ("D1216_dense_perm",   12, 16, "dense", True,  0.0, "kn", {}),  # 25% sparse
        # === Sampled 16:32 (n=32, k=16, 10K samples) ===
        ("D1632_sampled_dense", 16, 32, "dense", True, 0.0, "kn_sampled", {"n_samples": 10000}),
        # === SER + α=0.5 at the wider patterns ===
        ("S816_ser_a05",        8, 16, "ser",   True,  0.5, "kn", {}),
        ("S616_ser_a05",        6, 16, "ser",   True,  0.5, "kn", {}),
        ("S1216_ser_a05",      12, 16, "ser",   True,  0.5, "kn", {}),
    ]

    results = {"model": args.timm_name, "dense_top1": dense_top1, "ser_source_top1": ser_top1, "cells": []}
    best_cell = {"label": None, "pre_ft_top1": -1.0}

    for label, k, n, source, perm, alpha_ser, method, extra in cells:
        print(f"\n=== {label}  k:n={k}:{n}  src={source}  perm={perm}  α_ser={alpha_ser}  method={method} ===")
        t_start = time.time()
        student = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
        student.load_state_dict(dense_state if source == "dense" else ser_state)
        try:
            if method == "kn_sampled":
                fn = cert_aware_kn_sampled_for_conv  # only conv supported
                cert_stats = fn(student, calib_loader, n=n, k=k,
                                dense_state_dict=dense_state, n_calib_imgs=64,
                                free_restoration=True, permute_align=perm,
                                alpha_ser_prior=alpha_ser, log=False, device=device,
                                **extra)
            elif args.pipeline == "conv":
                cert_stats = cert_aware_kn_for_conv(student, calib_loader, n=n, k=k,
                                                    dense_state_dict=dense_state, n_calib_imgs=64,
                                                    free_restoration=True, permute_align=perm,
                                                    alpha_ser_prior=alpha_ser, log=False, device=device)
            else:
                cert_stats = cert_aware_kn_for_linear(student, calib_loader, n=n, k=k,
                                                      dense_state_dict=dense_state, n_calib_imgs=64,
                                                      free_restoration=True, permute_align=perm,
                                                      alpha_ser_prior=alpha_ser, log=False, device=device)
            t_eval = time.time()
            pre_ft_top1 = evaluate_top1(student, val_loader, device)
            n_layers = cert_stats.get("n_layers_modified", 0)
            print(f"  pre-FT top1 = {pre_ft_top1:.4f}  layers={n_layers}  cell_time={time.time()-t_start:.0f}s")

            if args.mask_save_dir:
                save_mask_snapshot(student, args.mask_save_dir, label)
            if args.ckpt_save_all_dir:
                Path(args.ckpt_save_all_dir).mkdir(parents=True, exist_ok=True)
                p = Path(args.ckpt_save_all_dir) / f"{label}.pt"
                torch.save({"state_dict": student.state_dict(), "cell": label,
                            "pre_ft_top1": pre_ft_top1, "k": k, "n": n,
                            "source": source, "alpha_ser": alpha_ser, "method": method},
                            p)
            if pre_ft_top1 > best_cell["pre_ft_top1"]:
                best_cell = {"label": label, "pre_ft_top1": pre_ft_top1}
                if args.ckpt_save_best:
                    Path(args.ckpt_save_best).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state_dict": student.state_dict(), "cell": label,
                                "pre_ft_top1": pre_ft_top1, "k": k, "n": n,
                                "source": source, "alpha_ser": alpha_ser},
                                args.ckpt_save_best)
                    print(f"  → new best, saved ckpt to {args.ckpt_save_best}")
        except Exception as e:
            print(f"  CELL FAILED: {e}")
            pre_ft_top1 = -1.0; n_layers = -1
        results["cells"].append({
            "label": label, "k": k, "n": n, "source": source, "perm": perm,
            "alpha_ser_prior": alpha_ser, "method": method, "kwargs": extra,
            "pre_ft_top1": pre_ft_top1, "n_layers_modified": n_layers,
            "elapsed_s": time.time() - t_start,
        })
        Path(args.output).write_text(json.dumps(results, indent=2))
        del student; torch.cuda.empty_cache()

    print(f"\n=== Advanced v2 complete ===")
    for c in sorted(results["cells"], key=lambda x: -x["pre_ft_top1"])[:10]:
        print(f"  {c['label']:25s} k:n={c['k']}:{c['n']} pre_ft={c['pre_ft_top1']:.4f}")
    print(f"BEST: {best_cell['label']} = {best_cell['pre_ft_top1']:.4f}")


if __name__ == "__main__":
    main()
