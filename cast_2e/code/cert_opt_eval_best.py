"""
cert_opt_eval_best.py - runs selected pre-FT methods identified
from the broader sweep. Saves artifacts for FT and benchmark stages.

Best methods by architecture (from advanced_v2 + earlier sweeps):
  ResNet50 (Conv pipeline):
    - 12:16 SER+α=0.5  (75% dense, 0.7564 = -0.51pp)
    - 10:16 dense+perm (62.5% dense, 0.7382)
    - 8:16  dense+perm (50% dense, 0.6556)
  ViT-B (Linear pipeline):
    - 6:12  SER+α=0.5  (50% dense, 0.7871)
    - 8:16  dense+perm (50% dense, 0.7741)
    - 12:16 SER+α=0.5  (75% dense)
  ConvNeXtV2 (Linear pipeline):
    - 10:16 dense+perm (62.5% dense, 0.8587 = -0.87pp)
    - 8:16  dense+perm (50% dense, 0.8248)
    - 12:16 dense+perm (75% dense)

Saves per-cell JSON, mask snapshots, and projected ckpts for all cells.
"""
import torch as _t; _t.backends.cudnn.enabled = False

import argparse, json, time, sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from project_kn_sparsity import cert_aware_kn_for_conv, cert_aware_kn_for_linear


CELLS_BY_ARCH = {
    "resnet": [
        # (label, k, n, source, perm, alpha_ser)
        ("D816_dense_perm",   8, 16, "dense", True, 0.0),
        ("D1016_dense_perm", 10, 16, "dense", True, 0.0),
        ("D1216_dense_perm", 12, 16, "dense", True, 0.0),
        ("S1216_ser_a05",    12, 16, "ser",   True, 0.5),
        ("S1016_ser_a05",    10, 16, "ser",   True, 0.5),
    ],
    "vitb": [
        ("D48_dense_perm",    4,  8, "dense", True, 0.0),
        ("D612_ser_a05",      6, 12, "ser",   True, 0.5),
        ("D612_dense_perm",   6, 12, "dense", True, 0.0),
        ("D816_dense_perm",   8, 16, "dense", True, 0.0),
        ("S1216_ser_a05",    12, 16, "ser",   True, 0.5),
    ],
    "convnext": [
        ("D816_dense_perm",   8, 16, "dense", True, 0.0),
        ("D1016_dense_perm", 10, 16, "dense", True, 0.0),
        ("D1216_dense_perm", 12, 16, "dense", True, 0.0),
        ("S1216_ser_a05",    12, 16, "ser",   True, 0.5),
    ],
}


def build_loaders(args, image_size, mean, std, crop_pct=0.875, interpolation="bicubic"):
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
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=64, shuffle=False,
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
    msd = model.state_dict(); total_n = loaded_n = 0
    for k, v in msd.items():
        total_n += v.numel()
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            loaded_n += v.numel()
    cov = loaded_n / max(1, total_n)
    if cov < 0.95: raise RuntimeError(f"coverage {cov:.4f} < 0.95")
    model.load_state_dict(sd, strict=False)


def save_mask_snapshot(model, save_dir, label):
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
    ap.add_argument("--mask-save-dir", required=True)
    ap.add_argument("--ckpt-save-all-dir", required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pipeline", choices=["conv", "linear"], default="conv")
    ap.add_argument("--arch-key", choices=list(CELLS_BY_ARCH.keys()), required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[best] device={device}, model={args.timm_name}, pipeline={args.pipeline}, arch_key={args.arch_key}")
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

    cells = CELLS_BY_ARCH[args.arch_key]
    results = {"model": args.timm_name, "dense_top1": dense_top1, "ser_source_top1": ser_top1, "cells": []}
    Path(args.ckpt_save_all_dir).mkdir(parents=True, exist_ok=True)

    for label, k, n, source, perm, alpha_ser in cells:
        print(f"\n=== {label}  k:n={k}:{n}  src={source}  perm={perm}  α_ser={alpha_ser} ===")
        t_start = time.time()
        student = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
        student.load_state_dict(dense_state if source == "dense" else ser_state)
        try:
            if args.pipeline == "conv":
                cert_stats = cert_aware_kn_for_conv(student, calib_loader, n=n, k=k,
                                                    dense_state_dict=dense_state, n_calib_imgs=64,
                                                    free_restoration=True, permute_align=perm,
                                                    alpha_ser_prior=alpha_ser, log=False, device=device)
            else:
                cert_stats = cert_aware_kn_for_linear(student, calib_loader, n=n, k=k,
                                                      dense_state_dict=dense_state, n_calib_imgs=64,
                                                      free_restoration=True, permute_align=perm,
                                                      alpha_ser_prior=alpha_ser, log=False, device=device)
            pre_ft_top1 = evaluate_top1(student, val_loader, device)
            n_layers = cert_stats.get("n_layers_modified", 0)
            print(f"  pre-FT top1 = {pre_ft_top1:.4f}  layers={n_layers}  cell_time={time.time()-t_start:.0f}s")

            save_mask_snapshot(student, args.mask_save_dir, label)
            p = Path(args.ckpt_save_all_dir) / f"{label}.pt"
            torch.save({"state_dict": student.state_dict(), "cell": label,
                        "pre_ft_top1": pre_ft_top1, "k": k, "n": n,
                        "source": source, "alpha_ser": alpha_ser, "method": "kn",
                        "perm": perm}, p)
        except Exception as e:
            print(f"  CELL FAILED: {e}")
            pre_ft_top1 = -1.0; n_layers = -1
        results["cells"].append({
            "label": label, "k": k, "n": n, "source": source, "perm": perm,
            "alpha_ser_prior": alpha_ser,
            "pre_ft_top1": pre_ft_top1, "n_layers_modified": n_layers,
            "elapsed_s": time.time() - t_start,
        })
        Path(args.output).write_text(json.dumps(results, indent=2))
        del student; torch.cuda.empty_cache()

    print(f"\n=== best cells complete ===")
    for c in sorted(results["cells"], key=lambda x: -x["pre_ft_top1"])[:10]:
        print(f"  {c['label']:25s} k:n={c['k']}:{c['n']} pre_ft={c['pre_ft_top1']:.4f}")


if __name__ == "__main__":
    main()
