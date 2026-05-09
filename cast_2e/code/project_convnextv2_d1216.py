"""
project_convnextv2_d1216.py — small wrapper that:
  1. Loads ConvNeXtV2-Base (timm dense weights).
  2. Runs cert-aware 12:16 projection (Linear pipeline, dense source, no-perm
     to keep weights functional without forward-pre-hooks for the FT runner).
  3. Saves projected student as student_pre_ft.pt at the canonical path.
The existing run_vit_tome_flop_reduction.py FT runner can then resume from
this checkpoint with --enable-2-4 swapped for plain mask-freeze (its
mask-extraction reads weight != 0 generically per line 308 of that file).
"""
import torch as _t; _t.backends.cudnn.enabled = False

import argparse, json, time, sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from project_kn_sparsity import cert_aware_kn_for_linear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm-name", default="convnextv2_base.fcmae_ft_in22k_in1k")
    ap.add_argument("--imagenet-val", required=True,
                    help="ImageFolder val (used for calibration AND post-projection eval)")
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--source", choices=["dense", "ser"], default="dense")
    ap.add_argument("--ser-ckpt", default=None)
    ap.add_argument("--alpha-ser", type=float, default=0.0)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda"
    print(f"[project_convnextv2] timm={args.timm_name}, k:n={args.k}:{args.n}, source={args.source}")
    import timm

    # 1. Build dense teacher (which is also the source for "dense" mode)
    teacher = timm.create_model(args.timm_name, pretrained=True).to(device).eval()
    cfg = timm.data.resolve_model_data_config(teacher)
    img_size = cfg["input_size"][-1]; mean = cfg["mean"]; std = cfg["std"]
    crop_pct = cfg.get("crop_pct", 0.875); interp = cfg.get("interpolation", "bicubic")

    val_tx = timm.data.create_transform(
        input_size=img_size, is_training=False, mean=mean, std=std,
        crop_pct=crop_pct, interpolation=interp,
    )
    val_ds = tv.datasets.ImageFolder(args.imagenet_val, transform=val_tx)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    calib_ds = tv.datasets.ImageFolder(args.imagenet_val, transform=val_tx)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=64, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True)

    dense_state = {n_: p.detach().clone() for n_, p in teacher.state_dict().items()}

    # 2. Build student
    student = timm.create_model(args.timm_name, pretrained=(args.source == "dense")).to(device).eval()
    if args.source == "ser":
        if not args.ser_ckpt:
            raise ValueError("--source ser requires --ser-ckpt")
        raw = torch.load(args.ser_ckpt, map_location="cpu", weights_only=False)
        sd = raw.get("model_state_dict") or raw.get("state_dict") or raw
        if any(k.startswith("module.") for k in sd):
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
        student.load_state_dict(sd, strict=False)
        print(f"  loaded SER from {args.ser_ckpt}")

    # Eval dense teacher
    def eval_top1(model, loader):
        model.eval(); c = t = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
                c += int((model(x).argmax(1) == y).sum().item()); t += y.size(0)
        return c / t

    teacher_top1 = eval_top1(teacher, val_loader)
    print(f"  teacher (dense) top1 = {teacher_top1:.4f}")

    # 3. Cert k:n projection on Linear layers (NO perm — keep weights functional
    #    without hooks so the downstream FT runner can load+freeze cleanly)
    print(f"\n=== cert_aware_kn_for_linear k:n={args.k}:{args.n}, perm=False ===")
    t0 = time.time()
    cert_stats = cert_aware_kn_for_linear(
        student, calib_loader, n=args.n, k=args.k,
        dense_state_dict=dense_state, n_calib_imgs=64,
        free_restoration=True, permute_align=False,
        alpha_ser_prior=args.alpha_ser, log=False, device=device,
    )
    print(f"  projection done in {time.time()-t0:.0f}s, layers_modified={cert_stats.get('n_layers_modified',0)}")

    # 4. Eval pre-FT
    pre_ft_top1 = eval_top1(student, val_loader)
    print(f"  pre-FT student top1 = {pre_ft_top1:.4f} (Δ vs teacher = {pre_ft_top1-teacher_top1:+.4f})")

    # 5. Save
    out = Path(args.out_ckpt); out.parent.mkdir(parents=True, exist_ok=True)
    nnz = sum(int((p != 0).sum()) for p in student.state_dict().values() if torch.is_tensor(p))
    total = sum(int(p.numel()) for p in student.state_dict().values() if torch.is_tensor(p))
    payload = {
        "state_dict": student.state_dict(),
        "cert_config": {"k": args.k, "n": args.n, "source": args.source,
                        "alpha_ser": args.alpha_ser, "permute_align": False,
                        "free_restoration": True, "pipeline": "linear"},
        "pre_ft_top1": pre_ft_top1, "teacher_top1": teacher_top1,
        "global_sparsity": 1.0 - nnz / max(1, total),
        "n_layers_modified": cert_stats.get("n_layers_modified", 0),
    }
    torch.save(payload, out)
    print(f"Saved: {out}")
    print(f"  global density: {nnz/max(1,total):.4f}")


if __name__ == "__main__":
    main()
