"""
run_vitb_ft_from_ckpt.py — 3-ep distillation FT on ViT-B starting from a
pre-projected (sparse) student ckpt. Mask is held fixed via re-zero after
every optimizer step.

Usage:
    python run_vitb_ft_from_ckpt.py \\
        --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k \\
        --student-ckpt /workspace/run_outputs/best_pod1/ckpts/vitb/D612_ser_a05.pt \\
        --imagenet-train /workspace/imagenet/train \\
        --imagenet-val /workspace/imagenet/val \\
        --output-dir /workspace/run_outputs/vitb_ft/D612_ser_a05 \\
        --epochs 3 --batch 256 --lr 5e-5 --distill-temp 2.0 --distill-alpha 0.5
"""
import torch as _t; _t.backends.cudnn.enabled = False

import argparse, json, time, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as T


def evaluate_top1(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            correct += (model(imgs).argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)
    return correct / total


def collect_masks(model):
    """Snapshot the zero-mask of every weight tensor that has any zeros."""
    masks = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                W = mod.weight.data
                m = (W != 0).to(W.dtype)
                if (m == 0).any():
                    masks[name] = m.clone()
    return masks


def reapply_masks(model, masks):
    with torch.no_grad():
        for name, mod in model.named_modules():
            if name in masks:
                mod.weight.data.mul_(masks[name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm-name", required=True)
    ap.add_argument("--student-ckpt", required=True)
    ap.add_argument("--imagenet-train", required=True)
    ap.add_argument("--imagenet-val", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--distill-alpha", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every-epoch", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[ft] device={device}, model={args.timm_name}, epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
    import timm

    # 1. Build dense teacher
    teacher = timm.create_model(args.timm_name, pretrained=True).to(device).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(teacher)
    img_size = cfg["input_size"][-1]; mean = cfg["mean"]; std = cfg["std"]
    crop_pct = cfg.get("crop_pct", 0.875); interp = cfg.get("interpolation", "bicubic")
    interp_map = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC}
    interp = interp_map.get(interp.lower(), T.InterpolationMode.BICUBIC)

    val_tx = T.Compose([
        T.Resize(int(img_size / crop_pct), interpolation=interp),
        T.CenterCrop(img_size), T.ToTensor(), T.Normalize(mean=mean, std=std),
    ])
    train_tx = T.Compose([
        T.RandomResizedCrop(img_size, interpolation=interp),
        T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize(mean=mean, std=std),
    ])
    val_ds = tv.datasets.ImageFolder(args.imagenet_val, transform=val_tx)
    train_ds = tv.datasets.ImageFolder(args.imagenet_train, transform=train_tx)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True,
                                                drop_last=True, persistent_workers=True)

    # 2. Build student from saved ckpt
    student = timm.create_model(args.timm_name, pretrained=False).to(device)
    raw = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict") or raw
    msd = student.state_dict()
    matched = 0; total_n = loaded_n = 0
    for k, v in msd.items():
        total_n += v.numel()
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            loaded_n += v.numel(); matched += 1
    coverage = loaded_n / max(1, total_n)
    print(f"  ckpt coverage: {coverage:.4f} ({matched} keys matched)")
    if coverage < 0.95:
        raise RuntimeError(f"ckpt coverage too low: {coverage:.4f}")
    student.load_state_dict(sd, strict=False)
    print(f"  ckpt cell={raw.get('cell')}, k:n={raw.get('k')}:{raw.get('n')}, "
          f"source={raw.get('source')}, alpha_ser={raw.get('alpha_ser')}, "
          f"cell_pre_ft_top1={raw.get('pre_ft_top1')}")

    # 3. Snapshot masks (zero pattern of student weights)
    masks = collect_masks(student)
    total_nnz = sum(int(m.sum().item()) for m in masks.values())
    total_params = sum(int(m.numel()) for m in masks.values())
    sparsity_avg = 1.0 - total_nnz / max(1, total_params)
    print(f"  collected masks for {len(masks)} layers, avg sparsity={sparsity_avg:.4f}")

    # 4. Eval baselines
    print("Eval baselines...")
    student.eval()
    pre_ft_top1 = evaluate_top1(student, val_loader, device)
    teacher_top1 = evaluate_top1(teacher, val_loader, device)
    print(f"  pre-FT student top1 = {pre_ft_top1:.4f}")
    print(f"  teacher top1        = {teacher_top1:.4f}")

    # 5. Optimizer
    optim = torch.optim.AdamW(student.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * len(train_loader))

    # 6. FT loop
    epoch_records = []
    print(f"Starting {args.epochs}-ep distill FT, alpha={args.distill_alpha}, T={args.distill_temp}, "
          f"steps/ep={len(train_loader)}")
    for epoch in range(1, args.epochs + 1):
        student.train()
        t_ep = time.time()
        cum_loss = 0.0; cum_correct = 0; cum_total = 0; n_step = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                t_logits = teacher(imgs)
            s_logits = student(imgs)

            T_temp = args.distill_temp
            kd = F.kl_div(F.log_softmax(s_logits / T_temp, dim=1),
                          F.softmax(t_logits / T_temp, dim=1),
                          reduction="batchmean") * (T_temp ** 2)
            ce = F.cross_entropy(s_logits, labels)
            loss = args.distill_alpha * kd + (1.0 - args.distill_alpha) * ce

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            reapply_masks(student, masks)  # mask freezing

            with torch.no_grad():
                cum_loss += float(loss.item())
                cum_correct += int((s_logits.argmax(dim=1) == labels).sum().item())
                cum_total += imgs.size(0)
                n_step += 1
            if n_step % args.log_every == 0:
                lr_now = optim.param_groups[0]["lr"]
                print(f"  ep{epoch}/{args.epochs} step={n_step}/{len(train_loader)} "
                      f"loss={cum_loss/n_step:.4f} acc={100*cum_correct/cum_total:.2f} "
                      f"lr={lr_now:.2e}", flush=True)

        # Per-epoch eval
        student.eval()
        post_ep_top1 = evaluate_top1(student, val_loader, device)
        rec = {
            "epoch": epoch,
            "train_loss": cum_loss / max(1, n_step),
            "train_acc_running": cum_correct / max(1, cum_total),
            "val_top1": post_ep_top1,
            "elapsed_s": time.time() - t_ep,
        }
        epoch_records.append(rec)
        print(f"  EPOCH {epoch}: train_loss={rec['train_loss']:.4f}  val_top1={post_ep_top1:.4f}  "
              f"elapsed={rec['elapsed_s']:.0f}s")

        if args.save_every_epoch:
            torch.save({"state_dict": student.state_dict(), "epoch": epoch,
                        "post_ft_top1": post_ep_top1,
                        "cell_meta": {"cell": raw.get("cell"), "k": raw.get("k"),
                                       "n": raw.get("n"), "source": raw.get("source"),
                                       "alpha_ser": raw.get("alpha_ser")}},
                       out / f"student_ep{epoch}.pt")

    final_top1 = epoch_records[-1]["val_top1"]
    final = {
        "model": args.timm_name,
        "student_ckpt": args.student_ckpt,
        "cell_meta": {"cell": raw.get("cell"), "k": raw.get("k"),
                      "n": raw.get("n"), "source": raw.get("source"),
                      "alpha_ser": raw.get("alpha_ser")},
        "pre_ft_top1": pre_ft_top1,
        "teacher_top1": teacher_top1,
        "post_ft_top1": final_top1,
        "delta_pre_to_post_pp": (final_top1 - pre_ft_top1) * 100.0,
        "delta_vs_teacher_pp": (final_top1 - teacher_top1) * 100.0,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "weight_decay": args.weight_decay,
        "distill_temp": args.distill_temp,
        "distill_alpha": args.distill_alpha,
        "n_layers_with_mask": len(masks),
        "ckpt_avg_sparsity": sparsity_avg,
        "epoch_records": epoch_records,
    }
    (out / "final_eval.json").write_text(json.dumps(final, indent=2))
    torch.save({"state_dict": student.state_dict(), "post_ft_top1": final_top1,
                "cell_meta": final["cell_meta"]}, out / "student_final.pt")
    print(f"\n=== FT COMPLETE ===  pre={pre_ft_top1:.4f} → post={final_top1:.4f}  "
          f"(Δ={final_top1-pre_ft_top1:+.4f}, vs teacher={final_top1-teacher_top1:+.4f})")
    print(f"Saved: {out}/final_eval.json, student_final.pt")


if __name__ == "__main__":
    main()
