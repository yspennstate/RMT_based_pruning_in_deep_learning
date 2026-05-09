"""
run_vitb_ft_inline.py — runs cert-aware k:n projection AND 3-ep distillation FT
in the same process, so the permutation forward-pre-hooks are never lost.

Use this when the cert projection uses --permute-align=True. The permutation
information is encoded as a forward-pre-hook on each layer that re-orders
the input channels — these hooks are NOT serialized in state_dict, so saving
+ reloading produces a non-functional ckpt (catastrophic accuracy drop).

This script avoids that by doing projection + FT in one process.

Usage:
    python run_vitb_ft_inline.py \\
        --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k \\
        --ser-checkpoint /workspace/sweep_ckpts/vit_base_patch16_224.augreg2_in21k_ft_in1k_keep_s35_classmag.pt \\
        --imagenet-train /workspace/imagenet/train \\
        --imagenet-val /workspace/imagenet/val \\
        --output-dir /workspace/run_outputs/vitb_ft/D612_ser_a05_inline \\
        --k 6 --n 12 --source ser --alpha-ser 0.5 \\
        --epochs 3 --batch 256 --lr 5e-5
"""
import torch as _t; _t.backends.cudnn.enabled = False

import argparse, json, time, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from project_kn_sparsity import cert_aware_kn_for_conv, cert_aware_kn_for_linear


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
    """Snapshot the zero-mask of every Conv/Linear weight that has any zeros."""
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


def save_perm_meta(model, save_path):
    """Save per-layer _cin_perm tensors so reload code can re-register hooks."""
    perm_dict = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "_cin_perm"):
            perm_dict[name] = mod._cin_perm.cpu().tolist()
    Path(save_path).write_text(json.dumps(perm_dict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm-name", required=True)
    ap.add_argument("--ser-checkpoint", required=True)
    ap.add_argument("--imagenet-train", required=True)
    ap.add_argument("--imagenet-val", required=True)
    ap.add_argument("--imagenet-train-for-calib", default=None,
                    help="defaults to --imagenet-val")
    ap.add_argument("--output-dir", required=True)
    # Cert k:n config
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--source", choices=["dense", "ser"], required=True)
    ap.add_argument("--alpha-ser", type=float, default=0.0)
    ap.add_argument("--permute-align", action="store_true", default=True)
    ap.add_argument("--no-permute-align", dest="permute_align", action="store_false")
    ap.add_argument("--free-restoration", action="store_true", default=True)
    ap.add_argument("--pipeline", choices=["conv", "linear"], default="linear")
    # FT config
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--distill-alpha", type=float, default=0.5)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every-epoch", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[ft_inline] device={device}, model={args.timm_name}")
    print(f"  cert: k:n={args.k}:{args.n}, source={args.source}, alpha_ser={args.alpha_ser}, "
          f"perm={args.permute_align}, free_restore={args.free_restoration}, pipeline={args.pipeline}")
    print(f"  ft: epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
    import timm

    # 1. Build dense teacher
    teacher = timm.create_model(args.timm_name, pretrained=True).to(device).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(teacher)
    img_size = cfg["input_size"][-1]; mean = cfg["mean"]; std = cfg["std"]
    crop_pct = cfg.get("crop_pct", 0.875); interp = cfg.get("interpolation", "bicubic")
    interp_map = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC}
    interp_mode = interp_map.get(interp.lower(), T.InterpolationMode.BICUBIC)

    # 2. Loaders (val + train + calib) using timm's canonical transforms.
    # train: RandAug-m9 + RandomErasing matches the augreg2 recipe the teacher was trained on.
    # val: deterministic resize + center crop with the model's exact crop_pct + interpolation.
    # calib: same as val (deterministic) so cert masks are reproducible.
    val_tx = timm.data.create_transform(
        input_size=img_size, is_training=False, mean=mean, std=std,
        crop_pct=crop_pct, interpolation=interp,
    )
    train_tx = timm.data.create_transform(
        input_size=img_size, is_training=True, mean=mean, std=std,
        interpolation=interp,
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25, re_mode="pixel", re_count=1,
        hflip=0.5, color_jitter=0.4,
    )
    print(f"  train transform: {train_tx}")
    print(f"  val transform:   {val_tx}")
    val_ds = tv.datasets.ImageFolder(args.imagenet_val, transform=val_tx)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    calib_ds = tv.datasets.ImageFolder(args.imagenet_train_for_calib or args.imagenet_val,
                                        transform=val_tx)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=64, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True)
    train_ds = tv.datasets.ImageFolder(args.imagenet_train, transform=train_tx)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True,
                                                drop_last=True, persistent_workers=True)

    # 3. Build student: source = dense (timm pretrained) or SER (load ckpt)
    student = timm.create_model(args.timm_name, pretrained=(args.source == "dense")).to(device).eval()
    if args.source == "ser":
        load_ser(student, args.ser_checkpoint)
    dense_state = {n: p.detach().clone() for n, p in teacher.state_dict().items()}

    # 4. Run cert-aware k:n projection in-place (registers _cin_perm hooks if perm)
    print(f"\n=== Running cert-aware {args.k}:{args.n} projection ===")
    t0 = time.time()
    if args.pipeline == "conv":
        cert_stats = cert_aware_kn_for_conv(
            student, calib_loader, n=args.n, k=args.k,
            dense_state_dict=dense_state, n_calib_imgs=64,
            free_restoration=args.free_restoration, permute_align=args.permute_align,
            alpha_ser_prior=args.alpha_ser, log=False, device=device,
        )
    else:
        cert_stats = cert_aware_kn_for_linear(
            student, calib_loader, n=args.n, k=args.k,
            dense_state_dict=dense_state, n_calib_imgs=64,
            free_restoration=args.free_restoration, permute_align=args.permute_align,
            alpha_ser_prior=args.alpha_ser, log=False, device=device,
        )
    print(f"  projection done in {time.time()-t0:.0f}s, layers_modified={cert_stats.get('n_layers_modified',0)}")

    # 5. Eval pre-FT
    print("Eval baselines...")
    pre_ft_top1 = evaluate_top1(student, val_loader, device)
    teacher_top1 = evaluate_top1(teacher, val_loader, device)
    print(f"  pre-FT student top1 = {pre_ft_top1:.4f}")
    print(f"  teacher top1        = {teacher_top1:.4f}")
    if pre_ft_top1 < 0.10:
        raise RuntimeError(f"pre-FT eval too low ({pre_ft_top1:.4f}), projection may be broken")

    # 6. Snapshot masks (post-projection) and save perm meta
    masks = collect_masks(student)
    total_nnz = sum(int(m.sum().item()) for m in masks.values())
    total_params = sum(int(m.numel()) for m in masks.values())
    sparsity_avg = 1.0 - total_nnz / max(1, total_params)
    print(f"  collected masks for {len(masks)} layers, avg sparsity={sparsity_avg:.4f}")
    save_perm_meta(student, out / "perm_meta.json")

    # Save the projected student ckpt (also for benchmarking)
    torch.save({"state_dict": student.state_dict(),
                "cell": f"D{args.k}{args.n}_{args.source}_a{int(args.alpha_ser*100):02d}",
                "k": args.k, "n": args.n, "source": args.source, "alpha_ser": args.alpha_ser,
                "permute_align": args.permute_align, "free_restoration": args.free_restoration,
                "pre_ft_top1": pre_ft_top1, "_perm_hook_required": args.permute_align},
               out / "student_pre_ft.pt")

    # 7. Optimizer + warmup + cosine scheduler.
    # Use param-group separation: no weight decay on biases, LayerNorm, or pos_embed.
    decay, no_decay = [], []
    for n_, p in student.named_parameters():
        if not p.requires_grad: continue
        if p.ndim <= 1 or "pos_embed" in n_ or "cls_token" in n_:
            no_decay.append(p)
        else:
            decay.append(p)
    optim = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.999))
    total_steps = args.epochs * len(train_loader)
    warmup = args.warmup_steps
    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # 8. FT loop
    print(f"Starting {args.epochs}-ep distill FT, alpha={args.distill_alpha}, T={args.distill_temp}, "
          f"steps/ep={len(train_loader)}")
    epoch_records = []
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
            ce = F.cross_entropy(s_logits, labels, label_smoothing=args.label_smoothing)
            loss = args.distill_alpha * kd + (1.0 - args.distill_alpha) * ce

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            reapply_masks(student, masks)

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

        student.eval()
        post_ep_top1 = evaluate_top1(student, val_loader, device)
        rec = {"epoch": epoch, "train_loss": cum_loss / max(1, n_step),
               "train_acc_running": cum_correct / max(1, cum_total),
               "val_top1": post_ep_top1, "elapsed_s": time.time() - t_ep}
        epoch_records.append(rec)
        print(f"  EPOCH {epoch}: train_loss={rec['train_loss']:.4f}  val_top1={post_ep_top1:.4f}  "
              f"elapsed={rec['elapsed_s']:.0f}s")
        if args.save_every_epoch:
            torch.save({"state_dict": student.state_dict(), "epoch": epoch,
                        "post_ft_top1": post_ep_top1,
                        "k": args.k, "n": args.n, "source": args.source, "alpha_ser": args.alpha_ser,
                        "permute_align": args.permute_align,
                        "_perm_hook_required": args.permute_align},
                       out / f"student_ep{epoch}.pt")

    final_top1 = epoch_records[-1]["val_top1"]
    final = {
        "model": args.timm_name,
        "cert_config": {"k": args.k, "n": args.n, "source": args.source,
                         "alpha_ser": args.alpha_ser, "permute_align": args.permute_align,
                         "free_restoration": args.free_restoration, "pipeline": args.pipeline},
        "ft_config": {"epochs": args.epochs, "lr": args.lr, "batch": args.batch,
                      "weight_decay": args.weight_decay, "distill_temp": args.distill_temp,
                      "distill_alpha": args.distill_alpha},
        "pre_ft_top1": pre_ft_top1,
        "teacher_top1": teacher_top1,
        "post_ft_top1": final_top1,
        "delta_pre_to_post_pp": (final_top1 - pre_ft_top1) * 100.0,
        "delta_vs_teacher_pp": (final_top1 - teacher_top1) * 100.0,
        "ckpt_avg_sparsity": sparsity_avg,
        "n_layers_with_mask": len(masks),
        "epoch_records": epoch_records,
    }
    (out / "final_eval.json").write_text(json.dumps(final, indent=2))
    torch.save({"state_dict": student.state_dict(), "post_ft_top1": final_top1,
                "k": args.k, "n": args.n, "source": args.source, "alpha_ser": args.alpha_ser,
                "permute_align": args.permute_align,
                "_perm_hook_required": args.permute_align},
               out / "student_final.pt")
    print(f"\n=== FT COMPLETE ===  pre={pre_ft_top1:.4f} → post={final_top1:.4f}  "
          f"(Δ={final_top1-pre_ft_top1:+.4f}, vs teacher={final_top1-teacher_top1:+.4f})")
    print(f"Saved: {out}/final_eval.json, student_final.pt, perm_meta.json")


if __name__ == "__main__":
    main()
