"""
run_resnet_cast_aws.py — single-file ResNet CAST-2E driver for AWS.

End-to-end:
  1. Load timm ResNet model with pretrained ImageNet weights (the dense teacher).
  2. Load SER s=0.35 student checkpoint (the unstructured-mask source).
  3. Apply Conv2d 2:4 projection (`project_conv_2_4`) — magnitude or cert-aware.
  4. Apply Linear 2:4 projection on the FC head (using the existing pattern).
  5. Distill-fine-tune for N epochs with the dense teacher.
  6. Evaluate top-1 on ImageNet val.
  7. Report sparsity + FLOP-only MAC counts (no Conv2d hardware-kernel speedup
     yet — that's deferred to a follow-up).

Designed to run on a single g5.xlarge or g6.xlarge spot instance in ~2-4 hours
per ResNet variant. Total budget for the 3 ResNet rows (resnet50, resnet50d,
resnet101d) ≈ $5-12 of the $160 AWS credit.

Usage:
    python run_resnet_cast_aws.py \\
        --timm-name resnet50.tv_in1k \\
        --ser-checkpoint /workspace/sweep_ckpts/resnet50.tv_in1k_keep_s35.pt \\
        --imagenet-root /workspace/imagenet \\
        --output-dir /workspace/cast_resnet/resnet50_$(date -u +%Y%m%dT%H%M%SZ) \\
        --epochs 3 \\
        --batch-size 64 \\
        --lr 1e-5 \\
        --method cert_aware \\
        --n-calib-imgs 256 \\
        --s3-backup-bucket cast-resnet-backup-973584726484
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local module
sys.path.insert(0, str(Path(__file__).parent))
from project_conv_2_4 import (
    magnitude_2_4_for_conv,
    cert_aware_2_4_for_conv,
    list_eligible_convs,
    merge_linear_and_conv_stats,
    collect_nonzero_masks,
    apply_masks,
    freeze_grad_at_masked,
    assert_2_4_legality,
)
from mac_counter import count_macs, sparse_exec_mac_estimate


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def s3_sync(local_dir: Path, bucket: str | None) -> None:
    if not bucket:
        return
    s3_uri = f"s3://{bucket}/{local_dir.name}/"
    log(f"S3 sync {local_dir} -> {s3_uri}")
    try:
        subprocess.run(
            ["aws", "s3", "sync", str(local_dir), s3_uri, "--quiet"],
            check=False, timeout=300,
        )
    except Exception as e:
        log(f"S3 sync failed: {e}")


def build_imagenet_loaders(root: Path, batch_train: int, batch_val: int,
                            num_workers: int = 4, image_size: int = 224,
                            mean: tuple = (0.485, 0.456, 0.406),
                            std: tuple = (0.229, 0.224, 0.225),
                            interpolation: str = "bilinear"):
    """Build standard ImageNet train + val loaders. Expects
       root/train/<class>/*.JPEG and root/val/<class>/*.JPEG layout.
       Uses the model's own data_config (image_size, mean, std) so that
       MAC counts and accuracy numbers are comparable to the timm baseline."""
    import torchvision.transforms as T
    import torchvision.datasets as D
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise RuntimeError(
            f"ImageFolder layout required at {root}/train and {root}/val. "
            f"Run aws_setup.sh which extracts the data into the expected layout."
        )
    norm = T.Normalize(mean=list(mean), std=list(std))
    tx_train = T.Compose([T.RandomResizedCrop(image_size), T.RandomHorizontalFlip(),
                          T.ToTensor(), norm])
    resize_short = int(image_size / 0.875)  # standard center-crop convention
    tx_val = T.Compose([T.Resize(resize_short), T.CenterCrop(image_size),
                        T.ToTensor(), norm])
    train_ds = D.ImageFolder(str(train_dir), transform=tx_train)
    val_ds = D.ImageFolder(str(val_dir), transform=tx_val)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_train, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_val, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_calibration_loader(root: Path, batch: int, num_workers: int = 4,
                              image_size: int = 224,
                              mean: tuple = (0.485, 0.456, 0.406),
                              std: tuple = (0.229, 0.224, 0.225)):
    """Deterministic calibration loader for the CAST pre-FT pattern search.
    NO RandomResizedCrop, NO horizontal flip, NO shuffle — just resize+center
    crop. This makes the cert-aware mask selection reproducible run-to-run."""
    import torchvision.transforms as T
    import torchvision.datasets as D
    train_dir = root / "train"
    if not train_dir.exists():
        raise RuntimeError(f"calibration data missing at {train_dir}")
    norm = T.Normalize(mean=list(mean), std=list(std))
    resize_short = int(image_size / 0.875)
    tx = T.Compose([T.Resize(resize_short), T.CenterCrop(image_size),
                    T.ToTensor(), norm])
    ds = D.ImageFolder(str(train_dir), transform=tx)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )


def load_ser_with_coverage_check(student: nn.Module, ser_path: str,
                                  min_coverage: float = 0.95) -> float:
    """Load SER source state_dict, asserting >=min_coverage of student tensor mass
    is matched. Strips common prefixes (`module.`)."""
    raw = torch.load(ser_path, map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict") or raw.get("state_dict") or raw
    if not isinstance(state, dict):
        raise RuntimeError(f"unrecognized checkpoint format at {ser_path}")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model_sd = student.state_dict()
    loaded_numel = 0
    total_numel = 0
    matched_keys = []
    for k, v in model_sd.items():
        total_numel += v.numel()
        if k in state and tuple(state[k].shape) == tuple(v.shape):
            loaded_numel += v.numel()
            matched_keys.append(k)
    coverage = loaded_numel / max(1, total_numel)
    log(f"  SER load: matched {len(matched_keys)}/{len(model_sd)} keys, "
        f"tensor coverage = {coverage:.4f}")
    if coverage < min_coverage:
        raise RuntimeError(f"SER ckpt load coverage {coverage:.4f} < {min_coverage}")
    student.load_state_dict(state, strict=False)
    return coverage


def evaluate_top1(model: nn.Module, val_loader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def project_linear_head_2_4_magnitude(model: nn.Module) -> dict:
    """Apply 2-of-4 magnitude masking on the FC head (mirrors the existing
    Linear-path behavior in the v11_pod_debug code)."""
    head = None
    head_name = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.shape[0] == 1000:
            head = m; head_name = n; break
    if head is None:
        return {"layers": {}, "linear_params": 0, "linear_nonzero_before": 0,
                "linear_nonzero_after": 0, "groups": 0,
                "groups_with_more_than_2_nonzero_after": 0,
                "linear_sparsity_before": 0.0, "linear_sparsity_after": 0.0,
                "mode": "linear_head_skipped_no_1000_class_layer"}
    W = head.weight.data
    Cout, Cin = W.shape
    cols_used = (Cin // 4) * 4
    Wleft = W[:, :cols_used]
    Wg = Wleft.reshape(Cout, cols_used // 4, 4)
    abs_g = Wg.abs()
    top2 = abs_g.topk(2, dim=-1).indices
    mask = torch.zeros_like(Wg)
    mask.scatter_(-1, top2, 1.0)
    Wm = (Wg * mask).reshape(Cout, cols_used)
    W_new = W.clone()
    W_new[:, :cols_used] = Wm
    head.weight.data.copy_(W_new)
    nnz_before = int((W != 0).sum().item())
    nnz_after = int((head.weight.data != 0).sum().item())
    return {
        "mode": "linear_head_magnitude_2_4",
        "include_head": True,
        "layers": {head_name: {
            "shape": list(W.shape),
            "params": W.numel(),
            "nonzero_before": nnz_before,
            "nonzero_after": nnz_after,
            "sparsity_before": 1 - nnz_before / W.numel(),
            "sparsity_after": 1 - nnz_after / W.numel(),
            "groups": (cols_used // 4) * Cout,
            "bad_groups_after": 0,
        }},
        "linear_params": W.numel(),
        "linear_nonzero_before": nnz_before,
        "linear_nonzero_after": nnz_after,
        "groups": (cols_used // 4) * Cout,
        "groups_with_more_than_2_nonzero_after": 0,
        "linear_sparsity_before": 1 - nnz_before / W.numel(),
        "linear_sparsity_after": 1 - nnz_after / W.numel(),
    }


def _build_manifest(args, image_size, mean, std, *,
                     dense_teacher_top1: float, ser_source_top1: float,
                     pre_ft_top1: float, post_ft_top1: float,
                     mac_summary: dict, merged_stats: dict, ser_coverage: float,
                     out_dir, start_ts: str) -> str:
    """Canonical YAML manifest for a single run. Required by run_all_resnets_aws.sh
    skip-detection and by paper-supplement reproducibility checklist."""
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).parent),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        pass
    end_ts = datetime.now(timezone.utc).isoformat()
    bad_groups = int(merged_stats.get("groups_with_more_than_2_nonzero_after", 0))
    lines = [
        f"# CAST-2E ResNet run manifest",
        f"run_id: {Path(out_dir).name}",
        f"git_commit: {git_commit}",
        f"engine: tradeswarm.cast_2e_resnet  # cast_2e_resnet_review/run_resnet_cast_aws.py",
        f"engine_version: 0.3.0",
        f"ts_started: {start_ts}",
        f"ts_completed: {end_ts}",
        f"image_size: {image_size}",
        f"data_inputs:",
        f"  ser_checkpoint:",
        f"    path: {args.ser_checkpoint}",
        f"    load_coverage_fraction: {ser_coverage:.6f}",
        f"  imagenet_root: {args.imagenet_root}",
        f"parameters:",
        f"  timm_name: {args.timm_name}",
        f"  method: {args.method}",
        f"  include_3x3_convs: {bool(args.include_3x3_convs)}",
        f"  free_restoration: {bool(not args.no_free_restore)}",
        f"  skip_head: {bool(args.skip_head)}",
        f"  epochs: {args.epochs}",
        f"  batch_size_train: {args.batch_size}",
        f"  batch_size_val: {args.batch_size_val}",
        f"  lr: {args.lr}",
        f"  distill_alpha: {args.distill_alpha}",
        f"  distill_temp: {args.distill_temp}",
        f"  label_smoothing: {args.label_smoothing}",
        f"  n_calib_imgs: {args.n_calib_imgs}",
        f"mac_report:",
        f"  dense_total_gmacs:    {mac_summary['dense_total_gmacs']:.6f}",
        f"  eligible_layer_count: {mac_summary['eligible_layer_count']}",
        f"  eligible_gmacs:       {mac_summary['eligible_gmacs']:.6f}",
        f"  eligible_fraction:    {mac_summary['eligible_fraction']:.6f}",
        f"  sparse_exec_gmacs:    {mac_summary['sparse_exec_total_gmacs']:.6f}",
        f"  mac_reduction_fraction: {mac_summary['mac_reduction_fraction']:.6f}",
        f"sparsity_after:",
        f"  conv_layers_modified: {merged_stats.get('n_layers_modified', 0)}",
        f"  conv_nnz_before: {merged_stats.get('linear_nonzero_before', 0)}",
        f"  conv_nnz_after:  {merged_stats.get('linear_nonzero_after', 0)}",
        f"  conv_sparsity_after: {merged_stats.get('linear_sparsity_after', 0):.6f}",
        f"  bad_groups_after: {bad_groups}",
        f"two_four_legality_check_after_each_epoch: PASS  # asserted in training loop, run aborts on failure",
        f"eval:",
        f"  dense_teacher_top1: {dense_teacher_top1:.6f}",
        f"  ser_source_top1:    {ser_source_top1:.6f}",
        f"  pre_ft_top1:        {pre_ft_top1:.6f}",
        f"  post_ft_top1:       {post_ft_top1:.6f}",
        f"  delta_vs_dense_teacher: {post_ft_top1 - dense_teacher_top1:+.6f}",
        f"  delta_vs_ser_source:    {post_ft_top1 - ser_source_top1:+.6f}",
        f"  delta_vs_pre_ft:        {post_ft_top1 - pre_ft_top1:+.6f}",
        f"acceptance:",
        f"  bad_groups_zero: {bad_groups == 0}",
        f"  ser_coverage_ge_95: {ser_coverage >= 0.95}",
        f"  post_ft_within_05pp_of_dense: {abs(post_ft_top1 - dense_teacher_top1) <= 0.005}",
        f"  status: {'PASS' if (bad_groups == 0 and ser_coverage >= 0.95 and abs(post_ft_top1 - dense_teacher_top1) <= 0.005) else 'REVIEW'}",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm-name", required=True)
    ap.add_argument("--ser-checkpoint", required=True, help="path to SER s=0.35 ckpt")
    ap.add_argument("--imagenet-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--batch-size-val", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--distill-alpha", type=float, default=0.5)
    ap.add_argument("--method", choices=["magnitude", "cert_aware"], default="cert_aware")
    ap.add_argument("--n-calib-imgs", type=int, default=256)
    ap.add_argument("--include-3x3-convs", action="store_true",
                    help="extend CAST to 3x3 convs (push reduction from ~25%% to ~50%%)")
    ap.add_argument("--no-free-restore", action="store_true",
                    help="ablation A3: disable slot_values free restoration. The 4-tuple "
                         "search uses only the SER-pruned weights, never the dense "
                         "pretrained values. Quantifies what free restoration buys.")
    ap.add_argument("--permute-align", action="store_true",
                    help="variant B: before 2:4 selection, compute an importance-aware "
                         "Cin permutation per layer that interleaves RMT-signal columns "
                         "across 4-tuples (so each 4-tuple has at least one signal column "
                         "the 2:4 mask can keep). Free at runtime via a forward pre-hook. "
                         "See PERMUTATION_ALIGN_DESIGN.md.")
    ap.add_argument("--skip-head", action="store_true", default=True,
                    help="skip 2:4 on the FC head (negligible MACs, default skip per Codex review)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--s3-backup-bucket", default="")
    ap.add_argument("--s3-backup-every-min", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_start_ts = datetime.now(timezone.utc).isoformat()
    log(f"output_dir={out_dir}  device={device}")

    # ---- Load models ----
    import timm
    log(f"Loading dense teacher: {args.timm_name} (pretrained)")
    teacher = timm.create_model(args.timm_name, pretrained=True).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    # Read the model's own data config (image_size, mean, std) — DO NOT hardcode 224
    data_cfg = timm.data.resolve_model_data_config(teacher)
    image_size = data_cfg["input_size"][-1]
    mean = data_cfg["mean"]; std = data_cfg["std"]
    log(f"  data_config: image_size={image_size}  mean={mean}  std={std}")

    log(f"Loading student from SER ckpt: {args.ser_checkpoint}")
    student = timm.create_model(args.timm_name, pretrained=False).to(device)
    coverage = load_ser_with_coverage_check(student, args.ser_checkpoint, min_coverage=0.95)
    student = student.to(device)

    # Save dense teacher copy (small, useful for resume)
    torch.save({"state_dict": teacher.state_dict()},
               out_dir / "teacher_dense.pt")

    # ---- Baseline evals BEFORE projection (per Codex review) ----
    # Need both dense_teacher_top1 and ser_source_top1 so the paper can
    # claim "post-FT top1 maintained vs DENSE", not just vs the projected
    # student (which has already lost some accuracy).
    log("Building val loader for baseline evals")
    _, val_loader_baseline = build_imagenet_loaders(
        Path(args.imagenet_root), args.batch_size, args.batch_size_val,
        args.num_workers, image_size=image_size, mean=mean, std=std,
    )
    log("Eval 1/2: dense teacher top1 (sanity vs. timm-published baseline)")
    dense_teacher_top1 = evaluate_top1(teacher, val_loader_baseline, device)
    log(f"  dense teacher top1 = {dense_teacher_top1:.4f}")
    log("Eval 2/2: SER source top1 (BEFORE 2:4 projection)")
    ser_source_top1 = evaluate_top1(student, val_loader_baseline, device)
    log(f"  SER source top1 = {ser_source_top1:.4f}")
    (out_dir / "baseline_eval.json").write_text(json.dumps({
        "dense_teacher_top1": dense_teacher_top1,
        "ser_source_top1": ser_source_top1,
        "ser_load_coverage_fraction": coverage,
        "ts": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    # ---- Apply 2:4 projection on Conv2d ----
    only_1x1 = not args.include_3x3_convs
    dense_state = dict(teacher.named_parameters())
    log(f"Eligible Conv2d layers (only_1x1={only_1x1}): "
        f"{len(list_eligible_convs(student, only_1x1=only_1x1))}")

    # Build the deterministic calibration loader once (used by cert-aware projection)
    if args.method == "cert_aware":
        log(f"Building DETERMINISTIC calibration loader (resize+center-crop, no aug, no shuffle)")
        calib_loader = build_calibration_loader(
            Path(args.imagenet_root), args.batch_size,
            num_workers=args.num_workers,
            image_size=image_size, mean=mean, std=std,
        )

    free_restoration = not args.no_free_restore
    if args.method == "magnitude":
        log(f"Conv 2:4 projection: magnitude  (only_1x1={only_1x1}  free_restore={free_restoration})")
        conv_stats = magnitude_2_4_for_conv(
            student, dense_state_dict=dense_state if free_restoration else None,
            free_restoration=free_restoration,
            only_1x1=only_1x1, log=True,
        )
    else:
        log(f"Conv 2:4 projection: cert-aware (covariance form)  "
            f"only_1x1={only_1x1}  free_restore={free_restoration}  "
            f"permute_align={args.permute_align}  calib={args.n_calib_imgs}")
        conv_stats = cert_aware_2_4_for_conv(
            student, calib_loader, dense_state_dict=dense_state if free_restoration else None,
            n_calib_imgs=args.n_calib_imgs, device=device,
            free_restoration=free_restoration, only_1x1=only_1x1,
            permute_align=args.permute_align, log=True,
        )

    # ---- Skip Linear head per Codex review (head MACs are ~0.1% of eligible) ----
    head_stats = {"layers": {}, "linear_params": 0, "linear_nonzero_before": 0,
                  "linear_nonzero_after": 0, "groups": 0,
                  "groups_with_more_than_2_nonzero_after": 0,
                  "linear_sparsity_before": 0.0, "linear_sparsity_after": 0.0,
                  "mode": "linear_head_skipped_per_codex_review_negligible_macs"}
    if not args.skip_head:
        log("Linear head 2:4 projection (magnitude)")
        head_stats = project_linear_head_2_4_magnitude(student)
    else:
        log("Skipping Linear head 2:4 projection (head MACs ~0.1% of eligible)")
    merged_stats = merge_linear_and_conv_stats(head_stats, conv_stats)
    (out_dir / "two_four_stats.json").write_text(
        json.dumps(merged_stats, indent=2, default=float))
    log(f"Net post-projection sparsity (eligible only): "
        f"{merged_stats['linear_sparsity_after']:.3f}  "
        f"({merged_stats['linear_nonzero_after']:,} of {merged_stats['linear_params']:,})")

    # ---- ASSERT 2:4 LEGALITY immediately after projection ----
    log("Asserting 2:4 legality of every projected layer...")
    legality = assert_2_4_legality(student, only_1x1=only_1x1,
                                    include_linear_head=not args.skip_head)
    log(f"  legality OK: {len(legality)} layers, all groups have exactly 2 NNZ")

    # ---- Snapshot masks NOW for FT freezing ----
    masks = collect_nonzero_masks(student, include_linear=not args.skip_head, only_1x1=only_1x1)
    log(f"  collected {len(masks)} layer masks for FT freezing")

    # ---- Exact hook-based MAC count (paper-ready numbers) ----
    log("Counting MACs via exact hook-based forward pass at 224x224")
    mac_report = count_macs(timm.create_model(args.timm_name, pretrained=False),
                            image_size=image_size, device=device)
    eligible_layer_names = list(merged_stats["layers"].keys())
    sparse_report = sparse_exec_mac_estimate(mac_report, eligible_layer_names)
    mac_summary = {
        "model": args.timm_name,
        "image_size": 224,
        "dense_total_gmacs": mac_report.dense_total / 1e9,
        "eligible_layer_count": len(eligible_layer_names),
        "eligible_gmacs": sparse_report.eligible_macs / 1e9,
        "eligible_fraction": sparse_report.eligible_fraction,
        "sparse_exec_total_gmacs": sparse_report.sparse_exec_estimate / 1e9,
        "mac_reduction_fraction": 1 - sparse_report.sparse_exec_estimate / sparse_report.dense_total,
        "by_layer": {n: {"type": L["type"], "macs": L["macs"], "eligible": L["eligible"],
                          "sparse_macs": L["sparse_macs"]}
                     for n, L in sparse_report.by_layer.items()},
    }
    (out_dir / "mac_report.json").write_text(json.dumps(mac_summary, indent=2, default=int))
    log(f"  dense:        {mac_summary['dense_total_gmacs']:>6.3f} GMACs")
    log(f"  eligible:     {mac_summary['eligible_layer_count']:>3} layers, "
        f"{mac_summary['eligible_gmacs']:>6.3f} GMACs ({mac_summary['eligible_fraction']*100:>4.1f}% of total)")
    log(f"  sparse-exec:  {mac_summary['sparse_exec_total_gmacs']:>6.3f} GMACs "
        f"({mac_summary['mac_reduction_fraction']*100:>4.1f}% reduction)")

    # ---- Build train + val loaders for FT (DIFFERENT from calibration loader: this
    #      one has RandomResizedCrop + flip + shuffle for training augmentation) ----
    log(f"Building FT train+val loaders (RandomResizedCrop+flip aug for train)")
    train_loader, val_loader = build_imagenet_loaders(
        Path(args.imagenet_root), args.batch_size, args.batch_size_val,
        args.num_workers, image_size=image_size, mean=mean, std=std,
    )

    log("Pre-FT eval on ImageNet val")
    pre_top1 = evaluate_top1(student, val_loader, device)
    log(f"  pre-FT top1 = {pre_top1:.4f}")
    (out_dir / "pre_ft_eval.json").write_text(
        json.dumps({"top1": pre_top1, "ts": datetime.now(timezone.utc).isoformat()},
                   indent=2))

    torch.save({"state_dict": student.state_dict()}, out_dir / "student_pre_ft.pt")
    s3_sync(out_dir, args.s3_backup_bucket)

    # ---- Distill fine-tune ----
    optim = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * len(train_loader))
    log(f"Starting {args.epochs}-epoch distill FT  "
        f"lr={args.lr}  alpha={args.distill_alpha}  T={args.distill_temp}")

    last_backup = time.time()
    for epoch in range(1, args.epochs + 1):
        student.train()
        t_start = time.time()
        cum_loss, cum_correct, cum_total = 0.0, 0, 0
        for step, (imgs, labels) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                t_logits = teacher(imgs)
            s_logits = student(imgs)
            T = args.distill_temp
            kd = F.kl_div(F.log_softmax(s_logits / T, dim=1),
                          F.softmax(t_logits / T, dim=1),
                          reduction="batchmean") * (T * T)
            ce = F.cross_entropy(s_logits, labels,
                                 label_smoothing=args.label_smoothing)
            loss = args.distill_alpha * kd + (1 - args.distill_alpha) * ce
            optim.zero_grad()
            loss.backward()
            # FREEZE 2:4 MASKS: zero gradient at masked-out positions BEFORE optimizer.step()
            freeze_grad_at_masked(student, masks)
            optim.step()
            # RE-APPLY MASKS in-place AFTER optimizer.step() — guards against any
            # weight-decay drift, momentum carry, or numerical noise.
            apply_masks(student, masks)
            sched.step()
            cum_loss += loss.item() * imgs.size(0)
            cum_correct += (s_logits.argmax(dim=1) == labels).sum().item()
            cum_total += imgs.size(0)
            if step % 100 == 0 or step == len(train_loader):
                cur_lr = sched.get_last_lr()[0]
                log(f"  train ep={epoch}/{args.epochs} step={step}/{len(train_loader)}  "
                    f"loss={cum_loss/cum_total:.4f}  "
                    f"acc={cum_correct/cum_total*100:.2f}  lr={cur_lr:.2e}")
            # Step-level checkpoint every 500 steps so a spot interruption
            # costs at most ~5 min of work, not a whole epoch
            if step % 500 == 0:
                step_ckpt = out_dir / "checkpoints" / "latest_step.pt"
                step_ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": student.state_dict(),
                    "optim": optim.state_dict(),
                    "sched": sched.state_dict(),
                    "epoch": epoch,
                    "step": step,
                }, step_ckpt)
            # S3 backup every N min
            if args.s3_backup_bucket and time.time() - last_backup > args.s3_backup_every_min * 60:
                s3_sync(out_dir, args.s3_backup_bucket)
                last_backup = time.time()
        ep_minutes = (time.time() - t_start) / 60.0
        log(f"  epoch {epoch} done. loss={cum_loss/cum_total:.4f}  "
            f"acc={cum_correct/cum_total*100:.2f}  minutes={ep_minutes:.1f}")
        # ASSERT 2:4 LEGALITY after every epoch — guarantees the masks held
        try:
            assert_2_4_legality(student, only_1x1=only_1x1,
                                 include_linear_head=not args.skip_head)
            log(f"  legality OK after epoch {epoch}")
        except AssertionError as e:
            log(f"  ERROR: 2:4 legality broke after epoch {epoch}: {e}")
            raise
        ckpt_path = out_dir / "checkpoints" / f"epoch{epoch}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": student.state_dict(), "epoch": epoch}, ckpt_path)
        s3_sync(out_dir, args.s3_backup_bucket)

    # ---- Final eval ----
    log("Final eval on ImageNet val")
    post_top1 = evaluate_top1(student, val_loader, device)
    log(f"  post-FT top1 = {post_top1:.4f}  "
        f"(delta_vs_dense_teacher: {post_top1 - dense_teacher_top1:+.4f}  "
        f"delta_vs_ser_source: {post_top1 - ser_source_top1:+.4f}  "
        f"delta_vs_pre_ft: {post_top1 - pre_top1:+.4f})")
    (out_dir / "final_eval.json").write_text(json.dumps({
        "model": args.timm_name,
        "method": args.method,
        "epochs": args.epochs,
        "include_3x3_convs": bool(args.include_3x3_convs),
        "free_restoration": bool(not args.no_free_restore),
        "skip_head": bool(args.skip_head),
        # Top-1 numbers — the meaningful ones for the paper:
        "dense_teacher_top1": dense_teacher_top1,   # what the dense pretrained timm model scores
        "ser_source_top1":    ser_source_top1,      # what the SER s=0.35 ckpt scores BEFORE projection
        "pre_ft_top1":        pre_top1,             # post-projection, pre-FT
        "post_ft_top1":       post_top1,            # post-projection, post-FT  (the headline number)
        "delta_vs_dense":     post_top1 - dense_teacher_top1,
        "delta_vs_ser":       post_top1 - ser_source_top1,
        "delta_vs_pre_ft":    post_top1 - pre_top1,
        "two_four_sparsity":  merged_stats['linear_sparsity_after'],
        "ser_load_coverage_fraction": coverage,
        "ts":                 datetime.now(timezone.utc).isoformat(),
    }, indent=2, default=float))

    # ---- Manifest (canonical run artifact, per Codex review) ----
    log("Writing manifest.yaml")
    manifest_text = _build_manifest(args, image_size, mean, std,
                                     dense_teacher_top1=dense_teacher_top1,
                                     ser_source_top1=ser_source_top1,
                                     pre_ft_top1=pre_top1,
                                     post_ft_top1=post_top1,
                                     mac_summary=mac_summary,
                                     merged_stats=merged_stats,
                                     ser_coverage=coverage,
                                     out_dir=out_dir,
                                     start_ts=run_start_ts)
    (out_dir / "manifest.yaml").write_text(manifest_text, encoding="utf-8")
    log(f"  manifest.yaml -> {out_dir / 'manifest.yaml'}")

    s3_sync(out_dir, args.s3_backup_bucket)
    log("DONE.")


if __name__ == "__main__":
    main()
