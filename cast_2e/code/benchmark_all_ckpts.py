"""
benchmark_all_ckpts.py — benchmark a DIRECTORY of projected-student ckpts.

For each ckpt in --ckpts-dir:
  1. NNZ count (true effective sparsity, including SER pre-zeros if free_restore=False)
  2. Theoretical FLOP reduction = 1 - effective_NNZ_in_eligible_layers / dense_eligible_params
  3. Dense-kernel throughput (running sparse weights through standard GEMM)
  4. PyTorch 2:4 sparse-kernel throughput (where applicable; only pure-2:4 layers)
  5. cuSparseLt note: 4:8/8:16 hardware speedup not measurable through PyTorch
     native; we report theoretical speedup based on NNZ ratio.

Usage:
    python benchmark_all_ckpts.py \\
        --timm-name resnet50.tv_in1k \\
        --ckpts-dir /workspace/run_outputs/ckpts_all/resnet50 \\
        --output /workspace/run_logs/benchmarks/resnet50_all.json \\
        --batch 128 --warmup 20 --iters 100
"""
from __future__ import annotations
import torch as _t; _t.backends.cudnn.enabled = False

import argparse, json, time, sys, glob
from pathlib import Path

import torch
import torch.nn as nn


def benchmark_throughput(model: nn.Module, device: str, batch: int, image_size: int,
                          warmup: int = 20, iters: int = 100) -> dict:
    model.eval()
    x = torch.randn(batch, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.time()
            _ = model(x)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
    times.sort()
    median_s = times[len(times)//2]
    return {"batch": batch, "image_size": image_size, "median_s": median_s,
            "images_per_s": batch / median_s}


def count_nnz_eligible(model: nn.Module, k: int, n: int) -> dict:
    """Count NNZ across Conv2d/Linear layers eligible for k:n (Cin % n == 0)."""
    eligible_params = 0; eligible_nnz = 0
    total_params = 0; total_nnz = 0
    layer_stats = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                W = mod.weight.data
                p = int(W.numel())
                nnz = int((W != 0).sum().item())
            total_params += p; total_nnz += nnz
            if isinstance(mod, nn.Conv2d):
                cin = W.shape[1]
            else:
                cin = W.shape[1]
            if cin % n == 0:
                eligible_params += p; eligible_nnz += nnz
            layer_stats[name] = {"params": p, "nnz": nnz,
                                  "sparsity": 1.0 - nnz/max(1,p),
                                  "eligible_kn": (cin % n == 0)}
    return {
        "total_params": total_params, "total_nnz": total_nnz,
        "global_sparsity": 1.0 - total_nnz / max(1, total_params),
        "eligible_params": eligible_params, "eligible_nnz": eligible_nnz,
        "eligible_sparsity": 1.0 - eligible_nnz / max(1, eligible_params),
        "n_layers": len(layer_stats),
    }


def try_apply_2_4_sparsity(model: nn.Module) -> dict:
    converted = 0; tried = 0
    try:
        from torch.sparse import to_sparse_semi_structured
    except ImportError:
        return {"supported": False}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            with torch.no_grad():
                W = mod.weight.data
                Cout, Cin = W.shape
                if Cin % 16 != 0:
                    continue
                tried += 1
                Wg = W.reshape(Cout, Cin // 4, 4)
                if ((Wg != 0).sum(dim=-1) == 2).all():
                    try:
                        sparse_w = to_sparse_semi_structured(W.cuda())
                        mod.weight = nn.Parameter(sparse_w, requires_grad=False)
                        converted += 1
                    except Exception:
                        pass
    return {"supported": True, "tried": tried, "converted": converted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm-name", required=True)
    ap.add_argument("--ckpts-dir", required=True, help="dir with .pt files")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--image-size", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    print(f"[benchmark_all] timm={args.timm_name}, ckpts_dir={args.ckpts_dir}")
    import timm

    # Dense baseline (once)
    dense = timm.create_model(args.timm_name, pretrained=True).to(device).eval()
    cfg = timm.data.resolve_model_data_config(dense)
    img_size = args.image_size or cfg["input_size"][-1]
    dense_thru = benchmark_throughput(dense, device, args.batch, img_size,
                                       warmup=args.warmup, iters=args.iters)
    dense_total_params = sum(p.numel() for p in dense.parameters())
    print(f"  dense: {dense_thru['images_per_s']:.1f} im/s, {dense_total_params:,} params")
    del dense; torch.cuda.empty_cache()

    ckpt_files = sorted(glob.glob(f"{args.ckpts_dir}/*.pt"))
    print(f"  found {len(ckpt_files)} ckpts")

    out = {
        "model": args.timm_name,
        "dense_throughput": dense_thru,
        "dense_total_params": dense_total_params,
        "image_size": img_size,
        "batch": args.batch,
        "cells": [],
    }

    for ckpt_path in ckpt_files:
        label = Path(ckpt_path).stem
        print(f"\n--- {label} ---")
        try:
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = raw.get("state_dict") or raw
            k = raw.get("k", 0); n = raw.get("n", 4)
            cell_pre_ft = raw.get("pre_ft_top1", -1.0)
            source = raw.get("source"); alpha = raw.get("alpha_ser")

            # Dense-kernel throughput on sparse weights (sanity)
            m = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
            m.load_state_dict(sd, strict=False)
            nnz = count_nnz_eligible(m, k=k, n=n)
            thru_dense = benchmark_throughput(m, device, args.batch, img_size,
                                                warmup=args.warmup, iters=args.iters)
            del m; torch.cuda.empty_cache()

            # 2:4 sparse-kernel throughput (only meaningful when k=2, n=4)
            thru_24 = None
            apply_24 = None
            if k == 2 and n == 4:
                m24 = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
                m24.load_state_dict(sd, strict=False)
                apply_24 = try_apply_2_4_sparsity(m24)
                if apply_24.get("converted", 0) > 0:
                    thru_24 = benchmark_throughput(m24, device, args.batch, img_size,
                                                     warmup=args.warmup, iters=args.iters)
                del m24; torch.cuda.empty_cache()

            theoretical_flop_reduction = nnz["eligible_sparsity"]  # fraction of MACs eliminated in eligible layers
            speedup_24 = (thru_24["images_per_s"] / dense_thru["images_per_s"]
                          if thru_24 else None)

            cell_record = {
                "label": label, "ckpt": str(ckpt_path),
                "k": k, "n": n, "source": source, "alpha_ser": alpha,
                "cell_pre_ft_top1": cell_pre_ft,
                "nnz_stats": nnz,
                "theoretical_flop_reduction_eligible": theoretical_flop_reduction,
                "throughput_dense_kernel": thru_dense,
                "throughput_2_4_kernel": thru_24,
                "kernel_speedup_2_4_vs_dense": speedup_24,
                "_2_4_conversion": apply_24,
                "_note_4_8_8_16": "PyTorch native does not support 4:8/8:16 sparse kernels; need cuSparseLt for hardware speedup. We report dense-kernel throughput + theoretical FLOP reduction.",
            }
            out["cells"].append(cell_record)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(out, indent=2))
            print(f"  pre_ft={cell_pre_ft:.4f}  k:n={k}:{n}  src={source}  "
                  f"sparsity_eligible={theoretical_flop_reduction:.4f}  "
                  f"dense-kernel={thru_dense['images_per_s']:.1f} im/s  "
                  f"2:4-kernel={thru_24['images_per_s']:.1f if thru_24 else 'NA'} im/s")
        except Exception as e:
            print(f"  FAILED: {e}")
            out["cells"].append({"label": label, "error": str(e)})
            Path(args.output).write_text(json.dumps(out, indent=2))

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
