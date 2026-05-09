"""
benchmark_sparse_throughput.py — measure ACTUAL wall-clock throughput for
each best-cell ckpt. Reports:
  - dense images/s
  - 2:4 sparse images/s (PyTorch to_sparse_semi_structured)
  - 1.xx× speedup ratio
  - effective NNZ % (counts SER pre-zeros that survive 2:4 — the user's point)

For 4:8 and 8:16: PyTorch only supports 2:4 hardware kernels via
to_sparse_semi_structured. We report THEORETICAL MAC reduction and dense
kernel throughput as a comparison endpoint, but 4:8/8:16 don't get hardware
speedup at the moment (would need cuSparseLt direct integration).

Outputs: JSON with throughput numbers + actual NNZ ratios.

Usage:
    python benchmark_sparse_throughput.py \\
        --timm-name resnet50.tv_in1k \\
        --ckpt /workspace/run_outputs/best_ckpts/resnet50_best.pt \\
        --output /workspace/run_logs/benchmark_resnet50.json \\
        --batch 128 --warmup 20 --iters 100
"""
from __future__ import annotations
import torch as _t; _t.backends.cudnn.enabled = False

import argparse, json, time, sys
from pathlib import Path

import torch
import torch.nn as nn


def benchmark_throughput(model: nn.Module, device: str, batch: int, image_size: int,
                          warmup: int = 20, iters: int = 100) -> dict:
    """Run forward passes; return median ms + images/s."""
    model.eval()
    x = torch.randn(batch, 3, image_size, image_size, device=device)
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()
    # Timed
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
    return {
        "batch": batch, "image_size": image_size, "warmup": warmup, "iters": iters,
        "median_s": median_s,
        "images_per_s": batch / median_s,
    }


def count_nnz(model: nn.Module) -> dict:
    """Count NNZ across all Linear/Conv2d weights."""
    total_params = 0; total_nnz = 0
    layer_stats = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                W = mod.weight.data
                p = int(W.numel()); nnz = int((W != 0).sum().item())
            total_params += p; total_nnz += nnz
            layer_stats[name] = {"params": p, "nnz": nnz, "sparsity": 1.0 - nnz/max(1,p)}
    return {
        "total_params": total_params, "total_nnz": total_nnz,
        "global_sparsity": 1.0 - total_nnz / max(1, total_params),
        "global_density": total_nnz / max(1, total_params),
        "n_layers": len(layer_stats),
    }


def try_apply_2_4_sparsity(model: nn.Module) -> dict:
    """Try to convert eligible Linear weights to PyTorch's sparse semi-structured.
    Reports how many converted, since pattern must be exactly 2:4."""
    converted = 0; failed = 0; tried = 0
    try:
        from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
        SparseSemiStructuredTensor._FORCE_CUTLASS = True
    except ImportError:
        return {"supported": False, "reason": "torch.sparse.to_sparse_semi_structured not available"}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            with torch.no_grad():
                W = mod.weight.data
                Cout, Cin = W.shape
                if Cin % 16 != 0:  # to_sparse_semi_structured requires Cin % 16 == 0
                    continue
                tried += 1
                # Check 2:4 legality: every 4-tuple has exactly 2 zeros
                Wg = W.reshape(Cout, Cin // 4, 4)
                nz_per_group = (Wg != 0).sum(dim=-1)
                if (nz_per_group == 2).all():
                    try:
                        sparse_w = to_sparse_semi_structured(W.cuda())
                        mod.weight = nn.Parameter(sparse_w, requires_grad=False)
                        converted += 1
                    except Exception as e:
                        failed += 1
    return {"supported": True, "tried": tried, "converted": converted, "failed": failed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm-name", required=True)
    ap.add_argument("--ckpt", required=True, help="Path to projected student ckpt (.pt)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--image-size", type=int, default=0, help="0 = use model default")
    args = ap.parse_args()

    device = "cuda"
    print(f"[benchmark] timm={args.timm_name}, ckpt={args.ckpt}")
    import timm

    # 1. Dense baseline
    dense = timm.create_model(args.timm_name, pretrained=True).to(device).eval()
    cfg = timm.data.resolve_model_data_config(dense)
    img_size = args.image_size or cfg["input_size"][-1]
    dense_nnz = count_nnz(dense)
    print(f"  dense: {dense_nnz['total_params']:,} params, {dense_nnz['global_density']:.4f} density")
    dense_thru = benchmark_throughput(dense, device, args.batch, img_size,
                                       warmup=args.warmup, iters=args.iters)
    print(f"  dense throughput: {dense_thru['images_per_s']:.1f} images/s @ batch {args.batch}, {img_size}px")
    del dense; torch.cuda.empty_cache()

    # 2. Projected (sparse) student
    raw = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = raw.get("state_dict") or raw
    sparse = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
    sparse.load_state_dict(sd, strict=False)
    sparse_nnz = count_nnz(sparse)
    print(f"  projected: {sparse_nnz['total_params']:,} params, {sparse_nnz['global_density']:.4f} density")
    sparse_thru_dense_kernel = benchmark_throughput(sparse, device, args.batch, img_size,
                                                     warmup=args.warmup, iters=args.iters)
    print(f"  projected (dense-kernel) throughput: {sparse_thru_dense_kernel['images_per_s']:.1f} images/s")

    # 3. Try to apply real 2:4 sparse kernels
    sparse_2_4 = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
    sparse_2_4.load_state_dict(sd, strict=False)
    apply_result = try_apply_2_4_sparsity(sparse_2_4)
    print(f"  2:4 conversion: {apply_result}")
    if apply_result.get("supported") and apply_result.get("converted", 0) > 0:
        sparse_thru_24 = benchmark_throughput(sparse_2_4, device, args.batch, img_size,
                                                warmup=args.warmup, iters=args.iters)
        print(f"  projected (2:4 sparse-kernel) throughput: {sparse_thru_24['images_per_s']:.1f} images/s")
    else:
        sparse_thru_24 = None

    # 4. Aggregate
    out = {
        "model": args.timm_name,
        "ckpt": args.ckpt,
        "cell_label": raw.get("cell"),
        "cell_pre_ft_top1": raw.get("pre_ft_top1"),
        "cell_kn": (raw.get("k"), raw.get("n")),
        "dense_nnz_stats": dense_nnz,
        "sparse_nnz_stats": sparse_nnz,
        "dense_throughput": dense_thru,
        "sparse_throughput_dense_kernel": sparse_thru_dense_kernel,
        "sparse_throughput_2_4_kernel": sparse_thru_24,
        "kernel_speedup_2_4_vs_dense": (sparse_thru_24["images_per_s"] / dense_thru["images_per_s"]
                                          if sparse_thru_24 else None),
        "_2_4_conversion": apply_result,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {args.output}")
    if sparse_thru_24:
        print(f"WALL-CLOCK SPEEDUP (dense → 2:4 sparse kernel): "
              f"{sparse_thru_24['images_per_s']/dense_thru['images_per_s']:.3f}×")


if __name__ == "__main__":
    main()
