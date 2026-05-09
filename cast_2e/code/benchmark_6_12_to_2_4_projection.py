"""
benchmark_6_12_to_2_4_projection.py — for 6:12 / 4:8 / 8:16 ckpts where PyTorch
has no native sparse kernel, project the mask down to 2:4 within each 4-block
and re-benchmark with PyTorch's `to_sparse_semi_structured`.

This gives the ACTUAL hardware speedup achievable on existing CUDA kernels
for any 50% N:M structured ckpt — a paper-quality speedup number.

Mathematically: a 6:12 mask is 50% sparse. We project it to 2:4 by, for each
12-element group split into 3 sub-blocks of 4: pick the 2 highest-magnitude
entries in each sub-block. This is a strict refinement of the 6:12 pattern
(every 2:4 pattern is a valid 6:12 pattern, but not vice-versa). Accuracy at
this projected mask is a LOWER BOUND on what 6:12 with optimal hardware would
achieve, and the speedup is the directly-measured 2:4 number.

Usage:
    python benchmark_6_12_to_2_4_projection.py \\
        --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k \\
        --student-ckpt /workspace/run_outputs/best_pod1/ckpts/vitb/D612_ser_a05.pt \\
        --output /workspace/run_outputs/benchmarks/D612_ser_a05_to_2_4_speedup.json
"""
import torch as _t; _t.backends.cudnn.enabled = False

import argparse, json, time
from pathlib import Path

import torch
import torch.nn as nn


def project_to_2_4(W: torch.Tensor) -> torch.Tensor:
    """For a Linear weight W [Cout, Cin], project to 2:4 within each 4-block.
    Keep the 2 highest-magnitude weights per 4-block; zero the rest."""
    Cout, Cin = W.shape
    if Cin % 4 != 0:
        return W
    Wg = W.reshape(Cout, Cin // 4, 4)
    abs_W = Wg.abs()
    top2_idx = abs_W.topk(2, dim=-1).indices            # [Cout, Cin/4, 2]
    mask = torch.zeros_like(Wg)
    mask.scatter_(-1, top2_idx, 1.0)
    Wm = Wg * mask
    return Wm.reshape(Cout, Cin)


def project_model_to_2_4(model: nn.Module) -> dict:
    """Walk Linear layers and project weights to 2:4 within each 4-block."""
    n_proj = 0; n_skip = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            with torch.no_grad():
                W = mod.weight.data
                if W.shape[1] % 16 == 0:                # to_sparse_semi_structured constraint
                    mod.weight.data.copy_(project_to_2_4(W))
                    n_proj += 1
                else:
                    n_skip += 1
    return {"n_projected": n_proj, "n_skipped_cin_not_div_16": n_skip}


def benchmark_throughput(model, device, batch=128, image_size=224, warmup=20, iters=100):
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
    return {"median_s": times[len(times)//2], "images_per_s": batch / times[len(times)//2]}


def count_nnz(model: nn.Module) -> dict:
    total_p = total_nnz = 0
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                W = m.weight.data
                total_p += W.numel()
                total_nnz += int((W != 0).sum().item())
    return {"params": total_p, "nnz": total_nnz, "sparsity": 1.0 - total_nnz / max(1, total_p)}


def apply_pytorch_2_4(model):
    try:
        from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
        SparseSemiStructuredTensor._FORCE_CUTLASS = True
    except ImportError:
        return {"supported": False}
    converted = tried = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            with torch.no_grad():
                W = mod.weight.data
                if W.shape[1] % 16 != 0: continue
                tried += 1
                Wg = W.reshape(W.shape[0], W.shape[1]//4, 4)
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
    ap.add_argument("--student-ckpt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    device = "cuda"
    print(f"[bench_proj] timm={args.timm_name}, ckpt={args.student_ckpt}")
    import timm
    cfg_model = timm.create_model(args.timm_name, pretrained=True).to(device).eval()
    cfg = timm.data.resolve_model_data_config(cfg_model)
    img_size = cfg["input_size"][-1]

    # 1. Dense baseline
    dense_thru = benchmark_throughput(cfg_model, device, args.batch, img_size,
                                       warmup=args.warmup, iters=args.iters)
    print(f"  dense: {dense_thru['images_per_s']:.1f} im/s")
    del cfg_model; torch.cuda.empty_cache()

    # 2. Original 6:12 ckpt (dense kernel)
    raw = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict") or raw
    m_orig = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
    m_orig.load_state_dict(sd, strict=False)
    nnz_orig = count_nnz(m_orig)
    thru_orig = benchmark_throughput(m_orig, device, args.batch, img_size,
                                      warmup=args.warmup, iters=args.iters)
    print(f"  original ckpt ({raw.get('cell','?')} k:n={raw.get('k')}:{raw.get('n')}): "
          f"sparsity={nnz_orig['sparsity']:.4f}, dense_kernel={thru_orig['images_per_s']:.1f} im/s")
    del m_orig; torch.cuda.empty_cache()

    # 3. Project to 2:4 within each 4-block, then apply PyTorch 2:4 kernel
    m_24 = timm.create_model(args.timm_name, pretrained=False).to(device).eval()
    m_24.load_state_dict(sd, strict=False)
    proj_stats = project_model_to_2_4(m_24)
    nnz_24 = count_nnz(m_24)
    print(f"  after 2:4 projection: {proj_stats}, sparsity={nnz_24['sparsity']:.4f}")

    apply_24 = apply_pytorch_2_4(m_24)
    print(f"  apply_2_4 kernel: {apply_24}")

    if apply_24.get("converted", 0) > 0:
        thru_24 = benchmark_throughput(m_24, device, args.batch, img_size,
                                        warmup=args.warmup, iters=args.iters)
        speedup = thru_24["images_per_s"] / dense_thru["images_per_s"]
        print(f"  2:4 sparse-kernel: {thru_24['images_per_s']:.1f} im/s  → SPEEDUP {speedup:.3f}× over dense")
    else:
        thru_24 = None; speedup = None
        print(f"  WARNING: no 2:4 conversion happened — projection failed")
    del m_24; torch.cuda.empty_cache()

    # 4. Save
    out_record = {
        "model": args.timm_name,
        "student_ckpt": args.student_ckpt,
        "original_cell": raw.get("cell"),
        "original_k_n": (raw.get("k"), raw.get("n")),
        "original_source": raw.get("source"),
        "original_alpha_ser": raw.get("alpha_ser"),
        "original_pre_ft_top1": raw.get("pre_ft_top1"),
        "image_size": img_size,
        "batch": args.batch,
        "dense_throughput": dense_thru,
        "original_ckpt_nnz_stats": nnz_orig,
        "original_ckpt_throughput_dense_kernel": thru_orig,
        "projected_2_4_nnz_stats": nnz_24,
        "projected_2_4_projection_stats": proj_stats,
        "projected_2_4_throughput_sparse_kernel": thru_24,
        "projected_2_4_speedup_vs_dense": speedup,
        "_2_4_kernel_apply": apply_24,
        "_note": (
            "The original ckpt (e.g. 6:12 cert-aware) gets dense-kernel throughput "
            "since PyTorch native only supports 2:4. We then project it down to 2:4 "
            "within each 4-block (top-2 magnitude per 4-tuple — a strict refinement of "
            "the original N:M pattern at 50%) and re-benchmark with PyTorch's "
            "to_sparse_semi_structured. The result is a directly-measured hardware "
            "speedup at 50% N:M structured sparsity, applicable to any 4:8/6:12/8:16 "
            "ckpt as a published lower bound."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out_record, indent=2))
    print(f"\nSaved: {args.output}")
    if speedup:
        print(f"PAPER NUMBER: {speedup:.3f}× wall-clock speedup at 50% N:M sparsity (PyTorch 2:4 kernel)")


if __name__ == "__main__":
    main()
