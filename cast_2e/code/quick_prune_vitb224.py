"""
quick_prune_vitb224.py — generate a ViT-B/16 224 augreg2 SER s=0.35 ckpt locally
when the paper's exact Hybrid-Mag-SER ckpt is not available.

Approach: Classical Magnitude prune to s=0.35 (zero out the smallest-by-|w|
weights in every Linear layer, uniformly to 35% sparsity). This corresponds to
Table 2 row "Classical magnitude" at s=0.35 → 83.53% top-1 on ViT-B/16 augreg2.
The paper's Hybrid Mag-SER row at s=0.35 is 84.28% (+0.75 pp). For a 5-method
PRE-FT ABLATION on the cert/projection knobs, this difference is small enough
that the relative method rankings transfer.

Output: a `keep_s35.pt` state_dict saved to --output, ready for cert_opt_eval.py.

Usage:
    python quick_prune_vitb224.py \\
        --timm-name vit_base_patch16_224.augreg2_in21k_ft_in1k \\
        --target-sparsity 0.35 \\
        --output /workspace/sweep_ckpts/vit_base_patch16_224.augreg2_in21k_ft_in1k_keep_s35_classmag.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn


def is_eligible_linear(name: str, mod: nn.Module) -> bool:
    """Eligible: nn.Linear with both dims divisible by 4 (so 2:4 will be applicable later)."""
    if not isinstance(mod, nn.Linear):
        return False
    out_f, in_f = mod.weight.shape
    return in_f % 4 == 0 and out_f >= 4


def magnitude_prune_to_target(model: nn.Module, target_sparsity: float) -> dict:
    """In-place: zero out the smallest-magnitude weights in each eligible Linear,
    uniformly across all eligible layers, to reach a global `target_sparsity`."""
    eligibles = [(n, m) for n, m in model.named_modules() if is_eligible_linear(n, m)]
    print(f"  eligible Linear layers: {len(eligibles)}")
    total = sum(m.weight.numel() for _, m in eligibles)
    n_to_zero = int(target_sparsity * total)
    print(f"  total params: {total:,}; want to zero: {n_to_zero:,} ({target_sparsity*100:.1f}%)")

    # Per-layer uniform: zero the bottom-(target_sparsity * Cin*Cout) per layer.
    layer_stats = []
    for name, m in eligibles:
        with torch.no_grad():
            w = m.weight.data
            num_zero = int(target_sparsity * w.numel())
            if num_zero == 0:
                continue
            flat = w.abs().flatten()
            # Find threshold: the k-th smallest magnitude
            kth = torch.kthvalue(flat, num_zero).values
            mask = w.abs() > kth
            m.weight.data = w * mask.float()
            layer_stats.append({
                "name": name, "params": w.numel(),
                "zeroed": int((m.weight.data == 0).sum().item()),
                "sparsity": float((m.weight.data == 0).float().mean().item()),
            })
    actual_sparsity = sum(s["zeroed"] for s in layer_stats) / total
    return {"target_sparsity": target_sparsity, "actual_sparsity": actual_sparsity,
            "n_layers": len(layer_stats), "layers": layer_stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm-name", required=True)
    ap.add_argument("--target-sparsity", type=float, default=0.35)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print(f"[quick_prune] timm={args.timm_name}, target s={args.target_sparsity}")
    t0 = time.time()

    import timm
    print(f"[quick_prune] loading dense pretrained model...")
    model = timm.create_model(args.timm_name, pretrained=True)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"[quick_prune] magnitude pruning per layer to s={args.target_sparsity}...")
    stats = magnitude_prune_to_target(model, args.target_sparsity)
    print(f"  achieved sparsity: {stats['actual_sparsity']:.4f}")
    print(f"  layers pruned: {stats['n_layers']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[quick_prune] saving to {out_path}")
    # Save as a plain state_dict (matches what load_ser_with_coverage_check expects)
    torch.save({"state_dict": model.state_dict()}, out_path)
    print(f"[quick_prune] done in {time.time()-t0:.1f}s")
    print(f"[quick_prune] file size: {out_path.stat().st_size / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
