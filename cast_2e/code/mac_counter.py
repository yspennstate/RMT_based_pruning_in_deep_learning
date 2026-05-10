"""
mac_counter.py — Exact hook-based per-layer MAC counting for any timm model.

For ResNet (and other Conv-heavy models), the existing CAST runner used a
"generic non-ViT fallback" that flat-counted Linear+Conv2d weight MACs with a
fixed 7×7 spatial assumption. That approximation is acceptable as a log
diagnostic, but paper tables should use actual forward shapes. This module
hooks the forward pass at 224×224 and computes exact per-layer MACs.

Conventions (same as the ViT path):
  - 1 MAC = 1 multiply-add
  - Conv2d MACs = Cout × Hout × Wout × (Cin/groups) × kH × kW
  - Linear MACs = Cin × Cout (per token, summed over batch=1)
  - BatchNorm / activations / pooling MACs are not counted by convention

Usage:
    from mac_counter import count_macs, sparse_exec_mac_estimate
    macs = count_macs(model, image_size=224)
    # macs.layers: dict[name] = {"type", "macs", "weight_shape"}
    # macs.dense_total: int
    sparse = sparse_exec_mac_estimate(macs, eligible_layer_names=["layer1.0.conv1", ...])
    print(sparse.dense_total, sparse.eligible_macs, sparse.sparse_exec_estimate)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class MacReport:
    image_size: int
    layers: dict[str, dict] = field(default_factory=dict)
    dense_total: int = 0
    counted_types: tuple[str, ...] = ("Conv2d", "Linear")


@dataclass
class SparseExecReport:
    dense_total: int
    eligible_layer_names: list[str]
    eligible_macs: int
    sparse_exec_estimate: int      # dense - 0.5 * eligible (analytical 2:4 saving)
    eligible_fraction: float
    sparsity_factor: float = 0.5   # 2:4 = 50% NNZ
    by_layer: dict[str, dict] = field(default_factory=dict)


def count_macs(model: nn.Module, image_size: int = 224,
               in_channels: int = 3, batch_size: int = 1,
               device: str | None = None) -> MacReport:
    """Run a single dummy forward pass at `image_size` and capture per-layer MACs."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    x = torch.zeros(batch_size, in_channels, image_size, image_size, device=device)

    layers: dict[str, dict] = {}
    handles = []
    seen_names = {id(m): n for n, m in model.named_modules()}

    def make_hook(name: str):
        def _hook(mod, inputs, output):
            try:
                if isinstance(mod, nn.Conv2d):
                    out_shape = output.shape  # [B, Cout, Hout, Wout]
                    Bh, Co, Ho, Wo = out_shape
                    Cin_per_group = mod.in_channels // mod.groups
                    kH, kW = mod.kernel_size
                    macs_per_image = Co * Ho * Wo * Cin_per_group * kH * kW
                    layers[name] = {
                        "type": "Conv2d",
                        "macs": macs_per_image,
                        "weight_shape": list(mod.weight.shape),
                        "out_shape": [int(s) for s in out_shape[1:]],
                        "groups": mod.groups,
                        "kernel_size": [kH, kW],
                    }
                elif isinstance(mod, nn.Linear):
                    # output: [..., Cout], input: [..., Cin]
                    in_shape = inputs[0].shape if inputs else None
                    # number of "token" positions
                    if in_shape is None:
                        n_tokens = 1
                    else:
                        n_tokens = 1
                        for s in in_shape[1:-1]:
                            n_tokens *= int(s)
                    macs_per_image = n_tokens * mod.in_features * mod.out_features
                    layers[name] = {
                        "type": "Linear",
                        "macs": macs_per_image,
                        "weight_shape": list(mod.weight.shape),
                        "n_tokens": n_tokens,
                    }
            except Exception as e:
                layers[name] = {"type": type(mod).__name__, "error": str(e), "macs": 0}
        return _hook

    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(make_hook(n)))

    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()

    total = sum(L["macs"] for L in layers.values())
    return MacReport(image_size=image_size, layers=layers, dense_total=total)


def sparse_exec_mac_estimate(report: MacReport,
                              eligible_layer_names: Iterable[str],
                              sparsity_factor: float = 0.5) -> SparseExecReport:
    """Compute the analytical sparse-exec MAC reduction.

    Eligible MACs are halved (since 2:4 keeps 2/4 of weights). All other layers
    keep their dense MAC contribution.

    Returns dense_total, eligible_macs, sparse_exec_estimate (= dense - 0.5*eligible),
    eligible_fraction.
    """
    eligible_set = set(eligible_layer_names)
    by_layer: dict[str, dict] = {}
    eligible = 0
    for name, L in report.layers.items():
        if name in eligible_set:
            by_layer[name] = {**L, "eligible": True,
                               "sparse_macs": int(L["macs"] * (1 - sparsity_factor))}
            eligible += L["macs"]
        else:
            by_layer[name] = {**L, "eligible": False, "sparse_macs": L["macs"]}
    sparse_total = report.dense_total - int(eligible * sparsity_factor)
    return SparseExecReport(
        dense_total=report.dense_total,
        eligible_layer_names=sorted(eligible_set),
        eligible_macs=eligible,
        sparse_exec_estimate=sparse_total,
        eligible_fraction=eligible / max(report.dense_total, 1),
        sparsity_factor=sparsity_factor,
        by_layer=by_layer,
    )


if __name__ == "__main__":
    import argparse, json, sys
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50.tv_in1k")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--out", default="-")
    p.add_argument("--include-1x1-only-eligible", action="store_true",
                   help="report sparse MACs if 1x1 convs + fc are CASTed (clean theory tie-in)")
    p.add_argument("--include-all-eligible", action="store_true",
                   help="report sparse MACs if ALL convs with Cin*kH*kW%%4==0 are CASTed (1x1+3x3+fc)")
    args = p.parse_args()
    try:
        import timm
    except ImportError:
        print("pip install timm"); sys.exit(1)
    model = timm.create_model(args.model, pretrained=False)
    report = count_macs(model, image_size=args.image_size)
    out = {
        "model": args.model,
        "image_size": args.image_size,
        "dense_total_macs": report.dense_total,
        "dense_total_gmacs": report.dense_total / 1e9,
        "n_counted_layers": len(report.layers),
        "by_type_breakdown": {
            t: sum(L["macs"] for L in report.layers.values() if L.get("type") == t)
            for t in ("Conv2d", "Linear")
        },
    }
    def _eligible_set(only_1x1: bool) -> list[str]:
        names = []
        for name, L in report.layers.items():
            if L.get("type") == "Conv2d" and L.get("groups") == 1:
                w = L["weight_shape"]; cin = w[1]; kH, kW = L.get("kernel_size", [0, 0])
                if cin < 4:
                    continue
                if only_1x1 and (kH, kW) != (1, 1):
                    continue
                # Need Cin*kH*kW divisible by 4 for the 2:4 partition along input
                if (cin * kH * kW) % 4 != 0:
                    continue
                names.append(name)
            elif L.get("type") == "Linear":
                names.append(name)
        return names

    if args.include_1x1_only_eligible or args.include_all_eligible:
        modes = []
        if args.include_1x1_only_eligible:
            modes.append(("1x1+fc", _eligible_set(only_1x1=True)))
        if args.include_all_eligible:
            modes.append(("all_eligible(1x1+3x3+fc)", _eligible_set(only_1x1=False)))
        out["eligibility_modes"] = {}
        for label, eligible_names in modes:
            sparse = sparse_exec_mac_estimate(report, eligible_names)
            out["eligibility_modes"][label] = {
                "eligible_layer_count": len(eligible_names),
                "eligible_macs": sparse.eligible_macs,
                "eligible_gmacs": sparse.eligible_macs / 1e9,
                "eligible_fraction": sparse.eligible_fraction,
                "sparse_exec_total_macs": sparse.sparse_exec_estimate,
                "sparse_exec_total_gmacs": sparse.sparse_exec_estimate / 1e9,
                "mac_reduction_fraction": 1 - sparse.sparse_exec_estimate / sparse.dense_total,
            }
        # For backward-compat keep top-level fields pointing at first mode
        first = next(iter(out["eligibility_modes"].values()))
        out["eligible_layer_count"] = first["eligible_layer_count"]
        out["eligible_macs"] = first["eligible_macs"]
        out["eligible_gmacs"] = first["eligible_gmacs"]
        out["eligible_fraction"] = first["eligible_fraction"]
        out["sparse_exec_total_macs"] = first["sparse_exec_total_macs"]
        out["sparse_exec_total_gmacs"] = first["sparse_exec_total_gmacs"]
        out["mac_reduction_fraction"] = first["mac_reduction_fraction"]
        out["eligible_layer_names"] = list(modes[0][1])
    text = json.dumps(out, indent=2, default=int)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"saved -> {args.out}")
