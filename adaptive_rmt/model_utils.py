from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def iter_target_layers_modules(model: nn.Module):
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            yield name, mod


def build_mask(model: nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    for name, mod in iter_target_layers_modules(model):
        with torch.no_grad():
            masks[name] = (mod.weight != 0).to(device)
    return masks


def apply_mask(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, mod in iter_target_layers_modules(model):
            if name in masks:
                mod.weight.mul_(masks[name].to(mod.weight.device, dtype=mod.weight.dtype))


def current_sparsity(model: nn.Module) -> float:
    zero = 0
    total = 0
    for _, mod in iter_target_layers_modules(model):
        w = mod.weight.detach()
        zero += int((w == 0).sum().item())
        total += w.numel()
    return zero / max(total, 1)


def load_anchor_checkpoint(
    model: nn.Module,
    checkpoint_path: Path | None,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if checkpoint_path is None or not checkpoint_path.exists():
        return None
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(data["model_state_dict"])
    masks = {k: v.to(device) for k, v in data.get("masks", {}).items()}
    return masks
