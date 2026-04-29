from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import halfnorm
import torch
import torch.nn as nn

from .model_utils import current_sparsity, iter_target_layers_modules
from .signals import LayerSignalState


def sv_prune_haar(
    weight: np.ndarray,
    splus: float,
    z_base: float,
    alpha: float | None = None,
    alpha_mean: float | None = None,
    power: float = 3.0,
    haar_max_error: float = 0.20,
) -> np.ndarray:
    def haar_halfnormal_fit_error(vector: np.ndarray) -> float:
        if vector.size < 16:
            return 1.0
        scaled_abs = np.abs(vector.astype(np.float64, copy=False)) * math.sqrt(vector.size)
        quantiles = np.asarray([0.20, 0.35, 0.50, 0.65, 0.80], dtype=np.float64)
        empirical = np.quantile(scaled_abs, quantiles)
        theoretical = halfnorm.ppf(quantiles)
        relative_error = np.abs(empirical - theoretical) / np.maximum(theoretical, 1.0e-6)
        return float(relative_error.mean())

    rows, cols = weight.shape
    z_eff = z_base
    if alpha is not None and alpha_mean is not None and alpha_mean > 1.0e-6:
        z_eff = z_base * (alpha / alpha_mean)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(weight).float().to(device)
    u_mat, singular_values, vt_mat = torch.linalg.svd(tensor, full_matrices=False)

    std_u = 1.0 / math.sqrt(rows)
    std_v = 1.0 / math.sqrt(cols)
    ratios = singular_values / splus if splus > 1.0e-12 else torch.ones_like(singular_values)
    bulk_mask = ratios < 1.0
    graduated = torch.zeros_like(singular_values)
    if bulk_mask.any():
        graduated[bulk_mask] = (1.0 - ratios[bulk_mask]).clamp(min=0).pow(power)

    haar_gates = torch.zeros_like(singular_values)
    bulk_indices = torch.nonzero(bulk_mask, as_tuple=False).flatten().tolist()
    for idx in bulk_indices:
        u_error = haar_halfnormal_fit_error(u_mat[:, idx].detach().cpu().numpy())
        v_error = haar_halfnormal_fit_error(vt_mat[idx, :].detach().cpu().numpy())
        mean_error = 0.5 * (u_error + v_error)
        haar_gates[idx] = max(0.0, 1.0 - mean_error / max(haar_max_error, 1.0e-12))

    graduated = graduated * haar_gates
    threshold_u = z_eff * std_u * graduated
    threshold_v = z_eff * std_v * graduated
    u_mat = torch.where(u_mat.abs() < threshold_u.unsqueeze(0), 0, u_mat)
    vt_mat = torch.where(vt_mat.abs() < threshold_v.unsqueeze(1), 0, vt_mat)

    result = (u_mat @ torch.diag(singular_values) @ vt_mat).cpu().numpy().astype(np.float32)
    del tensor, u_mat, singular_values, vt_mat
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def apply_sv_cleanup(
    model: nn.Module,
    signal_bank: dict[str, LayerSignalState],
    z_base: float,
    power: float,
    min_bulk_fraction: float,
    min_mp_like: float,
    haar_max_error: float,
) -> None:
    alphas = [state.alpha for state in signal_bank.values() if state.alpha is not None]
    alpha_mean = float(np.mean(alphas)) if alphas else None
    with torch.no_grad():
        for name, mod in iter_target_layers_modules(model):
            state = signal_bank[name]
            splus = state.live_splus if state.live_splus is not None else state.splus
            if splus is None:
                continue
            mp_like = max(0.0, 1.0 - state.mp_fit_error)
            if state.bulk_fraction < min_bulk_fraction or mp_like < min_mp_like:
                continue
            weight = mod.weight.detach().cpu().numpy()
            matrix = weight.reshape(weight.shape[0], -1) if weight.ndim == 4 else weight
            cleaned = sv_prune_haar(
                matrix,
                splus=splus,
                z_base=z_base,
                alpha=state.alpha,
                alpha_mean=alpha_mean,
                power=power,
                haar_max_error=haar_max_error,
            )
            mod.weight.copy_(torch.from_numpy(cleaned.reshape(weight.shape)).to(mod.weight.dtype))


@dataclass
class LayerPruningStats:
    name: str
    layer_weight: float
    active_before: int
    active_after: int
    pruned_now: int


@dataclass
class PruningResult:
    target_sparsity: float
    achieved_sparsity: float
    pruned_now: int
    total_params: int
    layer_stats: list[LayerPruningStats]


def incremental_prune_to_target(
    model: nn.Module,
    target_sparsity: float,
    layer_weights: dict[str, float],
) -> PruningResult:
    total_params = 0
    current_zero = 0
    layer_meta: list[dict] = []
    all_scores: list[np.ndarray] = []
    all_layer_ids: list[np.ndarray] = []
    all_local_ids: list[np.ndarray] = []

    for layer_idx, (name, mod) in enumerate(iter_target_layers_modules(model)):
        weight = mod.weight.detach().cpu().numpy()
        flat = weight.reshape(-1)
        active_idx = np.flatnonzero(flat != 0)
        total_params += flat.size
        current_zero += flat.size - active_idx.size

        layer_weight = float(layer_weights.get(name, 1.0))
        scores = np.abs(flat[active_idx]) / max(layer_weight, 1.0e-12)
        layer_meta.append(
            {
                "name": name,
                "module": mod,
                "shape": weight.shape,
                "flat": flat.copy(),
                "active_idx": active_idx,
                "active_before": int(active_idx.size),
                "layer_weight": layer_weight,
            }
        )
        if scores.size > 0:
            all_scores.append(scores)
            all_layer_ids.append(np.full(scores.shape, layer_idx, dtype=np.int64))
            all_local_ids.append(np.arange(scores.size, dtype=np.int64))

    target_zero = int(round(total_params * target_sparsity))
    additional_zero = max(0, target_zero - current_zero)

    if additional_zero == 0 or not all_scores:
        return PruningResult(
            target_sparsity=target_sparsity,
            achieved_sparsity=current_sparsity(model),
            pruned_now=0,
            total_params=total_params,
            layer_stats=[
                LayerPruningStats(
                    name=item["name"],
                    layer_weight=item["layer_weight"],
                    active_before=item["active_before"],
                    active_after=item["active_before"],
                    pruned_now=0,
                )
                for item in layer_meta
            ],
        )

    global_scores = np.concatenate(all_scores)
    global_layer_ids = np.concatenate(all_layer_ids)
    global_local_ids = np.concatenate(all_local_ids)
    additional_zero = min(additional_zero, global_scores.size)
    chosen = np.argpartition(global_scores, additional_zero - 1)[:additional_zero]

    by_layer: dict[int, list[int]] = defaultdict(list)
    for global_idx in chosen:
        by_layer[int(global_layer_ids[global_idx])].append(int(global_local_ids[global_idx]))

    layer_stats: list[LayerPruningStats] = []
    for layer_idx, item in enumerate(layer_meta):
        local_ids = by_layer.get(layer_idx, [])
        if local_ids:
            prune_positions = item["active_idx"][np.asarray(local_ids, dtype=np.int64)]
            item["flat"][prune_positions] = 0.0
        active_after = int(np.count_nonzero(item["flat"]))
        with torch.no_grad():
            reshaped = item["flat"].reshape(item["shape"])
            item["module"].weight.copy_(torch.from_numpy(reshaped).to(item["module"].weight.dtype))
        layer_stats.append(
            LayerPruningStats(
                name=item["name"],
                layer_weight=item["layer_weight"],
                active_before=item["active_before"],
                active_after=active_after,
                pruned_now=item["active_before"] - active_after,
            )
        )

    return PruningResult(
        target_sparsity=target_sparsity,
        achieved_sparsity=current_sparsity(model),
        pruned_now=additional_zero,
        total_params=total_params,
        layer_stats=layer_stats,
    )
