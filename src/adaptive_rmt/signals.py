from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import AdaptivePruningConfig
from .model_utils import iter_target_layers_modules
from .rmt_diagnostics import estimate_weight_rmt_snapshot


def classify_layer_type(name: str) -> str:
    lname = name.lower()
    if "attn" in lname:
        return "attn"
    if "mlp" in lname or ".fc" in lname:
        return "mlp"
    return "other"


def _zscore_dict(values: dict[str, float | None]) -> dict[str, float]:
    present = np.array([v for v in values.values() if v is not None], dtype=np.float64)
    if present.size == 0 or float(present.std()) < 1.0e-12:
        return {k: 0.0 for k in values}
    mu = float(present.mean())
    sigma = float(present.std())
    return {
        k: 0.0 if v is None else float((v - mu) / sigma)
        for k, v in values.items()
    }


def _effective_rank(weight: np.ndarray) -> float:
    singular_values = np.linalg.svd(weight, compute_uv=False)
    numerator = float(np.square(singular_values.sum()))
    denominator = float(np.square(singular_values).sum()) + 1.0e-12
    return numerator / denominator


def _lookup_metric(cache: dict[str, float], name: str) -> float | None:
    for cache_key, value in cache.items():
        if name in cache_key or cache_key in name:
            return float(value)
    return None


@dataclass
class LayerSignalState:
    name: str
    layer_type: str
    n_params: int
    splus: float | None
    alpha: float | None
    splus_z: float
    alpha_z: float
    baseline_frob: float
    baseline_effective_rank: float
    current_density: float = 1.0
    frob_ratio: float = 1.0
    effective_rank_ratio: float = 1.0
    grad_norm_ema: float = 0.0
    grad_percentile: float = 0.0
    lr_scale: float = 1.0
    live_sigma_sq: float | None = None
    live_splus: float | None = None
    mp_fit_error: float = 1.0
    bulk_fraction: float = 0.0
    spike_count: int = 0
    spike_mass_ratio: float = 0.0
    mean_outlier_ratio: float = 0.0
    live_splus_z: float = 0.0
    bulk_fraction_z: float = 0.0
    mp_like_z: float = 0.0
    spike_count_z: float = 0.0
    spike_mass_ratio_z: float = 0.0
    dynamic_randomness_z: float = 0.0
    last_rmt_refresh_step: int = -1
    left_spike_basis: np.ndarray | None = None
    right_spike_basis: np.ndarray | None = None
    spike_subspace_stability: float | None = None

    @property
    def headroom(self) -> float:
        return 0.5 * self.current_density + 0.5 * self.effective_rank_ratio


def load_signal_bank(
    model: nn.Module,
    cache_dir: Path,
    log_fn,
) -> dict[str, LayerSignalState]:
    splus_path = cache_dir / "rmt_splus_metrics.json"
    layer_stats_path = cache_dir / "rmt_layer_stats.json"

    splus_cache: dict[str, float] = {}
    alpha_cache: dict[str, float] = {}
    if splus_path.exists():
        with open(splus_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        flat = raw.get("splus", raw)
        splus_cache = {
            key: float(value)
            for key, value in flat.items()
            if isinstance(value, (int, float))
        }
        log_fn(f"loaded {len(splus_cache)} cached sigma+ values from {splus_path}")
    else:
        log_fn(f"WARNING: sigma+ cache missing at {splus_path}")

    if layer_stats_path.exists():
        with open(layer_stats_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        alpha_cache = {
            key: float(value["hill_alpha"])
            for key, value in raw.items()
            if isinstance(value, dict) and "hill_alpha" in value
        }
        log_fn(f"loaded {len(alpha_cache)} cached hill-alpha values from {layer_stats_path}")
    else:
        log_fn(f"WARNING: layer-stats cache missing at {layer_stats_path}")

    named_weights: dict[str, np.ndarray] = {}
    splus_values: dict[str, float | None] = {}
    alpha_values: dict[str, float | None] = {}

    for name, mod in iter_target_layers_modules(model):
        weight = mod.weight.detach().cpu().numpy()
        if weight.ndim == 4:
            weight = weight.reshape(weight.shape[0], -1)
        named_weights[name] = weight
        splus_values[name] = _lookup_metric(splus_cache, name)
        alpha_values[name] = _lookup_metric(alpha_cache, name)

    splus_z = _zscore_dict(splus_values)
    alpha_z = _zscore_dict(alpha_values)

    signal_bank: dict[str, LayerSignalState] = {}
    for name, weight in named_weights.items():
        signal_bank[name] = LayerSignalState(
            name=name,
            layer_type=classify_layer_type(name),
            n_params=int(weight.size),
            splus=splus_values[name],
            alpha=alpha_values[name],
            splus_z=splus_z[name],
            alpha_z=alpha_z[name],
            baseline_frob=float(np.linalg.norm(weight, ord="fro")),
            baseline_effective_rank=float(_effective_rank(weight)),
        )

    return signal_bank


def _fallback_randomness_score(state: LayerSignalState) -> float:
    if state.splus is not None:
        return state.splus_z
    if state.alpha is not None:
        return state.alpha_z
    return 0.0


def _current_randomness_score(state: LayerSignalState) -> float:
    if state.last_rmt_refresh_step >= 0:
        return state.dynamic_randomness_z
    return _fallback_randomness_score(state)


def _subspace_overlap(prev_basis: np.ndarray | None, curr_basis: np.ndarray | None) -> float | None:
    if prev_basis is None or curr_basis is None:
        return None
    if prev_basis.size == 0 or curr_basis.size == 0:
        return None
    k = min(prev_basis.shape[1], curr_basis.shape[1])
    if k <= 0:
        return None
    prev_proj = prev_basis[:, :k]
    curr_proj = curr_basis[:, :k]
    overlap = float(np.linalg.norm(prev_proj.T @ curr_proj, ord="fro") ** 2 / max(k, 1))
    return float(np.clip(overlap, 0.0, 1.0))


def refresh_live_rmt_metrics(
    model: nn.Module,
    signal_bank: dict[str, LayerSignalState],
    config: AdaptivePruningConfig,
    step_idx: int,
    log_fn,
    label: str,
) -> dict[str, float | int | list[str]]:
    if not config.dynamic_signals.enable_live_rmt_refresh:
        return {}

    live_splus_values: dict[str, float] = {}
    bulk_fraction_values: dict[str, float] = {}
    mp_like_values: dict[str, float] = {}
    spike_fraction_values: dict[str, float] = {}
    spike_mass_values: dict[str, float] = {}

    total_spike_count = 0
    bulk_values: list[float] = []
    spike_mass_ratio_values: list[float] = []
    spike_stability_values: list[float] = []
    outlier_ratio_values: list[float] = []

    for name, mod in iter_target_layers_modules(model):
        state = signal_bank[name]
        weight = mod.weight.detach().cpu().numpy()
        matrix = weight.reshape(weight.shape[0], -1) if weight.ndim == 4 else weight
        snapshot = estimate_weight_rmt_snapshot(
            matrix,
            alpha=config.dynamic_signals.rmt_refresh_alpha,
            max_basis_rank=config.milestone_rmt.subspace_max_rank,
        )
        metrics = snapshot.metrics
        left_stability = _subspace_overlap(state.left_spike_basis, snapshot.left_basis)
        right_stability = _subspace_overlap(state.right_spike_basis, snapshot.right_basis)
        valid_stabilities = [value for value in (left_stability, right_stability) if value is not None]
        state.spike_subspace_stability = (
            float(np.mean(valid_stabilities)) if valid_stabilities else None
        )
        state.live_sigma_sq = metrics.sigma_sq
        state.live_splus = metrics.splus
        state.mp_fit_error = metrics.mp_fit_error
        state.bulk_fraction = metrics.bulk_fraction
        state.spike_count = metrics.spike_count
        state.spike_mass_ratio = metrics.spike_mass_ratio
        state.mean_outlier_ratio = metrics.mean_outlier_ratio
        state.last_rmt_refresh_step = step_idx
        state.left_spike_basis = snapshot.left_basis
        state.right_spike_basis = snapshot.right_basis

        live_splus_values[name] = metrics.splus
        bulk_fraction_values[name] = metrics.bulk_fraction
        mp_like_values[name] = max(0.0, 1.0 - metrics.mp_fit_error)
        spike_fraction_values[name] = metrics.spike_count / max(min(matrix.shape), 1)
        spike_mass_values[name] = metrics.spike_mass_ratio

        total_spike_count += metrics.spike_count
        bulk_values.append(metrics.bulk_fraction)
        spike_mass_ratio_values.append(metrics.spike_mass_ratio)
        outlier_ratio_values.append(metrics.mean_outlier_ratio)
        if state.spike_subspace_stability is not None:
            spike_stability_values.append(state.spike_subspace_stability)

    live_splus_z = _zscore_dict(live_splus_values)
    bulk_fraction_z = _zscore_dict(bulk_fraction_values)
    mp_like_z = _zscore_dict(mp_like_values)
    spike_fraction_z = _zscore_dict(spike_fraction_values)
    spike_mass_ratio_z = _zscore_dict(spike_mass_values)

    for name, state in signal_bank.items():
        state.live_splus_z = live_splus_z[name]
        state.bulk_fraction_z = bulk_fraction_z[name]
        state.mp_like_z = mp_like_z[name]
        state.spike_count_z = spike_fraction_z[name]
        state.spike_mass_ratio_z = spike_mass_ratio_z[name]

        randomness_core = (
            config.dynamic_signals.rmt_randomness_splus_weight * state.live_splus_z
            + config.dynamic_signals.rmt_randomness_bulk_weight * state.bulk_fraction_z
            + config.dynamic_signals.rmt_randomness_mp_like_weight * state.mp_like_z
        )
        structured_relief = (
            config.dynamic_signals.rmt_structure_spike_mass_weight * state.spike_mass_ratio_z
            + config.dynamic_signals.rmt_structure_spike_count_weight * state.spike_count_z
        )
        state.dynamic_randomness_z = randomness_core - structured_relief

    random_ranked = sorted(
        signal_bank.values(),
        key=lambda item: item.dynamic_randomness_z,
        reverse=True,
    )
    spike_ranked = sorted(
        signal_bank.values(),
        key=lambda item: item.spike_mass_ratio,
        reverse=True,
    )
    summary: dict[str, float | int | list[str]] = {
        "avg_bulk_fraction": float(np.mean(bulk_values)) if bulk_values else 0.0,
        "avg_spike_mass_ratio": float(np.mean(spike_mass_ratio_values)) if spike_mass_ratio_values else 0.0,
        "avg_outlier_ratio": float(np.mean(outlier_ratio_values)) if outlier_ratio_values else 0.0,
        "avg_spike_stability": float(np.mean(spike_stability_values)) if spike_stability_values else None,
        "total_spike_count": int(total_spike_count),
        "top_random_layers": [item.name for item in random_ranked[:3]],
        "top_spike_layers": [item.name for item in spike_ranked[:3]],
    }
    log_fn(
        f"live-rmt {label}: avg_bulk={summary['avg_bulk_fraction']:.3f} "
        f"avg_spike_mass={summary['avg_spike_mass_ratio']:.3f} "
        f"avg_outlier_ratio={summary['avg_outlier_ratio']:.3f} "
        f"avg_spike_stability={summary['avg_spike_stability'] if summary['avg_spike_stability'] is not None else 'n/a'} "
        f"total_spikes={summary['total_spike_count']} "
        f"top_random={', '.join(summary['top_random_layers'])} "
        f"top_spiky={', '.join(summary['top_spike_layers'])}"
    )
    return summary


def refresh_dynamic_metrics(
    model: nn.Module,
    signal_bank: dict[str, LayerSignalState],
    config: AdaptivePruningConfig,
    step_idx: int,
) -> None:
    recompute_rank = step_idx % max(config.dynamic_signals.recompute_svd_every_steps, 1) == 0
    floor = config.dynamic_signals.min_effective_rank_ratio

    for name, mod in iter_target_layers_modules(model):
        state = signal_bank[name]
        weight = mod.weight.detach().cpu().numpy()
        flat = weight.reshape(-1)
        current_density = float(np.count_nonzero(flat)) / max(flat.size, 1)
        state.current_density = current_density

        matrix = weight.reshape(weight.shape[0], -1) if weight.ndim == 4 else weight
        current_frob = float(np.linalg.norm(matrix, ord="fro"))
        state.frob_ratio = current_frob / max(state.baseline_frob, 1.0e-12)

        if recompute_rank:
            current_rank = _effective_rank(matrix)
            state.effective_rank_ratio = max(
                floor,
                current_rank / max(state.baseline_effective_rank, 1.0e-12),
            )

        randomness_score = _current_randomness_score(state)
        raw_scale = 1.0 + config.dynamic_signals.lr_randomness_scale * randomness_score
        state.lr_scale = float(
            np.clip(
                raw_scale,
                config.dynamic_signals.lr_randomness_clip_min,
                config.dynamic_signals.lr_randomness_clip_max,
            )
        )


def update_gradient_statistics(
    model: nn.Module,
    signal_bank: dict[str, LayerSignalState],
    config: AdaptivePruningConfig,
) -> None:
    grad_values: list[float] = []
    for name, mod in iter_target_layers_modules(model):
        if mod.weight.grad is None:
            continue
        grad_norm = float(mod.weight.grad.detach().norm().item())
        state = signal_bank[name]
        state.grad_norm_ema = (
            config.dynamic_signals.grad_ema_decay * state.grad_norm_ema
            + (1.0 - config.dynamic_signals.grad_ema_decay) * grad_norm
        )
        grad_values.append(state.grad_norm_ema)

    if not grad_values:
        return

    sorted_values = np.sort(np.asarray(grad_values, dtype=np.float64))
    for state in signal_bank.values():
        idx = np.searchsorted(sorted_values, state.grad_norm_ema, side="right")
        state.grad_percentile = float(idx / max(len(sorted_values), 1))


def select_signal_regime(current_sparsity: float, config: AdaptivePruningConfig) -> tuple[str, float, float, float | None, float | None]:
    signal_cfg = config.signals
    if current_sparsity <= signal_cfg.low_sparsity_cutoff:
        return ("alpha", signal_cfg.alpha_beta, signal_cfg.alpha_sd, None, None)
    if current_sparsity <= 0.40:
        return ("splus", signal_cfg.low_splus_beta, signal_cfg.low_splus_sd, None, None)
    if current_sparsity <= signal_cfg.medium_sparsity_cutoff:
        return ("splus", signal_cfg.medium_splus_beta, signal_cfg.medium_splus_sd, None, None)
    return (
        "splus",
        signal_cfg.high_splus_beta,
        signal_cfg.high_splus_sd,
        signal_cfg.high_attn_beta,
        signal_cfg.high_mlp_beta,
    )


def compute_layer_budget_weights(
    signal_bank: dict[str, LayerSignalState],
    current_sparsity: float,
    config: AdaptivePruningConfig,
) -> dict[str, float]:
    signal_name, beta, s_decay, beta_attn, beta_mlp = select_signal_regime(current_sparsity, config)
    decay = max(0.0, 1.0 - current_sparsity / max(s_decay, 1.0e-12))
    floor = config.signals.min_layer_weight
    headroom_floor = config.dynamic_signals.headroom_floor
    weights: dict[str, float] = {}

    for name, state in signal_bank.items():
        live_z = _current_randomness_score(state)
        if signal_name == "alpha":
            z = 0.60 * state.alpha_z + 0.40 * live_z
        else:
            z = live_z
        if beta_attn is not None and beta_mlp is not None:
            if state.layer_type == "attn":
                beta_here = beta_attn
            elif state.layer_type == "mlp":
                beta_here = beta_mlp
            else:
                beta_here = beta
        else:
            beta_here = beta

        base = max(floor, 1.0 + beta_here * decay * z)
        headroom = max(headroom_floor, state.headroom)
        grad_gate = 1.0
        if state.grad_percentile >= config.dynamic_signals.grad_protection_percentile:
            grad_gate = config.dynamic_signals.grad_protection_scale

        bulk_push = 1.0 + 0.15 * max(state.bulk_fraction - 0.50, 0.0)
        spike_guard = 1.0 - 0.20 * max(state.spike_mass_ratio - 0.25, 0.0)
        weights[name] = max(floor, base * headroom * grad_gate * bulk_push * max(0.70, spike_guard))

    return weights


def aggregate_headroom(signal_bank: dict[str, LayerSignalState], config: AdaptivePruningConfig) -> float:
    if not signal_bank:
        return config.dynamic_signals.headroom_floor
    values = [max(config.dynamic_signals.headroom_floor, state.headroom) for state in signal_bank.values()]
    return float(np.mean(values))
