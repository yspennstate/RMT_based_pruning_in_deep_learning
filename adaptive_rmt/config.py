from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AdaptiveBounds:
    initial_delta_s: float = 0.00500
    delta_s_min: float = 0.00025
    delta_s_max: float = 0.00500
    initial_ft_steps: int = 256
    ft_steps_min: int = 160
    ft_steps_max: int = 4096
    initial_ft_lr: float = 7.0e-5
    ft_lr_min: float = 4.0e-5
    ft_lr_max: float = 1.5e-4


@dataclass
class RecoveryThresholds:
    hard_residual_drop_pp: float = 0.75
    soft_residual_drop_pp: float = 0.25
    easy_residual_drop_pp: float = 0.05
    easy_shock_pp: float = 0.25
    hard_loss_increase: float = 0.20
    soft_loss_increase: float = 0.08
    easy_loss_increase: float = 0.02
    hard_logit_drift: float = 0.30
    soft_logit_drift: float = 0.15
    hard_margin_flip_rate: float = 0.05
    soft_margin_flip_rate: float = 0.02


@dataclass
class ControllerMultipliers:
    recovery_delta_s: float = 0.35
    recovery_ft_steps: float = 1.90
    recovery_lr: float = 0.98
    soft_delta_s: float = 0.55
    soft_ft_steps: float = 1.45
    soft_lr: float = 1.00
    accelerate_delta_s: float = 1.00
    accelerate_ft_steps: float = 1.00
    accelerate_lr: float = 1.00


@dataclass
class ProbeConfig:
    probe_batches: int = 64
    probe_interval_steps: int = 48
    recovery_min_steps: int = 192
    recovery_target_drop_pp: float = 0.15
    recovery_loss_tolerance: float = 0.008
    full_eval_batches: int | None = None
    use_full_train_metrics: bool = False
    micro_full_epochs_base: int = 1
    micro_full_epochs_max: int = 3


@dataclass
class DeltaScheduleConfig:
    early_delta_s: float = 0.00500
    early_until_sparsity: float = 1.000
    medium_delta_s: float = 0.00500
    medium_until_sparsity: float = 1.000
    late_delta_s: float = 0.00500
    late_until_sparsity: float = 1.000
    deep_sparse_delta_s: float = 0.00500


@dataclass
class SignalConfig:
    alpha_beta: float = 0.50
    alpha_sd: float = 0.30
    low_splus_beta: float = 1.00
    low_splus_sd: float = 0.50
    medium_splus_beta: float = 1.25
    medium_splus_sd: float = 0.70
    high_splus_beta: float = 1.50
    high_splus_sd: float = 0.85
    high_attn_beta: float = 1.00
    high_mlp_beta: float = 2.00
    low_sparsity_cutoff: float = 0.20
    medium_sparsity_cutoff: float = 0.55
    min_layer_weight: float = 0.10


@dataclass
class DynamicSignalConfig:
    recompute_svd_every_steps: int = 5
    min_effective_rank_ratio: float = 0.35
    headroom_floor: float = 0.25
    enable_live_rmt_refresh: bool = True
    rmt_refresh_alpha: float = 0.25
    rmt_randomness_splus_weight: float = 0.40
    rmt_randomness_bulk_weight: float = 0.35
    rmt_randomness_mp_like_weight: float = 0.25
    rmt_structure_spike_mass_weight: float = 0.20
    rmt_structure_spike_count_weight: float = 0.10
    grad_ema_decay: float = 0.90
    grad_protection_percentile: float = 0.75
    grad_protection_scale: float = 0.70
    lr_randomness_scale: float = 0.12
    lr_randomness_clip_min: float = 0.85
    lr_randomness_clip_max: float = 1.20
    rmt_clip_randomness_scale: float = 0.10
    rmt_clip_spike_relief_scale: float = 0.10
    rmt_grad_interval_steps: int = 32
    rmt_grad_beta: float = 0.50
    rmt_grad_beta_bulk_scale: float = 0.10
    rmt_grad_beta_spike_scale: float = 0.08
    rmt_grad_min_gap_ratio: float = 1.35
    rmt_grad_max_rank: int = 8
    rmt_grad_min_matrix_dim: int = 64
    rmt_grad_min_bulk_erank_multiplier: float = 8.0


@dataclass
class MilestoneRMTConfig:
    subspace_max_rank: int = 4
    conservative_until_sparsity: float = 0.0
    conservative_delta_s: float = 1.00
    conservative_ft_steps: float = 1.00
    conservative_lr: float = 1.00
    bulk_collapse_bad_delta: float = -0.002
    bulk_collapse_good_delta: float = -0.010
    spike_mass_good_delta: float = 0.010
    outlier_ratio_good_delta: float = 0.03
    spike_stability_bad: float = 0.45
    spike_stability_good: float = 0.70
    collapse_delta_s: float = 0.65
    collapse_ft_steps: float = 1.40
    collapse_lr: float = 1.02
    stabilize_delta_s: float = 1.02
    stabilize_ft_steps: float = 0.98
    stabilize_lr: float = 1.00


@dataclass
class MilestoneFineTuneConfig:
    enabled: bool = True
    low_epochs: int = 1
    low_until_sparsity: float = 0.20
    medium_epochs: int = 2
    medium_until_sparsity: float = 0.45
    high_epochs: int = 3
    high_until_sparsity: float = 0.70
    lr_scale: float = 0.75


@dataclass
class RunConfig:
    model_name: str = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
    start_sparsity: float = 0.0
    target_sparsity: float = 0.70
    milestone_step: float = 0.05
    seed: int = 42
    batch_size_train: int = 128
    batch_size_val: int = 256
    train_probe_batch_size: int = 256
    num_workers: int = 4
    train_shuffle_buffer: int = 8192
    label_smoothing: float = 0.10
    weight_decay: float = 1.0e-6
    momentum: float = 0.90
    block_min_lr_fraction: float = 0.35
    block_warmup_fraction: float = 0.05
    global_grad_clip: float = 1.00
    optimizer_name: str = "sgd"
    enable_rmt_layer_budget: bool = True
    enable_rmt_live_refresh: bool = True
    enable_rmt_lr_scaling: bool = True
    enable_rmt_grad_shrink: bool = True
    enable_rmt_clip: bool = True
    output_dir: Path = Path("/workspace/finetune_results_adaptive")
    cache_dir: Path = Path(os.environ.get("RMT_CACHE", "./optuna_run/rmt_cache"))
    anchor_checkpoint: Path | None = None
    hf_snapshot_glob_train: str = "/workspace/hf_cache/hub/datasets--ILSVRC--imagenet-1k/snapshots/*/data/train-*.parquet"
    hf_snapshot_glob_val: str = "/workspace/hf_cache/hub/datasets--ILSVRC--imagenet-1k/snapshots/*/data/validation-*.parquet"
    enable_sv_cleanup: bool = False
    sv_cleanup_every_steps: int = 0
    sv_cleanup_z: float = 0.50
    sv_cleanup_power: float = 3.0
    sv_cleanup_min_bulk_fraction: float = 0.60
    sv_cleanup_min_mp_like: float = 0.50
    sv_cleanup_haar_max_error: float = 0.20


@dataclass
class AdaptivePruningConfig:
    bounds: AdaptiveBounds = field(default_factory=AdaptiveBounds)
    thresholds: RecoveryThresholds = field(default_factory=RecoveryThresholds)
    multipliers: ControllerMultipliers = field(default_factory=ControllerMultipliers)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    delta_schedule: DeltaScheduleConfig = field(default_factory=DeltaScheduleConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    dynamic_signals: DynamicSignalConfig = field(default_factory=DynamicSignalConfig)
    milestone_rmt: MilestoneRMTConfig = field(default_factory=MilestoneRMTConfig)
    milestone_finetune: MilestoneFineTuneConfig = field(default_factory=MilestoneFineTuneConfig)
    run: RunConfig = field(default_factory=RunConfig)
