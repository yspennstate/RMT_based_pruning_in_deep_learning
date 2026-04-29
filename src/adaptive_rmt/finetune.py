from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import AdaptivePruningConfig
from .controller import AdaptiveStepPlan
from .data import evaluate_model_metrics
from .model_utils import apply_mask, iter_target_layers_modules
from .signals import LayerSignalState, update_gradient_statistics


@dataclass
class AdaptiveBlockResult:
    steps_run: int
    final_train_acc: float
    best_probe_acc: float
    best_probe_loss: float
    stop_reason: str


@dataclass
class MilestoneFineTuneResult:
    epochs_run: int
    steps_run: int
    final_train_acc: float
    final_probe_acc: float
    final_probe_loss: float


def make_optimizer(
    model: nn.Module,
    config: AdaptivePruningConfig,
    lr: float,
    signal_bank: dict[str, LayerSignalState],
):
    parameter_groups = []
    seen_param_ids: set[int] = set()
    for name, mod in iter_target_layers_modules(model):
        params = [param for param in mod.parameters(recurse=False) if param.requires_grad]
        if not params:
            continue
        for param in params:
            seen_param_ids.add(id(param))
        parameter_groups.append(
            {
                "params": params,
                "lr": lr,
                "lr_scale": (
                    signal_bank[name].lr_scale
                    if config.run.enable_rmt_lr_scaling and name in signal_bank
                    else 1.0
                ),
            }
        )

    leftover = [param for param in model.parameters() if param.requires_grad and id(param) not in seen_param_ids]
    if leftover:
        parameter_groups.append({"params": leftover, "lr": lr, "lr_scale": 1.0})

    if config.run.optimizer_name.lower() == "adamw":
        return optim.AdamW(parameter_groups, lr=lr, weight_decay=config.run.weight_decay)
    return optim.SGD(
        parameter_groups,
        lr=lr,
        momentum=config.run.momentum,
        weight_decay=config.run.weight_decay,
    )


def block_lr_at(step: int, total_steps: int, base_lr: float, config: AdaptivePruningConfig) -> float:
    warmup = max(1, int(round(total_steps * config.run.block_warmup_fraction)))
    min_lr = base_lr * config.run.block_min_lr_fraction
    if step < warmup:
        return min_lr + (base_lr - min_lr) * step / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def apply_budget_aware_gradient_clip(
    model: nn.Module,
    signal_bank: dict[str, LayerSignalState],
    config: AdaptivePruningConfig,
) -> None:
    base_clip = config.run.global_grad_clip
    for name, mod in iter_target_layers_modules(model):
        grad = mod.weight.grad
        if grad is None:
            continue
        layer_state = signal_bank[name]
        headroom = max(config.dynamic_signals.headroom_floor, layer_state.headroom)
        layer_cap = base_clip * (0.50 + 0.50 * headroom)
        randomness_pressure = max(layer_state.dynamic_randomness_z, 0.0)
        structured_relief = max(layer_state.spike_mass_ratio_z, 0.0)
        layer_cap *= 1.0 - config.dynamic_signals.rmt_clip_randomness_scale * min(randomness_pressure, 2.0)
        layer_cap *= 1.0 + config.dynamic_signals.rmt_clip_spike_relief_scale * min(structured_relief, 2.0)
        layer_cap = float(np.clip(layer_cap, 0.35 * base_clip, 1.25 * base_clip))
        grad_norm = float(grad.norm().item())
        if grad_norm > layer_cap:
            grad.mul_(layer_cap / (grad_norm + 1.0e-12))


def apply_rmt_bulk_gradient_shrink(
    model: nn.Module,
    signal_bank: dict[str, LayerSignalState],
    config: AdaptivePruningConfig,
) -> dict[str, float]:
    stats: dict[str, float] = {"layers": 0, "bulk_erank_sum": 0.0, "spike_rank_sum": 0.0, "beta_sum": 0.0}
    beta_base = config.dynamic_signals.rmt_grad_beta
    min_gap_ratio = config.dynamic_signals.rmt_grad_min_gap_ratio
    max_rank = config.dynamic_signals.rmt_grad_max_rank
    min_dim = config.dynamic_signals.rmt_grad_min_matrix_dim
    erank_multiplier = config.dynamic_signals.rmt_grad_min_bulk_erank_multiplier

    for name, mod in iter_target_layers_modules(model):
        grad = mod.weight.grad
        if grad is None:
            continue
        matrix = grad
        original_shape = grad.shape
        if grad.ndim == 4:
            matrix = grad.reshape(grad.shape[0], -1)
        if min(matrix.shape) < min_dim:
            continue

        matrix_fp32 = matrix.detach().float()
        try:
            u_mat, singular_values, vt_mat = torch.linalg.svd(matrix_fp32, full_matrices=False)
        except RuntimeError:
            continue
        if singular_values.numel() <= 1:
            continue

        candidate_count = min(max_rank, singular_values.numel() - 1)
        gap_ratios = singular_values[:candidate_count] / singular_values[1 : candidate_count + 1].clamp_min(1.0e-12)
        best_idx = int(torch.argmax(gap_ratios).item())
        best_gap = float(gap_ratios[best_idx].item())
        if best_gap < min_gap_ratio:
            continue

        spike_rank = best_idx + 1
        bulk_singular_values = singular_values[spike_rank:]
        if bulk_singular_values.numel() == 0:
            continue

        bulk_erank = float(
            (bulk_singular_values.square().sum() / bulk_singular_values.max().square().clamp_min(1.0e-12)).item()
        )
        if bulk_erank <= erank_multiplier * max(spike_rank, 1):
            continue

        layer_state = signal_bank[name]
        randomness_pressure = max(layer_state.dynamic_randomness_z, 0.0)
        structured_relief = max(layer_state.spike_mass_ratio_z, 0.0)
        beta_here = beta_base
        beta_here -= config.dynamic_signals.rmt_grad_beta_bulk_scale * min(randomness_pressure, 2.0)
        beta_here -= 0.10 * max(layer_state.bulk_fraction - 0.50, 0.0)
        beta_here += config.dynamic_signals.rmt_grad_beta_spike_scale * min(structured_relief, 2.0)
        beta_here += 0.10 * max(layer_state.spike_mass_ratio - 0.25, 0.0)
        beta_here = max(0.25, min(0.80, beta_here))
        shrunk_singular_values = singular_values.clone()
        shrunk_singular_values[spike_rank:] *= beta_here
        filtered = (u_mat * shrunk_singular_values.unsqueeze(0)) @ vt_mat
        grad.copy_(filtered.reshape(original_shape).to(grad.dtype))

        stats["layers"] += 1
        stats["bulk_erank_sum"] += bulk_erank
        stats["spike_rank_sum"] += spike_rank
        stats["beta_sum"] += beta_here

    return stats


def run_finetune_block(
    *,
    model: nn.Module,
    masks: dict[str, torch.Tensor],
    train_loader,
    probe_loader_fn: Callable[[], object],
    device: torch.device,
    plan: AdaptiveStepPlan,
    config: AdaptivePruningConfig,
    signal_bank: dict[str, LayerSignalState],
    reference_probe_acc: float,
    reference_probe_loss: float,
    log_fn,
) -> AdaptiveBlockResult:
    optimizer = make_optimizer(model, config, plan.ft_lr, signal_bank)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.run.label_smoothing)

    best_probe = float("-inf")
    best_probe_loss = float("inf")
    steps_run = 0
    train_correct = 0
    train_total = 0
    stop_reason = "recovery_cap"
    min_recovery_steps = max(plan.ft_steps, config.probe.recovery_min_steps)
    max_steps = config.bounds.ft_steps_max
    lr_schedule_steps = min(max_steps, max(min_recovery_steps, int(math.ceil(min_recovery_steps * 1.5))))
    recovery_target_acc = reference_probe_acc - config.probe.recovery_target_drop_pp
    recovery_target_loss = reference_probe_loss + config.probe.recovery_loss_tolerance

    model.train()
    for batch_idx, (images, targets) in enumerate(train_loader):
        if steps_run >= max_steps:
            break

        current_lr = block_lr_at(steps_run, lr_schedule_steps, plan.ft_lr, config)
        for group in optimizer.param_groups:
            group["lr"] = current_lr * float(group.get("lr_scale", 1.0))

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        autocast_kwargs = {"device_type": device.type, "enabled": device.type == "cuda"}
        if device.type == "cuda":
            autocast_kwargs["dtype"] = torch.bfloat16
        with torch.amp.autocast(**autocast_kwargs):
            output = model(images)
            loss = criterion(output, targets)

        loss.backward()

        for name, mod in iter_target_layers_modules(model):
            if name in masks and mod.weight.grad is not None:
                mod.weight.grad.mul_(masks[name].to(mod.weight.grad.device, dtype=mod.weight.grad.dtype))

        if config.run.enable_rmt_grad_shrink and steps_run % max(config.dynamic_signals.rmt_grad_interval_steps, 1) == 0:
            rmt_stats = apply_rmt_bulk_gradient_shrink(model, signal_bank, config)
            if rmt_stats["layers"] > 0 and steps_run % 256 == 0:
                avg_erank = rmt_stats["bulk_erank_sum"] / rmt_stats["layers"]
                avg_rank = rmt_stats["spike_rank_sum"] / rmt_stats["layers"]
                avg_beta = rmt_stats["beta_sum"] / rmt_stats["layers"]
                log_fn(
                    f"  rmt-grad step={steps_run} layers={int(rmt_stats['layers'])} "
                    f"avg_bulk_erank={avg_erank:.2f} avg_spike_rank={avg_rank:.2f} avg_beta={avg_beta:.2f}"
                )
        update_gradient_statistics(model, signal_bank, config)
        if config.run.enable_rmt_clip:
            apply_budget_aware_gradient_clip(model, signal_bank, config)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.run.global_grad_clip)
        optimizer.step()
        apply_mask(model, masks)

        with torch.no_grad():
            prediction = output.argmax(dim=1)
            train_correct += int((prediction == targets).sum().item())
            train_total += int(targets.size(0))

        steps_run += 1
        if steps_run % 100 == 0:
            train_acc = 100.0 * train_correct / max(train_total, 1)
            log_fn(
                f"  micro-step {steps_run}/{max_steps} "
                f"loss={loss.item():.4f} train_acc={train_acc:.2f}% lr={current_lr:.2e}"
            )

        should_probe = steps_run % config.probe.probe_interval_steps == 0 or steps_run == plan.ft_steps
        if not should_probe:
            continue

        probe_acc, probe_loss = evaluate_model_metrics(
            model,
            probe_loader_fn(),
            device,
            log_fn=log_fn,
            label=f"probe step={steps_run}",
            max_batches=None if config.probe.use_full_train_metrics else config.probe.probe_batches,
        )
        if probe_acc > best_probe or probe_loss < best_probe_loss:
            best_probe = probe_acc
            best_probe_loss = probe_loss

        recovered_probe = probe_acc >= recovery_target_acc and probe_loss <= recovery_target_loss
        if steps_run >= min_recovery_steps and recovered_probe:
            stop_reason = "recovered_probe"
            log_fn(
                f"  recovered train probe at step={steps_run} "
                f"probe_acc={probe_acc:.2f}% target_acc={recovery_target_acc:.2f}% "
                f"probe_loss={probe_loss:.4f} target_loss={recovery_target_loss:.4f}"
            )
            break

    final_train_acc = 100.0 * train_correct / max(train_total, 1)
    if best_probe == float("-inf"):
        best_probe, best_probe_loss = evaluate_model_metrics(
            model,
            probe_loader_fn(),
            device,
            log_fn=log_fn,
            label="probe final",
            max_batches=None if config.probe.use_full_train_metrics else config.probe.probe_batches,
        )
    return AdaptiveBlockResult(
        steps_run=steps_run,
        final_train_acc=final_train_acc,
        best_probe_acc=best_probe,
        best_probe_loss=best_probe_loss,
        stop_reason=stop_reason,
    )


def milestone_macro_epochs_for_sparsity(config: AdaptivePruningConfig, sparsity: float) -> int:
    milestone_cfg = config.milestone_finetune
    if not milestone_cfg.enabled:
        return 0
    if sparsity <= milestone_cfg.low_until_sparsity + 1.0e-9:
        return milestone_cfg.low_epochs
    if sparsity <= milestone_cfg.medium_until_sparsity + 1.0e-9:
        return milestone_cfg.medium_epochs
    if sparsity <= milestone_cfg.high_until_sparsity + 1.0e-9:
        return milestone_cfg.high_epochs
    return 0


def run_milestone_finetune_epochs(
    *,
    model: nn.Module,
    masks: dict[str, torch.Tensor],
    train_loader,
    probe_loader_fn: Callable[[], object],
    device: torch.device,
    base_lr: float,
    epochs: int,
    config: AdaptivePruningConfig,
    signal_bank: dict[str, LayerSignalState],
    log_fn,
    label: str,
) -> MilestoneFineTuneResult:
    if epochs <= 0:
        probe_acc, probe_loss = evaluate_model_metrics(
            model,
            probe_loader_fn(),
            device,
            log_fn=log_fn,
            label=f"{label} probe final",
            max_batches=None if config.probe.use_full_train_metrics else config.probe.probe_batches,
        )
        return MilestoneFineTuneResult(
            epochs_run=0,
            steps_run=0,
            final_train_acc=0.0,
            final_probe_acc=probe_acc,
            final_probe_loss=probe_loss,
        )

    optimizer = make_optimizer(
        model,
        config,
        base_lr * config.milestone_finetune.lr_scale,
        signal_bank,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.run.label_smoothing)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, epochs * steps_per_epoch)
    steps_run = 0
    train_correct = 0
    train_total = 0
    final_probe_acc = 0.0
    final_probe_loss = 0.0

    model.train()
    for epoch_idx in range(epochs):
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_total = 0
        for step_in_epoch, (images, targets) in enumerate(train_loader):
            current_lr = block_lr_at(steps_run, total_steps, base_lr * config.milestone_finetune.lr_scale, config)
            for group in optimizer.param_groups:
                group["lr"] = current_lr * float(group.get("lr_scale", 1.0))

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            autocast_kwargs = {"device_type": device.type, "enabled": device.type == "cuda"}
            if device.type == "cuda":
                autocast_kwargs["dtype"] = torch.bfloat16
            with torch.amp.autocast(**autocast_kwargs):
                output = model(images)
                loss = criterion(output, targets)

            loss.backward()

            for name, mod in iter_target_layers_modules(model):
                if name in masks and mod.weight.grad is not None:
                    mod.weight.grad.mul_(masks[name].to(mod.weight.grad.device, dtype=mod.weight.grad.dtype))

            if config.run.enable_rmt_grad_shrink and steps_run % max(config.dynamic_signals.rmt_grad_interval_steps, 1) == 0:
                rmt_stats = apply_rmt_bulk_gradient_shrink(model, signal_bank, config)
                if rmt_stats["layers"] > 0 and steps_run % 256 == 0:
                    avg_erank = rmt_stats["bulk_erank_sum"] / rmt_stats["layers"]
                    avg_rank = rmt_stats["spike_rank_sum"] / rmt_stats["layers"]
                    avg_beta = rmt_stats["beta_sum"] / rmt_stats["layers"]
                    log_fn(
                        f"  {label} rmt-grad step={steps_run} layers={int(rmt_stats['layers'])} "
                        f"avg_bulk_erank={avg_erank:.2f} avg_spike_rank={avg_rank:.2f} avg_beta={avg_beta:.2f}"
                    )
            update_gradient_statistics(model, signal_bank, config)
            if config.run.enable_rmt_clip:
                apply_budget_aware_gradient_clip(model, signal_bank, config)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.run.global_grad_clip)
            optimizer.step()
            apply_mask(model, masks)

            with torch.no_grad():
                prediction = output.argmax(dim=1)
                correct_here = int((prediction == targets).sum().item())
                batch_total = int(targets.size(0))
                epoch_correct += correct_here
                epoch_total += batch_total
                train_correct += correct_here
                train_total += batch_total
                epoch_loss_sum += float(loss.item()) * batch_total

            steps_run += 1

        epoch_train_acc = 100.0 * epoch_correct / max(epoch_total, 1)
        epoch_train_loss = epoch_loss_sum / max(epoch_total, 1)
        final_probe_acc, final_probe_loss = evaluate_model_metrics(
            model,
            probe_loader_fn(),
            device,
            log_fn=log_fn,
            label=f"{label} epoch={epoch_idx + 1}/{epochs}",
            max_batches=None if config.probe.use_full_train_metrics else config.probe.probe_batches,
        )
        log_fn(
            f"  {label} epoch {epoch_idx + 1}/{epochs} done "
            f"train_loss={epoch_train_loss:.4f} train_acc={epoch_train_acc:.2f}%"
        )

    final_train_acc = 100.0 * train_correct / max(train_total, 1)
    return MilestoneFineTuneResult(
        epochs_run=epochs,
        steps_run=steps_run,
        final_train_acc=final_train_acc,
        final_probe_acc=final_probe_acc,
        final_probe_loss=final_probe_loss,
    )
