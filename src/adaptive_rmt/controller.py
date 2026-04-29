from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

from .config import AdaptivePruningConfig


class ControllerRegime(str, Enum):
    NORMAL = "normal"
    RECOVERY = "recovery"
    ACCELERATE = "accelerate"


@dataclass
class AdaptiveObservation:
    step_index: int
    reference_probe_acc: float
    reference_probe_loss: float
    post_prune_probe_acc: float
    post_prune_probe_loss: float
    post_ft_probe_acc: float
    post_ft_probe_loss: float
    current_sparsity: float
    ft_steps_run: int
    headroom: float
    bulk_change: float | None = None
    spike_mass_change: float | None = None
    spike_stability: float | None = None
    outlier_ratio_change: float | None = None
    post_prune_logit_drift: float | None = None
    post_prune_margin_flip_rate: float | None = None
    post_prune_margin_change: float | None = None

    @property
    def shock_pp(self) -> float:
        return self.reference_probe_acc - self.post_prune_probe_acc

    @property
    def residual_drop_pp(self) -> float:
        return self.reference_probe_acc - self.post_ft_probe_acc

    @property
    def recovery_pp(self) -> float:
        return self.post_ft_probe_acc - self.post_prune_probe_acc

    @property
    def residual_loss_increase(self) -> float:
        return self.post_ft_probe_loss - self.reference_probe_loss

    @property
    def shock_loss_increase(self) -> float:
        return self.post_prune_probe_loss - self.reference_probe_loss


@dataclass
class AdaptiveStepPlan:
    step_index: int
    current_sparsity: float
    target_sparsity: float
    delta_s: float
    ft_steps: int
    ft_lr: float
    regime: ControllerRegime
    milestone_sparsity: float | None
    headroom: float


@dataclass
class AdaptiveController:
    config: AdaptivePruningConfig
    delta_s: float = field(init=False)
    ft_steps: int = field(init=False)
    ft_lr: float = field(init=False)
    regime: ControllerRegime = field(default=ControllerRegime.NORMAL, init=False)
    history: list[AdaptiveObservation] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.delta_s = self.config.bounds.initial_delta_s
        self.ft_steps = self.config.bounds.initial_ft_steps
        self.ft_lr = self.config.bounds.initial_ft_lr

    def state_dict(self) -> dict:
        return {
            "delta_s": self.delta_s,
            "ft_steps": self.ft_steps,
            "ft_lr": self.ft_lr,
            "regime": self.regime.value,
            "history": [obs.__dict__ for obs in self.history],
        }

    def load_state_dict(self, state: dict) -> None:
        self.delta_s = float(state.get("delta_s", self.delta_s))
        self.ft_steps = int(state.get("ft_steps", self.ft_steps))
        self.ft_lr = float(state.get("ft_lr", self.ft_lr))
        self.regime = ControllerRegime(state.get("regime", ControllerRegime.NORMAL.value))
        self.history = [AdaptiveObservation(**item) for item in state.get("history", [])]

    def reset_policy_state(self) -> None:
        self.delta_s = self.config.bounds.initial_delta_s
        self.ft_steps = self.config.bounds.initial_ft_steps
        self.ft_lr = self.config.bounds.initial_ft_lr
        self.regime = ControllerRegime.NORMAL

    def plan(self, step_index: int, current_sparsity: float, headroom: float) -> AdaptiveStepPlan:
        bounds = self.config.bounds
        run_cfg = self.config.run

        remaining = max(0.0, run_cfg.target_sparsity - current_sparsity)
        if remaining <= 0.0:
            return AdaptiveStepPlan(
                step_index=step_index,
                current_sparsity=current_sparsity,
                target_sparsity=current_sparsity,
                delta_s=0.0,
                ft_steps=0,
                ft_lr=0.0,
                regime=self.regime,
                milestone_sparsity=None,
                headroom=headroom,
            )

        proposed_delta = self.delta_s
        schedule_cap = self._scheduled_delta_cap(current_sparsity)
        delta = self._clamp(proposed_delta, bounds.delta_s_min, bounds.delta_s_max)
        delta = min(delta, schedule_cap)
        delta = min(delta, remaining)

        ft_steps = int(round(self._clamp(self.ft_steps, bounds.ft_steps_min, bounds.ft_steps_max)))
        ft_lr = self._clamp(self.ft_lr, bounds.ft_lr_min, bounds.ft_lr_max)
        milestone = self._next_milestone(current_sparsity)
        target = current_sparsity + delta
        crossed = milestone is not None and target + 1.0e-9 >= milestone
        if crossed and milestone is not None:
            target = min(milestone, current_sparsity + remaining)
            delta = max(0.0, target - current_sparsity)

        return AdaptiveStepPlan(
            step_index=step_index,
            current_sparsity=current_sparsity,
            target_sparsity=target,
            delta_s=delta,
            ft_steps=ft_steps,
            ft_lr=ft_lr,
            regime=self.regime,
            milestone_sparsity=milestone if crossed else None,
            headroom=headroom,
        )

    def observe(self, observation: AdaptiveObservation) -> None:
        self.history.append(observation)
        thresholds = self.config.thresholds
        bounds = self.config.bounds
        mult = self.config.multipliers

        if (
            observation.residual_drop_pp > thresholds.hard_residual_drop_pp
            or observation.residual_loss_increase > thresholds.hard_loss_increase
            or (observation.post_prune_logit_drift is not None and observation.post_prune_logit_drift > thresholds.hard_logit_drift)
            or (observation.post_prune_margin_flip_rate is not None and observation.post_prune_margin_flip_rate > thresholds.hard_margin_flip_rate)
        ):
            self.regime = ControllerRegime.RECOVERY
            self.delta_s *= mult.recovery_delta_s
            self.ft_steps = int(math.ceil(self.ft_steps * mult.recovery_ft_steps))
            self.ft_lr *= mult.recovery_lr
        elif (
            observation.residual_drop_pp > thresholds.soft_residual_drop_pp
            or observation.residual_loss_increase > thresholds.soft_loss_increase
            or (observation.post_prune_logit_drift is not None and observation.post_prune_logit_drift > thresholds.soft_logit_drift)
            or (observation.post_prune_margin_flip_rate is not None and observation.post_prune_margin_flip_rate > thresholds.soft_margin_flip_rate)
        ):
            self.regime = ControllerRegime.NORMAL
            self.delta_s *= mult.soft_delta_s
            self.ft_steps = int(math.ceil(self.ft_steps * mult.soft_ft_steps))
            self.ft_lr *= mult.soft_lr
        elif (
            observation.residual_drop_pp <= thresholds.easy_residual_drop_pp
            and observation.shock_pp <= thresholds.easy_shock_pp
            and observation.residual_loss_increase <= thresholds.easy_loss_increase
        ):
            self.regime = ControllerRegime.NORMAL
            self.delta_s = bounds.initial_delta_s
            self.ft_steps = bounds.initial_ft_steps
            self.ft_lr = bounds.initial_ft_lr
        else:
            self.regime = ControllerRegime.NORMAL

        self.delta_s = self._clamp(self.delta_s, bounds.delta_s_min, bounds.delta_s_max)
        self.ft_steps = int(round(self._clamp(self.ft_steps, bounds.ft_steps_min, bounds.ft_steps_max)))
        self.ft_lr = self._clamp(self.ft_lr, bounds.ft_lr_min, bounds.ft_lr_max)

    def _next_milestone(self, current_sparsity: float) -> float | None:
        milestone_step = self.config.run.milestone_step
        next_bucket = math.floor((current_sparsity + 1.0e-9) / milestone_step) + 1
        milestone = round(next_bucket * milestone_step, 10)
        if milestone > self.config.run.target_sparsity + 1.0e-9:
            return None
        return milestone

    def _scheduled_delta_cap(self, current_sparsity: float) -> float:
        schedule = self.config.delta_schedule
        if current_sparsity < schedule.early_until_sparsity:
            return schedule.early_delta_s
        if current_sparsity < schedule.medium_until_sparsity:
            return schedule.medium_delta_s
        if current_sparsity < schedule.late_until_sparsity:
            return schedule.late_delta_s
        return schedule.deep_sparse_delta_s

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))
