from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def _mp_density_bounds(ndf: int, pdim: int, var: float = 1.0) -> tuple[float, float]:
    gamma = ndf / pdim
    inv_gamma_sqrt = math.sqrt(1.0 / gamma)
    a = var * (1.0 - inv_gamma_sqrt) ** 2
    b = var * (1.0 + inv_gamma_sqrt) ** 2
    return a, b


def _dmp(x: float, ndf: int, pdim: int, var: float = 1.0) -> float:
    gamma = ndf / pdim
    a, b = _mp_density_bounds(ndf, pdim, var)
    if x <= a or x >= b:
        return 0.0
    return gamma / (2.0 * math.pi * var * x) * math.sqrt(max(x - a, 0.0) * max(b - x, 0.0))


def _pmp(q: float, ndf: int, pdim: int, var: float = 1.0) -> float:
    gamma = ndf / pdim
    a, b = _mp_density_bounds(ndf, pdim, var)
    if q <= a:
        p = 0.0
    elif q >= b:
        p = 1.0
    else:
        p = quad(lambda x: _dmp(x, ndf, pdim, var), a, q)[0]
    if gamma < 1.0 and q >= 0.0:
        p += 1.0 - gamma
    return p


def _qmp(p: float, ndf: int, pdim: int, var: float = 1.0) -> float:
    svr = ndf / pdim
    a, b = _mp_density_bounds(ndf, pdim, var)
    if p <= 0.0:
        return 0.0 if svr <= 1.0 else a
    if p >= 1.0:
        return b
    if svr < 1.0:
        if p < 1.0 - svr:
            return 0.0
        if p == 1.0 - svr:
            return 0.0
    return brentq(lambda x: _pmp(x, ndf, pdim, var) - p, a, b)


def _mp_cdf_inner(gamma: float, sigma_sq: float, x: float) -> float:
    lp = sigma_sq * (1.0 + math.sqrt(gamma)) ** 2
    lm = sigma_sq * (1.0 - math.sqrt(gamma)) ** 2
    r = math.sqrt(max((lp - x) / max(x - lm, 1.0e-12), 0.0))
    f_val = math.pi * gamma + (1.0 / sigma_sq) * math.sqrt(max((lp - x) * (x - lm), 0.0))
    f_val += -(1.0 + gamma) * math.atan((r * r - 1.0) / max(2.0 * r, 1.0e-12))
    if gamma != 1.0:
        f_val += (1.0 - gamma) * math.atan(
            (lm * r * r - lp) / max(2.0 * sigma_sq * (1.0 - gamma) * r, 1.0e-12)
        )
    return f_val / (2.0 * math.pi * gamma)


def _mp_cdf(gamma: float, sigma_sq: float, sample_points: np.ndarray) -> np.ndarray:
    lp = sigma_sq * (1.0 + math.sqrt(gamma)) ** 2
    lm = sigma_sq * (1.0 - math.sqrt(gamma)) ** 2
    output: list[float] = []
    for x in sample_points:
        if gamma <= 1.0:
            if x < lm:
                output.append(0.0)
            elif x >= lp:
                output.append(1.0)
            else:
                output.append(_mp_cdf_inner(gamma, sigma_sq, float(x)))
        else:
            if x < lm:
                output.append((gamma - 1.0) / gamma)
            elif x >= lp:
                output.append(1.0)
            else:
                output.append((gamma - 1.0) / (2.0 * gamma) + _mp_cdf_inner(gamma, sigma_sq, float(x)))
    return np.asarray(output, dtype=np.float64)


def _empirical_cdf(sample_points: np.ndarray) -> np.ndarray:
    return np.asarray([i / len(sample_points) for i in range(len(sample_points))], dtype=np.float64)


@lru_cache(maxsize=128)
def _bema_unit_quantiles(pdim: int, ndf: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    p_tilde = min(pdim, ndf)
    lo = int(alpha * p_tilde)
    hi = int((1.0 - alpha) * p_tilde)
    if hi - lo < 8:
        lo = 0
        hi = p_tilde
    indices = np.arange(lo, hi, dtype=np.int64)
    quantiles = np.asarray([_qmp(i / p_tilde, ndf, pdim, 1.0) for i in indices], dtype=np.float64)
    return indices, quantiles


@dataclass
class RMTSpectralMetrics:
    sigma_sq: float
    splus: float
    mp_fit_error: float
    bulk_fraction: float
    spike_count: int
    spike_mass_ratio: float
    mean_outlier_ratio: float

    @property
    def bulk_randomness_score(self) -> float:
        mp_like = max(0.0, 1.0 - self.mp_fit_error)
        return mp_like * self.bulk_fraction


@dataclass
class RMTSpectralSnapshot:
    metrics: RMTSpectralMetrics
    left_basis: np.ndarray
    right_basis: np.ndarray


def _metrics_from_singular_values(
    singular_values: np.ndarray,
    rows: int,
    cols: int,
    alpha: float,
) -> RMTSpectralMetrics:
    p_dim = min(rows, cols)
    n_df = max(rows, cols)
    if singular_values.size == 0:
        return RMTSpectralMetrics(
            sigma_sq=0.0,
            splus=0.0,
            mp_fit_error=1.0,
            bulk_fraction=1.0,
            spike_count=0,
            spike_mass_ratio=0.0,
            mean_outlier_ratio=0.0,
        )

    eigenvalues = np.sort((singular_values ** 2) / max(n_df, 1))
    indices, unit_quantiles = _bema_unit_quantiles(p_dim, n_df, alpha)
    selected_eigs = eigenvalues[indices]
    numerator = float(np.dot(unit_quantiles, selected_eigs))
    denominator = float(np.dot(unit_quantiles, unit_quantiles)) + 1.0e-12
    sigma_sq = max(numerator / denominator, 1.0e-12)

    gamma = p_dim / max(n_df, 1)
    lambda_plus = sigma_sq * (1.0 + math.sqrt(gamma)) ** 2
    splus = math.sqrt(max(n_df, 1) * lambda_plus)

    central_empirical = alpha + (1.0 - 2.0 * alpha) * _empirical_cdf(selected_eigs)
    central_theoretical = _mp_cdf(gamma, sigma_sq, selected_eigs)
    mp_fit_error = float(np.linalg.norm(central_theoretical - central_empirical, ord=np.inf))

    bulk_mask = singular_values <= splus
    spike_mask = ~bulk_mask
    spike_energy = float(np.square(singular_values[spike_mask]).sum())
    total_energy = float(np.square(singular_values).sum()) + 1.0e-12
    outlier_ratios = singular_values[spike_mask] / max(splus, 1.0e-12)

    return RMTSpectralMetrics(
        sigma_sq=sigma_sq,
        splus=float(splus),
        mp_fit_error=mp_fit_error,
        bulk_fraction=float(bulk_mask.mean()),
        spike_count=int(spike_mask.sum()),
        spike_mass_ratio=spike_energy / total_energy,
        mean_outlier_ratio=float(outlier_ratios.mean()) if outlier_ratios.size else 0.0,
    )


def estimate_weight_rmt_metrics(weight_2d: np.ndarray, alpha: float = 0.25) -> RMTSpectralMetrics:
    rows, cols = weight_2d.shape
    singular_values = np.linalg.svd(weight_2d.astype(np.float64, copy=False), compute_uv=False)
    singular_values = np.asarray(singular_values, dtype=np.float64)
    return _metrics_from_singular_values(singular_values, rows, cols, alpha)


def estimate_weight_rmt_snapshot(
    weight_2d: np.ndarray,
    alpha: float = 0.25,
    max_basis_rank: int = 4,
) -> RMTSpectralSnapshot:
    rows, cols = weight_2d.shape
    u_mat, singular_values, vt_mat = np.linalg.svd(weight_2d.astype(np.float64, copy=False), full_matrices=False)
    singular_values = np.asarray(singular_values, dtype=np.float64)
    metrics = _metrics_from_singular_values(singular_values, rows, cols, alpha)
    basis_rank = min(max(metrics.spike_count, 0), max(max_basis_rank, 0))
    left_basis = np.asarray(u_mat[:, :basis_rank], dtype=np.float64, order="F")
    right_basis = np.asarray(vt_mat[:basis_rank, :].T, dtype=np.float64, order="F")
    return RMTSpectralSnapshot(
        metrics=metrics,
        left_basis=left_basis,
        right_basis=right_basis,
    )
