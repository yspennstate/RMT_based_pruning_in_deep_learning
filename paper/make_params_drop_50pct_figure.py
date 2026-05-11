from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Point:
    label: str
    family: str
    params_m: float
    drop_pp: float
    marker: str


POINTS = [
    Point("ViT-B 6:12", "ViT", 86.6, 1.37, "D"),
    Point("ViT-L 8:16", "ViT", 304.3, 0.51, "D"),
    Point("RN50 2:4", "ResNet", 25.6, 2.99, "o"),
    Point("RN50 2:4+perm", "ResNet", 25.6, 0.46, "s"),
    Point("RN50 8:16", "ResNet", 25.6, 0.26, "D"),
    Point("RN50d 2:4", "ResNet", 25.6, 2.47, "o"),
    Point("RN50d 2:4+perm", "ResNet", 25.6, 2.55, "s"),
    Point("RN50d 8:16", "ResNet", 25.6, 1.98, "D"),
    Point("RN101d 2:4", "ResNet", 44.6, 2.13, "o"),
    Point("RN101d 2:4+perm", "ResNet", 44.6, 1.67, "s"),
    Point("RN101d 8:16", "ResNet", 44.6, 1.34, "D"),
    Point("RN152d 2:4+perm", "ResNet", 60.2, 1.53, "s"),
    Point("CNv2-B 2:4", "ConvNeXtV2", 89.0, 1.25, "o"),
    Point("CNv2-B 8:16", "ConvNeXtV2", 89.0, 0.87, "D"),
]


COLORS = {
    "ViT": "#335c81",
    "ResNet": "#8f4f2f",
    "ConvNeXtV2": "#3f7d58",
}


OFFSETS = {
    "ViT-B 6:12": (5, 6),
    "ViT-L 8:16": (-52, 2),
    "RN50 2:4": (5, 3),
    "RN50 2:4+perm": (5, 0),
    "RN50 8:16": (5, -8),
    "RN50d 2:4": (5, -1),
    "RN50d 2:4+perm": (5, 5),
    "RN50d 8:16": (5, -3),
    "RN101d 2:4": (5, 2),
    "RN101d 2:4+perm": (5, 1),
    "RN101d 8:16": (5, -5),
    "RN152d 2:4+perm": (5, -1),
    "CNv2-B 2:4": (5, 4),
    "CNv2-B 8:16": (5, -5),
}


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    params = np.array([p.params_m for p in POINTS])
    drops = np.array([p.drop_pp for p in POINTS])
    log_params = np.log10(params)
    slope, intercept = np.polyfit(log_params, drops, deg=1)
    x_fit = np.linspace(params.min() * 0.9, params.max() * 1.1, 200)
    y_fit = slope * np.log10(x_fit) + intercept

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)

    for point in POINTS:
        ax.scatter(
            point.params_m,
            point.drop_pp,
            s=50,
            marker=point.marker,
            color=COLORS[point.family],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        dx, dy = OFFSETS[point.label]
        ax.annotate(
            point.label,
            (point.params_m, point.drop_pp),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.2,
            color="#222222",
        )

    ax.plot(x_fit, y_fit, color="#555555", linestyle="--", linewidth=1.1)

    family_handles = [
        ax.scatter([], [], color=color, s=45, marker="o", label=family)
        for family, color in COLORS.items()
    ]
    method_handles = [
        ax.scatter([], [], color="#777777", s=38, marker="o", label="2:4"),
        ax.scatter([], [], color="#777777", s=38, marker="s", label="2:4+perm"),
        ax.scatter([], [], color="#777777", s=38, marker="D", label="wider pattern"),
    ]
    ax.legend(handles=family_handles + method_handles, loc="upper right", frameon=True, framealpha=0.95)

    ax.set_xscale("log")
    ax.set_xlim(22, 340)
    ax.set_ylim(0, 3.25)
    ax.set_xlabel("Original dense parameters (millions)")
    ax.set_ylabel("Top-1 drop from dense (percentage points)")
    ax.grid(True, which="major", color="#d0d0d0", linewidth=0.7, alpha=0.75)
    ax.grid(True, which="minor", axis="x", color="#e6e6e6", linewidth=0.4, alpha=0.55)
    ax.set_axisbelow(True)

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_top1_vs_params_at_50pct_MAC.{ext}", dpi=300)


if __name__ == "__main__":
    main()
