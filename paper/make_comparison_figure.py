"""
Comparison figure: post-FT Top-1 vs structured-sparsity MAC reduction across
all baselines tabulated in tab:result_comparison + this paper's result
CAST rows.

Output: comparison_top1_vs_macred.png (raster, included in main.tex).
WDAC blocks matplotlib PDF export on this machine; PNG is fine.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (method, architecture, mac_reduction_pct, top1, ours, family)
data = [
    # DeiT-Base baselines reproduced by CP-ViT (head-to-head)
    ("VTP", "DeiT-B", 43.20, 80.70, False, "deit_head"),
    ("PoWER-BERT", "DeiT-B", 39.24, 80.17, False, "deit_head"),
    ("HVT", "DeiT-B", 44.78, 79.94, False, "deit_head"),
    ("CP-ViT", "DeiT-B", 41.62, 81.13, False, "deit_head"),
    # NViT / SViTE / Spartan
    ("NViT", "DeiT-B", 61.28, 83.29, False, "vit_pruning"),
    ("SViTE", "DeiT-B", 50.00, 81.51, False, "vit_pruning"),
    ("Spartan", "ViT-B", 90.00, 81.18, False, "vit_pruning"),
    # Hardware-friendly k:n structured-sparsity
    ("GPUSQ-ViT INT8", "DeiT-B", 50.00, 82.90, False, "kn_hw"),
    ("GPUSQ-ViT INT4", "DeiT-B", 50.00, 81.60, False, "kn_hw"),
    ("Beyond 2:4 (300e)", "DeiT-B", 75.00, 81.08, False, "kn_hw"),
    ("Beyond 2:4 (600e)", "DeiT-B", 75.00, 81.76, False, "kn_hw"),
    ("ELSA", "DeiT-B", 50.00, 81.60, False, "kn_hw"),
    ("LPViT", "DeiT-B", 50.00, 80.81, False, "kn_hw"),
    ("SERo", "DeiT-B", 53.85, 80.25, False, "kn_hw"),
    # This paper
    ("Hybrid Mag--SER", "ViT-B", 50.00, 83.37, True, "ours_vitb"),
    ("CAST 2:4+ToMe", "ViT-B", 59.81, 83.41, True, "ours_vitb"),
    ("CAST 6:12", "ViT-B", 50.00, 83.74, True, "ours_vitb"),
    ("CAST 8:16 dense+perm", "ViT-L", 50.00, 85.33, True, "ours_vitl"),
    ("CAST-conv+perm", "ResNet152d", 50.00, 81.33, True, "ours_conv"),
    ("CAST 8:16", "ConvNeXtV2-B", 50.00, 85.85, True, "ours_conv"),
    ("CAST 12:16", "ConvNeXtV2-B", 25.00, 86.35, True, "ours_conv"),
]

family_colors = {
    "deit_head": "#888888",
    "vit_pruning": "#5b8def",
    "kn_hw": "#e69138",
    "ours_vitb": "#c0392b",
    "ours_vitl": "#8e44ad",
    "ours_conv": "#16a085",
}
family_labels = {
    "deit_head": "DeiT-B head-to-head (VTP/PoWER/HVT/CP-ViT)",
    "vit_pruning": "ViT pruning baselines (NViT/SViTE/Spartan)",
    "kn_hw": "Hardware-friendly k:n baselines",
    "ours_vitb": "This paper, ViT-B/16",
    "ours_vitl": "This paper, ViT-L/16 (new result)",
    "ours_conv": "This paper, ConvNeXt / ResNet",
}

fig, ax = plt.subplots(figsize=(10, 7))

# Plot baselines
for fam in ["deit_head", "vit_pruning", "kn_hw"]:
    xs = [d[2] for d in data if d[5] == fam]
    ys = [d[3] for d in data if d[5] == fam]
    ax.scatter(
        xs,
        ys,
        s=70,
        c=family_colors[fam],
        marker="o",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.6,
        label=family_labels[fam],
    )

# Plot ours with stars
for fam in ["ours_vitb", "ours_vitl", "ours_conv"]:
    xs = [d[2] for d in data if d[5] == fam]
    ys = [d[3] for d in data if d[5] == fam]
    ax.scatter(
        xs,
        ys,
        s=180 if fam == "ours_vitl" else 130,
        c=family_colors[fam],
        marker="*" if fam == "ours_vitl" else "D",
        alpha=0.95,
        edgecolors="black",
        linewidths=1.0,
        label=family_labels[fam],
    )

# Annotate each point
for name, arch, x, y, ours, fam in data:
    if ours:
        if fam == "ours_vitl":
            ax.annotate(
                f"  {name}\n  ({arch}, {y:.2f}%)",
                (x, y),
                fontsize=9,
                fontweight="bold",
                color=family_colors[fam],
                ha="left",
                va="center",
                xytext=(8, 0),
                textcoords="offset points",
            )
        else:
            ax.annotate(
                f"  {name} ({arch})",
                (x, y),
                fontsize=8,
                fontweight="bold",
                color=family_colors[fam],
                ha="left",
                va="center",
                xytext=(8, 0),
                textcoords="offset points",
            )
    else:
        ax.annotate(
            f"  {name}",
            (x, y),
            fontsize=7.5,
            color="dimgray",
            ha="left",
            va="center",
            xytext=(6, 0),
            textcoords="offset points",
        )

# Dense baselines as horizontal dashed lines for reference
ax.axhline(y=85.84, color="#8e44ad", linestyle=":", alpha=0.45, linewidth=0.9)
ax.text(91, 85.92, "ViT-L/16 dense (85.84)", fontsize=7.5, color="#8e44ad", ha="right", va="bottom")
ax.axhline(y=85.11, color="#c0392b", linestyle=":", alpha=0.45, linewidth=0.9)
ax.text(91, 85.19, "ViT-B/16 dense (85.11)", fontsize=7.5, color="#c0392b", ha="right", va="bottom")
ax.axhline(y=86.72, color="#16a085", linestyle=":", alpha=0.45, linewidth=0.9)
ax.text(91, 86.80, "ConvNeXtV2-B dense (86.72)", fontsize=7.5, color="#16a085", ha="right", va="bottom")
ax.axhline(y=81.82, color="#888888", linestyle=":", alpha=0.45, linewidth=0.9)
ax.text(91, 81.90, "DeiT-B dense (81.82)", fontsize=7.5, color="dimgray", ha="right", va="bottom")

ax.set_xlabel("Structured-sparsity / MAC reduction (%)", fontsize=12)
ax.set_ylabel("ImageNet-1k Top-1 accuracy (%)", fontsize=12)
ax.set_title(
    "Top-1 accuracy vs.\\ MAC reduction across published ViT/DeiT pruning baselines\n"
    "and this paper's CAST rows (large markers = ours)",
    fontsize=12,
)
ax.set_xlim(20, 95)
ax.set_ylim(78.5, 88.5)
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(loc="lower left", fontsize=8, framealpha=0.95)

plt.tight_layout()
out = "comparison_top1_vs_macred.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
