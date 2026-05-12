"""
Comparison figure: accuracy drop from each method's own dense baseline vs.
reported sparsity or FLOP/MAC reduction for rows in tab:result_comparison and
the structured FLOP/MAC rows from tab:param_to_flop_followup.

Output: comparison_top1_vs_macred.png (raster, included in main.tex).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (method, architecture, reduction_pct, drop_from_dense_pp, ours, family, ft)
# Negative drop means the reported row is above its own dense baseline.
data = [
    # Published ViT-family weight-pruning baselines.
    ("NViT", "DeiT-B", 61.28, 0.07, False, "vit_pruning", "~300 ep + pruning steps"),
    ("SViTE", "DeiT-B", 50.00, 0.29, False, "vit_pruning", "~600 train ep"),
    ("Spartan", "ViT-B", 90.00, -1.12, False, "vit_pruning", "~300 train ep"),
    # DeiT-Base baselines reproduced by CP-ViT.
    ("VTP", "DeiT-B", 43.20, 1.12, False, "deit_head", "~30 FT ep"),
    ("PoWER-BERT", "DeiT-B", 39.24, 1.65, False, "deit_head", "~30 FT ep"),
    ("HVT", "DeiT-B", 44.78, 1.88, False, "deit_head", "~30 FT ep"),
    ("CP-ViT", "DeiT-B", 41.62, 0.69, False, "deit_head", "~30 FT ep"),
    # Hardware-friendly k:n structured-sparsity baselines with training/FT.
    ("GPUSQ INT8", "DeiT-B", 50.00, -1.10, False, "kn_hw", "~300 QAT/KD ep"),
    ("GPUSQ INT4", "DeiT-B", 50.00, 0.20, False, "kn_hw", "~300 QAT/KD ep"),
    ("Beyond 2:4 300e", "DeiT-B", 75.00, 0.76, False, "kn_hw", "~300 ep"),
    ("Beyond 2:4 600e", "DeiT-B", 75.00, 0.08, False, "kn_hw", "~600 train ep"),
    ("ELSA", "DeiT-B", 50.00, 0.00, False, "kn_hw", "~150 train ep"),
    ("SERo", "DeiT-B", 53.85, 1.55, False, "kn_hw", "~200 ep"),
    # Published CNN pruning baselines from tab:cnn_pruning_comparison.
    ("SFP", "ResNet-50", 41.80, 1.54, False, "cnn_pruning", "FT n/r"),
    ("HRank", "ResNet-50", 43.80, 1.17, False, "cnn_pruning", "FT n/r"),
    ("FPGM", "ResNet-50", 53.50, 1.32, False, "cnn_pruning", "FT n/r"),
    ("ResRep", "ResNet-50", 54.54, 0.00, False, "cnn_pruning", "~180 train ep"),
    ("DepGraph", "ResNet-50", 51.82, 0.32, False, "cnn_pruning", "FT n/r"),
    ("Isomorphic", "ResNet-50", 50.12, 0.22, False, "cnn_pruning", "~300 FT ep"),
    ("Isomorphic", "ResNet-101", 50.96, -0.05, False, "cnn_pruning", "~300 FT ep"),
    ("Isomorphic", "ResNet-152", 64.94, 0.47, False, "cnn_pruning", "~300 FT ep"),
    ("Isomorphic", "ConvNeXt-S", 44.79, 0.66, False, "cnn_pruning", "~300 FT ep"),
    ("Isomorphic", "ConvNeXt-T", 72.72, 1.64, False, "cnn_pruning", "~300 FT ep"),
    # This paper, unstructured and all structured FLOP/MAC rows.
    ("Hybrid Mag-SER", "ViT-B", 50.00, 1.70, True, "ours_vit", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "ViT-B/384", 50.00, 0.86, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "ViT-L", 50.00, 1.32, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "DeiT-T", 50.00, 4.47, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "DeiT-S", 50.00, 2.63, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "DeiT-B", 50.00, 1.68, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "Swin-T", 50.00, 1.40, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "ConvNeXt-B", 50.00, 1.04, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "ResNet50d", 50.00, 1.39, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "ResNet101d", 50.00, 0.70, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "Hiera-B+", 50.00, 2.15, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "ResNet18", 50.00, 0.64, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "ResNet34", 50.00, 0.40, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "ResNet50", 50.00, 0.37, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "DeiT-B adapt", 50.00, 1.85, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "Swin-T adapt", 50.00, 1.59, True, "ours_hybrid", "~1 ep/cycle"),
    ("Hybrid Mag-SER", "ConvNeXtV2-B", 50.00, 1.39, True, "ours_hybrid", "~1 ep/cycle"),
    ("CAST 2:4+ToMe", "ViT-B", 59.81, 1.70, True, "ours_vit", "~3 FT ep"),
    ("Magnitude 2:4+ToMe", "ViT-B", 59.81, 2.19, True, "ours_vit", "~3 FT ep"),
    ("CAST 6:12", "ViT-B", 50.00, 1.37, True, "ours_vit", "~3 FT ep"),
    ("CAST 2:4+ToMe", "ViT-L", 60.00, 1.47, True, "ours_vit", "~3 FT ep"),
    ("CAST 8:16", "ViT-L", 50.00, 0.51, True, "ours_vit", "~3 FT ep"),
    ("CAST 2:4+ToMe", "DeiT-B", 59.81, 1.32, True, "ours_deit", "~3 FT ep"),
    ("CAST 2:4+ToMe", "DeiT-S", 59.81, 2.89, True, "ours_deit", "~3 FT ep"),
    ("CAST 2:4+ToMe", "DeiT-T", 59.81, 6.28, True, "ours_deit", "~3 FT ep"),
    ("AlphaPruning-style", "ViT-B", 50.00, 2.20, True, "ours_vit", "~3 FT ep"),
    ("CAST-conv", "ResNet50", 48.50, 2.99, True, "ours_resnet", "~3 FT ep"),
    ("CAST-conv+perm", "ResNet50", 48.50, 0.46, True, "ours_resnet", "~3 FT ep"),
    ("CAST 8:16", "ResNet50", 50.00, 0.26, True, "ours_resnet", "~3 FT ep"),
    ("CAST-conv", "ResNet50d", 49.85, 2.47, True, "ours_resnet", "~3 FT ep"),
    ("CAST-conv+perm", "ResNet50d", 49.85, 2.55, True, "ours_resnet", "~3 FT ep"),
    ("CAST 8:16", "ResNet50d", 50.00, 1.98, True, "ours_resnet", "~3 FT ep"),
    ("CAST-conv", "ResNet101d", 50.00, 2.13, True, "ours_resnet", "~3 FT ep"),
    ("CAST-conv+perm", "ResNet101d", 50.00, 1.67, True, "ours_resnet", "~3 FT ep"),
    ("CAST 8:16", "ResNet101d", 50.00, 1.34, True, "ours_resnet", "~3 FT ep"),
    ("CAST-conv+perm", "ResNet152d", 50.00, 1.53, True, "ours_resnet", "~3 FT ep"),
    ("CAST 2:4", "ConvNeXtV2-B", 50.00, 1.25, True, "ours_convnext", "~3 FT ep"),
    ("CAST 8:16", "ConvNeXtV2-B", 50.00, 0.87, True, "ours_convnext", "~3 FT ep"),
    ("CAST 12:16", "ConvNeXtV2-B", 25.00, 0.37, True, "ours_convnext", "~3 FT ep"),
]

family_colors = {
    "deit_head": "#6f6f6f",
    "vit_pruning": "#3f72b5",
    "kn_hw": "#c87520",
    "cnn_pruning": "#5f7d35",
    "ours_vit": "#b8322a",
    "ours_hybrid": "#565656",
    "ours_deit": "#c24f93",
    "ours_resnet": "#13866f",
    "ours_convnext": "#7d3c98",
}
family_labels = {
    "deit_head": "Published structured FLOP pruning",
    "vit_pruning": "Published ViT weight pruning/training",
    "kn_hw": "Published hardware-friendly sparsity",
    "cnn_pruning": "Published CNN structural pruning",
    "ours_vit": "This paper, ViT rows",
    "ours_hybrid": "This paper, 50% Hybrid sweep",
    "ours_deit": "This paper, DeiT rows",
    "ours_resnet": "This paper, ResNet rows",
    "ours_convnext": "This paper, ConvNeXtV2 rows",
}
markers = {
    "deit_head": "o",
    "vit_pruning": "s",
    "kn_hw": "^",
    "cnn_pruning": "h",
    "ours_vit": "D",
    "ours_hybrid": "v",
    "ours_deit": "X",
    "ours_resnet": "P",
    "ours_convnext": "*",
}

fig, ax = plt.subplots(figsize=(12.2, 7.0))

plot_order = [
    "deit_head",
    "vit_pruning",
    "kn_hw",
    "cnn_pruning",
    "ours_hybrid",
    "ours_vit",
    "ours_deit",
    "ours_resnet",
    "ours_convnext",
]
for fam in plot_order:
    rows = [d for d in data if d[5] == fam]
    ax.scatter(
        [d[2] for d in rows],
        [d[3] for d in rows],
        s=135 if fam.startswith("ours") else 58,
        c=family_colors[fam],
        marker=markers[fam],
        alpha=0.9 if fam.startswith("ours") else 0.72,
        edgecolors="black",
        linewidths=0.8 if fam.startswith("ours") else 0.45,
        label=family_labels[fam],
        zorder=3 if fam.startswith("ours") else 2,
    )

labels = {
    ("Hybrid Mag-SER", "ViT-B"): ("Hybrid SER\nViT-B", -76, 14),
    ("CAST 2:4+ToMe", "ViT-B"): ("CAST 2:4+ToMe\nViT-B", 8, -18),
    ("Magnitude 2:4+ToMe", "ViT-B"): ("Mag 2:4+ToMe\nViT-B", -72, 18),
    ("CAST 6:12", "ViT-B"): ("CAST 6:12\nViT-B", 8, -28),
    ("CAST 8:16", "ViT-L"): ("CAST 8:16\nViT-L", 8, -18),
    ("CAST 2:4+ToMe", "DeiT-T"): ("DeiT-T", 8, 0),
    ("Hybrid Mag-SER", "ResNet34"): ("Hybrid 50%\nall-model sweep", -62, -38),
    ("CAST-conv", "ResNet50"): ("ResNet rows\n(all plotted)", -62, 26),
    ("CAST 8:16", "ResNet50"): ("best RN50", 7, -11),
    ("CAST-conv+perm", "ResNet152d"): ("RN152d perm", 8, 22),
    ("CAST 8:16", "ConvNeXtV2-B"): ("ConvNeXt 8:16", 8, 3),
    ("CAST 12:16", "ConvNeXtV2-B"): ("ConvNeXt 12:16", 8, 0),
    ("AlphaPruning-style", "ViT-B"): ("AlphaPruning-style\nViT-B", -88, 22),
    ("CP-ViT", "DeiT-B"): ("CP-ViT", 7, -8),
    ("NViT", "DeiT-B"): ("NViT", 7, 5),
    ("Spartan", "ViT-B"): ("Spartan", 7, 4),
    ("ELSA", "DeiT-B"): ("ELSA", 5, -2),
    ("Beyond 2:4 300e", "DeiT-B"): ("Beyond 2:4 300e", 7, 4),
    ("Beyond 2:4 600e", "DeiT-B"): ("Beyond 2:4 600e", 7, 4),
    ("SViTE", "DeiT-B"): ("SViTE", -38, 6),
    ("ResRep", "ResNet-50"): ("ResRep", 7, -10),
    ("Isomorphic", "ResNet-50"): ("Iso RN50", 7, -20),
    ("Isomorphic", "ConvNeXt-T"): ("Iso ConvNeXt-T", 7, 4),
}

for name, arch, x, y, ours, fam, ft in data:
    label_info = labels.get((name, arch))
    if label_info is None:
        continue
    else:
        label, dx, dy = label_info
    ax.annotate(
        label,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7.0 if not ours else 7.3,
        fontweight="bold" if ours else "normal",
        color=family_colors[fam] if ours else "#3d3d3d",
        ha="left" if dx >= 0 else "right",
        va="center",
        arrowprops=(
            dict(arrowstyle="-", color=family_colors[fam], lw=0.65, alpha=0.72)
            if ours and (abs(dx) > 28 or abs(dy) > 14)
            else None
        ),
    )

ax.axhline(0.0, color="black", linestyle="-", linewidth=0.9, alpha=0.55)
ax.text(22, -0.08, "no accuracy drop", fontsize=8, va="top", color="black")
ax.axvspan(48.5, 60.0, color="#eeeeee", alpha=0.43, zorder=0)
ax.text(54.2, 6.18, "main 50-60% region", fontsize=8, ha="center", color="#555555")

info = (
    "FT budget context:\n"
    "this paper: approx. 3 post-projection FT epochs\n"
    "Hybrid SER: approx. 1 epoch per pruning cycle\n"
    "external rows: original-method budget or n/r"
)
ax.text(
    0.985,
    0.965,
    info,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=8.0,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#9a9a9a", alpha=0.94),
)

ax.set_xlabel("Reported sparsity or FLOP/MAC reduction (%)", fontsize=11)
ax.set_ylabel("Top-1 drop from each row's dense baseline (pp)", fontsize=11)
ax.set_title(
    "Original-method comparison: accuracy drop vs. reported compression level\n"
    "Rows use their own dense baselines and FT/training budgets",
    fontsize=12,
)
ax.set_xlim(20, 94)
ax.set_ylim(-1.35, 6.75)
ax.grid(True, alpha=0.25, linestyle="--", zorder=0)
ax.legend(loc="upper left", fontsize=7.2, framealpha=0.96, ncol=1)

plt.tight_layout()
out = "comparison_top1_vs_macred.png"
plt.savefig(out, dpi=260, bbox_inches="tight")
print(f"Saved {out}")
