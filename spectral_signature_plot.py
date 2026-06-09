#!/usr/bin/env python3
"""Spectral-signature plot across all 138 features for the 8 training classes."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent
CSV = PROJECT_ROOT / "outputs_10epoch" / "training_sampled_values.csv"
OUT_DIR = PROJECT_ROOT / "outputs_10epoch"

CLASS_LABELS = {
    1: "Waterbody", 2: "Paddy", 3: "Built-up", 4: "Clouds",
    5: "Dense Vegetation", 6: "Sparse Vegetation", 7: "Ladang", 8: "Bareland",
}
CLASS_COLORS = {
    1: "#1f77b4", 2: "#2ca02c", 3: "#d62728", 4: "#7f7f7f",
    5: "#006400", 6: "#9acd32", 7: "#ff7f0e", 8: "#8c564b",
}

EPOCHS = ["march", "june", "aug", "sept", "jan25",
          "may25", "aug25", "sep25", "nov25", "mar26"]
PER_EPOCH = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8",
             "NDVI", "NDWI", "NDBI", "EVI"]
TEMPORAL = ["maxNDVI", "minNDVI", "stdNDVI", "ampNDVI",
            "maxNDBI", "minNDBI", "stdNDBI", "ampNDBI",
            "maxEVI",  "minEVI",  "stdEVI",  "ampEVI",
            "maxNDWI", "minNDWI", "stdNDWI", "ampNDWI"]
HEIGHT = ["tree_height_mean", "tree_height_std"]

df = pd.read_csv(CSV)
feat_cols = [c for c in df.columns if c != "class"]
assert len(feat_cols) == 138, f"expected 138 features, got {len(feat_cols)}"

# Per-class mean + std across all 138 features (in CSV column order).
grouped = df.groupby("class")[feat_cols]
means = grouped.mean()
stds = grouped.std()

# --- Plot ------------------------------------------------------------------
fig, (ax_spec, ax_full) = plt.subplots(
    2, 1, figsize=(22, 11),
    gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.35},
)

# Top panel: pure 8-band reflectance per epoch (mean per class), stacked epochs
x_spec = np.arange(len(EPOCHS) * 8)
for cls in sorted(means.index):
    y = [means.loc[cls, f"{ep}_b{b}"] for ep in EPOCHS for b in range(1, 9)]
    ax_spec.plot(x_spec, y, color=CLASS_COLORS[cls], lw=1.4,
                 label=f"{cls} — {CLASS_LABELS[cls]}")
for i, ep in enumerate(EPOCHS):
    ax_spec.axvline(i * 8 - 0.5, color="0.8", lw=0.6)
    ax_spec.text(i * 8 + 3.5, ax_spec.get_ylim()[1] if False else 0.0, "",
                 ha="center")
ax_spec.set_xticks([i * 8 + 3.5 for i in range(len(EPOCHS))])
ax_spec.set_xticklabels(EPOCHS, rotation=0)
ax_spec.set_xlim(-0.5, len(x_spec) - 0.5)
ax_spec.set_ylabel("Surface reflectance (normalized)")
ax_spec.set_title("Mean spectral signature per class — 8 PS bands × 10 epochs (80 bands)")
ax_spec.grid(alpha=0.3)
ax_spec.legend(ncol=4, fontsize=8, loc="upper right")

# Bottom panel: ALL 138 features, mean ± std shaded
x_all = np.arange(138)
for cls in sorted(means.index):
    m = means.loc[cls, feat_cols].values
    s = stds.loc[cls, feat_cols].values
    ax_full.plot(x_all, m, color=CLASS_COLORS[cls], lw=1.2,
                 label=f"{cls} — {CLASS_LABELS[cls]}")
    ax_full.fill_between(x_all, m - s, m + s,
                         color=CLASS_COLORS[cls], alpha=0.08, linewidth=0)

# Group dividers + labels
group_spans = []
idx = 0
for ep in EPOCHS:
    group_spans.append((idx, idx + len(PER_EPOCH), ep))
    idx += len(PER_EPOCH)
group_spans.append((idx, idx + len(TEMPORAL), "temporal"))
idx += len(TEMPORAL)
group_spans.append((idx, idx + len(HEIGHT), "height"))

for (a, b, name) in group_spans:
    ax_full.axvline(b - 0.5, color="0.75", lw=0.6)
    ax_full.text((a + b - 1) / 2, 1.02, name,
                 transform=ax_full.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=8, rotation=0, color="0.25")

ax_full.set_xticks(x_all)
ax_full.set_xticklabels(feat_cols, rotation=90, fontsize=5)
ax_full.set_xlim(-0.5, 137.5)
ax_full.set_ylabel("Feature value (normalized)")
ax_full.set_xlabel("Feature (138 total: 10 epochs × 12 + 16 temporal + 2 tree-height)")
ax_full.set_title("Mean ± 1σ per class across all 138 features")
ax_full.grid(alpha=0.3, axis="y")
ax_full.legend(ncol=4, fontsize=8, loc="upper right")

fig.suptitle("PlanetScope 10-epoch training samples — spectral signatures (n=810)",
             fontsize=14, y=0.995)

out_png = OUT_DIR / "spectral_signatures_138bands.png"
out_pdf = OUT_DIR / "spectral_signatures_138bands.pdf"
fig.savefig(out_png, dpi=180, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"wrote {out_png}")
print(f"wrote {out_pdf}")
