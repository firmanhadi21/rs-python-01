#!/usr/bin/env python3
"""Visualize class separability across all 138 features."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parent
CSV = PROJECT_ROOT / "outputs_10epoch" / "training_sampled_values.csv"
OUT = PROJECT_ROOT / "outputs_10epoch"

LABELS = {1:"Waterbody",2:"Paddy",3:"Built-up",4:"Clouds",
          5:"Dense Veg",6:"Sparse Veg",7:"Ladang",8:"Bareland"}
COLORS = {1:"#1f77b4",2:"#2ca02c",3:"#d62728",4:"#7f7f7f",
          5:"#006400",6:"#9acd32",7:"#ff7f0e",8:"#8c564b"}

df = pd.read_csv(CSV)
X  = df.drop(columns="class").values
y  = df["class"].values
cls = sorted(LABELS)
mu = {c: X[y==c].mean(0) for c in cls}
sd = {c: X[y==c].std(0)+1e-6 for c in cls}

def sep(i,j):
    return np.sqrt(((mu[i]-mu[j])**2 / (sd[i]**2 + sd[j]**2)).sum())

M = np.array([[sep(a,b) if a!=b else 0 for b in cls] for a in cls])
names = [LABELS[c] for c in cls]
iso = M.sum(1) / (len(cls)-1)            # mean distance to other classes
order = np.argsort(-iso)                 # most isolated first

# --- figure ----------------------------------------------------------------
fig = plt.figure(figsize=(18, 7.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.15], wspace=0.32)

# (1) Heatmap
ax1 = fig.add_subplot(gs[0,0])
Mm = np.ma.masked_equal(M, 0)
im = ax1.imshow(Mm, cmap="viridis")
ax1.set_xticks(range(len(cls))); ax1.set_yticks(range(len(cls)))
ax1.set_xticklabels(names, rotation=45, ha="right")
ax1.set_yticklabels(names)
for i in range(len(cls)):
    for j in range(len(cls)):
        if i==j: continue
        ax1.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                 color="white" if M[i,j] < M.max()*0.55 else "black", fontsize=8)
cb = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cb.set_label("Fisher-like distance (higher = easier)")
ax1.set_title("Pairwise class separability")

# (2) Isolation ranking (how distinct each class is overall)
ax2 = fig.add_subplot(gs[0,1])
ranked = [(names[i], iso[i], cls[i]) for i in order]
ys = np.arange(len(ranked))
ax2.barh(ys, [r[1] for r in ranked],
         color=[COLORS[r[2]] for r in ranked], edgecolor="0.2")
for yi, (nm, v, _) in zip(ys, ranked):
    ax2.text(v+0.2, yi, f"{v:.1f}", va="center", fontsize=9)
ax2.set_yticks(ys); ax2.set_yticklabels([r[0] for r in ranked])
ax2.invert_yaxis()
ax2.set_xlabel("Mean distance to other classes")
ax2.set_title("Class isolation (easiest → hardest)")
ax2.axvspan(0, 13, color="#ffcccc", alpha=0.3, zorder=0, label="harder (<13)")
ax2.axvspan(13, 17, color="#fff2cc", alpha=0.4, zorder=0, label="moderate")
ax2.axvspan(17, max(iso)+2, color="#d4edda", alpha=0.4, zorder=0, label="easy (≥17)")
ax2.legend(loc="lower right", fontsize=8)
ax2.set_xlim(0, max(iso)+2)

# (3) Confusion-pair graph: nodes at positions, edges = closeness
ax3 = fig.add_subplot(gs[0,2])
ax3.set_aspect("equal"); ax3.axis("off")
angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, len(cls), endpoint=False)
pos = {c: (np.cos(a), np.sin(a)) for c, a in zip(cls, angles)}

# Edges for the N closest pairs (smallest distance)
pairs = []
for i,a in enumerate(cls):
    for j,b in enumerate(cls):
        if j<=i: continue
        pairs.append((M[i,j], a, b))
pairs.sort()
top_close = pairs[:6]                    # 6 hardest pairs
dmin, dmax = pairs[0][0], pairs[-1][0]

for d, a, b in top_close:
    (x1,y1), (x2,y2) = pos[a], pos[b]
    # line width & alpha inverse to distance (closer = thicker red)
    t = (dmax - d) / (dmax - dmin + 1e-9)
    lw = 1 + 5*t
    ax3.plot([x1,x2],[y1,y2], color="#d62728", lw=lw, alpha=0.4+0.5*t, zorder=1)
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax3.text(mx, my, f"{d:.1f}", color="#7a1414", fontsize=8,
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

# Draw also top-3 MOST separated pairs in green for contrast
for d, a, b in pairs[-3:]:
    (x1,y1), (x2,y2) = pos[a], pos[b]
    ax3.plot([x1,x2],[y1,y2], color="#2ca02c", lw=1.2, alpha=0.35,
             linestyle=(0,(4,3)), zorder=0)

# Nodes
for c,(x,yy) in pos.items():
    ax3.scatter(x, yy, s=900, color=COLORS[c], edgecolor="black",
                linewidth=1.2, zorder=3)
    ax3.text(x*1.22, yy*1.22, LABELS[c], ha="center", va="center",
             fontsize=10, fontweight="bold")

ax3.set_xlim(-1.55,1.55); ax3.set_ylim(-1.4,1.4)
ax3.set_title("Confusion pairs — red = closest (hardest)\ngreen dashed = most separated")

fig.suptitle("Class separability across all 138 features  (n=810 samples)",
             fontsize=14, y=1.02)

png = OUT/"separability_overview.png"
pdf = OUT/"separability_overview.pdf"
fig.savefig(png, dpi=180, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print("wrote", png); print("wrote", pdf)
