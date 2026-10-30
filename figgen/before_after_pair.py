"""Poster variant of the P4 before/after Procrustes figure for an arbitrary same-dim pair.
Reads local caz_<concept>.json (paper_n250 tree); labels derived from the models actually used."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
sys.path.insert(0, str(Path.home() / "Source" / "Rosetta_Program"))
from rosetta_tools.viz_style import concept_color, THEME, apply_theme

DATA = Path.home() / "rosetta_data" / "paper_n250"
CONCEPTS = ["agency","authorization","causation","certainty","credibility","deception",
            "exfiltration","formality","moral_valence","negation","plurality","sarcasm",
            "sentiment","specificity","temporal_order","threat_severity","urgency"]

def key(mid): return mid.replace("/", "_").replace("-", "_")
def load(mid):
    out = {}
    for c in CONCEPTS:
        caz = json.load(open(DATA / key(mid) / f"caz_{c}.json"))
        ld = caz["layer_data"]; pk = ld["peak_layer"]
        out[c] = np.array(ld["metrics"][pk]["dom_vector"], dtype=np.float64)
    return out
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

mid_a, lab_a, mid_b, lab_b, outpath = sys.argv[1:6]
A, B = load(mid_a), load(mid_b)
MA = np.vstack([A[c] for c in CONCEPTS]); MB = np.vstack([B[c] for c in CONCEPTS])
assert MA.shape == MB.shape, (MA.shape, MB.shape)
R, _ = orthogonal_procrustes(MB, MA)
_, S, Vt = np.linalg.svd(MA, full_matrices=False); comps = Vt[:2]; evr = (S[:2]**2)/(S**2).sum()
a2d = MA @ comps.T; b_raw = MB @ comps.T; b_al = (MB @ R) @ comps.T
raw = np.mean([cos(A[c], B[c]) for c in CONCEPTS]); al = np.mean([cos(A[c], B[c] @ R) for c in CONCEPTS])
print(f"dim={MA.shape[1]} raw mean cos={raw:.3f} aligned mean cos={al:.3f}")

plt.rcParams.update({"font.size": 13})
fig, axes = plt.subplots(1, 2, figsize=(15, 7)); fig.patch.set_facecolor("white")
for ax, title, pts in [(axes[0], "Before alignment  (raw coordinate frames)", b_raw),
                       (axes[1], "After one Procrustes rotation  (shared by all 17 concepts)", b_al)]:
    for i, c in enumerate(CONCEPTS):
        col = concept_color(c); ax.plot([a2d[i,0], pts[i,0]], [a2d[i,1], pts[i,1]], color=col, lw=1.4, ls="--", alpha=0.5, zorder=2)
        ax.scatter(*a2d[i], color=col, s=220, zorder=5, edgecolors="white", linewidths=1.5)
        ax.scatter(*pts[i], facecolors="none", edgecolors=col, s=360, zorder=4, linewidths=2.8)
        ax.annotate(c.replace("_"," "), a2d[i], textcoords="offset points", xytext=(9,4), fontsize=10.5, color=col, zorder=6)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel(f"PC1  ({evr[0]:.0%} of {lab_a} variance)", fontsize=12.5)
    ax.set_ylabel(f"PC2  ({evr[1]:.0%} of {lab_a} variance)", fontsize=12.5)
    ax.axhline(0, color=THEME["spine"], lw=0.5, alpha=0.3); ax.axvline(0, color=THEME["spine"], lw=0.5, alpha=0.3)
    apply_theme(ax)
axes[0].annotate(f"mean same-concept cosine: {raw:.2f}", xy=(0.03,0.95), xycoords="axes fraction", va="top", fontsize=12.5,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=THEME["spine"], alpha=0.9))
axes[1].annotate(f"mean same-concept cosine: {al:.2f}", xy=(0.03,0.95), xycoords="axes fraction", va="top", fontsize=12.5,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=THEME["spine"], alpha=0.9))
la = plt.scatter([], [], color="#555", s=160, edgecolors="white", linewidths=1.5, label=f"{lab_a}  (filled)")
lb = plt.scatter([], [], facecolors="none", edgecolors="#555", s=240, linewidths=2.8, label=f"{lab_b}  (hollow)")
fig.legend(handles=[la, lb], loc="lower center", ncol=2, fontsize=13, bbox_to_anchor=(0.5, -0.01), frameon=False)
fig.suptitle(f"17 peak-layer concept directions, {lab_a} vs {lab_b}, projected into {lab_a}'s 2-D subspace",
             fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0.05, 1, 0.94]); plt.savefig(outpath, dpi=170, bbox_inches="tight"); print("saved", outpath)
