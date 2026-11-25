"""Transforme resultats_reels.csv en graphique pour le poster :
% de messages exacts selon la distance, une courbe par valence, un panneau par condition de bruit."""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = {2: "#2a78d6", 4: "#eb6834", 8: "#1baf7a", 16: "#eda100", 32: "#e87ba4"}

lignes = list(csv.DictReader(open("resultats_reels.csv", encoding="utf-8"), delimiter=";"))
stats = defaultdict(list)  # (bruit, M, distance) -> [0/1, ...]
for l in lignes:
    if l["exact"] == "":   # texte hors corpus : pas de comparaison possible
        continue
    stats[(l["bruit"], int(l["M"]), float(l["distance_m"]))].append(int(l["exact"]))

bruits = sorted({k[0] for k in stats})
fig, axes = plt.subplots(1, len(bruits), figsize=(7 * len(bruits), 4.6), squeeze=False)
for ax, bruit in zip(axes[0], bruits):
    for M in sorted({k[1] for k in stats if k[0] == bruit}):
        dist = sorted(k[2] for k in stats if k[0] == bruit and k[1] == M)
        pct = [100 * sum(stats[(bruit, M, d)]) / len(stats[(bruit, M, d)]) for d in dist]
        ax.plot(dist, pct, marker="o", lw=2.5, ms=8, color=SERIES[M], label=f"M = {M}")
    ax.set_title(f"Pièce : {bruit}", loc="left")
    ax.set_xlabel("Distance haut-parleur → micro (m)")
    ax.set_ylabel("Messages reçus exacts (%)")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
axes[0][0].legend(frameon=False)
fig.savefig("fig_tests_reels.png", dpi=300, bbox_inches="tight")

print(f"{len(lignes)} essais analysés.")
for k in sorted(stats):
    v = stats[k]
    print(f"bruit={k[0]:8s} M={k[1]:2d} d={k[2]:.1f} m : {sum(v)}/{len(v)} exacts")
