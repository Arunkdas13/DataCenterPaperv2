"""
stage3_phaseout_dual_cumulative.py

Produces figures/phaseout_dual_cumulative.png -- the two-panel (GWP20 |
GWP100) phase-out figure referenced by main.tex's fig:phaseout-cum. This
script was NOT part of the code the user's trimmed zip extracted from the
full project; stage3_v5_phaseout.py (included) only saves separate
single-metric cumulative plots. Rather than guess at a combining script,
this one just re-lays out the real, already-computed trajectory CSVs that
stage3_v5_phaseout.py writes to ../results/stage3/ -- no new modeling, no
new numbers, just the GWP20/GWP100 side-by-side presentation main.tex
actually uses (same two-panel layout as stage3_phaseB_dual_metric.py).

Run stage3_v5_phaseout.py first -- this script reads its CSV output.

Run: python3 stage3_phaseout_dual_cumulative.py
Saves: ../overleaf_final/figures/phaseout_dual_cumulative.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dcfair import gwp_from_pulse

GWP_VALUES = {"GWP20": gwp_from_pulse(20), "GWP100": gwp_from_pulse(100)}
PREFIX = "../results/stage3/stage3_v5_phaseout"
PEAK_YEAR, ZERO_YEAR = 2035, 2045
C_FIXED, C_RL = "#a8560a", "#215870"

fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=False)

for ax, (metric, gwp) in zip(axes, GWP_VALUES.items()):
    df_none = pd.read_csv(f"{PREFIX}_{metric}_none_trajectory.csv")
    df_fixed = pd.read_csv(f"{PREFIX}_{metric}_fixed_trajectory.csv")
    df_rl = pd.read_csv(f"{PREFIX}_{metric}_rl_trajectory.csv")

    for df, label, color, ls, lw in [(df_none, "No investment", "#8a8a80", ":", 1.8),
                                      (df_fixed, "Fixed 50/50 split", C_FIXED, "-", 2.6),
                                      (df_rl, f"RL ({metric})", C_RL, "-", 2.6)]:
        total = df["avoided_ch4"] * gwp + df["avoided_co2"]
        cum_gt = np.cumsum(total) / 1e9
        ax.plot(df["year"], cum_gt, color=color, lw=lw, linestyle=ls, label=label)

    ax.axvline(PEAK_YEAR, color="#5b3a7a", lw=1.0, ls=":")
    ax.axvline(ZERO_YEAR, color="#b23a2e", lw=1.2, ls=":")
    ax.set_xlabel("Year")
    ax.set_title(f"Scored under {metric}", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.6, loc="upper left")

axes[0].set_ylabel("Cumulative avoided CO2e (Gt)")
plt.suptitle("Phase-out scenario: cumulative avoided CO2e, fleet retires linearly 2035→2045", fontsize=13)
plt.tight_layout()
plt.savefig("../overleaf_final/figures/phaseout_dual_cumulative.png", dpi=220, bbox_inches="tight")
print("Saved phaseout_dual_cumulative.png")
