"""
Two-panel (GWP20 | GWP100) versions of the three Phase B, Q=1.5% figures,
for the paper's main text per the mentor's latest instruction (show both
horizons for both phases; only one training curve figure total, and this
is it). Reuses the exact already-saved, verified data from
stage3_v5_qsweep.py's Q=1.5% run -- no retraining, no new numbers:
  - stage3_v5_Q1p5_{metric}_{rl,fixed,none}_trajectory.csv
  - stage3_v5_Q1p5_{metric}_training_history.npz
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stage3_v5_real_leak_rates as v5
from dcfair import gwp_from_pulse

GWP_VALUES = {"GWP20": gwp_from_pulse(20), "GWP100": gwp_from_pulse(100)}
P = v5.P_LEAK
Q = 0.015
C_FIXED = "#a8560a"
C_RL = "#215870"

DATA = {}
for metric in ["GWP20", "GWP100"]:
    prefix = f"../results/stage3/stage3_v5_Q1p5_{metric}"
    DATA[metric] = dict(
        rl=pd.read_csv(f"{prefix}_rl_trajectory.csv"),
        fixed=pd.read_csv(f"{prefix}_fixed_trajectory.csv"),
        none=pd.read_csv(f"{prefix}_none_trajectory.csv"),
        hist=np.load(f"{prefix}_training_history.npz")["hist"],
    )

# ---------------------------------------------------------- Figure: allocation ratio (2-panel)
# SIMPLIFIED per mentor feedback: LDAR share + CCS share always sum to 1,
# so plotting both was showing the same information twice. Now just one
# line per policy (LDAR share; CCS share = 1 minus it, stated once in the
# caption/axis label instead of drawn twice). 2 lines/panel instead of 4.
# Y-axis fixed to the full [0,1] range on both panels (matches Fig. 2's
# Phase A convention exactly) instead of a per-panel zoomed range, so the
# divergence is shown at true scale, not visually inflated; the explicit
# end-of-horizon gap arrow still gives the exact number either way.
fig, axes = plt.subplots(1, 2, figsize=(17, 3.85), sharey=False)
fig.subplots_adjust(wspace=0.16, top=0.76, bottom=0.34, left=0.07, right=0.99)
handles_a, labels_a = None, None
for ax, metric in zip(axes, ["GWP20", "GWP100"]):
    df_fixed, df_rl = DATA[metric]["fixed"], DATA[metric]["rl"]
    ax.plot(df_fixed["year"], df_fixed["a_ldar_share"], color=C_FIXED, lw=3.0, ls="--", label="Fixed 50/50")
    ax.plot(df_rl["year"], df_rl["a_ldar_share"], color=C_RL, lw=3.6, label="RL (trained policy)")

    ax.set_ylim(-0.03, 1.03)

    end_share = df_rl["a_ldar_share"].iloc[-1]
    gap_pp = abs(end_share - 0.5) * 100
    end_year = df_rl["year"].iloc[-1]
    ax.annotate("", xy=(end_year, end_share), xytext=(end_year, 0.5),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.6))
    ax.annotate(f"{gap_pp:.0f} pp\ngap", xy=(end_year, (end_share + 0.5) / 2),
                xytext=(10, 0), textcoords="offset points", fontsize=15.9,
                color="#333", va="center", ha="left", fontweight="bold")

    ax.axhline(0.5, color="#bbb", lw=1.0, zorder=0)
    ax.set_xlabel("Year", fontsize=24.4)
    ax.set_title(f"{metric} reward", fontsize=25.6, fontweight="bold")
    ax.tick_params(labelsize=20.5)
    ax.margins(x=0.10)
    ax.grid(alpha=0.25)
    if handles_a is None:
        handles_a, labels_a = ax.get_legend_handles_labels()
axes[0].set_ylabel("LDAR share of\nthat year's budget", fontsize=22.0)
plt.suptitle("Phase B: LDAR share of the yearly budget at $Q=1.5\\%$ (CCS share = 1 minus this)", fontsize=22.0, y=0.99)
fig.legend(handles_a, labels_a, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=18.0, framealpha=0.95)
plt.savefig("../overleaf_final/figures/phaseB_dual_allocation_ratio.png", dpi=220, bbox_inches="tight")
print("Phase B dual-metric allocation ratio (simplified, 2 lines/panel) saved.")

# ---------------------------------------------------------- Figure: cumulative (2-panel)
fig, axes = plt.subplots(1, 2, figsize=(17, 3.55), sharey=False)
fig.subplots_adjust(wspace=0.16, top=0.76, bottom=0.34, left=0.07, right=0.97)
handles_c, labels_c = None, None
for ax, metric in zip(axes, ["GWP20", "GWP100"]):
    gwp = GWP_VALUES[metric]
    for key, label, color, ls, lw in [("none", "No investment", "#8a8a80", ":", 2.6),
                                        ("fixed", "Fixed 50/50", C_FIXED, "-", 3.4),
                                        ("rl", "RL", C_RL, "-", 3.4)]:
        df = DATA[metric][key]
        total = df["avoided_ch4"] * gwp + df["avoided_co2"]
        cum_gt = np.cumsum(total) / 1e9
        ax.plot(df["year"], cum_gt, color=color, lw=lw, linestyle=ls, label=label)
    ax.set_xlabel("Year", fontsize=24.4)
    ax.set_title(f"{metric} reward", fontsize=25.6, fontweight="bold")
    ax.tick_params(labelsize=20.5)
    ax.margins(x=0.06)
    ax.grid(alpha=0.25)
    if handles_c is None:
        handles_c, labels_c = ax.get_legend_handles_labels()
axes[0].set_ylabel("Cumulative avoided\nCO2e (Gt)", fontsize=22.0)
plt.suptitle("Phase B cumulative avoided emissions at $Q=1.5\\%$", fontsize=26.8, y=0.99)
fig.legend(handles_c, labels_c, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=18.3, framealpha=0.95)
plt.savefig("../overleaf_final/figures/phaseB_dual_cumulative.png", dpi=220, bbox_inches="tight")
print("Phase B dual-metric cumulative saved.")

# ---------------------------------------------------------- Figure: training curve (2-panel)
def sweep_ceiling(metric):
    gwp = GWP_VALUES[metric]
    v5.Q_LEAK = Q
    target_f = 1.0 - (Q / P if P > 0 else 0.0)
    target_f = max(target_f, 0.0)
    one_time = v5._solve_one_time_ldar_cost(target_f) if target_f > 0 else 0.0
    v5._ONE_TIME_LDAR_COST_TO_FLOOR = one_time
    v5.LDAR_OPEX_TO_FLOOR_BASE = v5.LEAK_REAPPEARANCE_RATE * one_time
    sweep = {}
    for share in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        rewards, _ = v5.run_fixed_split(share, gwp)
        sweep[share] = float(sum(rewards))
    return sweep


fig, axes = plt.subplots(1, 2, figsize=(17, 4.6), sharex=True)
fig.subplots_adjust(wspace=0.1, top=0.80, bottom=0.32, left=0.08, right=0.99)
handles_t, labels_t = None, None
for ax, metric in zip(axes, ["GWP20", "GWP100"]):
    h = DATA[metric]["hist"] / 1e9
    ep = np.arange(len(h))
    best_so_far = np.maximum.accumulate(h)
    sweep = sweep_ceiling(metric)
    fixed5050 = sweep[0.5] / 1e9
    ax.plot(ep, h, color="#6a3d9a", lw=0.8, alpha=0.30, label="RL, raw episode reward")
    ax.plot(ep, best_so_far, color="#6a3d9a", lw=3.4, label="RL, best seen so far")
    ax.axhline(fixed5050, color=C_FIXED, lw=2.4, ls="--", label="Fixed 50/50")
    ax.set_xlabel("Training episode", fontsize=24.4)
    ax.set_title(f"{metric} reward", fontsize=25.6, fontweight="bold")
    ax.tick_params(labelsize=20.5)
    ax.grid(alpha=0.25)
    if handles_t is None:
        handles_t, labels_t = ax.get_legend_handles_labels()
axes[0].set_ylabel("Discounted avoided\nCO2e (Gt)", fontsize=22.0)
plt.suptitle("Phase B RL training curve vs. Fixed 50/50 at $Q=1.5\\%$", fontsize=25.6, y=0.99)
fig.legend(handles_t, labels_t, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=17.1, framealpha=0.95)
plt.savefig("../overleaf_final/figures/phaseB_dual_training_curve.png", dpi=220, bbox_inches="tight")
print("Phase B dual-metric training curve saved.")
