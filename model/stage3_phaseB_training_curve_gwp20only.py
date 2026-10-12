"""
Phase B training-curve figure, further simplified per mentor feedback:
GWP20 is the paper's recommended/decision-relevant metric (Sec. 3, "Why
the horizon matters"), so the GWP100 training-curve panel is dropped
entirely rather than kept as a second, redundant chart. In its place: a
small summary table of RL's advantage over Fixed 50/50 under both
metrics, so GWP100 is still represented, just as numbers instead of a
duplicate chart. Same underlying, already-saved data, no retraining.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stage3_v5_real_leak_rates as v5
from dcfair import gwp_from_pulse

GWP_VALUES = {"GWP20": gwp_from_pulse(20), "GWP100": gwp_from_pulse(100)}
P = v5.P_LEAK
Q = 0.015
C_RL = "#215870"
C_FIXED = "#a8560a"

hist20 = np.load("../results/stage3/stage3_v5_Q1p5_GWP20_training_history.npz")["hist"]


def sweep_ceiling(metric):
    gwp = GWP_VALUES[metric]
    v5.Q_LEAK = Q
    target_f = 1.0 - (Q / P if P > 0 else 0.0)
    target_f = max(target_f, 0.0)
    one_time = v5._solve_one_time_ldar_cost(target_f) if target_f > 0 else 0.0
    v5._ONE_TIME_LDAR_COST_TO_FLOOR = one_time
    v5.LDAR_OPEX_TO_FLOOR_BASE = v5.LEAK_REAPPEARANCE_RATE * one_time
    rewards, _ = v5.run_fixed_split(0.5, gwp)
    return float(sum(rewards))


fig, ax = plt.subplots(figsize=(7.2, 3.9))
h20 = hist20 / 1e9
ep20 = np.arange(len(h20))
best20 = np.maximum.accumulate(h20)
fixed20 = sweep_ceiling("GWP20") / 1e9

ax.plot(ep20, h20, color=C_RL, lw=0.8, alpha=0.30, label="RL, raw episode reward")
ax.plot(ep20, best20, color=C_RL, lw=3.4, label="RL, best seen so far")
ax.axhline(fixed20, color=C_FIXED, lw=2.4, ls="--", label="Fixed 50/50")
# Best-so-far plateaus at episode ~806 (out of 3,000 trained); cropping the
# x-axis to 1,200 shows the full climb-and-plateau clearly instead of
# 1,800 extra empty-looking episodes of flat line -- the point (RL
# converges fast and stays converged) is what should be visible, not the
# unused tail of the training budget.
ax.set_xlim(0, 1200)
ax.set_xlabel("Training episode (of 3{,}000 total; converges by ~800)".replace("{,}", ","), fontsize=18)
ax.set_ylabel("Discounted avoided CO2e (Gt)", fontsize=17.5)
ax.set_title("Phase B training curve (GWP20)", fontsize=20, fontweight="bold")
ax.tick_params(labelsize=15.5)
ax.grid(alpha=0.25)
ax.legend(loc="lower right", fontsize=13, framealpha=0.95)
plt.tight_layout()
plt.savefig("../overleaf_final/figures/phaseB_training_curve_gwp20.png", dpi=220, bbox_inches="tight")
print("Phase B GWP20-only training curve saved.")
