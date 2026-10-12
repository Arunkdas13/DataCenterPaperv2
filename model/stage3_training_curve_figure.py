"""
Final figure: RL training curves vs. fixed-policy baselines.

X-axis: training episode (0-599).
Y-axis: that episode's total discounted avoided CO2e (Gt), same units/
        discounting as the reward the policy is actually trained on
        (see stage3_pipeline.train_policy / Stage3Env.step).

Baselines (CCS-first, 50/50 split, optimal-tied [LDAR-first=Greedy]) are
drawn as horizontal reference lines computed from the SAME discounted-total
quantity, evaluated deterministically (no training noise) -- so an
apples-to-apples comparison, not a training curve vs. a single bar.

Honest framing (do not overstate): the optimal-tied line is analytically
the best possible policy in this formulation (Sec 3.3's proof: LDAR is
cheaper than CCS everywhere in this project's cost range, so "always fund
whichever is cheaper" cannot be beaten). RL therefore cannot exceed it --
it can only converge onto it. What this figure actually shows and marks:
  1. RL clears the two weaker baselines (CCS-first, 50/50) from episode 0
     -- there is no crossing to mark because it never starts below them.
  2. The gap between RL's episode reward and the analytic optimum shrinks
     over training (2.7% -> 0.23% for GWP20; 1.3% -> ~0.3% for GWP100),
     i.e. RL is "on top" in the sense of tying the best achievable policy,
     not in the sense of surpassing an unbeatable ceiling.
"""
import numpy as np
import pandas as pd
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from dcfair import gwp_from_pulse

GWP20 = gwp_from_pulse(20)
GWP100 = gwp_from_pulse(100)
DISC = 0.02

# ---------------------------------------------------------------- baselines
df = pd.read_csv("../results/stage3/stage3_trajectory_all_policies.csv")


def disc_total(policy_key, metric):
    sub = df[df.policy == policy_key].sort_values("year")
    t = (sub["year"] - sub["year"].min()).values
    gwp = GWP20 if metric == "GWP20" else GWP100
    co2e = sub["co2_avoided"].values + gwp * sub["ch4_avoided"].values
    return float((co2e * (1 - DISC) ** t).sum())


BASELINES = {
    "GWP20": {
        "CCS_first": disc_total("CCS_first", "GWP20"),
        "Even_split": disc_total("Even_split", "GWP20"),
        "Optimal (LDAR-first = Greedy)": disc_total("LDAR_first", "GWP20"),
    },
    "GWP100": {
        "CCS_first": disc_total("CCS_first", "GWP100"),
        "Even_split": disc_total("Even_split", "GWP100"),
        "Optimal (LDAR-first = Greedy)": disc_total("LDAR_first", "GWP100"),
    },
}
BASE_COLORS = {"CCS_first": "#8a8a80", "Even_split": "#7fa8c9", "Optimal (LDAR-first = Greedy)": "#215870"}
BASE_LABELS = {"CCS_first": "CCS-first (never fund LDAR)", "Even_split": "50/50 split",
               "Optimal (LDAR-first = Greedy)": "Optimal (LDAR-first = Greedy) -- provable ceiling"}

# ---------------------------------------------------------------- RL history
hist = np.load("../results/stage3/stage3_training_history.npz")
HIST = {"GWP20": hist["hist20"], "GWP100": hist["hist100"]}
RL_COLOR = "#9c3226"

# CCS-first sits ~5x below the RL/50-50/optimal cluster, so a single linear
# axis squashes the actual training dynamics into a thin band at the top.
# Fixed with a broken y-axis per panel: a tall top strip for the cluster
# that matters (RL vs 50/50 vs optimal), a short bottom strip just to place
# CCS-first at its true value, with a diagonal break mark between them.
fig = plt.figure(figsize=(15, 6.6))
gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=0.08, wspace=0.14)
fig.subplots_adjust(top=0.83, bottom=0.27, left=0.08, right=0.98)
top_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
bot_axes = [fig.add_subplot(gs[1, 0], sharex=top_axes[0]), fig.add_subplot(gs[1, 1], sharex=top_axes[1])]

for col, (metric, title) in enumerate([("GWP20", "Trained against GWP20 reward"),
                                        ("GWP100", "Trained against GWP100 reward")]):
    ax, axb = top_axes[col], bot_axes[col]
    h = HIST[metric] / 1e9
    ep = np.arange(len(h))
    best_so_far = np.maximum.accumulate(h)
    ccs_gt = BASELINES[metric]["CCS_first"] / 1e9
    even_gt = BASELINES[metric]["Even_split"] / 1e9
    opt_gt = BASELINES[metric]["Optimal (LDAR-first = Greedy)"] / 1e9

    for a in (ax, axb):
        a.plot(ep, h, color=RL_COLOR, lw=0.6, alpha=0.35,
               label="RL, raw episode reward" if a is ax else None)
        a.plot(ep, best_so_far, color=RL_COLOR, lw=2.4,
               label="RL, best seen so far" if a is ax else None)
        for key, val in BASELINES[metric].items():
            a.axhline(val / 1e9, color=BASE_COLORS[key], lw=1.8, linestyle="--", alpha=0.9,
                       label=BASE_LABELS[key] if a is ax else None)

    top_lo = min(even_gt, h.min()) - (opt_gt - even_gt) * 0.5
    ax.set_ylim(top_lo, opt_gt + (opt_gt - even_gt) * 0.6)
    axb.set_ylim(ccs_gt - (opt_gt - even_gt) * 0.4, ccs_gt + (opt_gt - even_gt) * 0.4)

    ax.spines["bottom"].set_visible(False)
    axb.spines["top"].set_visible(False)
    ax.tick_params(labelbottom=False, bottom=False)
    d = 0.012
    kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=1)
    ax.plot((-d, +d), (-d * 2.2, +d * 2.2), **kw)
    ax.plot((1 - d, 1 + d), (-d * 2.2, +d * 2.2), **kw)
    kw.update(transform=axb.transAxes)
    axb.plot((-d, +d), (1 - d * 2.2, 1 + d * 2.2), **kw)
    axb.plot((1 - d, 1 + d), (1 - d * 2.2, 1 + d * 2.2), **kw)

    # mark episode 0: RL already clears the 50/50 baseline (visible in the top strip)
    ax.annotate("RL ahead of\n50/50 at ep 0",
                xy=(0, h[0]), xytext=(len(h) * 0.14, even_gt - (opt_gt - even_gt) * 0.15),
                fontsize=15.5, color=RL_COLOR,
                arrowprops=dict(arrowstyle="->", color=RL_COLOR, lw=1.4), ha="left")

    gap_start = (opt_gt - h[0]) / opt_gt * 100
    gap_end = (opt_gt - best_so_far[-1]) / opt_gt * 100
    ax.annotate(f"gap to ceiling: {gap_start:.1f}%$\\to${max(gap_end,0):.2f}%",
                xy=(len(h) - 1, best_so_far[-1]), xytext=(len(h) * 0.30, opt_gt + (opt_gt - even_gt) * 0.25),
                fontsize=15.5, color=BASE_COLORS["Optimal (LDAR-first = Greedy)"], ha="left",
                arrowprops=dict(arrowstyle="->", color=BASE_COLORS["Optimal (LDAR-first = Greedy)"], lw=1.3))

    axb.annotate("CCS-first sits $\\sim$5x lower (axis broken)",
                 xy=(len(h) * 0.5, ccs_gt + (opt_gt - even_gt) * 0.18), fontsize=15,
                 color=BASE_COLORS["CCS_first"], va="center", ha="center")

    ax.set_title(title, fontsize=20, fontweight="bold")
    axb.set_xlabel("Training episode", fontsize=17)
    ax.tick_params(labelsize=15)
    axb.tick_params(labelsize=15)
    ax.grid(alpha=0.25)
    axb.grid(alpha=0.25)

top_axes[0].set_ylabel("This episode's total discounted\navoided CO2e (Gt)", y=-0.15, fontsize=17)
handles_pa, labels_pa = top_axes[0].get_legend_handles_labels()
plt.suptitle("RL training curves vs. fixed-policy baselines", fontsize=21, y=0.96)
fig.legend(handles_pa, labels_pa, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3, fontsize=14, framealpha=0.95)
fig.text(0.5, -0.02,
         "Broken $y$-axis: CCS-first sits far below the rest. RL clears both weaker baselines from episode 0 and\n"
         "closes in on the analytic optimum (Sec.~3.3: LDAR is cheaper than CCS everywhere in this range).",
         ha="center", fontsize=14, color="#555")
plt.savefig("../results/stage3/stage3_training_curves_vs_baselines.png", dpi=150, bbox_inches="tight")
# Also save under the exact filename main.tex's appendix Fig.
# fig:phaseA-train-app includes (figures/phaseA_training_curve.png).
plt.savefig("../overleaf_final/figures/phaseA_training_curve.png", dpi=150, bbox_inches="tight")
print("Saved stage3_training_curves_vs_baselines.png")

for metric in ["GWP20", "GWP100"]:
    h = HIST[metric] / 1e9
    opt = BASELINES[metric]["Optimal (LDAR-first = Greedy)"] / 1e9
    print(metric, "ep0 gap%%:", (opt - h[0]) / opt * 100, "final best-so-far gap%%:",
          (opt - np.maximum.accumulate(h)[-1]) / opt * 100)
