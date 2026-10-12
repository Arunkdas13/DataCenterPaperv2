"""
Single-panel, GWP20-only versions of the three Phase A figures, for the
4-page paper's main text (per mentor-call figure list: Phase A gets its
own allocation-ratio + cumulative-emissions trajectory figures, plus one
training curve, GWP20 only -- no GWP100 panel needed in the main text).

Reuses the exact same verified data sources as the original two-panel
figures (stage3_trajectory_figures.py, stage3_training_curve_figure.py):
  - ../results/stage3/stage3_trajectory_all_policies.csv
  - ../results/stage3/stage3_training_history.npz
No retraining, no new numbers -- purely a re-plot of already-verified
data, restricted to one metric and one axis instead of two.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dcfair import gwp_from_pulse

GWP20 = gwp_from_pulse(20)
DISC = 0.02
METRIC = "GWP20"

all_df = pd.read_csv("../results/stage3/stage3_trajectory_all_policies.csv")

POLICY_ORDER = ["CCS_first", "Even_split", "OPTIMAL_TIE"]
LABELS = {"CCS_first": "CCS-first (never fund LDAR)",
          "Even_split": "50/50 split",
          "OPTIMAL_TIE": "LDAR-first = Greedy = RL (identical policy)"}
COLORS = {"CCS_first": "#8a8a80", "Even_split": "#7fa8c9", "OPTIMAL_TIE": "#215870"}
STYLES = {"CCS_first": ("-", 2.4), "Even_split": ("-", 2.4), "OPTIMAL_TIE": ("-", 2.8)}


def series_for(policy_key):
    if policy_key == "OPTIMAL_TIE":
        sub = all_df[all_df.policy == f"RL_{METRIC}"]
    else:
        sub = all_df[(all_df.policy == policy_key)]
    return sub.sort_values("year")


def score_under_metric(df):
    return df["co2_avoided"] + GWP20 * df["ch4_avoided"]


# ---------------------------------------------------------- Figure A: allocation ratio (GWP20 only)
fig, ax = plt.subplots(figsize=(7.2, 5.2))
for pk in POLICY_ORDER:
    df = series_for(pk)
    _, lw = STYLES[pk]
    ldar_share = df["ldar_share_of_budget"]
    ccs_share = 1.0 - ldar_share
    ax.plot(df["year"], ldar_share, color=COLORS[pk], linestyle="-", lw=lw, alpha=0.95,
            label=f"{LABELS[pk]} -- LDAR")
    ax.plot(df["year"], ccs_share, color=COLORS[pk], linestyle="--", lw=lw, alpha=0.6,
            label=f"{LABELS[pk]} -- CCS")
ax.axhline(0.5, color="#bbb", lw=0.8, zorder=0)
ax.set_xlabel("Year")
ax.set_ylabel("Fraction of that year's budget")
ax.set_title("Phase A allocation ratio, scored under GWP20 -- solid = LDAR share, dashed = CCS share", fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="center right", framealpha=0.95)
fig.text(0.5, -0.03,
         "Three of five candidate policies collapse onto one pair of lines (labeled) because they are literally\nidentical, not a rendering artifact.",
         ha="center", fontsize=8, color="#555")
plt.tight_layout()
plt.savefig("../overleaf_final/figures/phaseA_GWP20_allocation_ratio.png", dpi=150, bbox_inches="tight")
print("Phase A allocation ratio (GWP20-only) saved.")

# ---------------------------------------------------------- Figure B: cumulative avoided CO2e (GWP20 only)
fig, ax = plt.subplots(figsize=(7.2, 5.2))
for pk in POLICY_ORDER:
    df = series_for(pk)
    scored = score_under_metric(df)
    cum_gt = np.cumsum(scored) / 1e9
    ls, lw = STYLES[pk]
    ax.plot(df["year"], cum_gt, label=LABELS[pk], color=COLORS[pk], linestyle=ls, lw=lw, alpha=0.9)
    ax.annotate(f"{cum_gt.iloc[-1]:.1f} Gt", xy=(df["year"].iloc[-1], cum_gt.iloc[-1]),
                xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                color=COLORS[pk], fontweight="bold", va="center")
ax.set_xlabel("Year")
ax.set_ylabel("Cumulative avoided CO2e (Gt, undiscounted)")
ax.set_title("Phase A cumulative avoided emissions, scored under GWP20", fontsize=11.5)
ax.grid(alpha=0.25)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
fig.text(0.5, -0.03, "The tied line (RL = Greedy = LDAR-first) edges out the 50/50 split and far outperforms CCS-first.",
         ha="center", fontsize=8, color="#555")
plt.tight_layout()
plt.savefig("../overleaf_final/figures/phaseA_GWP20_cumulative.png", dpi=150, bbox_inches="tight")
print("Phase A cumulative (GWP20-only) saved.")

# ---------------------------------------------------------- Figure C: training curve (GWP20 only)


def disc_total(policy_key):
    sub = all_df[all_df.policy == policy_key].sort_values("year")
    t = (sub["year"] - sub["year"].min()).values
    co2e = sub["co2_avoided"].values + GWP20 * sub["ch4_avoided"].values
    return float((co2e * (1 - DISC) ** t).sum())


BASELINES = {
    "CCS_first": disc_total("CCS_first"),
    "Even_split": disc_total("Even_split"),
    "Optimal (LDAR-first = Greedy)": disc_total("LDAR_first"),
}
BASE_COLORS = {"CCS_first": "#8a8a80", "Even_split": "#7fa8c9", "Optimal (LDAR-first = Greedy)": "#215870"}
BASE_LABELS = {"CCS_first": "CCS-first (never fund LDAR)", "Even_split": "50/50 split",
               "Optimal (LDAR-first = Greedy)": "Optimal (LDAR-first = Greedy) -- provable ceiling"}

hist = np.load("../results/stage3/stage3_training_history.npz")
h = hist["hist20"] / 1e9
ep = np.arange(len(h))
best_so_far = np.maximum.accumulate(h)
RL_COLOR = "#9c3226"

ccs_gt = BASELINES["CCS_first"] / 1e9
even_gt = BASELINES["Even_split"] / 1e9
opt_gt = BASELINES["Optimal (LDAR-first = Greedy)"] / 1e9

fig = plt.figure(figsize=(7.6, 5.6))
gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.08)
ax = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[1, 0], sharex=ax)

for a in (ax, axb):
    a.plot(ep, h, color=RL_COLOR, lw=0.6, alpha=0.35,
           label="RL, raw episode reward" if a is ax else None)
    a.plot(ep, best_so_far, color=RL_COLOR, lw=2.4,
           label="RL, best seen so far" if a is ax else None)
    for key, val in BASELINES.items():
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

ax.annotate("RL already ahead\nof 50/50 at episode 0",
            xy=(0, h[0]), xytext=(len(h) * 0.14, even_gt - (opt_gt - even_gt) * 0.15),
            fontsize=8.5, color=RL_COLOR,
            arrowprops=dict(arrowstyle="->", color=RL_COLOR, lw=1.1), ha="left")

gap_start = (opt_gt - h[0]) / opt_gt * 100
gap_end = (opt_gt - best_so_far[-1]) / opt_gt * 100
ax.annotate(f"gap to ceiling: {gap_start:.1f}% at ep 0\n→ {max(gap_end,0):.2f}% by ep {len(h)-1}",
            xy=(len(h) - 1, best_so_far[-1]), xytext=(len(h) * 0.30, opt_gt + (opt_gt - even_gt) * 0.25),
            fontsize=8.5, color=BASE_COLORS["Optimal (LDAR-first = Greedy)"], ha="left",
            arrowprops=dict(arrowstyle="->", color=BASE_COLORS["Optimal (LDAR-first = Greedy)"], lw=1.0))

axb.annotate("CCS-first sits far below (~5x lower) -- axis is broken to keep this visible at all",
             xy=(len(h) * 0.5, ccs_gt - (opt_gt - even_gt) * 0.32), fontsize=7.8,
             color=BASE_COLORS["CCS_first"], va="center", ha="center")

ax.set_title("Phase A: RL training curve vs. fixed-policy baselines, GWP20 reward\n(broken y-axis: CCS-first sits far below the rest)", fontsize=11)
axb.set_xlabel("Training episode")
ax.set_ylabel("This episode's total discounted\navoided CO2e (Gt)")
ax.grid(alpha=0.25)
axb.grid(alpha=0.25)
ax.legend(fontsize=7.8, loc="lower right", framealpha=0.95)

fig.text(0.5, -0.04,
         "RL clears both weaker baselines from episode 0 and closes in on the analytic optimum\n(LDAR is cheaper than CCS everywhere in this cost range, so this ceiling is provably unbeatable).",
         ha="center", fontsize=8, color="#555")
plt.tight_layout()
plt.savefig("../overleaf_final/figures/phaseA_GWP20_training_curve.png", dpi=150, bbox_inches="tight")
print("Phase A training curve (GWP20-only) saved.")

gap0 = (opt_gt - h[0]) / opt_gt * 100
gapf = (opt_gt - best_so_far[-1]) / opt_gt * 100
print(f"GWP20 ep0 gap%: {gap0:.2f}  final best-so-far gap%: {gapf:.2f}")
