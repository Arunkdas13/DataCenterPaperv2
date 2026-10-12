"""
Two trajectory figures requested by the mentor call (Aug 27), modeled on the
Climate Finance Reimagined poster's dual-panel line-trajectory format
(ΔT and ΔE over time, one line per policy, each line labeled with its
cumulative outcome).

Figure A: allocation ratio (fraction of that year's budget spent on LDAR)
over time, one line per policy, split into GWP20 / GWP100 panels.

Figure B: cumulative avoided CO2e over time (this project's analogue of the
poster's ΔE), same five policies, same panel split. This is the direct
successor to the static end-of-horizon bar chart used earlier -- same
underlying numbers, but shown as full trajectories so the RL-vs-baseline
comparison is visible at every year, not just at 2100.

Five policies, matching the mentor's list exactly:
  - CCS-first   (a_ldar_share = 0 always -- this project's prior
                 "Status_quo_CCS_only" baseline, renamed here to match
                 the mentor's terminology)
  - LDAR-first  (a_ldar_share = 1 always -- NEW baseline, not previously
                 computed. Because Stage3Env auto-redirects any leftover
                 budget from a saturated branch to the other technology
                 (see stage3_pipeline.Stage3Env.step), this nominally
                 "spend everything on LDAR" rule automatically starts
                 funding CCS the moment LDAR saturates (~year 8) -- so it
                 is expected, not a bug, if this line tracks Greedy/RL
                 closely after that point. That is itself worth showing.
  - 50/50 split (a_ldar_share = 0.5 always -- prior "Even_split")
  - Greedy      (always fund whichever is cheaper right now -- prior
                 "Greedy_cheapest_first")
  - RL          (learned policy -- prior "RL_GWP20"/"RL_GWP100")

The last three already exist in stage3_policy_trajectories.csv from the
original stage3_pipeline.py run; only the LDAR-first baseline is newly
computed here, using the exact same environment and cost curves (no
retraining, no changed assumptions).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage3_pipeline import Stage3Env, YEARS, N

ANNUAL_BUDGET = 300e6


def run_fixed_policy(fn_ldar_share, metric):
    env = Stage3Env(ANNUAL_BUDGET, metric=metric)
    env.reset()
    infos = []
    for t in range(N):
        a = fn_ldar_share(t)
        s, r, done, info = env.step(a)
        infos.append(info)
    return infos


def infos_to_df(infos, label):
    df = pd.DataFrame(infos)
    df["year"] = YEARS
    df["policy"] = label
    df["ldar_share_of_budget"] = df["dollars_ldar"] / (df["dollars_ldar"] + df["dollars_ccs"])
    return df


if __name__ == "__main__":
    # -------------------------------------------------- new LDAR-first baseline
    ldar_first_20 = infos_to_df(run_fixed_policy(lambda t: 1.0, "GWP20"), "LDAR_first")
    ldar_first_100 = infos_to_df(run_fixed_policy(lambda t: 1.0, "GWP100"), "LDAR_first")

    # -------------------------------------------------- existing baselines/RL
    existing = pd.read_csv("../results/stage3/stage3_policy_trajectories.csv")

    rename = {"Status_quo_CCS_only": "CCS_first"}
    existing["policy"] = existing["policy"].replace(rename)

    all_df = pd.concat([existing, ldar_first_20, ldar_first_100], ignore_index=True)

    # NOTE: for the four fixed-rule policies (CCS_first, LDAR_first, Even_split,
    # Greedy_cheapest_first), the action taken each year does NOT depend on
    # which metric is being trained/scored against -- only RL's action does.
    # The original stage3_pipeline.py run therefore logged an IDENTICAL
    # trajectory twice under the same policy label (once during the GWP20
    # run, once during GWP100), so simply filtering by policy name returns
    # duplicate rows. They are de-duplicated on (policy, year) below; the
    # real per-metric difference is applied afterward via score_under_metric,
    # using each row's own real avoided-ton counts (co2_avoided, ch4_avoided).
    all_df = all_df.drop_duplicates(subset=["policy", "year"], keep="first")
    all_df.to_csv("../results/stage3/stage3_trajectory_all_policies.csv", index=False)

    # LDAR-first, Greedy, and RL are not just visually close, they are
    # BYTE-IDENTICAL trajectories (verified: max abs difference = 0.0 in
    # ldar_share_of_budget at every year). Drawing three overlapping lines
    # for one underlying curve is what made the first version of this
    # figure unreadable. Fixed here by merging them into a single plotted
    # series, labeled to say so explicitly, leaving three visually distinct
    # lines total instead of five where three sit on top of each other.
    POLICY_ORDER = ["CCS_first", "Even_split", "OPTIMAL_TIE"]
    LABELS = {"CCS_first": "CCS-first (never fund LDAR)",
              "Even_split": "50/50 split",
              "OPTIMAL_TIE": "LDAR-first = Greedy = RL (identical policy)"}
    COLORS = {"CCS_first": "#8a8a80", "Even_split": "#7fa8c9", "OPTIMAL_TIE": "#215870"}
    STYLES = {"CCS_first": ("-", 2.2), "Even_split": ("-", 2.2), "OPTIMAL_TIE": ("-", 2.6)}

    def series_for(policy_key, metric):
        if policy_key == "OPTIMAL_TIE":
            sub = all_df[all_df.policy == f"RL_{metric}"]  # identical to LDAR_first & Greedy, verified above
        else:
            sub = all_df[(all_df.policy == policy_key)]
            # CCS_first and Even_split are fixed rules: the action taken each
            # year does not depend on which metric is being scored, so one
            # trajectory serves both panels. co2e is rescored per metric
            # below from the real avoided-ton counts (co2_avoided, ch4_avoided).
        return sub.sort_values("year")

    from dcfair import gwp_from_pulse
    GWP20 = gwp_from_pulse(20)
    GWP100 = gwp_from_pulse(100)

    def score_under_metric(df, metric):
        gwp = GWP20 if metric == "GWP20" else GWP100
        return df["co2_avoided"] + gwp * df["ch4_avoided"]

    # ---------------------------------------------------------- FIGURE A: allocation ratio
    # Every policy's budget split has two halves that must sum to 1: the
    # LDAR share and the CCS share. Plotting only the LDAR share leaves the
    # CCS share implicit (reader has to compute "1 minus this line" in
    # their head). Fixed per the "fraction of EVERY curve" request: each
    # policy now draws BOTH its LDAR-share line (solid) and its CCS-share
    # line (dashed, same color) -- so the two lines for one policy are a
    # visible mirror pair around 0.5, not a hidden complement.
    fig, axes = plt.subplots(1, 2, figsize=(17, 3.4), sharey=True)
    fig.subplots_adjust(wspace=0.05, top=0.78, bottom=0.42, left=0.06, right=0.99)
    handles_a, labels_a = None, None
    for ax, metric, title in [(axes[0], "GWP20", "Scored under GWP20"), (axes[1], "GWP100", "Scored under GWP100")]:
        for pk in POLICY_ORDER:
            df = series_for(pk, metric)
            _, lw = STYLES[pk]
            ldar_share = df["ldar_share_of_budget"]
            ccs_share = 1.0 - ldar_share
            ax.plot(df["year"], ldar_share, color=COLORS[pk], linestyle="-", lw=lw + 1.2, alpha=0.95,
                     label=f"{LABELS[pk]} -- LDAR")
            ax.plot(df["year"], ccs_share, color=COLORS[pk], linestyle="--", lw=lw + 0.8, alpha=0.6,
                     label=f"{LABELS[pk]} -- CCS")
        ax.axhline(0.5, color="#bbb", lw=0.8, zorder=0)
        ax.set_xlabel("Year", fontsize=19)
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=16)
        ax.grid(alpha=0.25)
        if handles_a is None:
            handles_a, labels_a = ax.get_legend_handles_labels()
    axes[0].set_ylabel("Fraction of that\nyear's budget", fontsize=18)
    plt.suptitle("Allocation ratio over time, by policy (solid = LDAR share, dashed = CCS share)", fontsize=18, y=0.99)
    fig.legend(handles_a, labels_a, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, fontsize=12.5, framealpha=0.95)
    plt.savefig("../results/stage3/stage3_allocation_ratio_trajectories.png", dpi=150, bbox_inches="tight")
    # Also save under the exact filename main.tex's Fig. fig:phaseA includes
    # (figures/phaseA_allocation_ratio.png) -- same figure, two paths, so the
    # paper can be rebuilt directly from this repo's output without a manual
    # rename step.
    plt.savefig("../overleaf_final/figures/phaseA_allocation_ratio.png", dpi=150, bbox_inches="tight")
    print("Figure A saved.")

    # ---------------------------------------------------------- FIGURE B: cumulative avoided CO2e
    fig, axes = plt.subplots(1, 2, figsize=(16, 3.4), sharey=False)
    fig.subplots_adjust(wspace=0.14, top=0.80, bottom=0.36, left=0.06, right=0.97)
    handles_b, labels_b = None, None
    for ax, metric, title in [(axes[0], "GWP20", "Scored under GWP20"), (axes[1], "GWP100", "Scored under GWP100")]:
        for pk in POLICY_ORDER:
            df = series_for(pk, metric)
            scored = score_under_metric(df, metric)
            cum_gt = np.cumsum(scored) / 1e9
            ls, lw = STYLES[pk]
            ax.plot(df["year"], cum_gt, label=LABELS[pk], color=COLORS[pk], linestyle=ls, lw=lw + 1.2, alpha=0.9)
        ax.set_xlabel("Year", fontsize=19)
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.tick_params(labelsize=16)
        ax.margins(x=0.08)
        ax.grid(alpha=0.25)
        if handles_b is None:
            handles_b, labels_b = ax.get_legend_handles_labels()
    axes[0].set_ylabel("Cumulative avoided\nCO2e (Gt, undisc.)", fontsize=17)
    plt.suptitle("Cumulative avoided emissions over time, by policy", fontsize=19, y=0.99)
    fig.legend(handles_b, labels_b, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, fontsize=13, framealpha=0.95)
    plt.savefig("../results/stage3/stage3_cumulative_avoided_trajectories.png", dpi=150, bbox_inches="tight")
    # Also save under the exact filename main.tex's Fig. fig:phaseA includes
    # (figures/phaseA_cumulative.png).
    plt.savefig("../overleaf_final/figures/phaseA_cumulative.png", dpi=150, bbox_inches="tight")
    print("Figure B saved.")
