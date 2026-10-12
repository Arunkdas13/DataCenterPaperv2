"""
Phase-out scenario, requested directly: the real gas-fleet growth schedule
n(t) is unchanged through its 2035 peak (350 facilities, growth_factor=
5.07x), but then gas-powered data centers are retired/converted off gas
fleet-wide, linearly, reaching ZERO by 2045 -- not just "growth stops,"
the facilities and their emissions actually go away. Same v5 leak-rate
mechanic (real P=3.11%, Q=0%) run against this rise-then-fall n(t), same
6 figures as stage3_v5_figures.py (allocation ratio, cumulative, training
curve x {GWP20, GWP100}).

Growth-factor schedule:
  year <= 2035: unchanged real schedule (v5.growth_factor), peaks at
                5.0725x (350 facilities) exactly at 2035.
  2035 < year <= 2045: linear decline from that peak to 0.
  year > 2045: 0 (no gas-powered data centers left; BASE_CH4[t]*gf and
               BASE_CO2[t]*gf are both exactly zero here too, since the
               growth multiplier is what scales the emissions baseline --
               "emissions go too" per the phase-out framing).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stage3_v5_real_leak_rates as v5
from dcfair import gwp_from_pulse

GWP_VALUES = {"GWP20": gwp_from_pulse(20), "GWP100": gwp_from_pulse(100)}
P = v5.P_LEAK
C_FIXED = "#a8560a"
C_RL = "#215870"

PEAK_YEAR = 2035
ZERO_YEAR = 2045
_idx_2035 = list(v5.YEARS).index(PEAK_YEAR)
_PEAK_GF = v5.growth_factor(_idx_2035)  # 5.0725 -- unchanged from the real schedule
print(f"Peak growth_factor at {PEAK_YEAR} (unchanged from real schedule): {_PEAK_GF:.4f}")


_orig_growth_factor = v5.growth_factor   # save BEFORE monkeypatching, or this recurses forever


def growth_factor_phaseout(t):
    year = 2025 + t
    if year <= PEAK_YEAR:
        return _orig_growth_factor(t)       # real schedule, unchanged, 1.0 -> 5.0725x
    if year <= ZERO_YEAR:
        frac = (ZERO_YEAR - year) / (ZERO_YEAR - PEAK_YEAR)   # 1.0 at 2035 -> 0.0 at 2045
        return _PEAK_GF * frac
    # Economically zero (no gas fleet, no emissions) but FleetGrowthEnv._state
    # normalizes by growth_factor(N-1), so an exact 0.0 here divides by zero.
    # 1e-9x the peak is indistinguishable from "gone" in every dollar/ton
    # calculation downstream (BASE_CH4*1e-9x, required_to_floor*1e-9x, etc.
    # all round to ~0), it just keeps the state normalization finite.
    return _PEAK_GF * 1e-9


v5.growth_factor = growth_factor_phaseout
v5.Q_LEAK = 0.0   # match the last standalone Q=0% run for direct comparability

years = 2025 + np.arange(v5.N)
gf_check = np.array([growth_factor_phaseout(t) for t in range(v5.N)])
print("growth_factor at 2035/2040/2045/2046/2100:",
      dict(zip([2035, 2040, 2045, 2046, 2100],
               [gf_check[list(years).index(y)] for y in [2035, 2040, 2045, 2046, 2100]])))

all_meta = {"peak_growth_factor": _PEAK_GF, "peak_year": PEAK_YEAR, "zero_year": ZERO_YEAR, "metrics": {}}

for metric, gwp in GWP_VALUES.items():
    print(f"\n=== phaseout / {metric} ===")
    policy, hist = v5.train(gwp, n_episodes=3000, seed=0)
    _, rewards_rl, infos_rl = v5.run_episode(v5.FleetGrowthEnv(gwp=gwp), policy, greedy=True)
    rewards_fixed, infos_fixed = v5.run_fixed_split(0.5, gwp)
    env0 = v5.FleetGrowthEnv(gwp=gwp, annual_budget=0.0)
    s = env0.reset(); rewards_none, infos_none = [], []
    done = False
    while not done:
        s, r, done, info = env0.step(0.5)
        rewards_none.append(r); infos_none.append(info)

    df_rl = pd.DataFrame(infos_rl); df_rl["year"] = v5.YEARS
    df_fixed = pd.DataFrame(infos_fixed); df_fixed["year"] = v5.YEARS
    df_none = pd.DataFrame(infos_none); df_none["year"] = v5.YEARS

    total_rl = float(np.sum(rewards_rl))
    total_fixed = float(np.sum(rewards_fixed))
    undisc_rl = float(sum(i["avoided_co2"] + gwp * i["avoided_ch4"] for i in infos_rl)) / 1e9
    undisc_fixed = float(sum(i["avoided_co2"] + gwp * i["avoided_ch4"] for i in infos_fixed)) / 1e9
    disc_adv = (total_rl / total_fixed - 1) * 100
    undisc_adv = (undisc_rl / undisc_fixed - 1) * 100
    print(f"discounted adv={disc_adv:.2f}%  undiscounted adv={undisc_adv:.2f}%")
    print(f"LDAR share at 2035 (peak, idx={_idx_2035}): fixed={df_fixed['a_ldar_share'].iloc[_idx_2035]:.3f} "
          f"rl={df_rl['a_ldar_share'].iloc[_idx_2035]:.3f}")
    print(f"LDAR share at 2045 (zero, idx={list(years).index(2045)}): "
          f"rl={df_rl['a_ldar_share'].iloc[list(years).index(2045)]:.3f}")
    print(f"LDAR share at 2100 (post-zero): rl={df_rl['a_ldar_share'].iloc[-1]:.3f}")

    prefix = "../results/stage3/stage3_v5_phaseout"
    df_rl.to_csv(f"{prefix}_{metric}_rl_trajectory.csv", index=False)
    df_fixed.to_csv(f"{prefix}_{metric}_fixed_trajectory.csv", index=False)
    df_none.to_csv(f"{prefix}_{metric}_none_trajectory.csv", index=False)
    np.savez(f"{prefix}_{metric}_training_history.npz", hist=np.array(hist))

    all_meta["metrics"][metric] = dict(
        total_fixed=total_fixed, total_rl=total_rl, discounted_adv_pct=disc_adv,
        undiscounted_rl_gt=undisc_rl, undiscounted_fixed_gt=undisc_fixed, undiscounted_adv_pct=undisc_adv,
    )

    # -------- Figure 1: allocation ratio
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.plot(df_fixed["year"], df_fixed["a_ldar_share"], color=C_FIXED, lw=2.4, label="Fixed 50/50 -- LDAR share")
    ax.plot(df_fixed["year"], 1 - df_fixed["a_ldar_share"], color=C_FIXED, lw=2.0, ls="--", label="Fixed 50/50 -- CCS share")
    ax.plot(df_rl["year"], df_rl["a_ldar_share"], color=C_RL, lw=2.4, label=f"RL ({metric}) -- LDAR share")
    ax.plot(df_rl["year"], 1 - df_rl["a_ldar_share"], color=C_RL, lw=2.0, ls="--", label=f"RL ({metric}) -- CCS share")
    ax.axvline(PEAK_YEAR, color="#5b3a7a", lw=1.2, ls=":")
    ax.axvline(ZERO_YEAR, color="#b23a2e", lw=1.4, ls=":")
    ax.text(PEAK_YEAR + 0.3, 0.97, "2035: fleet peaks\n(350 facilities)", fontsize=7.6, color="#5b3a7a")
    ax.text(ZERO_YEAR + 0.5, 0.97, "2045: gas fleet\nfully retired", fontsize=7.6, color="#b23a2e")
    start, end2035 = df_rl["a_ldar_share"].iloc[0], df_rl["a_ldar_share"].iloc[_idx_2035]
    ax.annotate(f"RL: {start:.0%} LDAR / {1-start:.0%} CCS at start", xy=(df_rl["year"].iloc[0], start),
                xytext=(2038, min(max(start - 0.28, 0.06), 0.85)), fontsize=8.2, color=C_RL,
                arrowprops=dict(arrowstyle="->", color=C_RL, lw=1))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Year"); ax.set_ylabel("Fraction of that year's budget")
    ax.set_title(f"Allocation ratio, gas fleet phases out 2035→2045 -- {metric} reward", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.4, loc="center right")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{prefix}_{metric}_allocation_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -------- Figure 2: cumulative
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for df, label, color, ls, lw in [(df_none, "No investment", "#8a8a80", ":", 1.8),
                                       (df_fixed, "Fixed 50/50 split", C_FIXED, "-", 2.6),
                                       (df_rl, f"RL ({metric})", C_RL, "-", 2.6)]:
        total = df["avoided_ch4"] * gwp + df["avoided_co2"]
        cum_gt = np.cumsum(total) / 1e9
        ax.plot(df["year"], cum_gt, color=color, lw=lw, linestyle=ls, label=label)
        ax.annotate(f"{cum_gt.iloc[-1]:.1f} Gt", xy=(df["year"].iloc[-1], cum_gt.iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", fontsize=9, color=color, fontweight="bold", va="center")
    ax.axvline(PEAK_YEAR, color="#5b3a7a", lw=1.0, ls=":")
    ax.axvline(ZERO_YEAR, color="#b23a2e", lw=1.2, ls=":")
    sign = "beats" if undisc_adv >= 0 else "trails"
    ax.set_xlabel("Year"); ax.set_ylabel(f"Cumulative avoided CO2e (Gt, {metric}-weighted)")
    ax.set_title(f"{metric}, gas fleet phases out by 2045: RL {sign} fixed 50/50 by {abs(undisc_adv):.1f}% (undiscounted)", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{prefix}_{metric}_cumulative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -------- Figure 3: training curve vs baselines + fixed-share sweep
    sweep = {}
    for share in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        rewards, _ = v5.run_fixed_split(share, gwp)
        sweep[share] = float(sum(rewards))
    ldar_first, fixed5050, ccs_first = sweep[1.0], sweep[0.5], sweep[0.0]
    best_share = max(sweep, key=sweep.get)
    ceiling = sweep[best_share]

    h = np.array(hist) / 1e9
    ep = np.arange(len(h))
    best_so_far = np.maximum.accumulate(h)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.plot(ep, h, color="#6a3d9a", lw=0.6, alpha=0.30, label="RL, raw episode reward")
    ax.plot(ep, best_so_far, color="#6a3d9a", lw=2.6, label="RL, best seen so far")
    ax.axhline(0.0, color="#9c9c94", lw=1.4, ls="--", label="No investment")
    ax.axhline(ldar_first / 1e9, color="#b23a2e", lw=1.6, ls="--", label="LDAR-first (share=1.0)")
    ax.axhline(fixed5050 / 1e9, color=C_FIXED, lw=1.8, ls="--", label="Fixed 50/50")
    ax.axhline(ccs_first / 1e9, color=C_RL, lw=1.8, ls="--", label="CCS-first (share=0.0)")
    gap_end_pct = (ceiling - best_so_far[-1] * 1e9) / ceiling * 100 if ceiling > 0 else 0.0
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"This episode's total discounted avoided CO2e (Gt, {metric})")
    ax.set_title(f"v5 training curve, gas fleet phases out 2035→2045 -- {metric} reward", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.3, loc="best")
    fig.text(0.5, -0.03,
              f"Fixed-share sweep: best constant split is share_ldar={best_share:.1f} ({ceiling/1e9:.2f} Gt); "
              f"RL best-so-far ends {gap_end_pct:.1f}% below/above that ceiling. "
              f"Almost all avoidable mass happens before 2045 -- after that, the gas fleet (and its emissions) is gone, so allocation stops mattering.",
              ha="center", fontsize=8.0, color="#555")
    plt.tight_layout()
    plt.savefig(f"{prefix}_{metric}_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  best fixed share={best_share:.1f} (ceiling={ceiling/1e9:.2f} Gt), saved 3 figures for phaseout/{metric}")

with open("../results/stage3/stage3_v5_phaseout_meta.json", "w") as f:
    json.dump(all_meta, f, indent=2)
print("\nAll figures + meta saved.")
