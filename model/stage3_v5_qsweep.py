"""
Same v5 model (real P=3.11% gas-weighted fleet leak rate, real fleet
growth) run at two more Q floor values requested directly: Q=1.5% and
Q=3.1% (~= P, i.e. almost no floor benefit at all). For each Q, produces
the same 6 figures as stage3_v5_figures.py (allocation ratio, cumulative,
training curve x {GWP20, GWP100}) -- standalone sets, no cross-Q
comparison figure.
"""
import os
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

Q_SCENARIOS = {"Q1p5": 0.015, "Q3p1": 0.031}


def set_q(q_leak):
    v5.Q_LEAK = q_leak
    target_f = 1.0 - (q_leak / P if P > 0 else 0.0)
    target_f = max(target_f, 0.0)
    one_time = v5._solve_one_time_ldar_cost(target_f) if target_f > 0 else 0.0
    opex_base = v5.LEAK_REAPPEARANCE_RATE * one_time
    v5._ONE_TIME_LDAR_COST_TO_FLOOR = one_time
    v5.LDAR_OPEX_TO_FLOOR_BASE = opex_base
    return target_f, one_time, opex_base


all_meta = {}
for qkey, q_val in Q_SCENARIOS.items():
    target_f, one_time, opex_base = set_q(q_val)
    print(f"\n########## Q={q_val*100:.2f}% (P={P*100:.3f}%, target_f={target_f:.4f}) ##########")
    print(f"one-time cost=${one_time/1e6:.1f}M  annual opex=${opex_base/1e6:.1f}M/yr")

    all_meta[qkey] = {"Q_leak": q_val, "P_leak": P, "target_f": target_f,
                       "one_time_cost": one_time, "opex_base": opex_base, "metrics": {}}

    for metric, gwp in GWP_VALUES.items():
        print(f"\n=== {qkey} / {metric} ===")
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
        print(f"leak rate y0: fixed={infos_fixed[0]['leak_rate']:.4f} rl={infos_rl[0]['leak_rate']:.4f}")
        print(f"leak rate y75: fixed={infos_fixed[-1]['leak_rate']:.4f} rl={infos_rl[-1]['leak_rate']:.4f}")

        prefix = f"../results/stage3/stage3_v5_{qkey}_{metric}"
        df_rl.to_csv(f"{prefix}_rl_trajectory.csv", index=False)
        df_fixed.to_csv(f"{prefix}_fixed_trajectory.csv", index=False)
        df_none.to_csv(f"{prefix}_none_trajectory.csv", index=False)
        np.savez(f"{prefix}_training_history.npz", hist=np.array(hist))

        all_meta[qkey]["metrics"][metric] = dict(
            total_fixed=total_fixed, total_rl=total_rl, discounted_adv_pct=disc_adv,
            undiscounted_rl_gt=undisc_rl, undiscounted_fixed_gt=undisc_fixed, undiscounted_adv_pct=undisc_adv,
        )

        # -------- Figure 1: allocation ratio
        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        ax.plot(df_fixed["year"], df_fixed["a_ldar_share"], color=C_FIXED, lw=2.4, label="Fixed 50/50 -- LDAR share")
        ax.plot(df_fixed["year"], 1 - df_fixed["a_ldar_share"], color=C_FIXED, lw=2.0, ls="--", label="Fixed 50/50 -- CCS share")
        ax.plot(df_rl["year"], df_rl["a_ldar_share"], color=C_RL, lw=2.4, label=f"RL ({metric}) -- LDAR share")
        ax.plot(df_rl["year"], 1 - df_rl["a_ldar_share"], color=C_RL, lw=2.0, ls="--", label=f"RL ({metric}) -- CCS share")
        start, end = df_rl["a_ldar_share"].iloc[0], df_rl["a_ldar_share"].iloc[-1]
        ax.annotate(f"RL: {start:.0%} LDAR / {1-start:.0%} CCS at start", xy=(df_rl["year"].iloc[0], start),
                    xytext=(2035, min(max(start + 0.15, 0.08), 0.90)), fontsize=8.5, color=C_RL,
                    arrowprops=dict(arrowstyle="->", color=C_RL, lw=1))
        ax.annotate(f"{end:.0%} LDAR / {1-end:.0%} CCS by 2100", xy=(df_rl["year"].iloc[-1], end),
                    xytext=(2062, min(max(end + 0.12, 0.08), 0.90)), fontsize=8.5, color=C_RL,
                    arrowprops=dict(arrowstyle="->", color=C_RL, lw=1))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Year"); ax.set_ylabel("Fraction of that year's budget")
        ax.set_title(f"Allocation ratio, P={P*100:.2f}%/Q={q_val*100:.2f}% -- {metric} reward", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8.4, loc="center right")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(f"{prefix}_allocation_ratio.png", dpi=150, bbox_inches="tight")
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
        sign = "beats" if undisc_adv >= 0 else "trails"
        ax.set_xlabel("Year"); ax.set_ylabel(f"Cumulative avoided CO2e (Gt, {metric}-weighted)")
        ax.set_title(f"{metric}, Q={q_val*100:.2f}%: RL {sign} fixed 50/50 by {abs(undisc_adv):.1f}% (undiscounted)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9.5, loc="upper left")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(f"{prefix}_cumulative.png", dpi=150, bbox_inches="tight")
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
        ax.set_title(f"v5 training curve, P={P*100:.2f}%/Q={q_val*100:.2f}% -- {metric} reward", fontsize=12.5, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.3, loc="best")
        fig.text(0.5, -0.03,
                  f"Fixed-share sweep: best constant split is share_ldar={best_share:.1f} ({ceiling/1e9:.2f} Gt); "
                  f"RL best-so-far ends {gap_end_pct:.1f}% below/above that ceiling.",
                  ha="center", fontsize=8.3, color="#555")
        plt.tight_layout()
        plt.savefig(f"{prefix}_training_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  best fixed share={best_share:.1f} (ceiling={ceiling/1e9:.2f} Gt), saved 3 figures for {qkey}/{metric}")

with open("../results/stage3/stage3_v5_qsweep_meta.json", "w") as f:
    json.dump(all_meta, f, indent=2)
print("\nAll figures + meta saved.")
