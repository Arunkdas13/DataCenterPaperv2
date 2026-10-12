"""
Stage 3, v5: same fleet-growth + leak-rate-equilibrium model as v4, but
replacing the mentor-call PLACEHOLDER leak-rate parameters (P=40% do-
nothing, Q=10% floor -- explicitly flagged in v4 as illustrative, not
sourced) with REAL, DERIVED values:

  P_LEAK = 3.11% -- the actual gas-throughput-weighted average leak rate
  across this project's real 69-facility fleet, using the SAME
  basin-specific rates already verified in ch4_leak_verification_figure.py
  (Permian 9.4%, Chen et al. 2022; Appalachian 0.52%, 2025 AMI study;
  national average 2.3%, Alvarez et al. 2018). This is not a new
  assumption -- it's the real fleet's current weighted-average leak rate,
  computed directly from data already in this project:

      P = sum(gas_throughput_i * leak_rate_i) / sum(gas_throughput_i)
        = 0.031104  (3.11%)

  Q_LEAK = 0.0 -- the floor is fully sealed (zero leakage), the requested
  extreme target rather than an assumed non-zero floor.

Everything else (fleet growth n(t), the flow-not-stock leak-rate mechanic,
CCS mechanism, budget) is unchanged from v4. This script additionally
trains against BOTH GWP20 and GWP100 reward weightings (v4 only used
GWP100), since near-term vs long-term metric choice is this project's
other central finding and P is now small enough that which metric you
optimize for could plausibly matter more.
"""
import os
import json
import numpy as np
import pandas as pd

from stage3_pipeline import (BASE_CO2, BASE_CH4, GWP20, GWP100, YEARS, N,
                              ccs_marginal_cost, dollars_to_delta_f,
                              ldar_marginal_cost, LDAR_POTENTIAL_T_CO2E_GWP100,
                              CCS_POTENTIAL_T_CO2, CCS_MAX_CAPTURE, DISCOUNT_R)

np.random.seed(7)

ANNUAL_BUDGET = 300e6

# -------------------------------------------------- leak-rate equilibrium
# REAL, derived from this project's own 69-facility fleet data (see
# docstring above) -- not a placeholder, not looked up from a table.
_facility_df = pd.read_csv("../data/eip_74_facilities_processed.csv")
P_LEAK = float((_facility_df["gas_throughput_scf_yr"] * _facility_df["leak_rate"]).sum()
               / _facility_df["gas_throughput_scf_yr"].sum())   # 0.031104
Q_LEAK = 0.0                                                      # fully sealed, the requested extreme

LEAK_REAPPEARANCE_RATE = 0.10  # unchanged from v4 -- still an illustrative placeholder,
                                 # flagged the same way (no sourced annual reappearance-rate
                                 # study was found); disclosed here again for clarity.


def _solve_one_time_ldar_cost(target_f, tol=1e-4, max_iter=60):
    lo, hi = 0.0, 1e12
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        df, _ = dollars_to_delta_f(mid, 0.0, ldar_marginal_cost, LDAR_POTENTIAL_T_CO2E_GWP100)
        if df < target_f:
            lo = mid
        else:
            hi = mid
    return hi


# Q=0 means the full addressable leak fraction (100% of P) must be closed,
# not 75% as in v4's P=40%/Q=10% case.
_TARGET_F = 1.0 - (Q_LEAK / P_LEAK if P_LEAK > 0 else 0.0)   # = 1.0
_ONE_TIME_LDAR_COST_TO_FLOOR = _solve_one_time_ldar_cost(_TARGET_F)
LDAR_OPEX_TO_FLOOR_BASE = LEAK_REAPPEARANCE_RATE * _ONE_TIME_LDAR_COST_TO_FLOOR

# -------------------------------------------------------- fleet growth n(t)
# Identical to v4 (Aug 28 bug fix already applied): real dcmap.us Aug 2026
# dated pipeline schedule, added onto the real, MODELED 69-facility fleet.
N0 = 69.0
_CUMULATIVE_ADDITIONS = {
    2025: 0, 2026: 111, 2027: 193, 2028: 246, 2029: 254, 2030: 267,
    2031: 270, 2032: 273, 2033: 276, 2034: 278, 2035: 281,
}
_LAST_KNOWN_YEAR = 2035
_LAST_KNOWN_CUM = _CUMULATIVE_ADDITIONS[_LAST_KNOWN_YEAR]  # 281


def _fleet_size(year):
    y = min(year, _LAST_KNOWN_YEAR)
    cum = _CUMULATIVE_ADDITIONS.get(y, _LAST_KNOWN_CUM)
    return N0 + cum


def growth_factor(t):
    return _fleet_size(2025 + t) / N0


class FleetGrowthEnv:
    """Same mechanics as v4's FleetGrowthEnv, parameterized by which GWP
    horizon weights CH4 in the reward (GWP20 or GWP100) -- v4 only ever
    used GWP100."""

    def __init__(self, gwp=GWP100, annual_budget=ANNUAL_BUDGET):
        self.gwp = gwp
        self.annual_budget = annual_budget
        self.reset()

    def reset(self):
        self.t = 0
        self.f_ccs = 0.0
        return self._state()

    def _state(self):
        return np.array([
            self.t / (N - 1),
            self.f_ccs,
            growth_factor(self.t) / growth_factor(N - 1),
        ])

    def step(self, a_ldar_share):
        a_ldar_share = float(np.clip(a_ldar_share, 0.0, 1.0))
        dollars_ldar = a_ldar_share * self.annual_budget
        dollars_ccs = (1 - a_ldar_share) * self.annual_budget

        gf = growth_factor(self.t)
        required_to_floor = LDAR_OPEX_TO_FLOOR_BASE * gf
        frac_of_need = np.clip(dollars_ldar / required_to_floor, 0.0, 1.0) if required_to_floor > 0 else 0.0
        leak_rate = P_LEAK - (P_LEAK - Q_LEAK) * frac_of_need

        df_ccs, spent_ccs = dollars_to_delta_f(dollars_ccs, self.f_ccs, ccs_marginal_cost, CCS_POTENTIAL_T_CO2)
        self.f_ccs = min(self.f_ccs + df_ccs, 1.0)

        avoided_ch4 = BASE_CH4[self.t] * gf * (1.0 - leak_rate / P_LEAK if P_LEAK > 0 else 1.0)
        avoided_co2 = BASE_CO2[self.t] * gf * self.f_ccs * CCS_MAX_CAPTURE
        co2e_avoided = avoided_co2 + self.gwp * avoided_ch4
        reward = co2e_avoided * (1 - DISCOUNT_R) ** self.t

        info = dict(leak_rate=leak_rate, f_ccs=self.f_ccs, growth_factor=gf,
                    a_ldar_share=a_ldar_share, dollars_ldar=dollars_ldar,
                    dollars_ccs=dollars_ccs, avoided_ch4=avoided_ch4, avoided_co2=avoided_co2)
        self.t += 1
        done = self.t >= N
        return (self._state() if not done else None), reward, done, info


class Policy:
    def __init__(self, n_in=3, n_hidden=10, lr=0.02, std=0.15, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(n_in, n_hidden) * 0.3
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden) * 0.3
        self.b2 = 0.0
        self.lr = lr
        self.std = std

    def forward(self, s):
        h_pre = s @ self.W1 + self.b1
        h = np.tanh(h_pre)
        out_pre = h @ self.W2 + self.b2
        mean = 1 / (1 + np.exp(-out_pre))
        return mean, h

    def act(self, s, greedy=False):
        mean, h = self.forward(s)
        if greedy:
            return float(np.clip(mean, 0, 1)), None
        a = np.random.normal(mean, self.std)
        a = float(np.clip(a, 0.0, 1.0))
        return a, (s, h, mean, a)

    def update(self, cache, adv):
        gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
        gW2 = np.zeros_like(self.W2); gb2 = 0.0
        for (s, h, mean, a), av in zip(cache, adv):
            dlogp_dmean = (a - mean) / (self.std ** 2)
            dmean_dout = mean * (1 - mean)
            dout = dlogp_dmean * dmean_dout * av
            gW2 += h * dout; gb2 += dout
            dh = self.W2 * dout
            dh_pre = dh * (1 - h ** 2)
            gW1 += np.outer(s, dh_pre); gb1 += dh_pre
        n = len(cache)
        return gW1 / n, gb1 / n, gW2 / n, gb2 / n

    def apply_grad(self, g):
        gW1, gb1, gW2, gb2 = g
        self.W1 += self.lr * gW1; self.b1 += self.lr * gb1
        self.W2 += self.lr * gW2; self.b2 += self.lr * gb2

    def copy(self):
        p = Policy.__new__(Policy)
        p.W1 = self.W1.copy(); p.b1 = self.b1.copy()
        p.W2 = self.W2.copy(); p.b2 = self.b2
        p.lr = self.lr; p.std = self.std
        return p


def run_episode(env, policy, greedy=False):
    s = env.reset()
    cache, rewards, infos = [], [], []
    done = False
    while not done:
        a, tr = policy.act(s, greedy=greedy)
        s, r, done, info = env.step(a)
        rewards.append(r); infos.append(info)
        if not greedy:
            cache.append(tr)
    return cache, rewards, infos


def discounted_returns(rewards):
    return np.cumsum(rewards[::-1])[::-1]


def train(gwp, n_episodes=3000, eval_every=50, seed=0):
    env = FleetGrowthEnv(gwp=gwp)
    policy = Policy(seed=seed)
    best_policy = policy.copy()
    best_eval = -np.inf
    history = []
    for ep in range(n_episodes):
        cache, rewards, infos = run_episode(env, policy)
        returns = discounted_returns(rewards)
        returns_std = (returns - returns.mean()) / (returns.std() + 1e-8)
        g = policy.update(cache, returns_std)
        policy.apply_grad(g)
        history.append(float(np.sum(rewards)))
        if ep % eval_every == 0 or ep == n_episodes - 1:
            _, eval_rewards, _ = run_episode(FleetGrowthEnv(gwp=gwp), policy, greedy=True)
            eval_total = float(np.sum(eval_rewards))
            if eval_total > best_eval:
                best_eval = eval_total
                best_policy = policy.copy()
    return best_policy, history


def run_fixed_split(share, gwp):
    env = FleetGrowthEnv(gwp=gwp)
    s = env.reset()
    rewards, infos = [], []
    done = False
    while not done:
        s, r, done, info = env.step(share)
        rewards.append(r); infos.append(info)
    return rewards, infos


if __name__ == "__main__":
    os.makedirs("../results/stage3", exist_ok=True)
    print(f"P_LEAK (real, gas-weighted fleet average) = {P_LEAK:.6f} ({P_LEAK*100:.3f}%)")
    print(f"Q_LEAK (fully sealed target) = {Q_LEAK}")
    print(f"One-time cost to close full P->0 gap: ${_ONE_TIME_LDAR_COST_TO_FLOOR/1e6:.1f}M")
    print(f"Annual OPEX to hold near floor: ${LDAR_OPEX_TO_FLOOR_BASE/1e6:.1f}M/yr")

    all_meta = {}
    for metric_name, gwp in [("GWP20", GWP20), ("GWP100", GWP100)]:
        print(f"\n=== Training against {metric_name} (gwp={gwp:.2f}) ===")
        policy, hist = train(gwp, n_episodes=3000, seed=0)
        _, rewards_rl, infos_rl = run_episode(FleetGrowthEnv(gwp=gwp), policy, greedy=True)
        rewards_fixed, infos_fixed = run_fixed_split(0.5, gwp)
        env0 = FleetGrowthEnv(gwp=gwp, annual_budget=0.0)
        s = env0.reset(); rewards_none, infos_none = [], []
        done = False
        while not done:
            s, r, done, info = env0.step(0.5)
            rewards_none.append(r); infos_none.append(info)

        df_rl = pd.DataFrame(infos_rl); df_rl["year"] = YEARS
        df_fixed = pd.DataFrame(infos_fixed); df_fixed["year"] = YEARS
        df_none = pd.DataFrame(infos_none); df_none["year"] = YEARS

        total_rl = float(np.sum(rewards_rl))
        total_fixed = float(np.sum(rewards_fixed))
        # GWP-weighted (matches the reward formula exactly, minus the discount factor) --
        # NOT a raw sum of CH4 tons + CO2 tons, which would silently treat 1 ton CH4 as
        # equivalent to 1 ton CO2 (dimensionally wrong; caught during this run's review).
        undisc_rl = float(sum(i["avoided_co2"] + gwp * i["avoided_ch4"] for i in infos_rl)) / 1e9
        undisc_fixed = float(sum(i["avoided_co2"] + gwp * i["avoided_ch4"] for i in infos_fixed)) / 1e9

        print(f"Total reward -- No investment: {sum(rewards_none):.3e}")
        print(f"Total reward -- Fixed 50/50:   {total_fixed:.3e}")
        print(f"Total reward -- RL adaptive:   {total_rl:.3e}")
        print(f"RL advantage over fixed 50/50 (discounted):   {(total_rl/total_fixed-1)*100:.2f}%")
        print(f"RL advantage over fixed 50/50 (undiscounted): {(undisc_rl/undisc_fixed-1)*100:.2f}%")
        print(f"Leak rate at year 0:  fixed={infos_fixed[0]['leak_rate']:.4f}  rl={infos_rl[0]['leak_rate']:.4f}")
        print(f"Leak rate at year 75: fixed={infos_fixed[-1]['leak_rate']:.4f}  rl={infos_rl[-1]['leak_rate']:.4f}")

        df_rl.to_csv(f"../results/stage3/stage3_v5_{metric_name}_rl_trajectory.csv", index=False)
        df_fixed.to_csv(f"../results/stage3/stage3_v5_{metric_name}_fixed_trajectory.csv", index=False)
        df_none.to_csv(f"../results/stage3/stage3_v5_{metric_name}_none_trajectory.csv", index=False)
        np.savez(f"../results/stage3/stage3_v5_{metric_name}_training_history.npz", hist=np.array(hist))

        all_meta[metric_name] = dict(
            P_leak=P_LEAK, Q_leak=Q_LEAK,
            ldar_opex_to_floor_base=LDAR_OPEX_TO_FLOOR_BASE,
            one_time_ldar_cost_to_floor=_ONE_TIME_LDAR_COST_TO_FLOOR,
            total_none=float(sum(rewards_none)),
            total_fixed=total_fixed, total_rl=total_rl,
            discounted_adv_pct=(total_rl/total_fixed-1)*100,
            undiscounted_rl_gt=undisc_rl, undiscounted_fixed_gt=undisc_fixed,
            undiscounted_adv_pct=(undisc_rl/undisc_fixed-1)*100,
        )

    with open("../results/stage3/stage3_v5_meta.json", "w") as f:
        json.dump(all_meta, f, indent=2)
    print("\nSaved v5 results (both metrics).")
