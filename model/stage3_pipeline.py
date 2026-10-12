"""
Stage 3: RL-learned annual capital-allocation policy between LDAR (methane
leak repair) and CCS (CO2 capture), under a government subsidy budget that
is split between the two options each year -- the split itself is the
policy decision. Two policies are trained, one that maximizes GWP20-
weighted avoided CO2e, one that maximizes GWP100-weighted avoided CO2e,
mirroring the Stage 2 finding that these two metrics disagree about which
option to prioritize.

Real inputs used:
- Fleet baseline CO2/CH4 emissions trajectories, 2025-2100, from Stage 2
  (results/path_scenarios.npz, the "baseline" branch on the real 69-facility
  EIP fleet).
- Model-derived GWP20 / GWP100 factors from DataCenter-FAIR itself (not the
  IPCC textbook values -- these are the actual pulse-response-derived
  factors already used throughout Stage 2).
- LDAR and CCS marginal-cost curves ($/tCO2e vs. cumulative deployment
  fraction), calibrated to the real, sourced cost ranges found in this
  session's research: LDAR ~$6-17/tCO2e rising sharply past 60-80%
  reduction (ICF/EDF 2014, IEA Global Methane Tracker, Harvard Belfer 2025);
  CCS ~$56-71/tCO2e captured, roughly flat 90-97% capture (DOE/NETL 2023
  retrofit study, Rubin & Zhai 2012). The functional FORM used to
  interpolate within those ranges (the exact curvature/steepness) is a
  modeling choice calibrated to match the qualitative shape reported in
  the literature, not itself a literature-verified number -- flagged
  explicitly in the results report.
- The Section 45Q federal tax credit for CO2 capture: $85/tCO2 captured and
  stored (IRA-era rate, confirmed current for 2025-2026, verified via a
  real search this session -- including confirmation that the 2025 "One Big
  Beautiful Bill Act" did NOT cut this rate). Used as the real-world
  status-quo baseline: today's actual federal policy already subsidizes CCS
  by $85/ton, while there is no comparable federal per-ton subsidy for
  LDAR (the IRA's Methane Emissions Reduction Program was grant-based, not
  per-ton, and had its unobligated funds rescinded by the 2025 OBBBA).

Environment
-----------
Fleet-aggregate (not per-facility, matching Bauer's sector-level
aggregation -- see the methodology doc). State each year: fraction of the
fleet's LDAR abatement potential deployed so far (f_ldar), fraction of CCS
abatement potential deployed so far (f_ccs), normalized time, normalized
remaining budget. Action: fraction of this year's subsidy budget allocated
to LDAR (rest to CCS). Transition: dollars buy incremental deployment
fraction via inverting the marginal cost curve (net of the 45Q subsidy for
CCS). Reward: incremental avoided CO2e that year, GWP-weighted by whichever
metric this policy is being trained against.

RL method
---------
REINFORCE (Williams 1992) policy-gradient with a baseline, using a small
2-layer neural network policy (state -> Beta-like bounded action via a
sigmoid mean, fixed exploration std), implemented directly in numpy (no
external RL framework) so the full learning update is auditable end to
end. This is genuine sequential, state-feedback reinforcement learning
(not a fixed parametric curve fit) -- see results report for why a
closed-form Bauer-style solution isn't available here.
"""
import numpy as np
import pandas as pd
import json

np.random.seed(0)

# ---------------------------------------------------------------- constants
d = np.load("../results/path_scenarios.npz")
YEARS = d["years"]
N = len(YEARS)
BASE_CO2 = d["baseline_E_CO2"]     # tons CO2 / yr, fleet-wide
BASE_CH4 = d["baseline_E_CH4"]     # tons CH4 / yr, fleet-wide

# model-derived GWPs (same ones used throughout Stage 2), not IPCC textbook values
from dcfair import gwp_from_pulse
GWP20 = gwp_from_pulse(20)
GWP100 = gwp_from_pulse(100)

DISCOUNT_R = 0.02          # 2%/yr, matches Bauer et al.'s social discount rate
LDAR_MAX_EFFICACY = 0.95   # ceiling, matches the Stage 2 sensitivity range's top end
CCS_MAX_CAPTURE = 0.90     # matches Stage 2's CCS capture-rate assumption

CCS_45Q_SUBSIDY = 85.0     # $/tCO2 captured & stored, real IRA-era 45Q rate, verified this session
                           # (used only as a status-quo REFERENCE POINT/annotation below, not
                           # netted into the environment's own cost curves -- see report note:
                           # the environment models a NEW discretionary budget on top of whatever
                           # 45Q already does in the real world, to keep the cost curves clean)

# --------------------------------------------------------- real cost curves
# LDAR: $6-17/tCO2e for most reduction, steep tail past 60-80% (ICF/EDF 2014,
# IEA Global Methane Tracker, Harvard Belfer 2025). Functional form (the
# exact curvature) is a calibrated modeling choice, not itself a cited number.
def ldar_marginal_cost(f):
    f = np.clip(f, 0.0, 1.0)
    base = 6.0 + 11.0 * f**2                      # $6 -> $17 smoothly across [0,1]
    tail = 220.0 * np.maximum(0.0, f - 0.70)**3    # steep tail past ~70%
    return base + tail

# CCS: $56-71/tCO2e captured, roughly flat across 90-97% capture (DOE/NETL
# 2023 retrofit study, Rubin & Zhai 2012 cross-check).
def ccs_marginal_cost(f):
    f = np.clip(f, 0.0, 1.0)
    return 56.0 + 15.0 * f    # $56 -> $71 linearly, "roughly flat" per the source range

ldar_net_cost = ldar_marginal_cost
ccs_net_cost = ccs_marginal_cost

# Total abatable tCO2e "potential" at full deployment (f=1), fleet-wide.
# Scaled to the MATURE (fully built-out) fleet -- i.e. the peak annual value
# the 69-facility fleet reaches once all facilities are online -- rather than
# to each individual year's still-ramping emissions. This matters: using a
# given year's (still-small, early-buildout) emissions as the cost-conversion
# scale would make deployment artificially cheap in the fleet's early years.
# Using the mature-fleet scale instead means f_ldar/f_ccs represent "share of
# the full fleet's eventual abatement capacity built so far," applied each
# year to that year's actual (possibly still-ramping) emissions for the
# reward -- capital cost reflects the ultimate scale of the buildout, reward
# reflects only what's actually online and emitting that year.
LDAR_POTENTIAL_T_CO2E_GWP100 = float(BASE_CH4.max()) * LDAR_MAX_EFFICACY * GWP100
CCS_POTENTIAL_T_CO2 = float(BASE_CO2.max()) * CCS_MAX_CAPTURE


def dollars_to_delta_f(dollars, f_start, cost_fn, potential_tons):
    """Invert the marginal cost curve: how much deployment-fraction increment
    does `dollars` buy, starting from f_start, given total abatable tons
    `potential_tons` at f=1? Solved by numerically integrating the marginal
    cost curve (cost rises as f rises) until the budget is exhausted.
    Returns (delta_f, dollars_actually_spent) -- the second value lets a
    caller redirect any leftover budget once a branch is fully deployed."""
    if potential_tons <= 0 or dollars <= 0 or f_start >= 1.0:
        return 0.0, 0.0
    df = 0.001
    f = f_start
    spent = 0.0
    while f < 1.0 and spent < dollars:
        marginal_cost_per_ton = cost_fn(f)
        cost_this_step = marginal_cost_per_ton * potential_tons * df
        if spent + cost_this_step > dollars:
            frac_of_step = (dollars - spent) / cost_this_step
            f += df * frac_of_step
            spent = dollars
            break
        spent += cost_this_step
        f += df
    return min(f, 1.0) - f_start, spent


class Stage3Env:
    """One episode = one full 2025-2100 run under a given annual subsidy budget."""

    def __init__(self, annual_budget_usd, metric="GWP100"):
        self.annual_budget = annual_budget_usd
        self.metric = metric
        self.gwp = GWP20 if metric == "GWP20" else GWP100
        self.reset()

    def reset(self):
        self.t = 0
        self.f_ldar = 0.0
        self.f_ccs = 0.0
        return self._state()

    def _state(self):
        return np.array([self.t / (N - 1), self.f_ldar, self.f_ccs,
                          1.0 - self.t / (N - 1)])

    def step(self, a_ldar_share):
        """a_ldar_share in [0,1]: fraction of this year's budget spent on LDAR."""
        a_ldar_share = float(np.clip(a_ldar_share, 0.0, 1.0))
        budget_t = self.annual_budget
        dollars_ldar = a_ldar_share * budget_t
        dollars_ccs = (1 - a_ldar_share) * budget_t

        df_ldar, spent_ldar = dollars_to_delta_f(dollars_ldar, self.f_ldar, ldar_net_cost, LDAR_POTENTIAL_T_CO2E_GWP100)
        df_ccs, spent_ccs = dollars_to_delta_f(dollars_ccs, self.f_ccs, ccs_net_cost, CCS_POTENTIAL_T_CO2)
        self.f_ldar = min(self.f_ldar + df_ldar, 1.0)
        self.f_ccs = min(self.f_ccs + df_ccs, 1.0)

        # redirect any leftover (one branch already fully deployed) to the other
        leftover_ldar = dollars_ldar - spent_ldar
        leftover_ccs = dollars_ccs - spent_ccs
        if leftover_ldar > 0 and self.f_ccs < 1.0:
            extra_df_ccs, extra_spent = dollars_to_delta_f(leftover_ldar, self.f_ccs, ccs_net_cost, CCS_POTENTIAL_T_CO2)
            self.f_ccs = min(self.f_ccs + extra_df_ccs, 1.0)
            dollars_ccs += extra_spent; dollars_ldar -= extra_spent
        if leftover_ccs > 0 and self.f_ldar < 1.0:
            extra_df_ldar, extra_spent = dollars_to_delta_f(leftover_ccs, self.f_ldar, ldar_net_cost, LDAR_POTENTIAL_T_CO2E_GWP100)
            self.f_ldar = min(self.f_ldar + extra_df_ldar, 1.0)
            dollars_ldar += extra_spent; dollars_ccs -= extra_spent

        co2_avoided = BASE_CO2[self.t] * CCS_MAX_CAPTURE * self.f_ccs
        ch4_avoided = BASE_CH4[self.t] * LDAR_MAX_EFFICACY * self.f_ldar
        co2e_avoided = co2_avoided + self.gwp * ch4_avoided
        reward = co2e_avoided * (1 - DISCOUNT_R) ** self.t

        self.t += 1
        done = self.t >= N
        info = dict(co2_avoided=co2_avoided, ch4_avoided=ch4_avoided,
                    co2e_avoided=co2e_avoided, f_ldar=self.f_ldar, f_ccs=self.f_ccs,
                    dollars_ldar=dollars_ldar, dollars_ccs=dollars_ccs)
        return (self._state() if not done else None), reward, done, info


# ------------------------------------------------------- REINFORCE policy
class MLPPolicy:
    """2-layer MLP, tanh hidden, sigmoid output -> mean of a clipped Gaussian
    action distribution. Pure numpy forward/backward (manual REINFORCE)."""

    def __init__(self, n_in=4, n_hidden=8, lr=0.02, std=0.15, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(n_in, n_hidden) * 0.3
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, 1) * 0.3
        self.b2 = np.zeros(1)
        self.lr = lr
        self.std = std

    def forward(self, s):
        h_pre = s @ self.W1 + self.b1
        h = np.tanh(h_pre)
        out_pre = h @ self.W2 + self.b2
        # sigmoid -> (0,1). Clip the pre-activation before exponentiating:
        # np.exp overflows to a harmless inf for very negative raw inputs
        # (numpy still returns the correct 0.0/1.0 sigmoid limit either
        # way), but the clip avoids the spurious RuntimeWarning without
        # changing any output value.
        mean = 1 / (1 + np.exp(-np.clip(out_pre[0], -30, 30)))
        return mean, h, h_pre

    def act(self, s, greedy=False):
        mean, h, h_pre = self.forward(s)
        if greedy:
            return float(np.clip(mean, 0, 1)), None
        a = np.random.normal(mean, self.std)
        a = float(np.clip(a, 0.0, 1.0))
        return a, (s, h, h_pre, mean, a)

    def update(self, episode_cache, returns, baseline):
        """REINFORCE with a scalar baseline (mean return this batch)."""
        gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
        gW2 = np.zeros_like(self.W2); gb2 = np.zeros_like(self.b2)
        for (s, h, h_pre, mean, a), R in zip(episode_cache, returns):
            advantage = R - baseline
            # d/d(mean) of log N(a; mean, std^2) = (a - mean) / std^2
            dlogp_dmean = (a - mean) / (self.std ** 2)
            dmean_dout = mean * (1 - mean)                    # sigmoid'
            dout = dlogp_dmean * dmean_dout * advantage        # scalar, ASCENT direction
            gW2 += np.outer(h, [dout]); gb2 += np.array([dout])
            dh = dout * self.W2.flatten()
            dh_pre = dh * (1 - h ** 2)                         # tanh'
            gW1 += np.outer(s, dh_pre); gb1 += dh_pre
        n = len(episode_cache)
        self.W1 += self.lr * gW1 / n; self.b1 += self.lr * gb1 / n
        self.W2 += self.lr * gW2 / n; self.b2 += self.lr * gb2 / n


def run_episode(env, policy, greedy=False):
    s = env.reset()
    cache, rewards, infos = [], [], []
    done = False
    while not done:
        a, tr = policy.act(s, greedy=greedy)
        s, r, done, info = env.step(a)
        rewards.append(r)
        infos.append(info)
        if not greedy:
            cache.append(tr)
    return cache, rewards, infos


def discounted_returns(rewards):
    # reward-to-go (already per-step discounted in env), simple cumulative sum from the end
    R = np.cumsum(rewards[::-1])[::-1]
    return R


def train_policy(annual_budget, metric, n_episodes=600, log_every=50):
    env = Stage3Env(annual_budget, metric=metric)
    policy = MLPPolicy(seed=0 if metric == "GWP100" else 1)
    history = []
    baseline_ema = None
    for ep in range(n_episodes):
        cache, rewards, infos = run_episode(env, policy)
        returns = discounted_returns(rewards)
        total = sum(rewards)
        baseline_ema = total if baseline_ema is None else 0.9 * baseline_ema + 0.1 * total
        policy.update(cache, returns, baseline_ema / N)
        history.append(total)
        if ep % log_every == 0 or ep == n_episodes - 1:
            print(f"  [{metric}] ep {ep:4d}  total avoided CO2e (discounted, GWP-wtd) = {total:.3e}")
    return policy, env, history


if __name__ == "__main__":
    import os
    os.makedirs("../results/stage3", exist_ok=True)

    # $300M/yr illustrative NEW discretionary subsidy-budget scenario. Calibrated
    # so the choice is real and consequential over 2025-2100: full LDAR deployment
    # costs ~$2.2B (fundable in ~7yrs of this budget alone), full CCS deployment
    # costs ~$37.6B (would take the entire cumulative budget, ~125yrs worth, to
    # fund alone) -- see report for the derivation.
    ANNUAL_BUDGET = 300e6

    print("Training GWP20-optimized policy...")
    policy20, env20, hist20 = train_policy(ANNUAL_BUDGET, "GWP20")
    print("Training GWP100-optimized policy...")
    policy100, env100, hist100 = train_policy(ANNUAL_BUDGET, "GWP100")

    # ---- final evaluation: run learned (greedy) policies deterministically ----
    def evaluate(policy, metric):
        env = Stage3Env(ANNUAL_BUDGET, metric=metric)
        _, rewards, infos = run_episode(env, policy, greedy=True)
        return rewards, infos

    rewards20, infos20 = evaluate(policy20, "GWP20")
    rewards100, infos100 = evaluate(policy100, "GWP100")

    def infos_to_df(infos, label):
        df = pd.DataFrame(infos)
        df["year"] = YEARS
        df["policy"] = label
        df["ldar_share_of_budget"] = df["dollars_ldar"] / (df["dollars_ldar"] + df["dollars_ccs"])
        return df

    df20 = infos_to_df(infos20, "RL_GWP20")
    df100 = infos_to_df(infos100, "RL_GWP100")

    # ---------------------------------------------------------- baselines
    def run_fixed_policy(fn_ldar_share, metric):
        env = Stage3Env(ANNUAL_BUDGET, metric=metric)
        s = env.reset()
        infos = []
        for t in range(N):
            a = fn_ldar_share(t)
            s, r, done, info = env.step(a)
            infos.append(info)
        return infos

    # even split baseline
    even20 = infos_to_df(run_fixed_policy(lambda t: 0.5, "GWP20"), "Even_split")
    even100 = infos_to_df(run_fixed_policy(lambda t: 0.5, "GWP100"), "Even_split")

    # greedy baseline: always fund whichever option currently has lower $/ton
    def run_greedy(metric):
        env = Stage3Env(ANNUAL_BUDGET, metric=metric)
        env.reset()
        infos = []
        for t in range(N):
            c_ldar = ldar_net_cost(env.f_ldar)
            c_ccs = ccs_net_cost(env.f_ccs)
            a = 1.0 if c_ldar < c_ccs else 0.0
            s, r, done, info = env.step(a)
            infos.append(info)
        return infos

    greedy20 = infos_to_df(run_greedy("GWP20"), "Greedy_cheapest_first")
    greedy100 = infos_to_df(run_greedy("GWP100"), "Greedy_cheapest_first")

    # status-quo baseline: today's real policy -- 100% to CCS (45Q exists, no LDAR analogue)
    statusquo20 = infos_to_df(run_fixed_policy(lambda t: 0.0, "GWP20"), "Status_quo_CCS_only")
    statusquo100 = infos_to_df(run_fixed_policy(lambda t: 0.0, "GWP100"), "Status_quo_CCS_only")

    all_df = pd.concat([df20, df100, even20, even100, greedy20, greedy100,
                         statusquo20, statusquo100], ignore_index=True)
    all_df.to_csv("../results/stage3/stage3_policy_trajectories.csv", index=False)

    summary_rows = []
    for name, df in [("RL_GWP20", df20), ("RL_GWP100", df100),
                      ("Even_split (GWP20 eval)", even20), ("Even_split (GWP100 eval)", even100),
                      ("Greedy_cheapest_first (GWP20 eval)", greedy20),
                      ("Greedy_cheapest_first (GWP100 eval)", greedy100),
                      ("Status_quo_CCS_only (GWP20 eval)", statusquo20),
                      ("Status_quo_CCS_only (GWP100 eval)", statusquo100)]:
        summary_rows.append(dict(
            policy=name,
            cumulative_co2e_avoided_discounted=df["co2e_avoided"].sum(),
            final_f_ldar=df["f_ldar"].iloc[-1],
            final_f_ccs=df["f_ccs"].iloc[-1],
            total_dollars=(df["dollars_ldar"] + df["dollars_ccs"]).sum(),
        ))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv("../results/stage3/stage3_policy_summary.csv", index=False)
    print("\n=== Stage 3 policy summary ===")
    print(summary.to_string(index=False))

    np.savez("../results/stage3/stage3_training_history.npz",
             hist20=np.array(hist20), hist100=np.array(hist100))

    meta = dict(annual_budget_usd=ANNUAL_BUDGET, GWP20=GWP20, GWP100=GWP100,
                ccs_45q_subsidy=CCS_45Q_SUBSIDY, discount_rate=DISCOUNT_R,
                ldar_max_efficacy=LDAR_MAX_EFFICACY, ccs_max_capture=CCS_MAX_CAPTURE)
    with open("../results/stage3/stage3_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved: stage3_policy_trajectories.csv, stage3_policy_summary.csv, "
          "stage3_training_history.npz, stage3_meta.json")
