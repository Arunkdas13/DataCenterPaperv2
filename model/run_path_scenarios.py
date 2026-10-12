"""
Stage 2 (Path Planning): run DataCenter-FAIR on the real 69-facility EIP
fleet (see build_facility_emissions.py) under mitigation branches, 2025-
2100, and test directly whether the branches rank differently under a
near-term (GWP20) versus long-term (GWP100) weighting -- the temporal
trade-off diagnostic from Sartzetakis et al. (2025), applied here to the
actual proposed/permitted AI data center gas-turbine fleet instead of
narrative placeholder totals.

Branches
--------
1. baseline       - no mitigation; each facility emits its full CO2 and CH4
                     (stack + basin-specific upstream leak, see
                     build_facility_emissions.py) from its reported online
                     year through 2100.
2. ldar_{L}        - upstream methane leak rate cut by L% at each
                     facility's online year, for L in {50, 65, 80, 95}.
                     CO2 unaffected. A single-point LDAR efficacy number
                     could NOT be confirmed in a public-record research
                     pass (EPA's commonly-quoted "~80%" is for its entire
                     2023 methane rule package, not LDAR in isolation), so
                     rather than assert one guessed value, every plausible
                     value in this range is run and the crossover result
                     is checked for robustness across all of them.
3. ccs             - post-combustion CO2 capture at 90% (a standard
                     DOE/NETL assumption for amine-based capture),
                     retrofit 5 years after each facility's online year
                     (realistic permitting/construction lag). CH4
                     unaffected.
4. combined_{L}    - ldar_{L} from day one AND delayed CCS at the same
                     5-year lag -- the literal "sequencing" scenario.

Output
------
- results/path_scenarios.npz: full year-by-year trajectories per branch.
- results/path_scenarios_summary.csv: cumulative avoided CO2/CH4 and
  GWP20- vs GWP100-weighted avoided-CO2e per branch, relative to baseline.
- results/path_crossover_sensitivity.csv: crossover year (LDAR->CCS) under
  each metric, at each LDAR efficacy level -- the robustness check.
"""
import numpy as np
import pandas as pd
from dcfair import run_dcfair, gwp_from_pulse

YEARS = np.arange(2025, 2101)
N = len(YEARS)

LDAR_EFFICACY_LEVELS = [0.50, 0.65, 0.80, 0.95]
CCS_CAPTURE_RATE = 0.90      # standard post-combustion capture assumption
CCS_RETROFIT_LAG_YR = 5


def load_fleet():
    return pd.read_csv("../data/eip_74_facilities_processed.csv")


def fleet_trajectory(df, mitigate_ch4, ldar_efficacy, mitigate_co2):
    """Return (CO2_tons_per_yr[N], CH4_tons_per_yr[N]) summed across the fleet.
    mitigate_ch4 / mitigate_co2: bool flags for whether this branch applies
    the LDAR / CCS treatment. ldar_efficacy: fraction of upstream leak
    avoided, only used when mitigate_ch4 is True."""
    co2 = np.zeros(N)
    ch4 = np.zeros(N)
    for _, row in df.iterrows():
        online = int(row["online_year"])
        online_idx = online - YEARS[0]
        if online_idx >= N:
            continue
        online_idx = max(online_idx, 0)

        facility_co2_full = row["co2_tons_yr"]
        facility_stack_ch4 = row["stack_ch4_tons_yr"]
        facility_upstream_ch4_full = row["upstream_ch4_tons_yr"]

        for t in range(online_idx, N):
            year = YEARS[t]

            upstream = facility_upstream_ch4_full * (1 - ldar_efficacy) if mitigate_ch4 else facility_upstream_ch4_full
            ch4[t] += facility_stack_ch4 + upstream

            if mitigate_co2 and year >= online + CCS_RETROFIT_LAG_YR:
                co2[t] += facility_co2_full * (1 - CCS_CAPTURE_RATE)
            else:
                co2[t] += facility_co2_full

    return co2, ch4


def first_crossover(a, b, years):
    """First year at which series b (e.g. CCS) overtakes series a (e.g. LDAR)."""
    ahead = a > b
    for i in range(1, len(years)):
        if ahead[i - 1] and not ahead[i]:
            return int(years[i])
    return None


def run_all_branches():
    df = load_fleet()

    branches = {"baseline": dict(mitigate_ch4=False, ldar_efficacy=0.0, mitigate_co2=False),
                "ccs": dict(mitigate_ch4=False, ldar_efficacy=0.0, mitigate_co2=True)}
    for L in LDAR_EFFICACY_LEVELS:
        pct = int(L * 100)
        branches[f"ldar_{pct}"] = dict(mitigate_ch4=True, ldar_efficacy=L, mitigate_co2=False)
        branches[f"combined_{pct}"] = dict(mitigate_ch4=True, ldar_efficacy=L, mitigate_co2=True)

    results = {}
    for b, spec in branches.items():
        co2, ch4 = fleet_trajectory(df, **spec)
        r = run_dcfair(YEARS, co2, ch4)
        r["E_CO2"] = co2
        r["E_CH4"] = ch4
        results[b] = r
        print(f"[{b}] 2100: F_total={r['F_total'][-1]:.4e} W/m2, dT={r['dT'][-1]:.4e} K")

    np.savez("../results/path_scenarios.npz", years=YEARS,
             **{f"{b}_{k}": v for b in branches for k, v in results[b].items() if k != "years"})

    gwp20 = gwp_from_pulse(20)
    gwp100 = gwp_from_pulse(100)
    print(f"\nModel-derived GWP20={gwp20:.2f}, GWP100={gwp100:.2f}")

    base_co2_cum = results["baseline"]["E_CO2"].sum()
    base_ch4_cum = results["baseline"]["E_CH4"].sum()

    rows = []
    for b in branches:
        co2_cum = results[b]["E_CO2"].sum()
        ch4_cum = results[b]["E_CH4"].sum()
        avoided_co2 = base_co2_cum - co2_cum
        avoided_ch4 = base_ch4_cum - ch4_cum
        avoided_co2e_gwp20 = avoided_co2 + gwp20 * avoided_ch4
        avoided_co2e_gwp100 = avoided_co2 + gwp100 * avoided_ch4
        f_avoided_2045 = results["baseline"]["F_total"][2045 - YEARS[0]] - results[b]["F_total"][2045 - YEARS[0]]
        f_avoided_2100 = results["baseline"]["F_total"][-1] - results[b]["F_total"][-1]
        rows.append(dict(branch=b, avoided_CO2_tons_cum=avoided_co2, avoided_CH4_tons_cum=avoided_ch4,
                          avoided_CO2e_GWP20_tons=avoided_co2e_gwp20,
                          avoided_CO2e_GWP100_tons=avoided_co2e_gwp100,
                          F_avoided_2045_Wm2=f_avoided_2045, F_avoided_2100_Wm2=f_avoided_2100))
    summary = pd.DataFrame(rows)
    summary.to_csv("../results/path_scenarios_summary.csv", index=False)
    print("\n=== Branch comparison (relative to baseline) ===")
    print(summary.to_string(index=False))

    # ---- Sensitivity: crossover year (ldar_L -> ccs) at every LDAR efficacy level ----
    sens_rows = []
    for L in LDAR_EFFICACY_LEVELS:
        pct = int(L * 100)
        ldar_co2 = results[f"ldar_{pct}"]["E_CO2"]; ldar_ch4 = results[f"ldar_{pct}"]["E_CH4"]
        ccs_co2 = results["ccs"]["E_CO2"]; ccs_ch4 = results["ccs"]["E_CH4"]
        base_co2 = results["baseline"]["E_CO2"]; base_ch4 = results["baseline"]["E_CH4"]

        ldar_avoided_co2 = np.cumsum(base_co2 - ldar_co2); ldar_avoided_ch4 = np.cumsum(base_ch4 - ldar_ch4)
        ccs_avoided_co2 = np.cumsum(base_co2 - ccs_co2); ccs_avoided_ch4 = np.cumsum(base_ch4 - ccs_ch4)

        ldar20 = ldar_avoided_co2 + gwp20 * ldar_avoided_ch4
        ccs20 = ccs_avoided_co2 + gwp20 * ccs_avoided_ch4
        ldar100 = ldar_avoided_co2 + gwp100 * ldar_avoided_ch4
        ccs100 = ccs_avoided_co2 + gwp100 * ccs_avoided_ch4

        ldar_F = results["baseline"]["F_total"] - results[f"ldar_{pct}"]["F_total"]
        ccs_F = results["baseline"]["F_total"] - results["ccs"]["F_total"]

        sens_rows.append(dict(
            ldar_efficacy_pct=pct,
            crossover_GWP20=first_crossover(ldar20, ccs20, YEARS),
            crossover_GWP100=first_crossover(ldar100, ccs100, YEARS),
            crossover_raw_forcing=first_crossover(ldar_F, ccs_F, YEARS),
        ))
    sens = pd.DataFrame(sens_rows)
    sens.to_csv("../results/path_crossover_sensitivity.csv", index=False)
    print("\n=== Crossover year (LDAR -> CCS takes lead) vs. LDAR efficacy assumption ===")
    print(sens.to_string(index=False))
    print("\n(A crossover existing -- and staying within a similar few-year window --")
    print(" across the full 50-95% efficacy range means the finding does not depend on")
    print(" the one number that couldn't be pinned down in the literature search.)")

    return results, summary, sens


if __name__ == "__main__":
    import os
    os.makedirs("../results", exist_ok=True)
    run_all_branches()
