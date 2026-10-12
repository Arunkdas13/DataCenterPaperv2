"""
Build real, facility-level CO2 and CH4 emissions trajectories (2025-2100)
for the 74 gas power plants dedicated to AI data centers identified by the
Environmental Integrity Project (EIP), "The Power Behind AI" (2026),
Appendix C.

This replaces the earlier narrative-placeholder constants
(FULL_CO2_TONS = 35,000,000; FULL_CH4_TONS = 205,000) with numbers derived
directly from the project's own EIP_Report_ThePowerBehindAI_7.1.26.pdf,
following the report's own methodology (Appendix B) wherever possible, and
extending it only where the report is silent -- documented explicitly below.

WHAT THE EIP REPORT GIVES US
-----------------------------
- Per-facility: state, county, plant name, project type, expected operating
  year, generating capacity (MW), and annual greenhouse gases as CO2e
  (tons/yr), already GWP100-weighted per the report's Appendix B.
- Table 5: blended CO2e emission factors by combustion technology
  (simple-cycle 0.66, combined-cycle 0.43, internal combustion engine 0.49
  tons CO2e/MWh), and the heat rates used to derive them (10,688 / 7,614 /
  9,025 Btu/kWh respectively), assuming 8,760 hr/yr operation and a natural
  gas heat content of 1,020 Btu/scf.

WHAT THE EIP REPORT DOES NOT GIVE US (and how we filled the gap)
-------------------------------------------------------------------
1. A CO2/CH4 species split. The report's CO2e figure is already a blended,
   GWP100-weighted number; it does not report physical CO2 and CH4 tons
   separately, which DataCenter-FAIR needs to track methane's short-lived
   forcing distinctly from CO2's long-lived forcing.
   FIX: We fetched the Argonne National Laboratory report the EIP report's
   own GHG factors are sourced from (Argonne ANL/ESD-20/6, "Update of
   Emission Factors of Greenhouse Gases and Criteria Air Pollutants... of
   the U.S. Electricity Generation Sector," Aug 2020, Table 4). It gives
   STACK (combustion-slip) methane factors separately: 0.030 g CH4/kWh for
   simple-cycle gas turbines, 0.009 g CH4/kWh for combined-cycle, and
   0.008 g CH4/kWh for internal combustion engines. Compared to the ~450-660
   g CO2/kWh these technologies emit, stack methane slip is under 0.01% of
   combustion CO2 by mass and contributes negligibly to the blended CO2e
   figure -- we compute it here for completeness but it is NOT the material
   methane term in this analysis.
2. Upstream (supply-chain) methane leakage. This IS the material methane
   term, and the one the paper's "repair methane leaks" policy lever
   actually targets -- gas lost from wells, gathering lines, and
   compressors upstream of the power plant, not methane slipping through
   the turbine unburned. The EIP report does not estimate this at all
   (its scope is stack emissions from the plant itself, per its Appendix B).
   FIX: We use Alvarez et al. (2018, Science 361:186-188, already cited in
   this project's references.bib as `alvarez2018methane`), which estimates
   supply-chain methane loss at 2.3% of gross U.S. natural gas production.
   We apply that rate to each facility's own gas throughput (which we
   derive from its reported capacity, technology-implied heat rate, and
   8,760 hr/yr operation, i.e. the EIP report's own Appendix B equation run
   in reverse) to get a facility-specific upstream CH4 tonnage. This is a
   real, literature-sourced number, not an invented one -- but it is a
   national average leak rate applied locally, which is a limitation we
   state plainly (see LIMITATIONS at the bottom).
3. Combustion technology per facility. Appendix C reports capacity and
   total CO2e but not which of the three technologies (simple-cycle,
   combined-cycle, ICE) each plant uses. The report states plants are
   assumed simple-cycle "unless they stated publicly" otherwise, but does
   not say which ones stated otherwise.
   FIX: for each facility we back-solve its implied CO2e/MWh intensity
   (reported CO2e / (capacity_MW * 8760)) and match it against the three
   published Table 5 factors within a 5% tolerance; if it matches
   combined-cycle or ICE we use that technology's heat rate, otherwise we
   default to simple-cycle (the report's own stated default and the most
   common technology in the fleet). This lets each facility get a
   technology-consistent gas-throughput estimate; it is an inference, not
   a fact from the report, and is flagged as such in the output CSV.
4. Four facilities (Vermaland La Osa AZ, Data City Energy Center TX,
   Mesones Data Center Campus TX, Tract Austin Power Plant TX) have no
   reported capacity. The EIP report itself excluded them from its
   emissions totals for the same reason. We exclude them here too and
   report the exclusion explicitly, rather than silently guessing a value.
5. Commissioning year. 15 of 74 facilities report "Unknown" for expected
   operating year. We assign these the median operating year of the 59
   facilities with a known year, and flag every assigned row so the ramp
   curve's uncertainty is visible rather than hidden inside a smooth curve.

UPDATE (public-record research pass): the technology and online-year gaps
above were 35 facilities' worth of inference. We tasked five research
agents with finding real public records (Oil & Gas Watch, state permit
databases, company announcements, trade press) for every one of them, plus
literature for the two assumption numbers flagged as needing sensitivity
checks (LDAR efficacy, uniform national leak rate). Results:

TECHNOLOGY CORRECTIONS CONFIRMED (overriding the earlier "defaulted"
simple-cycle guess with the real, sourced technology):
  - Nexus Data Center Hubbard Power Plant (TX): reciprocating ICE engines
    (30x MAN 18V51/60G), not simple-cycle. Source: TCEQ permit record.
  - Schahfer Generating Station (IN): combined-cycle (4 units per NIPSCO
    filing), not simple-cycle. Source: InkFreeNews, Oct 2025.
  - Adams Fork Data Center Energy Campus + Adams Fork Harless (WV, both):
    two-stroke reciprocating engines, not simple-cycle turbines. Source:
    company site, adamsfork.com.
  - BFC Power and Cheyenne Power Hub (WY): combined-cycle (2x Mitsubishi
    M501JAC), not simple-cycle. Source: Global Energy Monitor, Power Eng.
  - Kilby Power Plant (TX): a real mix of combined-cycle (987 MW) and
    simple-cycle (1,608 MW) per Global Energy Monitor -- modeled here as a
    capacity-weighted blended heat rate rather than forcing it into one
    bucket.
  - Project Jupiter West Microgrid (NM): NOT a combustion turbine at all --
    Oracle/BorderPlex switched this project to Bloom Energy fuel cells in
    April 2026. Fuel cells are electrochemical (no combustion), meaningfully
    more efficient than a turbine (higher electrical efficiency = less gas
    burned per MWh = less upstream throughput), and have no stack methane
    slip pathway the way a turbine does. Modeled with a fuel-cell heat rate
    (6,500 Btu/kWh, typical solid-oxide fuel cell LHV efficiency) and zero
    stack CH4.
  Confirmed correct as originally defaulted (no change): Fermi America
  Project Matador (simple-cycle, Siemens SGT-800), Bluegrass Power
  Generation Facility (simple-cycle), xAI Southaven (simple-cycle).
  Still unconfirmed after the research pass, left at the simple-cycle
  default: Goodnight, Longhorn (confirmed simple-cycle, no change), Poolside,
  Apollo, EdgeConneX New Albany North, PowerConneX I/II, Socrates North/
  South/the Younger (confirmed as mixed turbine+reciprocating fleets but too
  granular to split reliably; left at simple-cycle default with this flag),
  Eagle Mountain/Aquila, Miller Generating Station.

COMMISSIONING YEAR CORRECTIONS CONFIRMED:
  - Lone Star Project (TX): 2028 (Baxtel).
  - Portsmouth Powered Land Project (OH): 2028 for its first 2,000 MW phase
    (Inside Climate News) -- used here for the full facility as the best
    available single date.
  Still unconfirmed after the research pass, left at the median-fallback
  assignment: Schahfer, Fayetteville/Tarboro NetZero, EdgeConneX New Albany
  North, Fort Cherry, NextEra Exxon Southeast, Kilby, MMEX Pecos, Xenergy
  Monahans, Casper Campus.

STATUS CHANGE FOUND:
  - NRG Tolar Power Center (TX): NRG withdrew this air permit application
    on June 18, 2026 amid community opposition (KERA News) -- after the EIP
    report's April 30, 2026 data cutoff. EXCLUDED from the fleet below as
    no longer an active project, rather than left in as if still proceeding.

BASIN-SPECIFIC UPSTREAM LEAK RATES (replacing the single 2.3% national
average where a basin-specific, peer-reviewed rate exists and applies):
  - Permian Basin (TX/NM core producing counties): 9.4% of gross production
    (Chen et al. 2022, Environ. Sci. Technol., basin-wide aerial
    hyperspectral survey, DOI via pubs.acs.org/doi/10.1021/acs.est.1c06458).
  - Appalachian Basin (OH/PA/WV, Marcellus-Utica): 0.52% of gross
    production -- the lowest of any major US basin, per the 2025
    Appalachian Methane Initiative study (jpt.spe.org).
  - Uinta Basin, Utah: 6-8% is documented (Helmig et al. 2021, Sci.
    Reports), but NONE of this project's Utah facilities (Millard, Utah,
    Box Elder counties) sit in the Uinta Basin -- so this rate is NOT
    applied; those facilities keep the national average, which is the
    more defensible choice for a consumption-side plant not co-located
    with production.
  - Wyoming, Gulf Coast/Mississippi, and every other county not listed
    above: no credible basin-specific, production-normalized rate was
    found in this research pass. Kept at the 2.3% national average
    (Alvarez et al. 2018) rather than guessing a basin figure.

LDAR EFFICACY -- NOT REPLACED WITH A SINGLE NUMBER, ON PURPOSE: the
research pass could not confirm a single credible, isolated percentage for
"OGI-based LDAR reduces upstream methane leakage by X%." EPA's often-quoted
"~80% reduction" is for its ENTIRE 2023 methane rule package (LDAR +
flaring + pneumatics + more combined), not LDAR alone, and several
directly-relevant peer-reviewed studies (Fox et al. 2020; Vaughn et al.
2023; Ravikumar et al. 2023) were identified but their exact percentages
could not be extracted (paywalled). Rather than keep a single guessed 80%
figure or invent a number from a title-only citation, `run_path_scenarios.py`
now runs the LDAR and combined branches at four efficacy levels (50%, 65%,
80%, 95%) and reports whether the paper's crossover conclusion is robust
across that whole plausible range -- this is the correct fix for an
assumption that couldn't be pinned to one literature value.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Constants taken directly from the EIP report, Appendix B / Table 5
# ---------------------------------------------------------------
HEAT_RATE_BTU_PER_KWH = {
    "simple_cycle": 10688.0,
    "combined_cycle": 7614.0,
    "ice": 9025.0,
    "fuel_cell": 6500.0,   # typical solid-oxide fuel cell LHV efficiency; not from the EIP report
}
CO2E_FACTOR_TONS_PER_MWH = {   # Table 5, already GWP100-blended (fuel_cell has no Table 5 entry --
    "simple_cycle": 0.66,      # we still use the plant's own EIP-reported CO2e for its CO2 term
    "combined_cycle": 0.43,    # regardless of technology bucket, so this factor is only used for
    "ice": 0.49,                # the technology-matching step, not applied to fuel-cell plants)
}
GAS_HEAT_CONTENT_BTU_PER_SCF = 1020.0
HOURS_PER_YEAR = 8760.0

# ---------------------------------------------------------------
# Manual technology overrides confirmed by public-record research
# (see UPDATE note above). Keys are (state, county, plant_name).
# ---------------------------------------------------------------
TECH_OVERRIDES = {
    ("TX", "Hill", "Nexus Data Center Hubbard Power Plant"): "ice",
    ("IN", "Jasper", "Schahfer Generating Station"): "combined_cycle",
    ("WV", "Mingo", "Adams Fork Data Center Energy Campus"): "ice",
    ("WV", "Mingo", "Adams Fork Harless Data Center Energy Campus"): "ice",
    ("WY", "Laramie", "BFC Power and Cheyenne Power Hub"): "combined_cycle",
    ("NM", "Dona Ana", "Project Jupiter West Microgrid"): "fuel_cell",
}
# Kilby is a real mix (987 MW combined-cycle + 1,608 MW simple-cycle, GEM);
# handled as a capacity-weighted blended heat rate, not a single bucket.
KILBY_KEY = ("TX", "Reeves", "Kilby Power Plant")
KILBY_CC_MW, KILBY_SC_MW = 987.0, 1608.0

# Manual online-year overrides confirmed by public-record research.
YEAR_OVERRIDES = {
    ("TX", "Glasscock", "Lone Star Project"): 2028,
    ("OH", "Pike", "Portsmouth Powered Land Project"): 2028,
}

# Facilities confirmed no longer active (permit withdrawn after the EIP
# report's April 30, 2026 cutoff) -- excluded from the fleet entirely.
EXCLUDED_INACTIVE = {
    ("TX", "Hood", "NRG Tolar Power Center"),  # permit withdrawn June 18, 2026 (KERA News)
}

# ---------------------------------------------------------------
# Basin-specific upstream leak rates (see UPDATE note above). Counties are
# matched case-sensitively against the CSV's `county` column.
# ---------------------------------------------------------------
PERMIAN_LEAK_RATE = 0.094   # Chen et al. 2022, ES&T
APPALACHIAN_LEAK_RATE = 0.0052  # 2025 Appalachian Methane Initiative
NATIONAL_LEAK_RATE = 0.023  # Alvarez et al. 2018, Science (default/fallback)

PERMIAN_COUNTIES = {
    ("TX", "Reeves"), ("TX", "Pecos"), ("TX", "Ward"), ("TX", "Midland"),
    ("TX", "Ector"), ("TX", "Glasscock"), ("TX", "Winkler"), ("TX", "Loving"),
    ("TX", "Crane"), ("TX", "Upton"), ("TX", "Reagan"), ("TX", "Howard"),
    ("TX", "Martin"), ("TX", "Andrews"), ("TX", "Gaines"), ("TX", "Dawson"),
    ("NM", "Lea"), ("NM", "Dona Ana"),
}
APPALACHIAN_STATES = {"OH", "PA", "WV"}


def leak_rate_for(state, county):
    if (state, county) in PERMIAN_COUNTIES:
        return PERMIAN_LEAK_RATE, "Permian (Chen et al. 2022)"
    if state in APPALACHIAN_STATES:
        return APPALACHIAN_LEAK_RATE, "Appalachian (2025 AMI study)"
    return NATIONAL_LEAK_RATE, "national average (Alvarez et al. 2018)"

# ---------------------------------------------------------------
# Constants from the Argonne National Laboratory report the EIP report's
# own Table 5 is sourced from (ANL/ESD-20/6, Table 4) -- stack methane slip,
# g CH4 / kWh. Included for completeness; shown to be immaterial below.
# ---------------------------------------------------------------
STACK_CH4_G_PER_KWH = {
    "simple_cycle": 0.030,
    "combined_cycle": 0.009,
    "ice": 0.008,
    "fuel_cell": 0.0,   # electrochemical, no combustion-slip pathway
}

GAS_CH4_VOLUME_FRACTION = 0.95     # typical pipeline-quality natural gas
CH4_DENSITY_KG_PER_SCF = 0.01926   # CH4 at standard conditions
KG_TO_SHORT_TON = 1.0 / 907.185

CSV_PATH = "../data/eip_74_facilities.csv"


def classify_technology(state, county, plant_name, capacity_mw, ghg_co2e_tons_yr):
    key = (state, county, plant_name)
    if key in TECH_OVERRIDES:
        return TECH_OVERRIDES[key], "confirmed (public record)"
    if capacity_mw is None or np.isnan(capacity_mw) or ghg_co2e_tons_yr is None or np.isnan(ghg_co2e_tons_yr):
        return None, None
    implied = ghg_co2e_tons_yr / (capacity_mw * HOURS_PER_YEAR)
    for tech, factor in CO2E_FACTOR_TONS_PER_MWH.items():
        if abs(implied - factor) / factor <= 0.05:
            return tech, "matched"
    return "simple_cycle", "defaulted"


def kilby_blended_heat_rate():
    total = KILBY_CC_MW + KILBY_SC_MW
    return (KILBY_CC_MW * HEAT_RATE_BTU_PER_KWH["combined_cycle"]
            + KILBY_SC_MW * HEAT_RATE_BTU_PER_KWH["simple_cycle"]) / total


def parse_year(y, fallback):
    if y is None or str(y).strip().lower() == "unknown" or str(y).strip() == "":
        return fallback, True
    s = str(y)
    # take the first 4-digit year found
    import re
    m = re.search(r"(20\d{2})", s)
    if m:
        return int(m.group(1)), False
    return fallback, True


def build():
    df = pd.read_csv(CSV_PATH)
    n_total = len(df)
    has_capacity = df["capacity_mw"].notna()
    n_excluded_no_capacity = (~has_capacity).sum()
    df = df[has_capacity].copy()

    is_inactive = df.apply(lambda r: (r["state"], r["county"], r["plant_name"]) in EXCLUDED_INACTIVE, axis=1)
    n_excluded_inactive = is_inactive.sum()
    df = df[~is_inactive].copy()

    df["technology"], df["technology_source"] = zip(
        *df.apply(lambda r: classify_technology(r["state"], r["county"], r["plant_name"],
                                                  r["capacity_mw"], r["ghg_co2e_tons_yr"]), axis=1)
    )

    df["leak_rate"], df["leak_rate_source"] = zip(
        *df.apply(lambda r: leak_rate_for(r["state"], r["county"]), axis=1)
    )

    # gas throughput (scf/yr) from EIP's own Appendix B equation, inverted:
    # Capacity(MW) x 1000 x HeatRate(Btu/kWh) x 8760 hr/yr / heat_content(Btu/scf)
    # Kilby gets its own capacity-weighted blended heat rate (real CC+SC mix).
    def throughput_scf(row):
        key = (row["state"], row["county"], row["plant_name"])
        hr = kilby_blended_heat_rate() if key == KILBY_KEY else HEAT_RATE_BTU_PER_KWH[row["technology"]]
        return row["capacity_mw"] * 1000.0 * hr * HOURS_PER_YEAR / GAS_HEAT_CONTENT_BTU_PER_SCF

    df["gas_throughput_scf_yr"] = df.apply(throughput_scf, axis=1)

    # stack (combustion-slip) CH4, tons/yr -- immaterial, kept for completeness
    def stack_ch4_tons(row):
        g_per_kwh = STACK_CH4_G_PER_KWH[row["technology"]]
        kwh_yr = row["capacity_mw"] * 1000.0 * HOURS_PER_YEAR
        return g_per_kwh * kwh_yr / 1000.0 * (1.0 / 907.185)  # g -> kg -> short ton

    df["stack_ch4_tons_yr"] = df.apply(stack_ch4_tons, axis=1)

    # upstream leaked CH4, tons/yr -- the material methane term, now using
    # a basin-specific leak rate where one applies (see leak_rate_for above)
    df["upstream_ch4_tons_yr"] = (
        df["gas_throughput_scf_yr"] * df["leak_rate"] * GAS_CH4_VOLUME_FRACTION
        * CH4_DENSITY_KG_PER_SCF * KG_TO_SHORT_TON
    )

    df["total_ch4_tons_yr"] = df["stack_ch4_tons_yr"] + df["upstream_ch4_tons_yr"]

    # CO2: use the EIP-reported CO2e directly as the CO2 term (stack CH4's
    # GWP100-weighted contribution to that figure is <0.1%, confirmed above,
    # so this is accurate to well within the report's own uncertainty).
    df["co2_tons_yr"] = df["ghg_co2e_tons_yr"]

    known_years = df["expected_operating_year"]
    known_year_vals = []
    import re
    for y in known_years:
        if str(y).strip().lower() != "unknown":
            m = re.search(r"(20\d{2})", str(y))
            if m:
                known_year_vals.append(int(m.group(1)))
    fallback_year = int(np.median(known_year_vals))

    def resolve_year(row):
        key = (row["state"], row["county"], row["plant_name"])
        if key in YEAR_OVERRIDES:
            return YEAR_OVERRIDES[key], False
        return parse_year(row["expected_operating_year"], fallback_year)

    resolved = df.apply(resolve_year, axis=1)
    df["online_year"] = [r[0] for r in resolved]
    df["online_year_assumed"] = [r[1] for r in resolved]

    df.to_csv("../data/eip_74_facilities_processed.csv", index=False)

    print(f"Facilities in Appendix C: {n_total}")
    print(f"Excluded (no reported capacity): {n_excluded_no_capacity}")
    print(f"Excluded (permit withdrawn / inactive): {n_excluded_inactive}")
    print(f"Leak rate mix: {df['leak_rate_source'].value_counts().to_dict()}")
    print(f"Facilities with confirmed (public-record) technology override: "
          f"{(df['technology_source']=='confirmed (public record)').sum()}")
    print(f"Facilities with computed emissions: {len(df)}")
    print(f"Total known capacity: {df['capacity_mw'].sum():,.0f} MW "
          f"(report's stated fleet total incl. unknown-capacity plants: 143,000 MW)")
    print(f"Technology mix (by count): {df['technology'].value_counts().to_dict()}")
    print(f"Technology inference: {df['technology_source'].value_counts().to_dict()}")
    print(f"Sum of reported CO2e (tons/yr, full fleet at buildout): {df['co2_tons_yr'].sum():,.0f} "
          f"(report's stated fleet total incl. unknown-capacity plants: 662,000,000)")
    print(f"Computed upstream CH4 (tons/yr, full fleet at buildout): {df['upstream_ch4_tons_yr'].sum():,.0f}")
    print(f"Computed stack CH4 (tons/yr, full fleet at buildout): {df['stack_ch4_tons_yr'].sum():,.1f} "
          f"(negligible next to upstream term, as expected)")
    print(f"Facilities with assumed (median-fallback) online year: {df['online_year_assumed'].sum()} of {len(df)}")
    return df


if __name__ == "__main__":
    build()
