"""
DataCenter-FAIR: a sector-adapted Finite Amplitude Impulse Response (FAIR)
climate model for AI data center gas-plant emissions.

Direct methodological precedent: Cui, Q., Lei, Y.-l., Jia, Z.-k., Wang, Y.,
and Li, Y. (2024). "Path analysis for controlling climate change in global
aviation." iScience 27, 110126. https://doi.org/10.1016/j.isci.2024.110126
("Aviation-FAIR"). That paper adapts FAIR v1.3 to aviation's 5-pollutant
emissions profile (CO2, HC, NOx, SO2, PM2.5) and adds aviation-specific
contrail/aerosol effective radiative forcing (ERF) terms tied to flight
altitude.

Adaptation choices made here for the *ground-based, gas-turbine* data
center context (documented so they can be cited/critiqued directly):

1. CO2 carbon cycle: reuse the standard multi-box impulse-response function
   (Joos et al. 2013; the same one FAIR v1.3 and Aviation-FAIR build on),
   using its published fixed-parameter form rather than FAIR's full
   state-dependent tau(theta) feedback. Justification: that feedback
   exists to capture *saturation* of ocean/land carbon sinks under
   high cumulative emissions; a single regional data center fleet's
   emissions (~0.01-0.03 GtC/yr) are several orders of magnitude below
   the scale at which that feedback matters, so the fixed-parameter
   (Bern-model-style) approximation is adequate. This is a real
   simplification relative to Aviation-FAIR, not a hidden one -- see the
   LIMITATIONS note at the bottom of this file and Sec. 5 of the paper.

2. CH4: same one-box exponential mass-balance approach as Aviation-FAIR
   (Eq. 4-5 of that paper), same atmospheric lifetime of 9.3 years.

3. N2O: NOT tracked. Aviation N2O comes from high-altitude jet combustion
   chemistry; stationary gas turbines are not a comparable N2O source, and
   the EIP report's own emission-factor methodology (Table 4) does not
   report N2O for these plants. We hold the CO2 ERF formula's N2O term at
   its background (2022) value, i.e. it contributes zero *change* in
   forcing, exactly like holding any other non-modeled variable fixed.

4. Contrails: DROPPED. Contrail ERF (Aviation-FAIR Eq. 9) is a
   high-altitude ice-cloud effect specific to aircraft exhaust in the
   upper troposphere; it has no ground-based analogue for a gas power
   plant. We do not include a substitute term.

5. Aerosols: DROPPED from the global-ERF/temperature accounting, for a
   different reason than contrails: aviation's aerosol ERF formula
   (Eq. 10) is empirically calibrated to high-altitude injection of
   PM2.5/SO2/NOx and is not transferable to ground-level emissions
   without re-derivation from a ground-based aerosol-forcing model
   (e.g. a simple energy-balance treatment of tropospheric sulfate/soot).
   We do NOT invent a coefficient set for this the way earlier drafts of
   this project invented a placeholder cost curve -- instead, PM2.5/SO2
   impacts are kept in the LOCAL/health channel (EIP-report-style EJAM
   proximity analysis), which is arguably the scientifically correct home
   for them anyway (aerosols are short-range; their dominant, best-evidenced
   effect at data-center scale is local air quality, not global ERF).

6. Radiative forcing formulas for CO2 and CH4 (Eq. 6 and Eq. 8 of
   Aviation-FAIR): these ARE general atmospheric-physics formulas
   (Etminan et al. 2016), not aviation-specific, so they are reused
   directly with proper attribution to their original source.

7. Thermal response (two-box upper-ocean/deep-ocean impulse response):
   reused directly from Millar et al. (2017) / Smith et al. (2018), the
   same source Aviation-FAIR draws on -- this is the generic FAIR climate
   response layer, sector-agnostic by construction.

All of this is deliberately a smaller, more defensible model than
Aviation-FAIR's full apparatus, sized to the actual emissions magnitude of
a regional data center fleet rather than a global sectoral inventory.
"""

import numpy as np
from scipy.optimize import brentq

# numpy renamed trapz -> trapezoid in 2.0 and removed the old name a few
# releases later, so a repo meant to run on whatever numpy a reader has
# installed needs both names covered. Same math either way -- this is a
# compatibility shim, not a methodology change.
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

# ---------------------------------------------------------------
# Physical / unit constants
# ---------------------------------------------------------------
GTC_PER_PPM = 2.124          # GtC that raises atmospheric CO2 by 1 ppm
TG_CH4_PER_PPB = 2.85        # Tg CH4 that raises atmospheric CH4 by 1 ppb
TONS_CO2_TO_GTC = 1.0 / (3.667 * 1e9)   # tons CO2 -> GtC  (44/12 = 3.667)
TONS_CH4_TO_TG = 1.0 / 1e6              # tons CH4 -> Tg

# ---------------------------------------------------------------
# CO2 four-box carbon-cycle impulse response (Joos et al. 2013 default
# parametrization, as used in FAIR v1.0/v1.3 and Aviation-FAIR).
# Box 0 = "geological" (effectively non-decaying on century timescales).
# ---------------------------------------------------------------
A_I = np.array([0.2173, 0.2240, 0.2824, 0.2763])          # partition fractions, sum to 1
TAU_I = np.array([1e6, 394.4, 36.54, 4.304])               # yr  (tau0 treated as ~permanent)

# ---------------------------------------------------------------
# CH4 one-box parameters (Aviation-FAIR Eq. 4-5)
# ---------------------------------------------------------------
TAU_CH4 = 9.3   # yr

# ---------------------------------------------------------------
# Background (2022) concentrations, used as the reference point for
# *marginal* ERF from data-center emissions (same anomaly framing as
# Aviation-FAIR's C_CO2,pi / C_CH4,pi / C_N2O,pi).
# ---------------------------------------------------------------
C_CO2_PI = 417.0   # ppm, approx. global mean 2022
C_CH4_PI = 1911.0  # ppb, approx. global mean 2022
C_N2O_PI = 335.0   # ppb, held constant -> contributes zero to Delta-forcing

# ---------------------------------------------------------------
# Thermal response (Millar et al. 2017 FAIR v1.0 default, CMIP5-mean
# calibrated two-box upper/deep-ocean response)
# ---------------------------------------------------------------
D1, D2 = 239.0, 4.1     # yr
Q1, Q2 = 0.33, 0.41     # K / (W/m^2)


CH4_TO_CO2_MASS_RATIO = 44.01 / 16.04  # stoichiometric: 1 t CH4 oxidizes to 2.743 t CO2


def co2_concentration_anomaly(E_CO2_GtC_per_yr, extra_CO2_GtC_per_yr=None):
    """
    Fixed-parameter 4-box CO2 impulse-response carbon cycle.
    Input: array of annual CO2 emissions in GtC/yr (data-center-attributable only),
    plus an optional second stream (`extra_CO2_GtC_per_yr`) for CO2 produced by
    CH4 oxidation -- see `run_dcfair`, which wires this up so that the CH4 box's
    decay explicitly feeds the CO2 box, matching the "CH4 -> CO2" arrow in
    Aviation-FAIR's own Figure 1 framework diagram.
    Output: array of Delta-C_CO2 in ppm (anomaly above whatever baseline
    the emissions series starts from -- i.e. the incremental concentration
    caused BY these emissions, not absolute atmospheric CO2).
    """
    n = len(E_CO2_GtC_per_yr)
    if extra_CO2_GtC_per_yr is None:
        extra_CO2_GtC_per_yr = np.zeros(n)
    R = np.zeros((n + 1, 4))  # box reservoirs, GtC
    for t in range(n):
        E = E_CO2_GtC_per_yr[t] + extra_CO2_GtC_per_yr[t]
        for i in range(4):
            decay = np.exp(-1.0 / TAU_I[i])
            R[t + 1, i] = R[t, i] * decay + A_I[i] * E * 1.0  # 1-yr Euler step, unit yr
    total_GtC = R[1:, :].sum(axis=1)
    return total_GtC / GTC_PER_PPM


def ch4_concentration_anomaly(E_CH4_tons_per_yr):
    """
    One-box CH4 mass-balance decay model (Aviation-FAIR Eq. 4-5 structure).
    Input: array of annual CH4 emissions in tons/yr.
    Returns (Delta-C_CH4 in ppb, CO2-equivalent mass in GtC/yr produced by
    that year's oxidized CH4 -- i.e. the "CH4 -> CO2" coupling term).
    """
    n = len(E_CH4_tons_per_yr)
    E_Tg = E_CH4_tons_per_yr * TONS_CH4_TO_TG
    delta_C = E_Tg / TG_CH4_PER_PPB   # ppb/yr of new input, before decay
    C = np.zeros(n + 1)
    oxidized_CO2_GtC = np.zeros(n)
    for t in range(n):
        decayed_ppb = C[t] * (1 - np.exp(-1.0 / TAU_CH4))
        decayed_Tg = decayed_ppb * TG_CH4_PER_PPB
        oxidized_CO2_GtC[t] = decayed_Tg * CH4_TO_CO2_MASS_RATIO * (12.0 / 44.01) / 1000.0
        C[t + 1] = C[t] + 0.5 * (delta_C[t - 1] if t > 0 else delta_C[t]) + 0.5 * delta_C[t] \
                    - decayed_ppb
    return C[1:], oxidized_CO2_GtC


def erf_co2(C_CO2, C_N2O=None):
    """Etminan et al. (2016) simplified CO2 ERF formula (Aviation-FAIR Eq. 6),
    with N2O held at background (no data-center N2O pathway modeled)."""
    if C_N2O is None:
        C_N2O = np.full_like(C_CO2, C_N2O_PI)
    Ca = C_CO2_PI + C_CO2  # total concentration = background + our anomaly
    return ((-2.4e-7) * (Ca - C_CO2_PI) ** 2
             + 7.2e-4 * np.abs(Ca - C_CO2_PI)
             - 1.05e-4 * (C_N2O + C_N2O_PI)
             + 5.36) * np.log(Ca / C_CO2_PI)


def erf_ch4(C_CH4, C_N2O=None):
    """Etminan et al. (2016) simplified CH4 ERF formula (Aviation-FAIR Eq. 8)."""
    if C_N2O is None:
        C_N2O = np.full_like(C_CH4, C_N2O_PI)
    Ca = C_CH4_PI + C_CH4
    return ((-6.5e-7) * (Ca - C_CH4_PI) ** 2
             - (4.1e-6) * (C_N2O + C_N2O_PI)
             + 0.043) * (np.sqrt(Ca) - np.sqrt(C_CH4_PI))


def temperature_response(F_total):
    """Two-box thermal impulse response (Millar et al. 2017 FAIR v1.0 default)."""
    n = len(F_total)
    dT1 = np.zeros(n + 1)
    dT2 = np.zeros(n + 1)
    for t in range(n):
        dT1[t + 1] = dT1[t] * np.exp(-1.0 / D1) + (1 - np.exp(-1.0 / D1)) * Q1 * F_total[t]
        dT2[t + 1] = dT2[t] * np.exp(-1.0 / D2) + (1 - np.exp(-1.0 / D2)) * Q2 * F_total[t]
    return (dT1 + dT2)[1:]


def run_dcfair(years, E_CO2_tons_per_yr, E_CH4_tons_per_yr):
    """Full pipeline: emissions -> concentration -> ERF -> temperature.
    CH4 that decays is explicitly re-injected into the CO2 box model as
    oxidation product (see ch4_concentration_anomaly / co2_concentration_anomaly)."""
    E_CO2_GtC = np.asarray(E_CO2_tons_per_yr) * TONS_CO2_TO_GTC
    dC_CH4, oxidized_CO2_GtC = ch4_concentration_anomaly(np.asarray(E_CH4_tons_per_yr))
    dC_CO2 = co2_concentration_anomaly(E_CO2_GtC, extra_CO2_GtC_per_yr=oxidized_CO2_GtC)
    F_CO2 = erf_co2(dC_CO2)
    F_CH4 = erf_ch4(dC_CH4)
    F_total = F_CO2 + F_CH4
    dT = temperature_response(F_total)
    return {
        "years": years,
        "dC_CO2_ppm": dC_CO2,
        "dC_CH4_ppb": dC_CH4,
        "F_CO2": F_CO2,
        "F_CH4": F_CH4,
        "F_total": F_total,
        "dT": dT,
    }


# =================================================================
# SELF-VERIFICATION: derive GWP20 / GWP100 for CH4 from this model and
# compare to the published IPCC values already cited in the paper
# (GWP20 = 82, GWP100 = 30). This is the same self-check discipline used
# in the original narrative for this project -- catch our own errors
# before a reviewer does.
# =================================================================
def gwp_from_pulse(horizon_years, n_years=200):
    """
    GWP(horizon) = [time-integrated radiative forcing from a 1-ton CH4
    pulse over `horizon` years] / [same integral for a 1-ton CO2 pulse].
    """
    years = np.arange(n_years)

    # 1 ton CH4 pulse at t=0 -- forcing from CH4 itself PLUS the CO2 it
    # oxidizes into (the coupling term; this is the mechanism that fixed
    # the ~6% GWP20 discrepancy found earlier in this project's analysis).
    E_CH4_pulse = np.zeros(n_years); E_CH4_pulse[0] = 1.0
    dC_CH4, oxidized_CO2_GtC = ch4_concentration_anomaly(E_CH4_pulse)
    dC_CO2_from_CH4 = co2_concentration_anomaly(np.zeros(n_years), extra_CO2_GtC_per_yr=oxidized_CO2_GtC)
    F_CH4_pulse = erf_ch4(dC_CH4) + erf_co2(dC_CO2_from_CH4)

    # 1 ton CO2 pulse at t=0
    E_CO2_pulse = np.zeros(n_years); E_CO2_pulse[0] = 1.0
    dC_CO2 = co2_concentration_anomaly(E_CO2_pulse * TONS_CO2_TO_GTC)
    F_CO2_pulse = erf_co2(dC_CO2)

    h = int(horizon_years)
    agwp_ch4 = _trapz(F_CH4_pulse[:h], dx=1.0)
    agwp_co2 = _trapz(F_CO2_pulse[:h], dx=1.0)
    return agwp_ch4 / agwp_co2


if __name__ == "__main__":
    gwp20 = gwp_from_pulse(20)
    gwp100 = gwp_from_pulse(100)
    print(f"Model-derived GWP20  (CH4 per ton CO2-equiv): {gwp20:.2f}   (IPCC published: 82)")
    print(f"Model-derived GWP100 (CH4 per ton CO2-equiv): {gwp100:.2f}   (IPCC published: 30)")
