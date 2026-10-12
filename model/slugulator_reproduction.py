"""
slugulator_reproduction.py

WHAT THIS FILE IS
------------------
A line-by-line Python port of the University of Chicago "Slugulator" climate
model, originally written in CoffeeScript for a browser tool
(https://climatemodels.uchicago.edu/slugulator/slugulator.html), source file
"slugulator 4.coffee".

Every constant and every formula below is copied directly from that source
file, with the exact line number it came from noted in a comment. NOTHING in
this script is fit, tuned, or hand-picked to match a target curve. This
script computes the model's "Concentrations" plot and "Radiative Forcings
(W/m2)" plot for the tool's DEFAULT inputs:

    CO2 spike size   = 1 Gton C   (slugCO2, coffee line 7)
    CH4 spike size   = 1 Gton C   (slugCH4, coffee line 8)
    Climate sensitivity = 3 deg C per CO2 doubling (coffee line 9)
    CH4 forcing efficacy = 1.4    (coffee line 10)
    Plot time range  = 100 years  (coffee line 23, selectors.plotTime.selected)

WHY THIS MATTERS FOR THE PAPER
-------------------------------
This script is a *verification* reference, not a component of the paper's
own DataCenter-FAIR model. Its only job is to prove that the Slugulator
citation is being described honestly: the model's own physics -- not any
hand-picked ratio -- is what produces methane's sharp, short-lived forcing
spike versus CO2's small, slowly-building plateau.

Running this script from scratch (no cached numbers, no shortcuts) gives:
    peak CH4 radiative forcing (0-100 yr window)  ~ 0.42  W/m^2
    CO2 radiative forcing at year 100             ~ 0.02  W/m^2
which independently reproduces Figure 1 in the paper. That match falls
straight out of the real equations below -- there is no
`co2_forcing(t) = 0.047 * methane_forcing(t)` shortcut anywhere in this
file, and there should never be one: computing CO2's forcing this way from
a fixed ratio to methane would be circular (it assumes the answer to get
the answer) and would silently break the moment slug sizes, climate
sensitivity, or the CH4 efficacy input are changed. The formulas below
instead compute CO2 and CH4 forcing completely independently, exactly as
the original tool does, and the ~4-5% ratio falls out as a *result*, not an
input.

HOW TO RUN
----------
    pip install numpy matplotlib
    python3 slugulator_reproduction.py

This will print the computed peak/plateau numbers to the terminal and save
two PNG figures in the same folder:
    slugulator_concentrations.png
    slugulator_radiative_forcing.png
"""

import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# SECTION 1 -- MODEL CONSTANTS
# Every value here is copied verbatim from Application.runModel() in
# slugulator_4.coffee. The comment after each line gives the exact
# source line number so this can be checked against the original file.
# =====================================================================

GTON_PER_PPM = 2.0          # Gt of Carbon equivalent to 1 ppm of atmospheric
                             # concentration.                          (coffee line 92)

OPACITY_CO2 = 4.0           # Radiative forcing, in W/m^2, produced by one
                             # doubling of atmospheric CO2. This is the
                             # standard "4 W/m^2 per doubling" logarithmic
                             # CO2 forcing constant.                    (coffee line 93)

CLIMATE_SENSITIVITY = 3.0   # Equilibrium warming (deg C) for a doubling of
                             # CO2. Default value in the tool's input box.
                             # (Only used for the temperature panels, which
                             # this script does not reproduce -- we only
                             # need Concentrations and Radiative Forcing.)
                             #                                          (coffee line 9)

EFFICACY_CH4 = 1.4          # Multiplier on methane's raw radiative forcing
                             # to account for atmospheric chemical
                             # feedbacks (e.g. water vapor) that amplify
                             # CH4's direct forcing. Default input value.
                             #                                          (coffee line 10)

INITIAL_ATM_CH4 = 1.6       # ppm. Pre-spike atmospheric methane
                             # concentration used by the tool (already
                             # includes anthropogenic background).      (coffee line 96)

AGE_EXPONENT = 0.26         # Exponent governing how methane's own
                             # atmospheric lifetime lengthens as
                             # atmospheric CH4 concentration rises
                             # (a simplified OH-feedback effect).        (coffee line 97)

LIFETIME0 = 7.5              # years. Methane's atmospheric e-folding
                             # lifetime at the pre-anthropogenic baseline
                             # concentration.                           (coffee line 98)

INITIAL_ATM_CO2 = 280.0      # ppm. Pre-industrial atmospheric CO2
                             # concentration used as the reference point
                             # for the logarithmic forcing formula.     (coffee line 100)

DRAWDOWN_TIME = [300.0, 5000.0, 400000.0]   # years. Three e-folding
                             # timescales of a simplified multi-box
                             # ocean/land carbon-cycle drawdown of a CO2
                             # pulse (fast / medium / very slow uptake).
                             #                                          (coffee line 101)

DRAWDOWN_FRAC = [0.75, 0.15, 0.10]   # Fraction of the initial CO2 pulse
                             # that decays with each of the three
                             # timescales above (these three fractions
                             # sum to 1.0).                             (coffee line 102)

# ---- Default "slug" (pulse) sizes, exactly as pre-filled in the tool ----
SLUG_CO2_GTC = 1.0          # Gton Carbon added as CO2 at year 0.       (coffee line 7)
SLUG_CH4_GTC = 1.0          # Gton Carbon added as CH4 at year 0.       (coffee line 8)

# ---- Plot window, matching the tool's default "Show 100 years" setting ----
PLOT_TIME_YEARS = 100       # coffee line 23: selectors.plotTime.selected = [100]


# =====================================================================
# SECTION 2 -- BUILD THE TIME GRID
# The original tool uses a growing time step (0.5 yr steps that multiply
# by 1.25 each iteration, capped at 5 yr) so that a single run can cover
# up to a million years without needing millions of tiny steps. We
# reproduce that exact stepping rule (coffee lines 114-127) rather than
# substituting an arbitrary evenly-spaced grid, so the time points this
# script evaluates are identical in construction to the original tool's.
# =====================================================================

def build_year_grid():
    """Reproduce coffee lines 114-127 exactly."""
    year = 1.0
    years = [0.0, year]
    time_step = 0.5
    time_step_max = 5.0
    n_steps = int(20 + 1_000_000 / time_step_max)   # coffee line 118

    for _ in range(n_steps):
        year += time_step
        time_step *= 1.25
        if time_step > time_step_max:
            time_step = time_step_max
        if year < 1e6:
            years.append(year)

    # coffee line 126: also force-include the plot-time-selector milestone
    # years, so the tool always has an exact data point at each preset
    # plot range (10, 25, 50, 100, 500, ... years).
    milestone_years = [10, 25, 50, 100, 500, 1000, 10000, 100000, 1000000]
    years = years + [float(y) for y in milestone_years]

    years = sorted(set(years))   # coffee line 127 sorts; de-duplicating
                                  # here is a no-op for the physics since
                                  # repeated identical year values would
                                  # just recompute the same numbers twice.
    return np.array(years)


# =====================================================================
# SECTION 3 -- RUN THE MODEL
# This is a direct transcription of the main loop, coffee lines 129-188.
# One important detail preserved on purpose: the methane "lifetime"
# feedback for the CURRENT time step uses the atmospheric CH4
# concentration computed on the PREVIOUS time step (this is exactly how
# the coffee variable `atmCH4` carries its value across loop iterations
# before being overwritten). This lag is part of the original model's
# numerics, not a bug -- we reproduce it faithfully rather than "fixing"
# it, since the goal here is an exact replication, not a reinterpretation.
# =====================================================================

def run_slugulator(slug_co2_gtc=SLUG_CO2_GTC, slug_ch4_gtc=SLUG_CH4_GTC,
                    efficacy_ch4=EFFICACY_CH4):
    years = build_year_grid()

    # coffee line 104: atmCH4 is seeded (before the main loop starts)
    # using the FULL post-spike concentration. This seed value is what
    # the very first loop iteration uses to compute the initial lifetime
    # feedback.
    atm_ch4 = INITIAL_ATM_CH4 + slug_ch4_gtc / GTON_PER_PPM

    atm_co2_arr = np.zeros_like(years)
    atm_ch4_arr = np.zeros_like(years)
    rf_co2_arr = np.zeros_like(years)
    rf_ch4_arr = np.zeros_like(years)

    for i, year in enumerate(years):
        # ---- methane concentration + methane radiative forcing ----
        # coffee lines 131-140
        lifetime = LIFETIME0 * (atm_ch4 / INITIAL_ATM_CH4) ** AGE_EXPONENT
        atm_ch4 = INITIAL_ATM_CH4
        slug_co2_from_ch4 = 0.0
        if year > 0:
            # Remaining atmospheric CH4 from the spike, exponential decay
            # with the (concentration-dependent) lifetime computed above.
            atm_ch4 += slug_ch4_gtc / GTON_PER_PPM * np.exp(-year / lifetime)
            # The CH4 that HAS decayed is chemically oxidized to CO2, so
            # it gets added to the CO2 budget below.
            slug_co2_from_ch4 = slug_ch4_gtc / GTON_PER_PPM * (1 - np.exp(-year / lifetime))

        # Methane radiative forcing formula (coffee lines 137-140): a
        # quadratic-in-log fit, scaled by the efficacy factor.
        rf_ch4 = (-1.0569 * (np.log(INITIAL_ATM_CH4) - np.log(atm_ch4))
                  + -0.16019 * (np.log(INITIAL_ATM_CH4) ** 2 - np.log(atm_ch4) ** 2))
        rf_ch4 *= efficacy_ch4

        # ---- CO2 concentration + CO2 radiative forcing ----
        # coffee lines 142-151
        slug_co2 = slug_co2_gtc / GTON_PER_PPM + slug_co2_from_ch4
        remaining_co2_fraction = 0.0
        for t_decay, frac in zip(DRAWDOWN_TIME, DRAWDOWN_FRAC):
            remaining_co2_fraction += frac * np.exp(-year / t_decay)

        atm_co2 = INITIAL_ATM_CO2
        if year > 0:
            atm_co2 += slug_co2 * remaining_co2_fraction

        # Standard logarithmic CO2 forcing: 4 W/m^2 per doubling.
        rf_co2 = OPACITY_CO2 * np.log(atm_co2 / INITIAL_ATM_CO2) / np.log(2)

        atm_co2_arr[i] = atm_co2
        atm_ch4_arr[i] = atm_ch4
        rf_co2_arr[i] = rf_co2
        rf_ch4_arr[i] = rf_ch4

    return years, atm_co2_arr, atm_ch4_arr, rf_co2_arr, rf_ch4_arr


# =====================================================================
# SECTION 4 -- RUN, PRINT VERIFICATION NUMBERS, PLOT
# =====================================================================

if __name__ == "__main__":
    years, atm_co2, atm_ch4, rf_co2, rf_ch4 = run_slugulator()

    # Restrict to the plot window (default: 0-100 years), matching the
    # tool's default "Show 100 years" selector (coffee line 23).
    window = years <= PLOT_TIME_YEARS
    yrs_w = years[window]
    co2_conc_w = atm_co2[window]
    ch4_conc_w = atm_ch4[window]
    rf_co2_w = rf_co2[window]
    rf_ch4_w = rf_ch4[window]

    # ---- verification printout: no hand-picked numbers, just what the
    # ---- equations above actually produce ----
    peak_idx = np.argmax(rf_ch4_w)
    print("=" * 70)
    print("VERIFICATION (computed directly from the equations above,")
    print("no fitting or hand-picked constants):")
    print(f"  Peak CH4 radiative forcing in 0-{PLOT_TIME_YEARS} yr window: "
          f"{rf_ch4_w[peak_idx]:.4f} W/m^2, at year {yrs_w[peak_idx]:.2f}")
    print(f"  CO2 radiative forcing at year {PLOT_TIME_YEARS}: "
          f"{rf_co2_w[-1]:.4f} W/m^2")
    ratio = rf_co2_w[-1] / rf_ch4_w[peak_idx] * 100
    print(f"  Ratio (CO2 at year {PLOT_TIME_YEARS}) / (peak CH4): {ratio:.2f}%")
    print("  This ratio is a RESULT of the physics above, not an input:")
    print("  nowhere in this script is CO2 forcing defined as a fixed")
    print("  percentage of methane forcing.")
    print("=" * 70)

    # ---- Figure 1: Concentrations (matches the tool's "Concentrations"
    # ---- plot choice, dual y-axis: ppm CO2 left, ppm CH4 right, exactly
    # ---- as the original ComboChart does) ----
    fig, ax1 = plt.subplots(figsize=(7, 5))
    color_co2 = "#4C72B0"
    color_ch4 = "#DD8452"

    ax1.set_xlabel("Years")
    ax1.set_ylabel("Atmospheric CO$_2$ (ppm)", color=color_co2)
    l1, = ax1.plot(yrs_w, co2_conc_w, color=color_co2, linewidth=2, label="pCO$_2$")
    ax1.tick_params(axis="y", labelcolor=color_co2)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Atmospheric CH$_4$ (ppm)", color=color_ch4)
    l2, = ax2.plot(yrs_w, ch4_conc_w, color=color_ch4, linewidth=2, label="pCH$_4$")
    ax2.tick_params(axis="y", labelcolor=color_ch4)

    ax1.set_title("Slugulator reproduction: Concentrations\n"
                   "(1 GtC CO$_2$ + 1 GtC CH$_4$ pulse at year 0)")
    ax1.legend(handles=[l1, l2], loc="upper right")
    fig.tight_layout()
    fig.savefig("slugulator_concentrations.png", dpi=150)
    print("\nSaved slugulator_concentrations.png")

    # ---- Figure 2: Radiative Forcings (both series on one axis since
    # ---- both are already in the same units) ----
    fig2, ax = plt.subplots(figsize=(7, 5))
    ax.plot(yrs_w, rf_co2_w, color=color_co2, linewidth=2, label="Rad. Forc. CO$_2$")
    ax.plot(yrs_w, rf_ch4_w, color=color_ch4, linewidth=2, label="Rad. Forc. CH$_4$")
    ax.set_xlabel("Years")
    ax.set_ylabel("Radiative forcing (W/m$^2$)")
    ax.set_title("Slugulator reproduction: Radiative Forcing\n"
                  "(1 GtC CO$_2$ + 1 GtC CH$_4$ pulse at year 0)")
    ax.legend(loc="upper right")
    ax.axhline(0, color="gray", linewidth=0.5)
    fig2.tight_layout()
    fig2.savefig("slugulator_radiative_forcing.png", dpi=150)
    print("Saved slugulator_radiative_forcing.png")
