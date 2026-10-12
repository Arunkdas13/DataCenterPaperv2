"""
figure2_pipeline_diagram.py

Figure 2 in the paper -- the DataCenter-FAIR model pipeline. This is a
structural diagram, not a data plot, so there's nothing to "run" against
real numbers the way the other figures are -- it's a labeled box diagram
showing the four stages the climate model moves through (Fleet Emissions ->
Atmospheric Concentration -> Radiative Forcing -> Temperature Response),
each arrow citing the real source for that modeling step:

  Joos et al. 2013      -- carbon-cycle drawdown (the multi-box CO2 decay
                            used in dcfair.py and in slugulator_reproduction.py)
  Etminan et al. 2016   -- CO2 / CH4 radiative forcing formulas
  Millar et al. 2017 /
  Smith et al. 2018     -- FAIR two-box temperature-response layer

adapted from Cui et al. 2024's Aviation-FAIR framework (the direct
methodological precedent this project's DataCenter-FAIR model is built on).

Run: python3 figure2_pipeline_diagram.py
Saves: figure2_pipeline_diagram.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(11, 3.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2)
ax.axis('off')

boxes = [
    ("Fleet\nEmissions", 0.3),
    ("Atmospheric\nConcentration", 2.9),
    ("Radiative\nForcing", 5.5),
    ("Temperature\nResponse", 8.1),
]
citations = [
    "Joos et al. 2013\n(carbon-cycle\ndrawdown)",
    "Etminan et al. 2016\n(CO2/CH4 forcing\nformulas)",
    "Millar et al. 2017 /\nSmith et al. 2018\n(FAIR temp. response)",
]

box_w, box_h = 1.6, 1.0
for label, x in boxes:
    rect = mpatches.FancyBboxPatch((x, 0.5), box_w, box_h,
                                    boxstyle="round,pad=0.05,rounding_size=0.08",
                                    linewidth=1.5, edgecolor="#2b4a6b", facecolor="#dbe7f3")
    ax.add_patch(rect)
    ax.text(x + box_w / 2, 0.5 + box_h / 2, label, ha="center", va="center", fontsize=10, fontweight="bold")

for i, (label, x) in enumerate(boxes[:-1]):
    x0 = x + box_w
    x1 = boxes[i + 1][1]
    ax.annotate("", xy=(x1 - 0.05, 1.0), xytext=(x0 + 0.05, 1.0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#2b4a6b"))
    ax.text((x0 + x1) / 2, 1.55, citations[i], ha="center", va="bottom", fontsize=7.5, color="#555")

ax.set_title("DataCenter-FAIR pipeline (adapted from Cui et al. 2024, Aviation-FAIR)", fontsize=11)
plt.tight_layout()
plt.savefig("figure2_pipeline_diagram.png", dpi=150, bbox_inches="tight")
print("Saved figure2_pipeline_diagram.png")
