"""Simple standalone n(t) chart for the walkthrough intro (Sec 2): real,
dated U.S. data-center pipeline schedule (dcmap.us, Aug 2026 tracker)
applied on top of this project's real, modeled 69-facility research fleet, replacing the
earlier illustrative "linear to 700 by 2100" placeholder."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

years = list(range(2025, 2101))
cum = {2025: 0, 2026: 111, 2027: 193, 2028: 246, 2029: 254, 2030: 267,
       2031: 270, 2032: 273, 2033: 276, 2034: 278, 2035: 281}
n = []
last = 0
for y in years:
    if y in cum:
        last = cum[y]
    n.append(69 + last)

fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.plot(years, n, color="#5b3a7a", lw=2.8)
ax.axvline(2035, color="#9c3226", lw=1.1, linestyle=":")
ax.scatter([2025], [69], color="#5b3a7a", zorder=5, s=55)
ax.annotate("69 facilities\n(this project's modeled research fleet, 2025)", xy=(2025, 69),
            xytext=(2038, 40), fontsize=9.5, color="#5b3a7a",
            arrowprops=dict(arrowstyle="->", color="#5b3a7a"))
ax.scatter([2035], [350], color="#5b3a7a", zorder=5, s=55)
ax.annotate("350 facilities by 2035\n(69 + 281 real, dated pipeline additions)", xy=(2035, 350),
            xytext=(2040, 300), fontsize=9.5, color="#5b3a7a", ha="left",
            arrowprops=dict(arrowstyle="->", color="#5b3a7a"))
ax.annotate("held flat 2036-2100 --\nno dated facilities past 2035\nin the public pipeline\n(a real reporting limit,\nlikely an undercount)", xy=(2070, 350),
            xytext=(2050, 190), fontsize=9, color="#9c3226", ha="left",
            arrowprops=dict(arrowstyle="->", color="#9c3226"))

ax.set_xlabel("Year")
ax.set_ylabel("Number of data centers in the fleet, n(t)")
ax.set_title("The fleet this project models keeps growing: n(t)", fontsize=13, fontweight="bold")
ax.grid(alpha=0.25)
ax.set_ylim(0, 420)
ax.set_xlim(2023, 2102)
plt.tight_layout(rect=[0, 0.13, 1, 1])
fig.text(0.5, 0.005,
         "Real, dated pipeline schedule (dcmap.us, Aug 2026 tracker): 111 facilities completing in 2026, 82 in 2027, 53 in 2028,\n"
         "tapering to single digits by 2029 onward -- 281 total additions with known completion dates through 2035. This project's\n"
         "69-facility research fleet (the same facilities modeled in Part A) is grown by these real counts, not a smooth or invented rate.",
         ha="center", fontsize=8.3, color="#555")
plt.savefig("../results/stage3/fleet_growth_nt_simple.png", dpi=150, bbox_inches="tight")
# Also save under the exact filename main.tex's Fig. fig:nt (left panel)
# includes (figures/nt_mw_dual_axis.png).
plt.savefig("../overleaf_final/figures/nt_mw_dual_axis.png", dpi=150, bbox_inches="tight")
print("saved")
