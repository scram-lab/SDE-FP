import matplotlib.pyplot as plt
from itertools import cycle
from pathlib import Path

SAVEPATH = Path("../figures/geometry")
SAVEPATH.mkdir(exist_ok=True, parents=True)

colorer = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

XMIN, XINTERFACE, XMAX = 0.0, 0.75, 3.0

fig, ax = plt.subplots(figsize=(8, 1.5))

# region shading
ax.axvspan(xmin=XMIN, xmax=XINTERFACE, color=next(colorer))
ax.axvspan(xmin=XINTERFACE, xmax=XMAX, color=next(colorer))
ax.axvline(XINTERFACE, color="k", ls="--")

# setting the text stuff
YMIN, YMAX = ax.get_ylim()
text = "Region {0}\n$\\eta={1}$\n$\\Sigma_t={2:.1f}$"
ax.text((XINTERFACE - XMIN) / 2 + XMIN, (YMAX - YMIN) / 2, text.format("I", "10^{-4}", 100), ha="center", va="center")
ax.text((XMAX - XINTERFACE) / 2 + XINTERFACE, (YMAX - YMIN) / 2, text.format("II", "10^{-2}", 50), ha="center", va="center")

# pencil beam (arrow outside of axes is a smidge anoying )
ax.annotate(
    '', xy=(0.0, (YMAX-YMIN) / 2), xytext=(-.45, (YMAX - YMIN )/ 2),
    arrowprops={"arrowstyle":'->', "color":'black'},
)
ax.text(
    -0.25, 1.05 * (YMAX - YMIN) / 2 ,
    'Pencil\nBeam',
    ha='center', va='bottom',
)

ax.grid(False)
ax.set_yticks([])
ax.set_xbound(XMIN, XMAX)
ax.set_ybound(YMIN, YMAX)
ax.set_xlabel("Axial Depth  [cm]")
fig.savefig(SAVEPATH / "1d_geometry", bbox_inches="tight")
plt.show()