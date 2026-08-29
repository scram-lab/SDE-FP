import pathlib
import h5py
import matplotlib.pyplot as plt
plt.rcParams["lines.linewidth"] = 1.0
import seaborn as sns
cmap = sns.color_palette("flare", as_cmap=True)
import numpy as np

REFERENCE_PATH = pathlib.Path("../reference")
REFERENCE_FILE = REFERENCE_PATH / "reference_sn.h5"

OUTPUTS_PATH = pathlib.Path("../outputs")
OUTPUT_FILE = OUTPUTS_PATH / "mu_c_range_depC.h5"

SAVE_PATH = pathlib.Path("../figures")
SAVE_PATH.mkdir(exist_ok=True)
NORM_ORDERS = [2, np.inf] 

# Leakes results from screen grab
LEAKEAS = np.array([
    0, 2.365591397849462,
    0.06013878180416343, 2.575757575757576,
    0.12027756360832698, 2.6686217008797652,
    0.1804163454124904, 2.732160312805474,
    0.24286815728604472, 2.7859237536656893,
    0.3006939090208173, 2.869012707722385,
    0.35851966075558983, 2.9032258064516125,
    0.42097147262914425, 2.966764418377321,
    0.4787972243639167, 3.0254154447702835,
    0.5412490362374711, 3.093841642228739,
    0.5990747879722436, 3.1524926686217007,
    0.659213569776407, 3.2160312805474094,
    0.7193523515805704, 3.255131964809384,
    0.7794911333847341, 3.729227761485826,
    0.8396299151888975, 4.139784946236559,
    0.899768696993061, 4.252199413489736,
    0.9599074787972244, 4.232649071358749,
    1.0200462606013878, 4.1544477028347995,
    1.0801850424055512, 4.056695992179863,
    1.1403238242097147, 3.929618768328446,
    1.200462606013878, 3.826979472140762,
    1.2582883577486508, 3.7145650048875853,
    1.320740169622205, 3.6021505376344085,
    1.3785659213569776, 3.4848484848484844,
    1.4410177332305318, 3.3773216031280544,
    1.4988434849653045, 3.2404692082111435,
    1.5612952968388587, 3.12316715542522,
    1.6191210485736316, 3.0254154447702835,
    1.679259830377795, 2.917888563049853,
    1.7393986121819585, 2.7956989247311825,
    1.799537393986122, 2.673509286412512,
    1.8619892058596763, 2.575757575757576,
    1.9198149575944488, 2.4682306940371457,
    1.9799537393986122, 2.3509286412512216,
    2.0400925212027756, 2.2189638318670575,
    2.1002313030069395, 2.1016617790811343,
    2.160370084811103, 1.9843597262952102,
    2.2205088666152664, 1.8768328445747797,
    2.28064764841943, 1.7595307917888565,
    2.3407864302235932, 1.6471163245356792,
    2.4009252120277567, 1.5004887585532747,
    2.46106399383192, 1.3929618768328451,
    2.5188897455666925, 1.290322580645161,
    2.579028527370856, 1.1779081133919842,
    2.6391673091750194, 1.0557184750733137,
    2.699306090979183, 0.9335288367546433,
    2.7594448727833463, 0.8260019550342133,
    2.8218966846569007, 0.7086999022482887,
    2.879722436391673, 0.591397849462366,
    2.9398612181958366, 0.4740957966764423,
    3, 0.29814271749755594
]).reshape(-1, 2) 
REF_BX, REF_BOLTZMANN = LEAKEAS[:, 0], LEAKEAS[:, 1]

# Reference
REF_SOLN_DICT = {}
with h5py.File(REFERENCE_FILE, "r") as f:
    f.visititems(
        lambda name, obj: REF_SOLN_DICT.__setitem__(
            name, obj[()] if isinstance(obj, h5py.Dataset) else None
        )
    )
REF_X, REF_FP = REF_SOLN_DICT["z_centers"], REF_SOLN_DICT["phi"]
REF_MU, REF_FP_PSI = REF_SOLN_DICT["mu"], REF_SOLN_DICT["psi"]

# SDE
SDE_MU_C = set()
SDE_SOLN_DICT = {}
with h5py.File(OUTPUT_FILE, "r") as sde_file:
    for method, solver_solns in sde_file.items():
        SDE_SOLN_DICT[method.title()] = {}
        for mu_c, muc_sets in solver_solns.items():
            SDE_MU_C.add(float(mu_c))
            SDE_SOLN_DICT[method.title()][float(mu_c)] = {
                "x": muc_sets["x_centers"][()],
                "mu": muc_sets["mu_centers"][()],
                "phi": muc_sets["phi"][()],
                "psi": muc_sets["psi"][()],
                "time": muc_sets["time"][()][0],
            }
LINESTYLES = {method: ['--', '-.', ":"][i] for i, method in enumerate(SDE_SOLN_DICT.keys())}
SDE_MU_C = sorted(SDE_MU_C)
METHODS = sorted(SDE_SOLN_DICT.keys())
#SDE_MU_C = [-1.0, 1.0]

error_arrays = {order: np.zeros((len(SDE_SOLN_DICT), len(SDE_MU_C))) for order in NORM_ORDERS}
time_array = np.zeros((len(SDE_SOLN_DICT), len(SDE_MU_C)))
fom_arrays = {order: np.zeros((len(SDE_SOLN_DICT), len(SDE_MU_C))) for order in NORM_ORDERS}
for i, method in enumerate(METHODS):
    method_dict = SDE_SOLN_DICT[method]
    #ls = LINESTYLES[method]
    fig, ax = plt.subplots()
    err_fig, err_ax = plt.subplots()
    for j, mu_c in enumerate(SDE_MU_C):
        soln_dict = method_dict[mu_c]
        color = cmap((mu_c + 1) / 2)
        x = soln_dict["x"]
        phi = soln_dict["phi"]
        label = "SDE Fokker-Planck" if mu_c == -1.0 else "MC Boltzmann"
        ax.plot(x, phi, label=label)# color=color) 
        err_ax.plot(x, (phi - REF_FP) / REF_FP, label=label)

        time_array[i, j] = soln_dict["time"]
        for order in NORM_ORDERS:
            error = np.linalg.norm(phi - REF_FP, order) / np.linalg.norm(REF_FP, order)
            error_arrays[order][i, j] = error
            fom_arrays[order][i, j] = 1 / error**2 / soln_dict["time"]

    ax.plot(REF_X, REF_FP, label = r"Reference $S_{64}$ FP")#color="#0072B2", marker="x", ms=2.0) 
    ax.plot(REF_BX, REF_BOLTZMANN, linestyle="none", marker = ".", fillstyle="none", label = "Reference MC Boltzmann")

    ax.set_title(method)
    ax.set_ylabel(r"$\phi$  $\left(cm^{2}\cdot s^{-1}\right)$")
    err_ax.set_title(f"{method} against Reference FP")
    err_ax.set_ylabel("Relative Error")
    for _ax in [ax, err_ax]:
        _ax.axvline(0.75, color="k", ls="--", label="Material Interface")
        _ax.set_xlabel(r"X Position  $\left(cm\right)$")
        _ax.legend()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    #fig.colorbar(sm, ax=ax, label=r"$\mu_c$")
    fig.savefig(SAVE_PATH / ("phi_" + method))
    err_fig.savefig(SAVE_PATH / ("err_" + method))

err_fig, err_ax = plt.subplots()
fom_fig, fom_ax = plt.subplots()
time_fig, time_ax = plt.subplots()
counter = 0
for order in NORM_ORDERS:
  for i, method in enumerate(METHODS):
      marker = ["o", "^", "x", "*"][counter]
      counter += 1
      err_ax.plot(SDE_MU_C, error_arrays[order][i], marker=marker, ms = 2, label = rf"$L_{{{order if order is not np.inf else r"\infty"}}}$: " + method)
      if order == 2:
          fom_ax.plot(SDE_MU_C, fom_arrays[order][i], marker=marker, label = method, ms = 2)
          time_ax.plot(SDE_MU_C, time_array[i], marker=marker, label=method, ms = 2)

for ax in [err_ax, fom_ax, time_ax]:
    ax.set_xlabel(r"$\mu_c$") 
    ax.legend()
err_ax.set_ylabel("Relative Error")
err_fig.savefig(SAVE_PATH / "error")
fom_ax.set_ylabel(r"Figure of Merit  $\left[\frac{1}{t_{r}\sigma_{r}^{2}}\right]$")
fom_fig.savefig(SAVE_PATH / "fom")
time_ax.set_ylabel("Time  [s]")
time_fig.savefig(SAVE_PATH / "time")
plt.show()