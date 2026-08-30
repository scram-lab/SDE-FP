import pathlib
import h5py
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
cmap = sns.color_palette("flare", as_cmap=True)
import pandas as pd 
import numpy as np

REFERENCE_PATH = pathlib.Path("../reference")
REFERENCE_FILE = REFERENCE_PATH / "reference_sn.h5"

OUTPUT_PATH = pathlib.Path("../outputs")
OUTPUT_FILE = OUTPUT_PATH / "logspace_e-4_e0.h5"

SAVE_PATH = pathlib.Path("../figures/ds_variation")
SAVE_PATH.mkdir(exist_ok=True, parents=True)
NORM_ORDERS = [2, np.inf] 
# Reference
REF_SOLN_DICT = {}
with h5py.File(REFERENCE_FILE, "r") as f:
    f.visititems(
        lambda name, obj: REF_SOLN_DICT.__setitem__(
            name, obj[()] if isinstance(obj, h5py.Dataset) else None
        )
    )
REF_X, REF_PHI = REF_SOLN_DICT["z_centers"], REF_SOLN_DICT["phi"]
REF_MU, REF_PSI = REF_SOLN_DICT["mu"], REF_SOLN_DICT["psi"]

# SDE
SDE_DS = set()
SDE_DS_2_SCALE = {}
SDE_SOLN_DICT = {}
with h5py.File(OUTPUT_FILE, "r") as sde_file:
    for method, solver_solns in sde_file.items():
        SDE_SOLN_DICT[method.title()] = {}
        for ds, ds_sets in solver_solns.items():
            SDE_DS.add(float(ds))
            SDE_SOLN_DICT[method.title()][float(ds)] = {
                "x": ds_sets["x_centers"][()],
                "mu": ds_sets["mu_centers"][()],
                "phi": ds_sets["phi"][()],
                "psi": ds_sets["psi"][()],
                "time": ds_sets["time"][()][0],
            }
            SDE_DS_2_SCALE[float(ds)] = ds_sets["scale"][()][0]

SDE_DS = sorted(SDE_DS, reverse=True)
select_DS = [SDE_DS[i] for i in range(len(SDE_DS)) if not i % ((len(SDE_DS) - 1)/4) and SDE_DS[i] != max(SDE_DS)]

error_arrays = {order: np.zeros((len(SDE_SOLN_DICT), len(SDE_DS))) for order in NORM_ORDERS}
for i, (method, ds_dict) in enumerate(SDE_SOLN_DICT.items()):
    fig, ax = plt.subplots()
    err_fig, err_ax = plt.subplots()
    for j, ds in enumerate(SDE_DS):
        scale = SDE_DS_2_SCALE[ds]
        x = ds_dict[ds]["x"]
        phi = ds_dict[ds]["phi"]
        ax.plot(x, phi, color = cmap(scale))
        if ds in select_DS:
            ## Plot w markers
            err_ax.plot(x, (phi - REF_PHI) / REF_PHI, label=rf"$\Delta_{{s}}$ = {scale}$\cdot\lambda_{{\text{{mfp,tr}}}}$")
        for order in NORM_ORDERS:
            error_arrays[order][i, j] = np.linalg.norm(phi - REF_PHI, order) / np.linalg.norm(REF_PHI, order)

    # normal figure
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$\Delta_s$ Relative to $\Sigma_{tr}$")
    ax.plot(REF_X, REF_PHI, label=r"Reference $S_{64}$ Fokker-Planck", color="#0072B2", ls="--", lw=1.5)
    ax.axvline(0.75, color="k", ls="--")
    ax.legend()
    ax.set_xlabel(r"X Position  $\left(cm\right)$")
    ax.set_ylabel(r"$\phi$  $\left(cm^{2}\cdot s^{-1}\right)$")
    fig.savefig(SAVE_PATH / f"phi_{method}")

    # error figure
    err_ax.axvline(0.75, color="k", ls="--")
    err_ax.legend()
    err_ax.set_xlabel(r"X Position  $\left(cm\right)$")
    err_ax.set_ylabel("Relative Error (to reference)")
    err_fig.savefig(SAVE_PATH / f"ds_slice_error_{method}.pdf")

time_array = np.zeros((len(SDE_SOLN_DICT), len(SDE_DS)))
for i, (method, ds_dict) in enumerate(SDE_SOLN_DICT.items()):
    for j, ds in enumerate(SDE_DS):
        time_array[i, j] = ds_dict[ds]["time"]

scales = [float(SDE_DS_2_SCALE[ds]) for ds in SDE_DS]
markers = cycle(["o", "x", "^", "s"])
err_fig, err_ax = plt.subplots()
for i, method in enumerate(SDE_SOLN_DICT.keys()):
    times = time_array[i]
    for order in NORM_ORDERS:
        errors = error_arrays[order][i]
        err_ax.loglog(
            scales,  errors,
            marker=next(markers),
            ms=3.5,
            label=f"{method}, $L_{{{r"\infty" if order is np.inf else order}}}$",
        )
err_ax.legend()
err_ax.set_xlabel(r"$\Delta_s$ scale to $\lambda_{{\text{{mfp,tr}}}}$")
err_ax.set_ylabel("Relative Error (to reference)")
err_fig.savefig(SAVE_PATH/"rel_err")

plt.show()