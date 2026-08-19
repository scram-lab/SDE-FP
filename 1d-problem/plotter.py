import pathlib
import h5py
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

SAVE_PATH = pathlib.Path("figures")
SAVE_PATH.mkdir(exist_ok=True)
NORM_ORDERS = [2, np.inf] 
# Reference
REF_SOLN_DICT = {}
with h5py.File("reference_sn.h5", "r") as f:
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
with h5py.File("sde_output.h5", "r") as sde_file:
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
select_DS = [SDE_DS[i] for i in range(len(SDE_DS)) if not i % (len(SDE_DS)/2)] + [SDE_DS[-1]]

fig, ax = plt.subplots()
ds_err_plot = {ds: plt.subplots() for ds in select_DS}
error_arrays = {order: np.zeros((len(SDE_SOLN_DICT), len(SDE_DS))) for order in NORM_ORDERS}
for i, (method, ds_dict) in enumerate(SDE_SOLN_DICT.items()):
    for j, ds in enumerate(SDE_DS):
        x = ds_dict[ds]["x"]
        phi = ds_dict[ds]["phi"]
        if ds in select_DS:
            ax.plot(x, phi, label=f"{method}, DS={ds}")
            ds_err_plot[ds][1].plot(x, (phi - REF_PHI) / REF_PHI, label=method)
            ds_err_plot[ds][1].set_title(rf"$\Delta_{{s}}$ = {SDE_DS_2_SCALE[ds]}$\cdot\lambda_{{\text{{mfp,tr}}}}$")
        for order in NORM_ORDERS:
            error_arrays[order][i, j] = np.linalg.norm(phi - REF_PHI, order) / np.linalg.norm(REF_PHI, order)
df = pd.DataFrame(error_arrays[np.inf], index=[*SDE_SOLN_DICT.keys()], columns=[str(SDE_DS_2_SCALE[ds]) for ds in SDE_DS])
df.columns.name = "DS Scale"
print(df)

ax.plot(REF_X, REF_PHI, color="k", label="Reference")
ax.axvline(0.75, color="k", ls="--", label="Material Interface")
ax.legend()
ax.set_xlabel(r"X Position  $\left(cm\right)$")
ax.set_ylabel(r"$\phi$  $\left(cm^{2}\cdot s^{-1}\right)$")
fig.savefig(SAVE_PATH / "phi")

for ds in select_DS:
    fig, ax = ds_err_plot[ds]
    ax.axvline(0.75, color="k", ls="--", label="Material Interface")
    ax.legend()
    ax.set_xlabel(r"X Position  $\left(cm\right)$")
    ax.set_ylabel("Relative Error (to reference)")
    fig.savefig(SAVE_PATH / f"error_{SDE_DS_2_SCALE[ds]}.pdf")
    if ds not in select_DS: plt.close(fig)

time_array = np.zeros((len(SDE_SOLN_DICT), len(SDE_DS)))
for i, (method, ds_dict) in enumerate(SDE_SOLN_DICT.items()):
    for j, ds in enumerate(SDE_DS):
        time_array[i, j] = ds_dict[ds]["time"]
df = pd.DataFrame(time_array, index=[*SDE_SOLN_DICT.keys()], columns=[str(SDE_DS_2_SCALE[ds]) for ds in SDE_DS])
df.columns.name = "DS Scale"
print(df)

scales = [float(SDE_DS_2_SCALE[ds]) for ds in SDE_DS]
err_fig, err_ax = plt.subplots()
fom_fig, fom_ax = plt.subplots()
for i, method in enumerate(SDE_SOLN_DICT.keys()):
    times = time_array[i]
    for order in NORM_ORDERS:
        errors = error_arrays[order][i]
        err_ax.loglog(scales,  errors, label=method + f" $L_{{{r"\infty" if order is np.inf else order}}}$")
    fom_ax.loglog(scales, 1/ (times * errors**2), label=method)
for ax in [err_ax, fom_ax]:
    ax.legend()
    ax.set_xlabel(r"$\Delta_s$ scale to $\lambda_{{\text{{mfp,tr}}}}$")
err_ax.set_ylabel("Relative Error (to reference)")
err_fig.savefig(SAVE_PATH/"rel_err")
fom_ax.set_ylabel(r"Figure of Merit  $\left[\frac{1}{t_{r}\sigma_{r}^{2}}\right]$")
fom_fig.savefig(SAVE_PATH/"fom")

plt.show()