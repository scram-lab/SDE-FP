import h5py
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

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

fig, ax = plt.subplots()
ds_err_plot = {ds: plt.subplots() for ds in SDE_DS}
for method, ds_dict in SDE_SOLN_DICT.items():
    for ds, soln_dict in ds_dict.items():
        ax.plot(soln_dict["x"], soln_dict["phi"], label=f"{method}, DS={ds}")
        ds_err_plot[ds][1].plot(soln_dict["x"], (soln_dict["phi"] - REF_PHI) / REF_PHI, label=method)
        ds_err_plot[ds][1].set_title(rf"$\Delta_{{s}}$ = {SDE_DS_2_SCALE[ds]}$\cdot\Sigma_{{tr}}$")

ax.plot(REF_X, REF_PHI, label="Reference")
ax.axvline(0.75, color="k", ls="--", label="Material Interface")
ax.legend()
ax.set_xlabel(r"X Position  $\left(cm\right)$")
ax.set_ylabel(r"$\phi$  $\left(cm^{2}\cdot s^{-1}\right)$")
fig.savefig("phi")

for ds, (fig, ax) in ds_err_plot.items():
    ax.axvline(0.75, color="k", ls="--", label="Material Interface")
    ax.legend()
    ax.set_xlabel(r"X Position  $\left(cm\right)$")
    ax.set_ylabel("Relative Error (to reference)")
    fig.savefig(f"error_{SDE_DS_2_SCALE[ds]}.pdf")
plt.show()

time_array = np.zeros((len(SDE_SOLN_DICT), len(SDE_DS)))
for i, (method, ds_dict) in enumerate(SDE_SOLN_DICT.items()):
    for j, (ds, subdict) in enumerate(ds_dict.items()):
        time_array[i, j] = subdict["time"]
df = pd.DataFrame(time_array, index=[*SDE_SOLN_DICT.keys()], columns=[str(SDE_DS_2_SCALE[ds]) for ds in ds_dict.keys()])
df.columns.name = "DS Scale"
print(df)
        
