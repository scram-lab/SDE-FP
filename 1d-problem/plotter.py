import h5py
import matplotlib.pyplot as plt

# Reference
REF_SOLN_DICT = {}
with h5py.File("reference_sn.h5", "r") as f:
    f.visititems(
        lambda name, obj: REF_SOLN_DICT.__setitem__(
            name, obj[()] if isinstance(obj, h5py.Dataset) else None
        )
    )
REF_X, REF_PHI = REF_SOLN_DICT["z_centers"], REF_SOLN_DICT["phi"]

# SDE
SDE_DS = set()
SDE_SOLN_DICT = {}
with h5py.File("sde_output.h5", "r") as sde_file:
    for method, solver_solns in sde_file.items():
        SDE_SOLN_DICT[method.title()] = {}
        for ds, ds_sets in solver_solns.items():
            SDE_DS.add(float(ds))
            SDE_SOLN_DICT[method.title()][float(ds)] = {
                "x": ds_sets["x_centers"][()],
                "phi": ds_sets["phi"][()],
                "psi": ds_sets["psi"][()]
            }

fig, ax = plt.subplots()
ds_err_plot = {ds: plt.subplots() for ds in SDE_DS}
for method, ds_dict in SDE_SOLN_DICT.items():
    for ds, soln_dict in ds_dict.items():
        ax.plot(soln_dict["x"], soln_dict["phi"], label=f"{method}, DS={ds}")
        ds_err_plot[ds][1].plot(soln_dict["x"], (soln_dict["phi"] - REF_PHI) / REF_PHI, label=method)

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
    fig.savefig(f"error_{ds}.pdf")
plt.show()
