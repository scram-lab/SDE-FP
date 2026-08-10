import h5py
import matplotlib.pyplot as plt

# Reference
REF_SOLN_DICT = {}
with h5py.File("reference_sn.h5", "r") as f:
    f.visititems(
        lambda name, obj: REF_SOLN_DICT.__setitem__(name, obj[()] if isinstance(obj, h5py.Dataset) else None)
    )
REF_X, REF_PHI = REF_SOLN_DICT["z_centers"], REF_SOLN_DICT["phi"]

# SDE 
SDE_DS = []
with h5py.File("sde_output.h5", "r") as f:
    f.visititems(
        lambda name, obj: SDE_DS.append(name if isinstance(obj, h5py.Group) else None)
    )
    SDE_DS = [ds for ds in SDE_DS if ds]

fig, ax = plt.subplots()
err_fig, err_ax = plt.subplots()
for ds_name in SDE_DS:
  with h5py.File("sde_output.h5", "r") as f:
    group = f[ds_name]
    x = group["x_centers"][()]
    phi = group["phi"][()]
  ax.plot(x, phi, label =f"DS={ds_name[3:]}")
  err_ax.plot(x, (phi - REF_PHI) / REF_PHI, label=f"DS={ds_name[3:]}")

ax.plot(REF_X, REF_PHI, label="Reference")
ax.axvline(0.75, color="k", ls="--",label="Material Interface")
err_ax.axvline(0.75, color="k", ls="--",label="Material Interface")
ax.legend()
err_ax.legend()
ax.set_xlabel(r"X Position  $\left[cm\right]$")
err_ax.set_xlabel(r"X Position  $\left[cm\right]$")
ax.set_ylabel(r"$\phi$  $\left[cm^{2}\cdot s^{-1}\right]$")
err_ax.set_ylabel("Relative Error (to reference)")
fig.savefig("phi")
err_fig.savefig("error")
plt.show()