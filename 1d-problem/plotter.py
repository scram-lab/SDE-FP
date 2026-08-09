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
for ds_name in SDE_DS:
  with h5py.File("sde_output.h5", "r") as f:
    group = f[ds_name]
    x = group["x_centers"][()]
    phi = group["phi"][()]
  ax.plot(x, phi, label =f"DS={ds_name[3:]}")

ax.plot(REF_X, REF_PHI, label="Reference")
ax.axvline(0.75, color="k", ls="--",label="Material Interface")
ax.legend()
ax.set_xlabel(r"X Position  $\left[cm\right]$")
ax.set_ylabel(r"$\phi$  $\left[cm^{2}\cdot s^{-1}\right]$")
plt.show()