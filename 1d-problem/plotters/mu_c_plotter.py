import sys, pathlib, itertools, h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
cmap = sns.color_palette("flare", as_cmap=True)

sys.path.append("../reference")
from leakas_boltzmann import LEAKAS_BX, LEAKAS_BOLTZMANN

REFERENCE_PATH = pathlib.Path("../reference")
REFERENCE_FILE = REFERENCE_PATH / "reference_sn.h5"

OUTPUTS_PATH = pathlib.Path("../outputs/muc_variation")
OUTPUT_FILE = OUTPUTS_PATH / "logspace_muc.h5"

SAVE_PATH = pathlib.Path("../figures/muc_variation")
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
REF_X, REF_FP = REF_SOLN_DICT["z_centers"], REF_SOLN_DICT["phi"]
REF_MU, REF_FP_PSI = REF_SOLN_DICT["mu"], REF_SOLN_DICT["psi"]

# SDE
SDE_MU_C = set()
SDE_SCALE = set()
SDE_TRIAL_MUC = {}
SDE_TRIAL_SCALE = {}
SDE_SOLN_DICT = {}
with h5py.File(OUTPUT_FILE, "r") as sde_file:
    for method, solver_solns in sde_file.items():
        SDE_SOLN_DICT[method.title()] = {}
        for trial_name, soln_sets in solver_solns.items():
            trial_id = int(trial_name.split(" ")[-1])

            mu_c = soln_sets["mu_c"][()][0]
            SDE_MU_C.add(mu_c)
            SDE_TRIAL_MUC[trial_id] = mu_c

            scale = soln_sets["scale"][()][0]
            SDE_SCALE.add(scale)
            SDE_TRIAL_SCALE[trial_id] = scale

            SDE_SOLN_DICT[method.title()][trial_id] = {
                "x": soln_sets["x_centers"][()],
                "mu": soln_sets["mu_centers"][()],
                "phi": soln_sets["phi"][()],
                "psi": soln_sets["psi"][()],
                "time": soln_sets["time"][()][0],
            }

SDE_MU_C = sorted(list(SDE_MU_C))
MIN_MUC = min(SDE_MU_C[1:])
SDE_SCALE = sorted(list(SDE_SCALE))

runtimes = {method:{scale:np.zeros(len(SDE_MU_C)-1) for scale in SDE_SCALE} for method in SDE_SOLN_DICT.keys()}
errors = {method:{scale:np.zeros(len(SDE_MU_C)-1) for scale in SDE_SCALE} for method in SDE_SOLN_DICT.keys()}
for method, trial_dict in SDE_SOLN_DICT.items():
    if method == "Milstein":
        plots = {scale: plt.subplots() for scale in SDE_SCALE}
    for trial_id, soln_dict in trial_dict.items():
        muc = SDE_TRIAL_MUC[trial_id]
        if muc == -1: 
            continue
        muc_idx = SDE_MU_C.index(muc) - 1

        scale = SDE_TRIAL_SCALE[trial_id]

        x = soln_dict["x"]
        phi = soln_dict["phi"]
        l2_err = np.linalg.norm(phi - REF_FP, 2) / np.linalg.norm(REF_FP, 2)
        errors[method][scale][muc_idx] = l2_err
        runtimes[method][scale][muc_idx] = soln_dict["time"]

        if method != "Milstein":
            continue
        relative_muc = (muc - MIN_MUC) / (1 - MIN_MUC)
        ax = plots[scale][1]
        ax.plot(x, phi, color=cmap(relative_muc), zorder=1+muc) #order by muc

    if method != "Milstein":
        continue

    for scale, (fig, ax) in plots.items():
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=MIN_MUC, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=r"$\mu_c$")

        ax.axvline(0.75, color="k", ls="--")
        ax.set_xlabel(r"X Position  $\left(cm\right)$")
        ax.set_ylabel(r"$\phi$  $\left(cm^{2}\cdot s^{-1}\right)$")
        ax.set_title(fr"$\Delta_s={scale}\cdot\Sigma_{{tr}}$")
        fig.savefig(SAVE_PATH / f"phi-{scale}.pdf")
plt.show()

markers = itertools.cycle(["o", "x", "^", "s", "*"])
for method in SDE_SOLN_DICT.keys():
    time_fig, time_ax = plt.subplots()
    boltzmann_runtimes = []
    for scale, runtime_arr in runtimes[method].items():
        time_ax.plot(
            SDE_MU_C[1:], runtime_arr,
            marker=next(markers),
            ms=3.5,
            label = rf"$\Delta_s={scale}\cdot\Sigma_{{tr}}$",
            zorder= 3 - scale,
        )
        boltzmann_runtimes.append(runtime_arr[-1])

    ave_boltzmann = np.average(boltzmann_runtimes)
    time_ax.axhline(ave_boltzmann, color="k", linestyle="--", zorder=2, label="Boltzmann Runtime")
    time_ax.legend()
    
    time_ax.set_title(f"{method} Runtimes")
    time_ax.set_ylabel("Runtime  [s]")
    time_ax.set_xlabel(r"$\mu_c$")
    time_fig.savefig(SAVE_PATH / f"runtime_{method.lower()}")

plt.show()