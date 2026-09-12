import sys, pathlib, itertools, h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
cmap = sns.color_palette("flare", as_cmap=True)

sys.path.append("../reference")
from leakas_boltzmann import LEAKAS_BX, LEAKAS_BOLTZMANN

REFERENCE_PATH = pathlib.Path("../reference")
REFERENCE_FP_FILE = REFERENCE_PATH / "reference_sn.h5"

OUTPUTS_PATH = pathlib.Path("../outputs/muc_variation")
OUTPUT_FILE = OUTPUTS_PATH / "logspace_muc.h5"

SAVE_PATH = pathlib.Path("../figures/muc_variation")
SAVE_PATH.mkdir(exist_ok=True, parents=True)

NORM_ORDERS = [2, np.inf] 

# Reference
REF_FP_SOLN_DICT = {}
with h5py.File(REFERENCE_FP_FILE, "r") as f:
    f.visititems(
        lambda name, obj: REF_FP_SOLN_DICT.__setitem__(
            name, obj[()] if isinstance(obj, h5py.Dataset) else None
        )
    )
REF_X, REF_FP = REF_FP_SOLN_DICT["z_centers"], REF_FP_SOLN_DICT["phi"]
REF_MU, REF_FP_PSI = REF_FP_SOLN_DICT["mu"], REF_FP_SOLN_DICT["psi"]

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
            if "Reference Boltzmann" in trial_name:
                continue
            trial_id = int(trial_name.split(" ")[-1])

            mu_c = soln_sets["mu_c"][()][0]
            SDE_MU_C.add(mu_c)
            SDE_TRIAL_MUC[trial_id] = mu_c

            scale = soln_sets["scale"][()][0]
            SDE_SCALE.add(scale)
            SDE_TRIAL_SCALE[trial_id] = scale

            x = soln_sets["x_centers"][()]
            phi = soln_sets["phi"][()]
            SDE_SOLN_DICT[method.title()][trial_id] = {
                "x": x,
                "mu": soln_sets["mu_centers"][()],
                "phi": phi,
                "psi": soln_sets["psi"][()],
                "time": soln_sets["time"][()][0],
            }
    ref_soln_set = sde_file["milstein/Reference Boltzmann (solution)"]
    REF_BOLTZMANN = ref_soln_set["phi"][()]
    ref_time_set = sde_file["milstein/Reference Boltzmann (time)"]
    REF_BOLTZMANN_TIME = ref_time_set["time"][()][0]
SDE_SOLN_DICT.pop("Euler-Maruyama")

SDE_MU_C = sorted(list(SDE_MU_C))
MIN_MUC = min(SDE_MU_C[1:])
SDE_SCALE = sorted(list(SDE_SCALE))

PLOT_SCALES = [0.1, 0.01]
SELECT_MU_C = [0.9, 0.99]

runtimes = {method:{scale:np.zeros(len(SDE_MU_C)-1) for scale in SDE_SCALE} for method in SDE_SOLN_DICT.keys()}
errors = {method:{scale:np.zeros(len(SDE_MU_C)-1) for scale in SDE_SCALE} for method in SDE_SOLN_DICT.keys()}
for method, trial_dict in SDE_SOLN_DICT.items():
    plots = {scale: plt.subplots() for scale in PLOT_SCALES}
    err_fig, err_ax = plt.subplots()
    for trial_id, soln_dict in sorted(trial_dict.items(), key= lambda x: x[0], reverse=True):
        muc = SDE_TRIAL_MUC[trial_id]
        muc_idx = SDE_MU_C.index(muc) - 1

        scale = SDE_TRIAL_SCALE[trial_id]

        x = soln_dict["x"]
        phi = soln_dict["phi"]

        if muc != -1: 
            l2_err = np.linalg.norm(phi - REF_BOLTZMANN, 2) / np.linalg.norm(REF_BOLTZMANN, 2)
            errors[method][scale][muc_idx] = l2_err
            runtimes[method][scale][muc_idx] = soln_dict["time"]

        if muc in SELECT_MU_C and scale in PLOT_SCALES:
            plots[scale][1].plot(x, phi, label=rf"$\mu_c={muc}$")
        if muc == 0.997:
            err_ax.plot(
                x, (phi - REF_BOLTZMANN) / REF_BOLTZMANN,
                label = rf"$\Delta_s={scale}\cdot\lambda_{{\text{{mfp,tr}}}}$",
                zorder=scale + 1
            )

    for scale, (fig, ax) in plots.items():
        ax.plot(
            REF_X, REF_BOLTZMANN,
            ls="-.",
            label=r"$\mu_c=1$"
        )
        ax.plot(
            LEAKAS_BX, LEAKAS_BOLTZMANN,
            marker="o",
            fillstyle="none",
            ms=4.5,
            linestyle="none",
            label="Leakas et. al MC Boltzmann",
            zorder=-1,
        )

        ax.axvline(0.75, color="k", ls="--")
        ax.set_xlabel(r"X Position  $\left(cm\right)$")
        ax.set_ylabel(r"$\phi$  $\left(cm^{2}\cdot s^{-1}\right)$")
        ax.legend()
        fig.savefig(SAVE_PATH / f"phi-{scale}.pdf")

    err_ax.axvline(0.75, color="k", ls="--")
    err_ax.set_xlabel(r"X Position  $\left(cm\right)$")
    err_ax.set_ylabel(r"Relative Difference to $\mu_c=1$")
    err_ax.legend()
    err_fig.savefig(SAVE_PATH / "select_errors")

markers = itertools.cycle(["o", "x", "^", "s", "*"])
for method in SDE_SOLN_DICT.keys():
    time_fig, time_ax = plt.subplots()
    err_fig, err_ax = plt.subplots()

    positions = range(len(SDE_MU_C[1:]))
    for scale, err_arr in errors[method].items():
        err_ax.plot(
            positions, err_arr,
            marker=next(markers),
            ms=3.5,
            label = rf"$\Delta_s={scale}\cdot\lambda_{{\text{{mfp,tr}}}}$",
            zorder=1+scale
        )
    for scale, runtime_arr in runtimes[method].items():
        time_ax.plot(
            positions, runtime_arr,
            marker=next(markers),
            ms=3.5,
            label = rf"$\Delta_s={scale}\cdot\lambda_{{\text{{mfp,tr}}}}$",
            zorder= 3 - scale,
        )

    err_ax.set_ylabel("Relative $L_2$ Error")

    time_ax.axhline(REF_BOLTZMANN_TIME, color="k", linestyle="--", zorder=2, label=r"$\mu_c=1$ Runtime") 
    time_ax.set_ylabel("Runtime  [s]")

    for ax in [time_ax, err_ax]:
        ax.set_xlabel(r"$\mu_c$")
        ax.legend()
        ax.set_xticks(positions)
        ax.set_xticklabels([f'{v:g}' for v in SDE_MU_C[1:]], rotation=45, ha='right')

    err_fig.savefig(SAVE_PATH / f"error_{method.lower()}")
    time_fig.savefig(SAVE_PATH / f"runtime_{method.lower()}")

plt.show()