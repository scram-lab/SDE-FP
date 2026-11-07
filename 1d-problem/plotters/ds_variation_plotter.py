import pathlib
import h5py
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
cmap = sns.color_palette("flare", as_cmap=True)
import numpy as np
import scipy as sp

REFERENCE_PATH = pathlib.Path("../reference")
REFERENCE_FILE = REFERENCE_PATH / "reference_sn.h5"

OUTPUT_PATH = pathlib.Path("../outputs/ds_variation")
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
SDE_SOLN_DICT = {}
SDE_SCALES = set()
with h5py.File(OUTPUT_FILE, "r") as sde_file:
    for method, solver_solns in sde_file.items():
        SDE_SOLN_DICT[method.title()] = {}
        for trial_name, soln_sets in solver_solns.items():
            if "Reference Boltzmann" in trial_name:
                continue
            trial_id = int(trial_name.split(" ")[-1])
            scale = float(soln_sets["scale"][()][0])
            SDE_SOLN_DICT[method.title()][trial_id] = {
                "x": soln_sets["x_centers"][()],
                "mu": soln_sets["mu_centers"][()],
                "phi": soln_sets["phi"][()],
                "psi": soln_sets["psi"][()],
                "time": soln_sets["time"][()][0],
                "scale":scale,
            }
            SDE_SCALES.add(scale)

SDE_SOLN_DICT.pop("Euler-Maruyama")
SDE_SCALES = np.array(sorted(SDE_SCALES))
SDE_SOLN_DICT["Milstein"] = dict(sorted(SDE_SOLN_DICT["Milstein"].items(), key=lambda kd: kd[1]["scale"]))
select_scale = [SDE_SCALES[i] for i in range(len(SDE_SCALES)) if not i % ((len(SDE_SCALES) - 1)/4)]

error_arrays = {order: np.zeros((len(SDE_SOLN_DICT), len(SDE_SCALES))) for order in NORM_ORDERS}
for i, (method, trial_dict) in enumerate(SDE_SOLN_DICT.items()):
    print(method)
    fig, ax = plt.subplots()
    err_fig, err_ax = plt.subplots()
    for trial_id, soln_dict in trial_dict.items():
        scale = soln_dict["scale"]
        x = soln_dict["x"]
        phi = soln_dict["phi"]
        ax.plot(x, phi, color = cmap(scale), zorder= 2 - scale)
        if scale in select_scale:
            ## Plot w markers
            err_ax.plot(x, (phi - REF_PHI) / REF_PHI, label=rf"$\Delta_{{s}}$ = {scale}$\cdot\lambda_{{\text{{mfp,tr}}}}$")
        
        error_arrays[2][i, trial_id] =  np.linalg.norm(phi - REF_PHI, 2) / np.linalg.norm(REF_PHI, 2)
        arg_index = np.argmax(phi - REF_PHI)
        error_arrays[np.inf][i, trial_id] = np.linalg.norm(phi - REF_PHI, np.inf) / REF_PHI[arg_index]

    # normal figure
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$\Delta_s$ Relative to $\lambda_{\text{mfp,tr}}$")
    ax.plot(REF_X, REF_PHI, label=r"Reference $S_{64}$ Fokker-Planck", color="#0072B2", ls="--", lw=1.5)
    ax.axvline(0.75, color="k", ls="--")
    ax.legend()
    ax.set_xlabel(r"Z Position  $\left(cm\right)$")
    ax.set_ylabel(r"$\phi$  $\left(cm^{2}\cdot s^{-1}\right)$")
    fig.savefig(SAVE_PATH / f"phi_{method}")

    # error figure
    err_ax.axvline(0.75, color="k", ls="--")
    handles, labels = err_ax.get_legend_handles_labels()
    err_ax.legend(handles[::-1], labels[::-1])
    err_ax.set_xlabel(r"Z Position  $\left(cm\right)$")
    err_ax.set_ylabel("Relative Error (to reference)")
    err_fig.savefig(SAVE_PATH / f"ds_slice_error_{method}.pdf")

time_array = np.zeros((len(SDE_SOLN_DICT), len(SDE_SCALES)))
for i, (method, trial_dict) in enumerate(SDE_SOLN_DICT.items()):
    for trial_id, soln_dict in trial_dict.items():
        time_array[i, trial_id] = soln_dict["time"]

markers = cycle(["o", "x", "^", "s"])
err_fig, err_ax = plt.subplots()
for i, method in enumerate(SDE_SOLN_DICT.keys()):
    times = time_array[i]
    for order in NORM_ORDERS:
        errors = error_arrays[order][i]
        rev_SDE_SCALES = SDE_SCALES[::-1]
        err_ax.loglog(
            rev_SDE_SCALES,  errors,
            marker=next(markers),
            ms=3.5,
            label=f"{method}, $L_{{{r"\infty" if order is np.inf else order}}}$",
        )
        if order == 2:
            where = np.where(rev_SDE_SCALES > 1e-2)
            def log(x, a, b):
                return a * np.sqrt(x) + b
            popt, _ = sp.optimize.curve_fit(log, rev_SDE_SCALES[where], errors[where],bounds=([0, -np.inf], [np.inf, np.inf]),)
            err_ax.loglog(
                rev_SDE_SCALES[where], log(rev_SDE_SCALES[where], *popt),
                label=rf"${popt[0]:.3e}\cdot\sqrt{{\Delta_s}} + {popt[1]:.3e}$"
            )
err_ax.legend()
err_ax.set_xlabel(r"$\Delta_s$ scale to $\lambda_{{\text{{mfp,tr}}}}$")
err_ax.set_ylabel("Relative Error (to reference)")
err_fig.savefig(SAVE_PATH/"rel_err")

plt.show()