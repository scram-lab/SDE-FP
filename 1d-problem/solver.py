import numpy as np
import h5py
import math
from colorama import Fore, init
init(True)
import matplotlib.pyplot as plt
from numba import njit

BOUNDS = np.array([0.0, 0.75, 3.0])
SIGMA_TR = np.array([1.6422722822026428e-01, 3.6612717220096727e+00])
SIGMA_T = [100.0, 50.0]
MU_0 = 1.0
X0 = 0.0
X_MIN = BOUNDS[0]
X_MAX = BOUNDS[-1]

# =========================================================================== #
# Reference Solution
# =========================================================================== #
REF_SOLN_DICT = {}
with h5py.File("reference_sn.h5", "r") as f:
    f.visititems(
        lambda name, obj: REF_SOLN_DICT.__setitem__(name, obj[()] if isinstance(obj, h5py.Dataset) else None)
    )
REF_X, REF_PHI = REF_SOLN_DICT["z_centers"], REF_SOLN_DICT["phi"]


# =========================================================================== #
# Numerical
# =========================================================================== #
# helper for clipping, njit doesnt like np.clip apparently
@njit
def clip(a, min_, max_):
  return min( max(a, min_), max_)

@njit
def hist_tally(m, x_prev, weight, psi_arr, ds, dx):
  n_x_bins, n_mu_bins = psi_arr.shape

  mu_index = int(math.floor((m + 1.0) / 2.0 * n_mu_bins))
  mu_index = clip(mu_index, 0, n_mu_bins - 1)

  x_new = x_prev + m * ds

  if abs(m) < 1.0e-14:
    ix = int(math.floor((x_prev - X_MIN) / dx))
    ix = clip(ix, 0, n_x_bins - 1)
    psi_arr[ix, mu_index] += weight * ds
    return x_new

  x = x_prev
  s_remaining = ds

  while s_remaining > 0.0:
    ix = int(math.floor((x - X_MIN) / dx))
    ix = clip(ix, 0, n_x_bins - 1)

    if m > 0.0:
      x_face = X_MIN + (ix + 1) * dx
      dist_to_face = x_face - x
    else:
      x_face = X_MIN + ix * dx
      dist_to_face = x - x_face

    if x_face <= X_MIN or x_face >= X_MAX:
      ds_to_boundary = dist_to_face / abs(m)
      psi_arr[ix, mu_index] += weight * min(ds_to_boundary, s_remaining)
      break

    ds_to_face = dist_to_face / abs(m)

    if ds_to_face >= s_remaining:
      psi_arr[ix, mu_index] += weight * s_remaining
      s_remaining = 0.0
    else:
      psi_arr[ix, mu_index] += weight * ds_to_face
      s_remaining -= ds_to_face

      if m > 0.0:
        x = x_face + 1.0e-12 * dx
      else:
        x = x_face - 1.0e-12 * dx

  return x_new


@njit
def euler_marayuma_psi(DS=0.2, N_X=300, N_MU=100, N_HISTORIES=1000):
  _psi = np.zeros((N_X, N_MU))

  DMU = 2 / N_MU
  DX = (X_MAX - X_MIN) / N_X

  for h in range(N_HISTORIES):
    mn = MU_0
    xn = X0
    psin = 1.0
    while True:
      xi = np.random.normal()
      TRANSPORT_XS = SIGMA_TR[np.searchsorted(BOUNDS, xn) - 1]
      mn = mn - TRANSPORT_XS*mn*DS + math.sqrt(TRANSPORT_XS * (1 - mn**2)*DS) * xi
      mn = clip(mn, -1, 1)
      xnew = hist_tally(mn, xn, psin, _psi, DS, DX)
      if xnew < X_MIN or xnew > X_MAX:
        break
      xn = xnew


  _psi /= N_HISTORIES * DMU * DX
  return _psi, DX, DMU


# =========================================================================== #
# Plotting
# =========================================================================== #
ds_array = [0.02, 0.01, 0.005, 0.0025]

figs = []
axes = []

phi_fig, phi_ax = plt.subplots()
err_fig, err_ax = plt.subplots()
for ds in ds_array:
  euler_marayuma, DX, DMU = euler_marayuma_psi(DS=ds, N_HISTORIES=1_000_000)
  nx, nmu = euler_marayuma.shape
  x_edges = np.linspace(X_MIN, X_MAX, nx + 1)
  x_vals = 0.5 * (x_edges[:-1] + x_edges[1:])

  phi = np.sum(euler_marayuma, axis=1) * DMU
  phi_ax.plot(x_vals, phi, label=fr"$\Delta_s={ds}$")
  err = np.abs((phi - REF_PHI) / REF_PHI)
  err_ax.semilogy(x_vals, err, label=fr"$\Delta_s={ds}$")
  print(Fore.GREEN + f"Finished ds={ds}")

phi_ax.plot(REF_X, REF_PHI, label="Reference")

phi_ax.set_xlabel(r"x-position")
phi_ax.set_ylabel(r"$\phi$  $\left[\cm^{2}\cdot s^{-1}\right]$")
err_ax.set_xlabel(r"x-position")
err_ax.set_ylabel(r"Error [%]")
phi_ax.legend()
err_ax.legend()

phi_fig.savefig("phi")
err_fig.savefig("relative_error")
plt.show()

