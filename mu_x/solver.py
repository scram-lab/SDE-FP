import numpy as np
import math
from colorama import Fore, init
init(True)
import matplotlib.pyplot as plt
from numba import njit
from numpy.polynomial.legendre import Legendre, legval

SIGMA_TR = 0.5
SIGMA_T = 1.0
MU_0 = 0.0
X0 = 0.0
X_MIN = 0.0
X_MAX = 2.0
LEGENDRE_ORDER = 10 


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
def euler_marayuma_psi(DS=0.2, N_X=200, N_MU=100, N_HISTORIES=1000):
  _psi = np.zeros((N_X, N_MU))

  DMU = 2 / N_MU
  DX = (X_MAX - X_MIN) / N_X
  ATTENUATION = np.exp(-SIGMA_T * DS)

  for h in range(N_HISTORIES):
    mn = MU_0
    xn = X0
    psin = 1.0
    while True:
      xi = np.random.normal()
      mn = mn - SIGMA_TR*mn*DS + math.sqrt(SIGMA_TR * (1 - mn**2)*DS) * xi
      mn = clip(mn, -1, 1)
      psin *= ATTENUATION
      xnew = hist_tally(mn, xn, psin, _psi, DS, DX)
      if xnew < X_MIN or xnew > X_MAX:
        break
      xn = xnew


  _psi /= N_HISTORIES * DMU * DX
  return _psi, DX, DMU


# =========================================================================== #
# Plotting
# =========================================================================== #
x_vals_plot = np.linspace(X_MIN, X_MAX, 17)
x_vals_plot[0] = (x_vals_plot[0] + x_vals_plot[1])/2
x_vals_plot[-1] = (x_vals_plot[-1] + x_vals_plot[-2])/2
ds_array = [0.2, 0.1, 0.02, 0.01]

figs = []
axes = []
for xval in x_vals_plot:
  fig, ax = plt.subplots()
  figs.append(fig)
  axes.append(ax)
  ax.set_title(rf"$\psi(x={xval}, mu)$")

phi_fig, phi_ax = plt.subplots()
for ds in ds_array:
  euler_marayuma, DX, DMU = euler_marayuma_psi(DS=ds, N_HISTORIES=1_000_000)
  nx, nmu = euler_marayuma.shape
  x_edges = np.linspace(X_MIN, X_MAX, nx + 1)
  x_vals = 0.5 * (x_edges[:-1] + x_edges[1:])
  mu_edges = np.linspace(-1.0, 1.0, nmu + 1)
  mu_vals = 0.5 * (mu_edges[:-1] + mu_edges[1:])


  for i, (ax, xval) in enumerate(zip(axes, x_vals_plot)):
    xindex = clip(int(nx * (xval - X_MIN) / (X_MAX - X_MIN)), 0, nx - 1)
    psix = euler_marayuma[xindex]
    ax.plot(
      mu_vals, psix / np.sum(psix),
      label=rf"Euler, $\Delta s$={ds}"
    )
  
  phi = np.sum(euler_marayuma, axis=1) * DMU
  phi_ax.plot(x_vals, phi, label=fr"$\Delta_s={ds}$")
  print(Fore.GREEN + f"Finished ds={ds}$")

for fig, ax, xval in zip(figs, axes, x_vals_plot):
  ax.set_xlabel(r"$\mu$")
  ax.legend()
  fig.savefig(rf"psi-x-{str(xval).replace(".", "_")}")

phi_ax.set_xlabel(r"x-position")
phi_ax.set_title(r"$\phi(x) (with attenuation)$")
phi_ax.legend()
#phi_ax.set_yscale("log")
#phi_ax.set_ylim(1e-4, 1.0)
phi_fig.savefig("phi_w_attenuation")

plt.show()

