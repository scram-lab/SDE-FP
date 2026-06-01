import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numpy.polynomial.legendre import Legendre, legval

SIGMA_TR = 0.5
MU_0 = 0.0
S_MAX = 2.0
LEGENDRE_ORDER = 10 

# =========================================================================== #
# Analytical
# =========================================================================== #
def ana_psi(mu, s, legendre_order=1000):
  LEGENDRE_PN = [Legendre.basis(n) for n in range(legendre_order+1)]
  mu, s = np.broadcast_arrays(mu, s)
  res = np.full_like(mu, 0.5)
  for l in range(1, legendre_order+1):
    _sum = (2*l+1) / 2 * LEGENDRE_PN[l](mu)
    _sum *= LEGENDRE_PN[l](MU_0) * np.exp(-l*(l+1) / 2 * SIGMA_TR * s)
    res += _sum
  return res


# =========================================================================== #
# Numerical
# =========================================================================== #
# helper for clipping, njit doesnt like np.clip apparently
@njit
def clip(a, min_, max_):
  return min( max(a, min_), max_)

@njit
def hist_tally(n, m, psi):
  n_mu_bins = psi.shape[1]

  mu_index = int(np.floor((m + 1) / 2 * n_mu_bins))
  mu_index = clip(mu_index, 0, n_mu_bins - 1)

  psi[n, mu_index] += 1

@njit
def leg_tally(n, m, psi, l_moments):
  m = clip(m, -1, 1)
  l_moments[0] = 1
  l_moments[1] = m
  for l in range(2, LEGENDRE_ORDER + 1):
    l_moments[l] = ((2*l - 1)*m*l_moments[l-1] - (l - 1)*l_moments[l-2]) / l
  for l in range(LEGENDRE_ORDER + 1):
    psi[n, l] += l_moments[l]

@njit
def euler_marayuma_psi(N_S=1001, N_MU=100, N_HISTORIES=1000):
  #_psi = np.zeros((N_S, LEGENDRE_ORDER + 1))
  _psi = np.zeros((N_S, N_MU))
  #l_moments = np.zeros(LEGENDRE_ORDER + 1)

  DS = S_MAX / (N_S - 1)
  DMU = 2 / N_MU

  for h in range(N_HISTORIES):
    mn = MU_0
    hist_tally(0, mn, _psi)
    #leg_tally(0, mn, _psi, l_moments)
    for n in range(1, N_S):
      xi = np.random.normal()
      mn = mn - SIGMA_TR*mn*DS + np.sqrt(SIGMA_TR * (1 - mn**2)*DS) * xi
      mn = clip(mn, -1, 1)
      hist_tally(n, mn, _psi)
      #leg_tally(n, mn, _psi, l_moments)

  for n in range(N_S):
  #   for l in range(LEGENDRE_ORDER + 1):
  #       _psi[n, l] *= (2*l + 1) / (2.0 * N_HISTORIES)
    _psi[n, :] /= N_HISTORIES * DMU

  return _psi

@njit
def milstein_psi(N_S=1001, N_MU=100, N_HISTORIES=1000):
  #_psi = np.zeros((N_S, LEGENDRE_ORDER + 1))
  _psi = np.zeros((N_S, N_MU))
  #l_moments = np.zeros(LEGENDRE_ORDER + 1)

  DS = S_MAX / (N_S - 1)
  DMU = 2.0 / N_MU

  for h in range(N_HISTORIES):
    mn = MU_0
    hist_tally(0, mn, _psi)
    #leg_tally(0, mn, _psi, l_moments)
    for n in range(1, N_S):
      xi = np.random.normal()
      dw = np.sqrt(DS) * xi
      a = -SIGMA_TR * mn
      b = np.sqrt(SIGMA_TR * (1 - mn**2))
      b_times_db = -SIGMA_TR * mn
      mn = mn + (a * DS) + (b * dw) + 0.5*(b_times_db) * (dw**2 - DS) 
      mn = clip(mn, -1, 1)
      hist_tally(n, mn, _psi)
      #leg_tally(n, mn, _psi, l_moments)

  for n in range(N_S):
  #   for l in range(LEGENDRE_ORDER + 1):
  #       _psi[n, l] *= (2*l + 1) / (2.0 * N_HISTORIES)
      _psi[n, :] /= N_HISTORIES * DMU

  return _psi

@njit
def order2_weak(N_S=1001, N_MU=100, N_HISTORIES=1000):
  # Chapter 5, EQ 1.7 Kloeden, Platen, Schurz
  _psi = np.zeros((N_S, N_MU))

  DS = S_MAX / (N_S - 1)
  DMU = 2.0 / N_MU
  
  for h in range(N_HISTORIES):
    mn = MU_0
    hist_tally(0, mn, _psi)
    for n in range(N_S):
      xi = np.random.normal()
      eta = np.random.normal()
      dw = np.sqrt(DS) * xi
      dz = 0.5 * DS**1.5 * (xi + eta / np.sqrt(3.0))  #Chapter 1, eq 3.13 Kloeden, Platen, Schurz
      a = -SIGMA_TR * mn
      ap = -SIGMA_TR
      b = np.sqrt(SIGMA_TR * (1.0 - mn**2))
      bp = -SIGMA_TR * mn / b
      b_bp = -SIGMA_TR * mn
      bpp_bsq = -SIGMA_TR**2 / b

      mn = (
        mn
        + a*DS 
        + b*dw
        + 0.5 * b_bp * (dw**2 - DS)
        + ap*b*dz
        + 0.5 * (a*ap) * DS**2  # second derivative of a is 0, so 1/2 a'' ^2 drops
        + (a*bp + 0.5*bpp_bsq) * (dw*DS - dz)
      )
      hist_tally(n, mn, _psi)

  for n in range(N_S):
      _psi[n, :] /= N_HISTORIES * DMU

  return _psi

def evaluate_psi(psi_l, mu):
  N_S, _ = psi_l.shape
  num_psi_vals = np.zeros((N_S, len(mu)))
  for n in range(N_S):
    num_psi_vals[n, :] = legval(mu, psi_l[n, :])
  #num_psi_vals = np.clip(num_psi_vals / np.max(num_psi_vals), 0, 1)
  return num_psi_vals

# =========================================================================== #
# Plotting
# =========================================================================== #
mu_vals = np.linspace(-1.0, 1.0, 500)
s_vals_plot = [1.0, 2.0]
ns_array = [11, 101]

figs = []
err_figs = []
axes = []
err_axes = []
anas = []
for s_val_plot in s_vals_plot:
  fig, ax = plt.subplots()
  figs.append(fig)
  axes.append(ax)
  analytical = ana_psi(mu_vals, s_val_plot)
  anas.append(analytical)
  ax.plot(mu_vals, analytical, color="k", label="Analytical")
  ax.set_title(f"$S={s_val_plot}$")
  fig, ax = plt.subplots()
  ax.set_title(f"$S={s_val_plot}$") 
  err_figs.append(fig)
  err_axes.append(ax)

for ns in ns_array:
  s_vals = np.linspace(0.0, S_MAX, ns)
  s_indices = np.where(np.isin(s_vals, s_vals_plot))[0]
  euler_marayuma = euler_marayuma_psi(N_S=ns, N_HISTORIES=10_000_000)
  #euler_marayuma = evaluate_psi(euler_marayuma_l, mu_vals)
  milstein = milstein_psi(N_S=ns, N_HISTORIES=10_000_000)
  #milstein = evaluate_psi(milstein_psi_l, mu_vals)
  taylor2 = order2_weak(N_S=ns, N_HISTORIES=10_000_000)
  nmu = euler_marayuma.shape[1]
  mu_edges = np.linspace(-1.0, 1.0, nmu + 1)
  mu_vals = 0.5 * (mu_edges[:-1] + mu_edges[1:])

  for i, (ax, err_ax, sidx) in enumerate(zip(axes, err_axes, s_indices)):
    analytical_hist = ana_psi(mu_vals, s_vals_plot[i])
    ax.plot(
      mu_vals, euler_marayuma[sidx],
      label=rf"Euler, $\Delta s$={S_MAX/(ns-1)}"
    )
    err_ax.plot(
      mu_vals, np.abs(euler_marayuma[sidx] - analytical_hist) / analytical_hist, 
      label=rf"Euler, $\Delta s$={S_MAX/(ns-1)}"
    )
    ax.plot(
      mu_vals, milstein[sidx],
      label=rf"Milstein, $\Delta s$={S_MAX/(ns-1)}"
    )
    err_ax.plot(
      mu_vals, np.abs(milstein[sidx] - analytical_hist) / analytical_hist, 
      label=rf"Milstein, $\Delta s$={S_MAX/(ns-1)}"
    )
    ax.plot(
      mu_vals, taylor2[sidx],
      label=rf"Weak 2, $\Delta s$={S_MAX/(ns-1)}"
    )
    err_ax.plot(
      mu_vals, np.abs(taylor2[sidx] - analytical_hist) / analytical_hist, 
      label=rf"Weak 2, $\Delta s$={S_MAX/(ns-1)}"
    )

for _axes in [axes, err_axes]:
  for ax in _axes:
    ax.set_xlabel(r"$\mu$")
    ax.legend()
    #ax.set_yscale("log")
for s, fig, err_fig in zip(s_vals_plot, figs, err_figs):
  name = f"s-{s}-".replace(".", "_")
  fig.savefig(name+"psi")
  err_fig.savefig(name + "error")
plt.show()

