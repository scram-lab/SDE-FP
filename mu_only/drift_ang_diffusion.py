import pathlib
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numpy.polynomial.legendre import Legendre, legval

SAVEPATH = pathlib.Path("figures")
SAVEPATH.mkdir(exist_ok=True)

SIGMA_TR = 100
MFP = 1 / SIGMA_TR
MU_0 = 0.0
S_MAX = 10 * MFP
LEGENDRE_ORDER = 10 

# =========================================================================== #
# Analytical
# =========================================================================== #
def ana_psi(mu, s, legendre_order=1000):
  mu, s = np.broadcast_arrays(mu, s)

  #skip order 0
  res = np.full(mu.shape, 0.5)

  p0_mu, p1_mu = np.ones_like(mu), mu.copy()
  p0_0, p1_0 = 1.0, MU_0 

  # l = 1 term
  res += (3/2) * p1_0 * np.exp(-1*2/2 * SIGMA_TR * s) * p1_mu

  p_prev, p_cur = p0_mu, p1_mu
  p_prev0, p_cur0 = p0_0, p1_0

  for l in range(2, legendre_order + 1):
    p_next = ((2*l - 1) * mu * p_cur - (l - 1) * p_prev) / l
    p_next0 = ((2*l - 1) * MU_0 * p_cur0 - (l - 1) * p_prev0) / l

    res += (2*l + 1) / 2 * p_next0 * np.exp(-l*(l+1) / 2 * SIGMA_TR * s) * p_next

    p_prev, p_cur = p_cur, p_next
    p_prev0, p_cur0 = p_cur0, p_next0

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
# =========================================================================== #
n_histories = 100_000

scales = np.array([0.025 ,0.1, 1.0])
ds_values = scales * MFP
ns_values = np.asarray(S_MAX / ds_values, dtype=int) + 1

n_mu = 200
mu_edges = np.linspace(-1.0, 1.0, n_mu + 1)
mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])

for scale, ns, ds in zip(scales, ns_values, ds_values):
    s_grid = np.linspace(0.0, S_MAX, ns)

    print("Generating numerical solution")
    psi = milstein_psi(N_S=ns, N_MU=n_mu, N_HISTORIES=n_histories)

    s_mesh, mu_grid = np.meshgrid(s_grid, mu_centers, indexing="ij")
    print("Generating analytical solution")
    ana = ana_psi(mu_grid, s_mesh)

    print("Plotting")
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.abs(psi - ana) / ana
    rel_err = np.where(ana > 1e-12, rel_err, np.nan)

    fig, ax = plt.subplots()
    colorplot = ax.pcolormesh(mu_centers, s_grid / MFP, rel_err, shading="auto", vmin=0, vmax=1)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\lambda_{\text{mfp,tr}}$")
    ax.grid(False)
    fig.colorbar(colorplot, ax=ax, label="Relative error")
    fig.savefig(SAVEPATH / f"err_heatmap_{scale}.pdf")
plt.show()
