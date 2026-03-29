#!/usr/bin/env python3
"""
figure2_EOT_heatmaps.py
========================
Generates Figure 2 from:

  Eilertsen, J., Schnell, S. & Walcher, S. (2025).
  "Ehrlich occupancy time: Beyond k_off to a complete residence time framework."
  Journal of Pharmacokinetics and Pharmacodynamics.

Figure 2 (EOT_heatmaps_high_res.pdf / .png):
  High-resolution heatmap analysis of EOT_infinity and its analytical bounds
  across the (k3/k2, b0/Kd) parameter plane (panels A–I).

  Panel A  — Converged EOT_infinity values.
  Panel B  — Upper bound relative error.
  Panel C  — Lower bound relative error.
  Panel D  — Which bound is tighter (log ratio of errors).
  Panel E  — Relative bound width.
  Panel F  — Sensitivity d log(EOT) / d log(k3/k2).
  Panels G–I — Line plots of EOT_infinity vs k3/k2 for b0/Kd = 0.1, 1.0, 10.0.

Numerical method
----------------
ODEs are integrated with scipy.integrate.solve_ivp, method 'RK45'
(explicit Runge–Kutta, order 4/5), rtol = atol = 1e-10.
The EOT_infinity integral is evaluated via numpy.trapezoid (trapezoidal rule).
The 40×40 computed grid is bicubic-interpolated to 200×200 for display.

Requirements
------------
See requirements.txt.  Python >= 3.10.

Usage
-----
    python figure2_EOT_heatmaps.py

Output files (written to the current directory):
    EOT_heatmaps_high_res.pdf
    EOT_heatmaps_high_res.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline, make_interp_spline
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------
plt.rcParams['font.size']       = 10
plt.rcParams['font.family']     = 'sans-serif'
plt.rcParams['axes.labelsize']  = 11
plt.rcParams['axes.titlesize']  = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi']      = 150
plt.rcParams['savefig.dpi']     = 300


# ---------------------------------------------------------------------------
# Core computation functions (shared with figure3_EOT_parameter_slices.py)
# ---------------------------------------------------------------------------

def compute_EOT_converged(k1, k2, k3, a0, b0, tol=1e-6, max_T_factor=100):
    """
    Compute EOT_infinity by integrating the pseudo-first-order ODE
    (equation 49 in the paper) until the running integral converges.

        dc/dt = -[k1*b0*exp(-k3*t) + k2]*c + k1*a0*b0*exp(-k3*t),  c(0)=0

    Parameters
    ----------
    k1 : float  Association rate constant (M^-1 s^-1).
    k2 : float  Dissociation rate constant (s^-1).
    k3 : float  Drug elimination rate constant (s^-1).
    a0 : float  Initial receptor concentration (M).
    b0 : float  Initial drug concentration (M).
    tol : float Relative convergence tolerance (default 1e-6).
    max_T_factor : int
                Maximum integration time as a multiple of 1/min(k2,k3).

    Returns
    -------
    EOT_current : float   Converged EOT_infinity (s).
    t           : ndarray Time array from the final integration pass.
    f           : ndarray Fractional occupancy f(t) = c(t)/a0.
    converged   : bool    True if the convergence criterion was satisfied.
    """
    time_scale   = 1.0 / min(k2, k3)
    T_max        = 10.0 * time_scale
    num_points   = 2000
    EOT_previous = None
    EOT_current  = None
    converged    = False

    def ode_rhs(t, y):
        c    = y[0]
        b    = b0 * np.exp(-k3 * t)
        dcdt = k1 * (a0 - c) * b - k2 * c
        return [dcdt]

    for _ in range(5):
        t_eval = np.linspace(0, T_max, num_points)
        sol    = solve_ivp(ode_rhs, (0.0, T_max), [0.0],
                           method='RK45', t_eval=t_eval,
                           rtol=1e-10, atol=1e-10)
        t = sol.t
        c = sol.y[0]
        f = c / a0

        EOT_current = np.trapezoid(f, t)

        if EOT_previous is not None:
            rel_change = abs(EOT_current - EOT_previous) / max(EOT_current, 1e-30)
            if rel_change < tol:
                converged = True
                break

        if f[-1] < 1e-8:
            converged = True
            break

        EOT_previous = EOT_current
        T_max       *= 2.0
        if T_max > max_T_factor * time_scale:
            break

    return EOT_current, t, f, converged


def compute_EOT_bounds(k1, k2, k3, b0):
    """
    Analytical upper and lower bounds for EOT_infinity
    (equations 59 and 63 in the paper).

        EOT_upper = b0 / (Kd * k3)
        EOT_lower = b0 / ((b0 + Kd) * k3),  where Kd = k2/k1.

    Returns
    -------
    EOT_upper : float
    EOT_lower : float
    """
    Kd = k2 / k1
    return b0 / (Kd * k3),  b0 / ((b0 + Kd) * k3)


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------

def generate_figure2():
    """
    Build Figure 2: high-resolution heatmap analysis.
    Returns the matplotlib Figure object.
    """
    k1 = 1e6;  k2 = 1.0
    Kd = k2 / k1

    n_coarse = 40;  n_fine = 200

    k3_k2_coarse = np.logspace(-2,  2,   n_coarse)
    b0_Kd_coarse = np.logspace(-1,  1.7, n_coarse)
    k3_k2_fine   = np.logspace(-2,  2,   n_fine)
    b0_Kd_fine   = np.logspace(-1,  1.7, n_fine)

    EOT_exact_mat = np.zeros((n_coarse, n_coarse))
    EOT_upper_mat = np.zeros((n_coarse, n_coarse))
    EOT_lower_mat = np.zeros((n_coarse, n_coarse))

    print(f"Computing EOT on {n_coarse}x{n_coarse} grid ...")
    for i, k3_k2 in enumerate(k3_k2_coarse):
        if i % 10 == 0:
            print(f"  Row {i}/{n_coarse} ...")
        for j, b0_Kd in enumerate(b0_Kd_coarse):
            k3 = k3_k2 * k2
            b0 = b0_Kd * Kd
            a0 = 0.1 * Kd           # eps = a0/b0 kept at 0.1

            EOT, _, _, _       = compute_EOT_converged(k1, k2, k3, a0, b0)
            EOT_up, EOT_lo     = compute_EOT_bounds(k1, k2, k3, b0)
            EOT_exact_mat[i, j] = EOT
            EOT_upper_mat[i, j] = EOT_up
            EOT_lower_mat[i, j] = EOT_lo

    print(f"Interpolating to {n_fine}x{n_fine} grid ...")
    lc = np.log10(k3_k2_coarse);  ld = np.log10(b0_Kd_coarse)
    lf = np.log10(k3_k2_fine);    lg = np.log10(b0_Kd_fine)

    def _interp(mat):
        spl = RectBivariateSpline(lc, ld, np.log10(mat + 1e-30), kx=3, ky=3, s=0)
        return 10 ** spl(lf, lg)

    EOT_exact_fine = _interp(EOT_exact_mat)
    EOT_upper_fine = _interp(EOT_upper_mat)
    EOT_lower_fine = _interp(EOT_lower_mat)

    print("Rendering Figure 2 ...")
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    def _heatmap(ax, data, title, clabel, norm=None, cmap='viridis'):
        if norm is None:
            norm = LogNorm(vmin=data.min(), vmax=data.max())
        im = ax.pcolormesh(b0_Kd_fine, k3_k2_fine, data,
                           norm=norm, cmap=cmap,
                           shading='gouraud', rasterized=True)
        ax.set_xscale('log');  ax.set_yscale('log')
        ax.set_xlabel('$b_0/K_d$');  ax.set_ylabel('$k_3/k_2$')
        ax.set_title(title, fontweight='bold')
        cb = plt.colorbar(im, ax=ax, label=clabel, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15, linewidth=0.5)

    # A
    _heatmap(fig.add_subplot(gs[0, 0]),
             EOT_exact_fine,
             r'A: Converged $\mathrm{EOT}_\infty$',
             r'$\mathrm{EOT}_\infty$ (s)')
    # B
    _heatmap(fig.add_subplot(gs[0, 1]),
             np.abs(EOT_upper_fine - EOT_exact_fine) / EOT_exact_fine,
             'B: Upper Bound Relative Error', 'Relative Error',
             norm=LogNorm(vmin=0.01, vmax=10), cmap='RdYlBu_r')
    # C
    _heatmap(fig.add_subplot(gs[0, 2]),
             np.abs(EOT_exact_fine - EOT_lower_fine) / EOT_exact_fine,
             'C: Lower Bound Relative Error', 'Relative Error',
             norm=LogNorm(vmin=0.01, vmax=10), cmap='RdYlBu_r')

    # D
    ax4 = fig.add_subplot(gs[1, 0])
    ratio = np.log10(
        np.abs(EOT_upper_fine - EOT_exact_fine) /
        (np.abs(EOT_exact_fine - EOT_lower_fine) + 1e-30))
    im4 = ax4.pcolormesh(b0_Kd_fine, k3_k2_fine, ratio,
                         norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2),
                         cmap='coolwarm', shading='gouraud', rasterized=True)
    ax4.set_xscale('log');  ax4.set_yscale('log')
    ax4.set_xlabel('$b_0/K_d$');  ax4.set_ylabel('$k_3/k_2$')
    ax4.set_title('D: Which Bound is Tighter', fontweight='bold')
    plt.colorbar(im4, ax=ax4,
                 label=r'$\log_{10}$(Upper Error / Lower Error)',
                 pad=0.02).ax.tick_params(labelsize=8)
    ax4.grid(True, alpha=0.15, linewidth=0.5)
    kw = dict(fontsize=8, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax4.text(0.05, 0.95, 'Lower\ntighter', transform=ax4.transAxes,
             color='darkblue', va='top', **kw)
    ax4.text(0.85, 0.05, 'Upper\ntighter', transform=ax4.transAxes,
             color='darkred', **kw)

    # E
    bw = (EOT_upper_fine - EOT_lower_fine) / EOT_exact_fine
    _heatmap(fig.add_subplot(gs[1, 1]),
             bw, 'E: Relative Bound Width',
             '(Upper − Lower) / EOT',
             norm=LogNorm(vmin=bw.min(), vmax=bw.max()), cmap='YlOrRd')

    # F
    _heatmap(fig.add_subplot(gs[1, 2]),
             np.abs(np.gradient(np.log10(EOT_exact_fine), axis=0)),
             r'F: Sensitivity to $k_3/k_2$',
             r'$|\partial\log(\mathrm{EOT})/\partial\log(k_3/k_2)|$',
             norm=LogNorm(), cmap='plasma')

    # G–I: line plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, b0_Kd_val in enumerate([0.1, 1.0, 10.0]):
        ax = fig.add_subplot(gs[2, idx])
        j  = np.argmin(np.abs(b0_Kd_coarse - b0_Kd_val))

        k3_sm = np.logspace(-2, 2, 200)

        def _smooth(yvec):
            spl = make_interp_spline(
                np.log10(k3_k2_coarse), np.log10(yvec), k=3)
            return 10 ** spl(np.log10(k3_sm))

        ax.loglog(k3_sm, _smooth(EOT_exact_mat[:, j]), '-',
                  color=colors[idx], linewidth=2.5, label='Converged EOT')
        ax.loglog(k3_sm, _smooth(EOT_upper_mat[:, j]), '--',
                  color='red',   linewidth=1.5, alpha=0.7, label='Upper bound')
        ax.loglog(k3_sm, _smooth(EOT_lower_mat[:, j]), '--',
                  color='green', linewidth=1.5, alpha=0.7, label='Lower bound')
        ax.fill_between(k3_sm,
                        _smooth(EOT_lower_mat[:, j]),
                        _smooth(EOT_upper_mat[:, j]),
                        alpha=0.15, color='gray')
        ax.scatter(k3_k2_coarse[::5], EOT_exact_mat[::5, j],
                   s=20, color=colors[idx], alpha=0.5, zorder=5)
        ax.set_xlabel('$k_3/k_2$')
        ax.set_ylabel(r'$\mathrm{EOT}_\infty$ (s)')
        ax.set_title(f'{chr(71+idx)}: $b_0/K_d$ = {b0_Kd_val}',
                     fontweight='bold')
        ax.set_xlim([0.01, 100])
        ax.set_ylim([min(EOT_exact_mat[:, j].min(),
                         EOT_lower_mat[:, j].min()) * 0.5,
                     max(EOT_exact_mat[:, j].max(),
                         EOT_upper_mat[:, j].max()) * 2])
        ax.legend(loc='best', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('\n' + '='*60)
    print('FIGURE 2 — High-resolution EOT heatmaps')
    print('='*60 + '\n')
    fig = generate_figure2()
    fig.savefig('EOT_heatmaps_high_res.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('EOT_heatmaps_high_res.png', dpi=300, bbox_inches='tight')
    print('\nSaved: EOT_heatmaps_high_res.pdf / .png')
    print('='*60 + '\n')
