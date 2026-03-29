#!/usr/bin/env python3
"""
figure_EOT_bounds.py
====================
Generates Figures 2 and 3 from:

  Eilertsen, J., Schnell, S. & Walcher, S. (2025).
  "Ehrlich occupancy time: Beyond k_off to a complete residence time framework."
  Journal of Pharmacokinetics and Pharmacodynamics.

Figure 2 (EOT_heatmaps_high_res.pdf / .png):
  High-resolution heatmap analysis of EOT_infinity bounds across parameter space
  (panels A–I).

Figure 3 (EOT_parameter_slices.pdf / .png):
  Systematic analysis of EOT behaviour and bound quality, including time
  evolution panels (panels A–F).

Numerical method
----------------
All ODEs are integrated with scipy.integrate.solve_ivp.
  - Parameter-sweep computations (compute_EOT_converged):
      solver = 'RK45' (explicit Runge–Kutta, order 4/5)
      rtol = atol = 1e-10
  - Time-evolution panels D–F (generate_comparison_slices):
      solver = 'Radau' (implicit, L-stable; used because the slow-clearance
      case k3/k2 = 0.01 produces a stiff system)
      rtol = 1e-10, atol = 1e-12

The EOT_infinity integral is computed from the numerical solution c(t) via
numpy.trapezoid (the trapezoidal rule).  Integration continues until the
running value changes by less than tol = 1e-6 per iteration, or until
f(t_end) < 1e-8 (system fully dissociated).

Requirements
------------
See requirements.txt.  Run with Python >= 3.10.

Usage
-----
    python figure_EOT_bounds.py

Output files (written to the current directory):
    EOT_heatmaps_high_res.pdf / .png   (Figure 2)
    EOT_parameter_slices.pdf  / .png   (Figure 3)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import make_interp_spline
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, TwoSlopeNorm
import seaborn as sns        # noqa: F401  (imported for style consistency)
from matplotlib import cm    # noqa: F401
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
# Core computation functions
# ---------------------------------------------------------------------------

def compute_EOT_converged(k1, k2, k3, a0, b0, tol=1e-6, max_T_factor=100):
    """
    Compute EOT_infinity by integrating the pseudo-first-order ODE until
    the running integral converges.

    The reduced ODE (eq. 49 in the paper) is:
        dc/dt = -[k1*b0*exp(-k3*t) + k2]*c + k1*a0*b0*exp(-k3*t),  c(0)=0

    where b(t) = b0*exp(-k3*t) is the leading-order (O(eps^0)) approximation
    for the drug concentration under the pseudo-first-order assumption
    eps = a0/b0 << 1.

    Parameters
    ----------
    k1 : float  Association rate constant (M^-1 s^-1).
    k2 : float  Dissociation rate constant (s^-1).
    k3 : float  Drug elimination rate constant (s^-1).
    a0 : float  Initial receptor concentration (M).
    b0 : float  Initial drug concentration (M).
    tol : float Convergence tolerance on the relative change in EOT between
                successive doubling intervals (default 1e-6).
    max_T_factor : int
                Maximum integration time expressed as a multiple of the
                natural time scale 1/min(k2, k3) (default 100).

    Returns
    -------
    EOT_current : float   Converged EOT_infinity value (s).
    t           : ndarray Time array used in the final integration pass.
    f           : ndarray Fractional occupancy f(t) = c(t)/a0.
    converged   : bool    True if the convergence criterion was met.
    """
    time_scale = 1.0 / min(k2, k3)
    T_max      = 10.0 * time_scale
    num_points = 2000          # dense grid for accurate trapezoidal integration

    converged    = False
    EOT_previous = None
    EOT_current  = None

    def ode_rhs(t, y):
        """Right-hand side of the reduced pseudo-first-order ODE."""
        c    = y[0]
        b    = b0 * np.exp(-k3 * t)
        dcdt = k1 * (a0 - c) * b - k2 * c
        return [dcdt]

    for _ in range(5):
        t_eval = np.linspace(0, T_max, num_points)

        sol = solve_ivp(
            ode_rhs,
            t_span=(0.0, T_max),
            y0=[0.0],
            method='RK45',
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-10,
            dense_output=False,
        )

        t = sol.t
        c = sol.y[0]
        f = c / a0

        # EOT(T) = (1/a0) * integral_0^T c(t) dt = integral_0^T f(t) dt
        EOT_current = np.trapezoid(f, t)

        # Convergence check: relative change between successive passes
        if EOT_previous is not None:
            rel_change = abs(EOT_current - EOT_previous) / max(EOT_current, 1e-30)
            if rel_change < tol:
                converged = True
                break

        # Early exit: system fully dissociated
        if f[-1] < 1e-8:
            converged = True
            break

        EOT_previous = EOT_current
        T_max *= 2.0

        if T_max > max_T_factor * time_scale:
            break

    return EOT_current, t, f, converged


def compute_EOT_bounds(k1, k2, k3, b0):
    """
    Compute the analytical upper and lower bounds for EOT_infinity
    (equations 59 and 63 in the paper).

        EOT_upper = b0 / (Kd * k3)          [eq. 59]
        EOT_lower = b0 / ((b0 + Kd) * k3)   [eq. 63]

    where Kd = k2 / k1.

    Parameters
    ----------
    k1, k2, k3 : float  Rate constants (M^-1 s^-1; s^-1; s^-1).
    b0          : float  Initial drug concentration (M).

    Returns
    -------
    EOT_upper : float
    EOT_lower : float
    """
    Kd        = k2 / k1
    EOT_upper = b0 / (Kd * k3)
    EOT_lower = b0 / ((b0 + Kd) * k3)
    return EOT_upper, EOT_lower


# ---------------------------------------------------------------------------
# Figure 2: High-resolution heatmaps
# ---------------------------------------------------------------------------

def generate_high_resolution_heatmaps():
    """
    Generate Figure 2: high-resolution heatmap analysis of EOT_infinity
    and its analytical bounds across the (k3/k2, b0/Kd) parameter plane.

    Panels A–F: heatmaps (40x40 computed, bicubic-interpolated to 200x200).
    Panels G–I: line plots for b0/Kd = 0.1, 1.0, 10.0.
    """
    # Base kinetic parameters
    k1 = 1e6    # M^-1 s^-1
    k2 = 1.0    # s^-1
    Kd = k2 / k1

    # Grid sizes
    n_coarse = 40    # points actually computed
    n_fine   = 200   # points after bicubic interpolation

    # Parameter ranges (log-spaced)
    k3_k2_coarse  = np.logspace(-2,  2,  n_coarse)
    b0_Kd_coarse  = np.logspace(-1,  1.7, n_coarse)
    k3_k2_fine    = np.logspace(-2,  2,  n_fine)
    b0_Kd_fine    = np.logspace(-1,  1.7, n_fine)

    # Storage arrays
    EOT_exact_mat = np.zeros((n_coarse, n_coarse))
    EOT_upper_mat = np.zeros((n_coarse, n_coarse))
    EOT_lower_mat = np.zeros((n_coarse, n_coarse))

    print(f"Computing EOT on {n_coarse}x{n_coarse} grid "
          f"(eps = a0/b0 = 0.1 throughout) ...")

    for i, k3_k2 in enumerate(k3_k2_coarse):
        if i % 10 == 0:
            print(f"  Row {i}/{n_coarse} ...")
        for j, b0_Kd in enumerate(b0_Kd_coarse):
            k3 = k3_k2 * k2
            b0 = b0_Kd * Kd
            # Pseudo-first-order validity: a0/b0 = 0.1 (eps = 0.1)
            a0 = 0.1 * Kd

            EOT_exact, _, _, _ = compute_EOT_converged(k1, k2, k3, a0, b0)
            EOT_upper, EOT_lower = compute_EOT_bounds(k1, k2, k3, b0)

            EOT_exact_mat[i, j] = EOT_exact
            EOT_upper_mat[i, j] = EOT_upper
            EOT_lower_mat[i, j] = EOT_lower

    # Bicubic interpolation in log-space for smooth visualisation
    print(f"Interpolating to {n_fine}x{n_fine} grid ...")
    log_k3_k2_c = np.log10(k3_k2_coarse)
    log_b0_Kd_c = np.log10(b0_Kd_coarse)
    log_k3_k2_f = np.log10(k3_k2_fine)
    log_b0_Kd_f = np.log10(b0_Kd_fine)

    def interpolate_log(mat):
        spl = RectBivariateSpline(
            log_k3_k2_c, log_b0_Kd_c,
            np.log10(mat + 1e-30), kx=3, ky=3, s=0
        )
        return 10 ** spl(log_k3_k2_f, log_b0_Kd_f)

    EOT_exact_fine = interpolate_log(EOT_exact_mat)
    EOT_upper_fine = interpolate_log(EOT_upper_mat)
    EOT_lower_fine = interpolate_log(EOT_lower_mat)

    # -------------------------------------------------------------------
    # Build figure
    # -------------------------------------------------------------------
    print("Rendering Figure 2 ...")
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    def _heatmap(ax, data, title, label, norm=None, cmap='viridis'):
        if norm is None:
            norm = LogNorm(vmin=data.min(), vmax=data.max())
        im = ax.pcolormesh(b0_Kd_fine, k3_k2_fine, data,
                           norm=norm, cmap=cmap,
                           shading='gouraud', rasterized=True)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('$b_0/K_d$'); ax.set_ylabel('$k_3/k_2$')
        ax.set_title(title, fontweight='bold')
        cb = plt.colorbar(im, ax=ax, label=label, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        return im

    # A: Converged EOT_inf
    _heatmap(fig.add_subplot(gs[0, 0]),
             EOT_exact_fine,
             r'A: Converged $\mathrm{EOT}_\infty$',
             r'$\mathrm{EOT}_\infty$ (s)')

    # B: Upper bound relative error
    rel_err_upper = np.abs(EOT_upper_fine - EOT_exact_fine) / EOT_exact_fine
    _heatmap(fig.add_subplot(gs[0, 1]),
             rel_err_upper,
             'B: Upper Bound Relative Error',
             'Relative Error',
             norm=LogNorm(vmin=0.01, vmax=10),
             cmap='RdYlBu_r')

    # C: Lower bound relative error
    rel_err_lower = np.abs(EOT_exact_fine - EOT_lower_fine) / EOT_exact_fine
    _heatmap(fig.add_subplot(gs[0, 2]),
             rel_err_lower,
             'C: Lower Bound Relative Error',
             'Relative Error',
             norm=LogNorm(vmin=0.01, vmax=10),
             cmap='RdYlBu_r')

    # D: Which bound is tighter (log ratio of errors)
    ax4 = fig.add_subplot(gs[1, 0])
    upper_dist  = np.abs(EOT_upper_fine - EOT_exact_fine)
    lower_dist  = np.abs(EOT_exact_fine - EOT_lower_fine)
    bound_ratio = np.log10(upper_dist / (lower_dist + 1e-30))
    norm4 = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    im4   = ax4.pcolormesh(b0_Kd_fine, k3_k2_fine, bound_ratio,
                           norm=norm4, cmap='coolwarm',
                           shading='gouraud', rasterized=True)
    ax4.set_xscale('log'); ax4.set_yscale('log')
    ax4.set_xlabel('$b_0/K_d$'); ax4.set_ylabel('$k_3/k_2$')
    ax4.set_title('D: Which Bound is Tighter', fontweight='bold')
    cb4 = plt.colorbar(im4, ax=ax4,
                       label=r'$\log_{10}$(Upper Error / Lower Error)', pad=0.02)
    cb4.ax.tick_params(labelsize=8)
    ax4.grid(True, alpha=0.15, linewidth=0.5)
    kw = dict(fontsize=8, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax4.text(0.05, 0.95, 'Lower\ntighter', transform=ax4.transAxes,
             color='darkblue', verticalalignment='top', **kw)
    ax4.text(0.85, 0.05, 'Upper\ntighter', transform=ax4.transAxes,
             color='darkred', **kw)

    # E: Relative bound width
    bound_width = (EOT_upper_fine - EOT_lower_fine) / EOT_exact_fine
    _heatmap(fig.add_subplot(gs[1, 1]),
             bound_width,
             'E: Relative Bound Width',
             '(Upper − Lower) / EOT',
             norm=LogNorm(vmin=bound_width.min(), vmax=bound_width.max()),
             cmap='YlOrRd')

    # F: Sensitivity d log(EOT) / d log(k3/k2)
    grad = np.gradient(np.log10(EOT_exact_fine), axis=0)
    _heatmap(fig.add_subplot(gs[1, 2]),
             np.abs(grad),
             r'F: Sensitivity to $k_3/k_2$',
             r'$|\partial\log(\mathrm{EOT})/\partial\log(k_3/k_2)|$',
             norm=LogNorm(),
             cmap='plasma')

    # G–I: Line plots for selected b0/Kd values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, b0_Kd_val in enumerate([0.1, 1.0, 10.0]):
        ax = fig.add_subplot(gs[2, idx])

        j_coarse = np.argmin(np.abs(b0_Kd_coarse - b0_Kd_val))
        EOT_sl  = EOT_exact_mat[:, j_coarse]
        up_sl   = EOT_upper_mat[:, j_coarse]
        lo_sl   = EOT_lower_mat[:, j_coarse]

        # Smooth curves via log-log cubic spline
        k3_smooth = np.logspace(-2, 2, 200)

        def _smooth(y_data):
            spl = make_interp_spline(
                np.log10(k3_k2_coarse), np.log10(y_data), k=3)
            return 10 ** spl(np.log10(k3_smooth))

        ax.loglog(k3_smooth, _smooth(EOT_sl), '-',
                  color=colors[idx], linewidth=2.5, label='Converged EOT')
        ax.loglog(k3_smooth, _smooth(up_sl), '--',
                  color='red', linewidth=1.5, alpha=0.7, label='Upper bound')
        ax.loglog(k3_smooth, _smooth(lo_sl), '--',
                  color='green', linewidth=1.5, alpha=0.7, label='Lower bound')
        ax.fill_between(k3_smooth, _smooth(lo_sl), _smooth(up_sl),
                        alpha=0.15, color='gray')
        ax.scatter(k3_k2_coarse[::5], EOT_sl[::5],
                   s=20, color=colors[idx], alpha=0.5, zorder=5)

        ax.set_xlabel('$k_3/k_2$')
        ax.set_ylabel(r'$\mathrm{EOT}_\infty$ (s)')
        ax.set_title(f'{chr(71+idx)}: $b_0/K_d$ = {b0_Kd_val}',
                     fontweight='bold')
        ax.set_xlim([0.01, 100])
        y_lo = min(EOT_sl.min(), lo_sl.min()) * 0.5
        y_hi = max(EOT_sl.max(), up_sl.max()) * 2
        ax.set_ylim([y_lo, y_hi])
        ax.legend(loc='best', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Parameter slices and time evolution
# ---------------------------------------------------------------------------

def generate_comparison_slices():
    """
    Generate Figure 3: systematic analysis of EOT behaviour along parameter
    slices and time-evolution panels.

    Panels A–C: parameter sweeps (solve_ivp RK45).
    Panels D–F: time evolution of EOT(t) (solve_ivp Radau, suitable for
                the stiff slow-clearance case k3/k2 = 0.01).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    k1 = 1e6;  k2 = 1.0
    Kd = k2 / k1
    # Pseudo-first-order validity: a0/b0 = 0.01 (eps = 0.01)
    a0 = 0.01 * Kd

    k3_k2_arr  = np.logspace(-2, 2, 100)
    b0_Kd_arr  = np.logspace(-1, 2, 100)
    colors_b0  = plt.cm.viridis(np.linspace(0.2, 0.9, 5))
    colors_k3  = plt.cm.plasma(np.linspace(0.2, 0.9, 5))

    # ------------------------------------------------------------------
    # Panel A: EOT_inf vs k3/k2 for different b0/Kd
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    for i, b0_Kd in enumerate([0.1, 0.5, 1.0, 5.0, 10.0]):
        b0   = b0_Kd * Kd
        eots = [compute_EOT_converged(k1, k2, k3_k2*k2, a0, b0,
                                      tol=1e-5)[0]
                for k3_k2 in k3_k2_arr]
        ax.loglog(k3_k2_arr, eots, linewidth=2,
                  color=colors_b0[i], label=f'$b_0/K_d$ = {b0_Kd}')
    ax.set_xlabel('$k_3/k_2$')
    ax.set_ylabel(r'$\mathrm{EOT}_\infty$ (s)')
    ax.set_title('A: Effect of Clearance Rate', fontweight='bold')
    ax.legend(loc='best', fontsize=8)

    # ------------------------------------------------------------------
    # Panel B: EOT_inf vs b0/Kd for different k3/k2
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    for i, k3_k2 in enumerate([0.01, 0.1, 1.0, 10.0, 100.0]):
        k3   = k3_k2 * k2
        eots = [compute_EOT_converged(k1, k2, k3, a0, b0_Kd*Kd,
                                      tol=1e-5)[0]
                for b0_Kd in b0_Kd_arr]
        ax.loglog(b0_Kd_arr, eots, linewidth=2,
                  color=colors_k3[i], label=f'$k_3/k_2$ = {k3_k2}')
    ax.set_xlabel('$b_0/K_d$')
    ax.set_ylabel(r'$\mathrm{EOT}_\infty$ (s)')
    ax.set_title('B: Effect of Drug Concentration', fontweight='bold')
    ax.legend(loc='best', fontsize=8)

    # ------------------------------------------------------------------
    # Panel C: Bound tightness (relative errors)
    # ------------------------------------------------------------------
    ax = axes[0, 2]
    for i, b0_Kd in enumerate([0.1, 1.0, 10.0]):
        b0  = b0_Kd * Kd
        up_err, lo_err = [], []
        for k3_k2 in k3_k2_arr:
            k3 = k3_k2 * k2
            EOT, _, _, _       = compute_EOT_converged(k1, k2, k3, a0, b0,
                                                        tol=1e-5)
            EOT_up, EOT_lo     = compute_EOT_bounds(k1, k2, k3, b0)
            up_err.append(abs(EOT_up - EOT) / EOT)
            lo_err.append(abs(EOT    - EOT_lo) / EOT)
        c = colors_b0[i * 2]
        ax.loglog(k3_k2_arr, up_err, '--', linewidth=1.5, color=c, alpha=0.7)
        ax.loglog(k3_k2_arr, lo_err, '-',  linewidth=2.0, color=c,
                  label=f'$b_0/K_d$ = {b0_Kd}')
    ax.set_xlabel('$k_3/k_2$')
    ax.set_ylabel('Relative Error')
    ax.set_title('C: Bound Tightness Analysis', fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.text(0.02, 0.98, 'Solid: Lower bound\nDashed: Upper bound',
            transform=ax.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ------------------------------------------------------------------
    # Panels D–F: Time evolution of EOT(t)
    # Radau solver is used here because the slow-clearance case
    # (k3/k2 = 0.01, so k2 >> k3) is stiff.
    # ------------------------------------------------------------------
    cases = [
        (0.01,  10.0, 'Slow clearance'),
        (1.0,   10.0, 'Moderate clearance'),
        (100.0, 10.0, 'Fast clearance'),
    ]

    for idx, (k3_k2, b0_Kd, title) in enumerate(cases):
        ax = axes[1, idx]
        k3     = k3_k2 * k2
        b0     = b0_Kd * Kd
        T_max  = 20.0 / min(k2, k3)
        t_eval = np.linspace(0, T_max, 1000)

        def ode_rhs_te(t, y, _k1=k1, _k2=k2, _k3=k3, _a0=a0, _b0=b0):
            c    = max(y[0], 0.0)
            b    = _b0 * np.exp(-_k3 * t)
            dcdt = _k1 * (_a0 - c) * b - _k2 * c
            return [dcdt]

        sol = solve_ivp(ode_rhs_te, [0, T_max], [0.0], method='Radau',
                        t_eval=t_eval, rtol=1e-10, atol=1e-12)
        t = sol.t
        f = np.maximum(sol.y[0] / a0, 0.0)

        # Cumulative EOT(t)
        EOT_cum = np.zeros_like(t)
        for ii in range(1, len(t)):
            EOT_cum[ii] = np.trapezoid(f[:ii+1], t[:ii+1])

        EOT_up, EOT_lo = compute_EOT_bounds(k1, k2, k3, b0)

        t_dim = t * min(k2, k3)       # dimensionless time axis
        ax.plot(t_dim, EOT_cum, 'b-', linewidth=2.5, label='EOT(t)')
        ax.axhline(EOT_up, color='r', linestyle='--', linewidth=1.5,
                   alpha=0.7, label='Upper bound')
        ax.axhline(EOT_lo, color='g', linestyle='--', linewidth=1.5,
                   alpha=0.7, label='Lower bound')
        ax.fill_between([0, t_dim[-1]], EOT_lo, EOT_up,
                        alpha=0.1, color='gray')

        ax.set_xlabel(r'Dimensionless time ($t \cdot \min(k_2,\,k_3)$)')
        ax.set_ylabel('EOT(t) (s)')
        ax.set_title(f'{chr(68+idx)}: {title}', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.text(0.98, 0.02,
                f'$k_3/k_2$ = {k3_k2}\n$b_0/K_d$ = {b0_Kd}',
                transform=ax.transAxes, fontsize=7, ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('\n' + '='*60)
    print('EOT FIGURE GENERATION  (Figures 2 and 3)')
    print('='*60 + '\n')

    print('Generating Figure 2: high-resolution heatmaps ...')
    fig1 = generate_high_resolution_heatmaps()
    fig1.savefig('EOT_heatmaps_high_res.pdf', dpi=300, bbox_inches='tight')
    fig1.savefig('EOT_heatmaps_high_res.png', dpi=300, bbox_inches='tight')
    print('  Saved: EOT_heatmaps_high_res.pdf / .png')

    print('\nGenerating Figure 3: parameter slices ...')
    fig2 = generate_comparison_slices()
    fig2.savefig('EOT_parameter_slices.pdf', dpi=300, bbox_inches='tight')
    fig2.savefig('EOT_parameter_slices.png', dpi=300, bbox_inches='tight')
    print('  Saved: EOT_parameter_slices.pdf / .png')

    print('\n' + '='*60)
    print('Done.  Four files written to the current directory.')
    print('='*60 + '\n')
