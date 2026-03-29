#!/usr/bin/env python3
"""
figure3_EOT_parameter_slices.py
================================
Generates Figure 3 from:

  Eilertsen, J., Schnell, S. & Walcher, S. (2025).
  "Ehrlich occupancy time: Beyond k_off to a complete residence time framework."
  Journal of Pharmacokinetics and Pharmacodynamics.

Figure 3 (EOT_parameter_slices.pdf / .png):
  Systematic analysis of EOT behaviour and bound quality.

  Panel A — EOT_infinity vs k3/k2 for different b0/Kd (clearance effect).
  Panel B — EOT_infinity vs b0/Kd for different k3/k2 (concentration effect).
  Panel C — Bound tightness: relative errors of upper and lower bounds.
  Panel D — Time evolution EOT(t), slow clearance   (k3/k2 = 0.01).
  Panel E — Time evolution EOT(t), moderate clearance (k3/k2 = 1.0).
  Panel F — Time evolution EOT(t), fast clearance    (k3/k2 = 100).

Numerical methods
-----------------
Panels A–C: scipy.integrate.solve_ivp, method 'RK45', rtol = atol = 1e-10.

Panels D–F: scipy.integrate.solve_ivp, method 'Radau' (implicit, L-stable).
  The slow-clearance case (k3/k2 = 0.01) is stiff because k2 >> k3;
  Radau is more efficient and accurate for such problems.
  rtol = 1e-10, atol = 1e-12.

The EOT(t) integral is computed via numpy.trapezoid (trapezoidal rule).

Requirements
------------
See requirements.txt.  Python >= 3.10.

Usage
-----
    python figure3_EOT_parameter_slices.py

Output files (written to the current directory):
    EOT_parameter_slices.pdf
    EOT_parameter_slices.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
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
# Core computation functions (shared with figure2_EOT_heatmaps.py)
# ---------------------------------------------------------------------------

def compute_EOT_converged(k1, k2, k3, a0, b0, tol=1e-6, max_T_factor=100):
    """
    Compute EOT_infinity by integrating the pseudo-first-order ODE
    (equation 49 in the paper) until the running integral converges.

        dc/dt = -[k1*b0*exp(-k3*t) + k2]*c + k1*a0*b0*exp(-k3*t),  c(0)=0

    Parameters
    ----------
    k1, k2, k3 : float  Rate constants (M^-1 s^-1; s^-1; s^-1).
    a0, b0      : float  Initial concentrations (M).
    tol         : float  Relative convergence tolerance (default 1e-6).
    max_T_factor: int    Maximum time horizon multiple (default 100).

    Returns
    -------
    EOT_current : float
    t           : ndarray
    f           : ndarray  Fractional occupancy f(t) = c(t)/a0.
    converged   : bool
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
        t = sol.t;  c = sol.y[0];  f = c / a0
        EOT_current = np.trapezoid(f, t)

        if EOT_previous is not None:
            if abs(EOT_current - EOT_previous) / max(EOT_current, 1e-30) < tol:
                converged = True;  break

        if f[-1] < 1e-8:
            converged = True;  break

        EOT_previous = EOT_current
        T_max       *= 2.0
        if T_max > max_T_factor * time_scale:
            break

    return EOT_current, t, f, converged


def compute_EOT_bounds(k1, k2, k3, b0):
    """
    Analytical bounds for EOT_infinity (equations 59 and 63 in the paper).

    Returns
    -------
    EOT_upper, EOT_lower : float
    """
    Kd = k2 / k1
    return b0 / (Kd * k3),  b0 / ((b0 + Kd) * k3)


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------

def generate_figure3():
    """
    Build Figure 3: parameter slices and time-evolution panels.
    Returns the matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    k1 = 1e6;  k2 = 1.0
    Kd = k2 / k1
    # Pseudo-first-order validity: eps = a0/b0 = 0.01
    a0 = 0.01 * Kd

    k3_k2_arr  = np.logspace(-2, 2, 100)
    b0_Kd_arr  = np.logspace(-1, 2, 100)
    colors_b0  = plt.cm.viridis(np.linspace(0.2, 0.9, 5))
    colors_k3  = plt.cm.plasma(np.linspace(0.2, 0.9, 5))

    # ------------------------------------------------------------------
    # Panel A: EOT_inf vs k3/k2 for several b0/Kd values
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    for i, b0_Kd in enumerate([0.1, 0.5, 1.0, 5.0, 10.0]):
        b0   = b0_Kd * Kd
        eots = [compute_EOT_converged(k1, k2, r*k2, a0, b0, tol=1e-5)[0]
                for r in k3_k2_arr]
        ax.loglog(k3_k2_arr, eots, linewidth=2,
                  color=colors_b0[i], label=f'$b_0/K_d$ = {b0_Kd}')
    ax.set_xlabel('$k_3/k_2$')
    ax.set_ylabel(r'$\mathrm{EOT}_\infty$ (s)')
    ax.set_title('A: Effect of Clearance Rate', fontweight='bold')
    ax.legend(loc='best', fontsize=8)

    # ------------------------------------------------------------------
    # Panel B: EOT_inf vs b0/Kd for several k3/k2 values
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    for i, k3_k2 in enumerate([0.01, 0.1, 1.0, 10.0, 100.0]):
        k3   = k3_k2 * k2
        eots = [compute_EOT_converged(k1, k2, k3, a0, r*Kd, tol=1e-5)[0]
                for r in b0_Kd_arr]
        ax.loglog(b0_Kd_arr, eots, linewidth=2,
                  color=colors_k3[i], label=f'$k_3/k_2$ = {k3_k2}')
    ax.set_xlabel('$b_0/K_d$')
    ax.set_ylabel(r'$\mathrm{EOT}_\infty$ (s)')
    ax.set_title('B: Effect of Drug Concentration', fontweight='bold')
    ax.legend(loc='best', fontsize=8)

    # ------------------------------------------------------------------
    # Panel C: Bound tightness (relative errors of upper and lower bounds)
    # ------------------------------------------------------------------
    ax = axes[0, 2]
    for i, b0_Kd in enumerate([0.1, 1.0, 10.0]):
        b0 = b0_Kd * Kd
        up_err, lo_err = [], []
        for k3_k2 in k3_k2_arr:
            k3 = k3_k2 * k2
            EOT, _, _, _   = compute_EOT_converged(k1, k2, k3, a0, b0, tol=1e-5)
            EOT_up, EOT_lo = compute_EOT_bounds(k1, k2, k3, b0)
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
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ------------------------------------------------------------------
    # Panels D–F: Time evolution of EOT(t)
    # Radau is used because the slow-clearance case (k3/k2 = 0.01)
    # is stiff (k2 >> k3); Radau handles this efficiently.
    # ------------------------------------------------------------------
    cases = [
        (0.01,  10.0, 'Slow clearance'),
        (1.0,   10.0, 'Moderate clearance'),
        (100.0, 10.0, 'Fast clearance'),
    ]

    for idx, (k3_k2, b0_Kd, title) in enumerate(cases):
        ax    = axes[1, idx]
        k3    = k3_k2 * k2
        b0    = b0_Kd * Kd
        T_max = 20.0 / min(k2, k3)

        def ode_rhs_te(t, y,
                       _k1=k1, _k2=k2, _k3=k3, _a0=a0, _b0=b0):
            c    = max(y[0], 0.0)
            b    = _b0 * np.exp(-_k3 * t)
            dcdt = _k1 * (_a0 - c) * b - _k2 * c
            return [dcdt]

        sol = solve_ivp(ode_rhs_te, [0, T_max], [0.0],
                        method='Radau',
                        t_eval=np.linspace(0, T_max, 1000),
                        rtol=1e-10, atol=1e-12)
        t = sol.t
        f = np.maximum(sol.y[0] / a0, 0.0)

        # Cumulative EOT(t)
        EOT_cum = np.zeros_like(t)
        for ii in range(1, len(t)):
            EOT_cum[ii] = np.trapezoid(f[:ii+1], t[:ii+1])

        EOT_up, EOT_lo = compute_EOT_bounds(k1, k2, k3, b0)
        t_dim = t * min(k2, k3)    # dimensionless time axis

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
    print('FIGURE 3 — EOT parameter slices and time evolution')
    print('='*60 + '\n')
    fig = generate_figure3()
    fig.savefig('EOT_parameter_slices.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('EOT_parameter_slices.png', dpi=300, bbox_inches='tight')
    print('\nSaved: EOT_parameter_slices.pdf / .png')
    print('='*60 + '\n')
