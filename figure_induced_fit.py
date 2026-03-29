#!/usr/bin/env python3
"""
figure_induced_fit.py
=====================
Generates Figure 5 from:

  Eilertsen, J., Schnell, S. & Walcher, S. (2025).
  "Ehrlich occupancy time: Beyond k_off to a complete residence time framework."
  Journal of Pharmacokinetics and Pharmacodynamics.

Figure 5 (FIG_induced_fit.pdf):
  Effect of the induced-fit isomerisation ratio k3/k4 on fractional
  target occupancy f_infinity.

Method
------
f_infinity is computed analytically using equation (38) in the paper:

    Kd_star = Kd / (1 + k3/k4)
    f_inf   = b0 / (Kd_star + b0)

No ODE integration is required for this figure.

Usage
-----
    python figure_induced_fit.py

Output
------
    FIG_induced_fit.pdf   (Figure 5)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'serif'


# ---------------------------------------------------------------------------
# Analytical result (eq. 38)
# ---------------------------------------------------------------------------

def f_infinity_induced_fit(b0, k3_k4_ratio, Kd):
    """
    Equilibrium fractional occupancy for the induced-fit mechanism
    (equation 38 in the paper).

        Kd_star = Kd / (1 + k3/k4)
        f_inf   = b0 / (Kd_star + b0)

    Parameters
    ----------
    b0           : float       Drug concentration (same units as Kd).
    k3_k4_ratio  : array-like  Forward/reverse isomerisation ratio k3/k4.
    Kd           : float       Intrinsic dissociation constant.

    Returns
    -------
    ndarray  Fractional occupancy (dimensionless, in [0, 1]).
    """
    ratio   = np.asarray(k3_k4_ratio)
    Kd_star = Kd / (1.0 + ratio)
    return b0 / (Kd_star + b0)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def generate_figure():
    """Build and return Figure 5."""
    Kd = 1.0   # µM (arbitrary units; figure is dimensionless in b0/Kd)

    b0_vals = [0.1 * Kd, 1.0 * Kd, 10.0 * Kd]
    colors  = ['blue', 'orange', 'green']
    labels  = ['$b_0 = 0.1K_d$', '$b_0 = K_d$', '$b_0 = 10K_d$']

    # Isomerisation ratio axis: 0.01 to 100 (four orders of magnitude)
    k3_k4 = np.logspace(-2, 2, 200)

    fig, ax = plt.subplots(figsize=(10, 6))

    for b0, color, label in zip(b0_vals, colors, labels):
        ax.plot(k3_k4, f_infinity_induced_fit(b0, k3_k4, Kd),
                color=color, label=label, linewidth=2)

    # Dashed reference: no induced fit (k3/k4 → 0), b0 = Kd → f_inf = 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5,
               label=r'No induced fit ($k_3=0,\; b_0=K_d$)')

    ax.set_xscale('log')
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('$k_3/k_4$', fontsize=12)
    ax.set_ylabel('$f_{\\infty}$', fontsize=12)
    ax.legend()
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = generate_figure()
    fig.savefig('FIG_induced_fit.pdf')
    print("Saved: FIG_induced_fit.pdf")
