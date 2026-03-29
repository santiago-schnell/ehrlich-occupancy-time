#!/usr/bin/env python3
"""
figure_concentration_dependence.py
===================================
Generates Figure 4 from:

  Eilertsen, J., Schnell, S. & Walcher, S. (2025).
  "Ehrlich occupancy time: Beyond k_off to a complete residence time framework."
  Journal of Pharmacokinetics and Pharmacodynamics.

Figure 4 (FIG_concentration_dependence.pdf):
  Predicted concentration-dependence of EOT_infinity.
  Shows how EOT_inf varies with drug concentration b0 for three elimination
  rates k3, illustrating the transition from linear to plateau behaviour
  near b0 = Kd.

Method
------
EOT_infinity is computed analytically using the lower bound formula
(equation 63 in the paper):

    EOT_inf_lower = b0 / ((b0 + Kd) * k3)

This is the exact expression used in the figure; no ODE integration is
required for this plot.

Usage
-----
    python figure_concentration_dependence.py

Output
------
    FIG_concentration_dependence.pdf   (Figure 4)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'serif'


# ---------------------------------------------------------------------------
# Analytical bound (eq. 63)
# ---------------------------------------------------------------------------

def eot_infinity_lower(b0, Kd, k3):
    """
    Lower bound for EOT_infinity (eq. 63 in the paper).

        EOT_inf_lower = b0 / ((b0 + Kd) * k3)

    Parameters
    ----------
    b0 : array-like  Initial drug concentration (M).
    Kd : float       Dissociation constant (M).
    k3 : float       Drug elimination rate constant (s^-1).

    Returns
    -------
    ndarray  Lower bound values (s).
    """
    return np.asarray(b0) / ((np.asarray(b0) + Kd) * k3)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def generate_figure():
    """Build and return Figure 4."""
    # Parameters (matching paper Section 4.7 / Appendix B)
    k1 = 1e6          # M^-1 s^-1
    k2 = 0.1          # s^-1
    Kd_M  = k2 / k1   # Molar  (1e-7 M)
    Kd_uM = Kd_M * 1e6  # µM  (0.1 µM)

    k3_values = [0.01, 0.1, 1.0]   # s^-1
    colors    = ['blue', 'orange', 'green']
    markers   = ['v', 'o', '^']    # down-triangle, circle, up-triangle
    labels    = [f'$k_3 = {k3}$ s$^{{-1}}$' for k3 in k3_values]

    # Drug concentration axis: 1 nM to 10 µM, 20 log-spaced points
    b0_M  = np.logspace(-9, -5, 20)
    b0_uM = b0_M * 1e6

    fig, ax = plt.subplots(figsize=(10, 6))

    for k3, color, label, marker in zip(k3_values, colors, labels, markers):
        eot = eot_infinity_lower(b0_M, Kd_M, k3)
        ax.scatter(b0_uM, eot,
                   color=color, label=label, marker=marker, s=50)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$b_0$ ($\mu$M)', fontsize=12)
    ax.set_ylabel(r'$\mathrm{EOT}_\infty$ (s)', fontsize=12)
    ax.legend()
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = generate_figure()
    fig.savefig('FIG_concentration_dependence.pdf')
    print("Saved: FIG_concentration_dependence.pdf")
