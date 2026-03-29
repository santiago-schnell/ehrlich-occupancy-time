# Ehrlich Occupancy Time — Reproducible Figures

Code to reproduce all numerical figures in:

> Eilertsen, J., Schnell, S. & Walcher, S. (2025).  
> **Ehrlich occupancy time: Beyond *k*_off to a complete residence time framework.**  
> *Journal of Pharmacokinetics and Pharmacodynamics.*  
> DOI: [to be added upon publication]

---

## What this repository contains

| File | Produces |
|------|----------|
| `figure_EOT_bounds.py` | Figure 2 (`EOT_heatmaps_high_res`) and Figure 3 (`EOT_parameter_slices`) |
| `figure_concentration_dependence.py` | Figure 4 (`FIG_concentration_dependence`) |
| `figure_induced_fit.py` | Figure 5 (`FIG_induced_fit`) |
| `requirements.txt` | Python package dependencies |

Figures 1 (conceptual schematic) is a hand-drawn illustration and is not
generated computationally.

---

## Quick start

### Step 1 — Check your Python version

Open a terminal and type:

```bash
python --version
```

You need **Python 3.10 or newer**.  If you have an older version, download
Python from <https://www.python.org/downloads/>.

---

### Step 2 — Download the code

**Option A — with Git (recommended):**

```bash
git clone https://github.com/schnell-lab/ehrlich-occupancy-time.git
cd ehrlich-occupancy-time
```

**Option B — without Git:**

Click the green **Code** button on the GitHub page, choose
**Download ZIP**, unzip the file, then open a terminal and navigate
into the unzipped folder:

```bash
cd ehrlich-occupancy-time-main   # folder name may vary
```

---

### Step 3 — Create an isolated environment (strongly recommended)

This prevents the packages needed here from conflicting with packages
you may already have installed.

```bash
python -m venv .venv
```

Activate the environment:

- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (Command Prompt):**
  ```bat
  .venv\Scripts\activate.bat
  ```
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

You will see `(.venv)` appear at the start of your command prompt,
confirming the environment is active.

---

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs NumPy, SciPy, Matplotlib, and Seaborn.  It takes
roughly one to two minutes on a typical internet connection.

---

### Step 5 — Generate the figures

Run each script from the repository folder.  Output files are written
to the **current directory**.

#### Figure 2 and Figure 3 (main numerical validation)

```bash
python figure_EOT_bounds.py
```

Expected output in the terminal:

```
============================================================
EOT FIGURE GENERATION  (Figures 2 and 3)
============================================================

Generating Figure 2: high-resolution heatmaps ...
  Computing EOT on 40x40 grid ...
  Row 0/40 ...
  Row 10/40 ...
  Row 20/40 ...
  Row 30/40 ...
  Interpolating to 200x200 grid ...
  Rendering Figure 2 ...
  Saved: EOT_heatmaps_high_res.pdf / .png

Generating Figure 3: parameter slices ...
  Saved: EOT_parameter_slices.pdf / .png

============================================================
Done.  Four files written to the current directory.
============================================================
```

> **Runtime:** approximately 5–15 minutes depending on your computer,
> because this script integrates over 1,600 parameter combinations
> on a 40×40 grid.

#### Figure 4 (concentration dependence)

```bash
python figure_concentration_dependence.py
```

Runs in under one second.  Produces `FIG_concentration_dependence.pdf`.

#### Figure 5 (induced fit)

```bash
python figure_induced_fit.py
```

Runs in under one second.  Produces `FIG_induced_fit.pdf`.

---

## Numerical methods

### ODE system

All scripts that solve differential equations use the reduced
pseudo-first-order ODE (equation 49 in the paper):

$$\dot{c} = -\bigl[k_1 b_0 e^{-k_3 t} + k_2\bigr]\, c
            + k_1 a_0 b_0 e^{-k_3 t}, \qquad c(0) = 0$$

where $b(t) = b_0 e^{-k_3 t}$ is the leading-order approximation for the
drug concentration under the pseudo-first-order condition
$\varepsilon = a_0/b_0 \ll 1$.

### Solvers

| Context | Solver | Tolerances |
|---------|--------|-----------|
| Parameter sweeps (Figures 2–3, panels A–C) | `scipy.integrate.solve_ivp`, method `RK45` (explicit Runge–Kutta 4/5) | rtol = atol = 10⁻¹⁰ |
| Time-evolution panels (Figure 3, panels D–F) | `scipy.integrate.solve_ivp`, method `Radau` (implicit, L-stable) | rtol = 10⁻¹⁰, atol = 10⁻¹² |

The `Radau` solver is used for the time-evolution panels because the
slow-clearance case ($k_3/k_2 = 0.01$) produces a stiff system where
$k_2 \gg k_3$ and the explicit `RK45` would require extremely small steps.

### EOT integral

The Ehrlich occupancy time integral

$$\mathrm{EOT}(T) = \int_0^T f(t)\,\mathrm{d}t, \qquad f(t) = c(t)/a_0$$

is evaluated from the numerical solution using **NumPy's trapezoidal
rule** (`numpy.trapezoid`).  For the parameter sweeps, integration
continues by doubling the time horizon until the running value changes by
less than $10^{-6}$ (relative), or until $f(t_\text{end}) < 10^{-8}$
(system fully dissociated).

### Analytical bounds

Figures 4 and 5 use closed-form analytical expressions only — no
numerical integration is involved.

---

## Software environment

The published figures were produced with:

| Package | Version |
|---------|---------|
| Python | 3.12 |
| NumPy | 2.4.2 |
| SciPy | 1.17.0 |
| Matplotlib | 3.10.8 |
| Seaborn | 0.13.2 |

**NumPy ≥ 2.0 is required** because the scripts use
`numpy.trapezoid`, which was introduced in NumPy 2.0 (June 2024).
If you have NumPy 1.x, replace every occurrence of `np.trapezoid`
with `np.trapz` (the legacy API, still available in NumPy 1.x).

---

## Repository structure

```
ehrlich-occupancy-time/
├── README.md
├── requirements.txt
├── .gitignore
├── figure_EOT_bounds.py                ← Figures 2 and 3
├── figure_concentration_dependence.py  ← Figure 4
└── figure_induced_fit.py               ← Figure 5
```

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{Eilertsen2025EOT,
  author  = {Eilertsen, Justin and Schnell, Santiago and Walcher, Sebastian},
  title   = {Ehrlich occupancy time: Beyond $k_{\rm off}$ to a complete
             residence time framework},
  journal = {Journal of Pharmacokinetics and Pharmacodynamics},
  year    = {2025},
  doi     = {to be added}
}
```

---

## Contact

For questions about the code, please open a
[GitHub Issue](../../issues).  
For questions about the mathematics or pharmacology, contact
Santiago Schnell at
[santiago.schnell@dartmouth.edu](mailto:santiago.schnell@dartmouth.edu).
