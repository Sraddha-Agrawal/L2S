import numpy as np
from pathlib import Path
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize
from numpy.linalg import lstsq
from scipy.optimize import root_scalar, minimize_scalar

#def load_bi_pes(input_dir="Input_files", x_saddle=0.23589477841686, energy_in_ev=0.004180159285619251):
#    energies = np.loadtxt(Path(input_dir) / "toten.dat")
#    x = np.linspace(-0.5, 0.9, len(energies))
#    x = x - x_saddle
#    bi_yinterp = InterpolatedUnivariateSpline(x, np.array(energies) / energy_in_ev, k=5)
#    neg_bi_yinterp = InterpolatedUnivariateSpline(x, -np.array(energies) / energy_in_ev, k=5)
#
#    bi_xmin = minimize(bi_yinterp, [0.0], bounds=[(-0.1-x_saddle, 0.1-x_saddle)]).x[0]
#    bi_xmax = minimize(neg_bi_yinterp, [0.0], bounds=[(-0.1, 0.1)]).x[0]
#    return x, bi_yinterp, bi_xmin, bi_xmax

def load_bi_pes(input_dir="Input_files",
                energy_in_ev=0.004180159285619251,
                x_range=(-0.5, 0.9),
                saddle_bracket=(0.05, 0.45)):
    """
    Load Bi PES and automatically shift coordinate so saddle is at x=0.

    Returns
    -------
    x_shifted : array
        shifted coordinate grid (Å)
    bi_yinterp : spline
        PES spline in INTERNAL energy units
    bi_xmin : float
        minimum position (shifted Å)
    bi_xmax : float
        saddle position (shifted Å, always 0)
    """

    # -----------------------------
    # Load PES
    # -----------------------------
    energies = np.loadtxt(Path(input_dir) / "toten.dat")
    x_raw = np.linspace(x_range[0], x_range[1], len(energies))

    # spline in INTERNAL energy units
    U_raw = InterpolatedUnivariateSpline(
        x_raw, energies / energy_in_ev, k=5
    )

    # -----------------------------
    # Find saddle automatically
    # Solve U'(x) = 0
    # -----------------------------
    res = root_scalar(
        lambda xx: float(U_raw(xx, nu=1)),
        bracket=saddle_bracket,
        method="brentq"
    )

    x_saddle = float(res.root)

    # -----------------------------
    # Shift coordinate so saddle = 0
    # -----------------------------
    x_shifted = x_raw - x_saddle

    bi_yinterp = InterpolatedUnivariateSpline(
        x_shifted, energies / energy_in_ev, k=5
    )

    # -----------------------------
    # Find minimum in shifted coords
    # -----------------------------
    res_min = minimize_scalar(
        lambda xx: float(bi_yinterp(xx)),
        bounds=(-0.6, -0.02),
        method="bounded"
    )

    bi_xmin = float(res_min.x)

    # saddle is exactly zero after shift
    bi_xmax = 0.0

    return x_shifted, bi_yinterp, bi_xmin, bi_xmax

def load_epsprime_poly(input_dir="Input_files/chi_files", all_pos2=None, x_grid=None, energy_in_ev=0.004180159285619251):
    if all_pos2 is None:
        all_pos2 = [4, 12, 16, 24, 28, 32, 38, 42, 48, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 68, 73, 80, 88, 96]

    baseDir = Path(input_dir)
    bi_eps_2eV = [
        row[1] + 1j * row[2]
        for i in all_pos2
        for row in np.loadtxt(baseDir / f"chi_{i}.dat", delimiter=" ")
        if 1.90 <= row[0] <= 2.05
    ]
    bi_eps_2eV = np.array(bi_eps_2eV)

    x = x_grid[all_pos2]
    y = bi_eps_2eV.real

    num = 9
    X_poly = np.column_stack([x**i for i in range(0, num, 2)])
    coefficients, _, _, _ = lstsq(X_poly, y, rcond=None)

    poly_derivative_func = lambda x: sum((2*i)*c*x**(2*i-1) for i, c in enumerate(coefficients[1:], start=1))
    return poly_derivative_func

def compute_bi_phfreq_in_thz(bi_yinterp, bi_xmin, bi_mass, freq_in_thz):
    bi_phomega = np.sqrt(bi_yinterp(bi_xmin, nu=2) / bi_mass)
    return bi_phomega * freq_in_thz / (2*np.pi)
