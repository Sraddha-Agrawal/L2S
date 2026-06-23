from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

from scipy.interpolate import InterpolatedUnivariateSpline, RBFInterpolator
from scipy.optimize import brentq


# ============================================================
# Bismuth parameters
# ============================================================
BI_MASS = 208.9804
ENERGY_IN_EV = 0.004180159285619251
FREQ_IN_THZ = 6.3507799295888985
T_IN_PS = 1.0 / FREQ_IN_THZ


# ============================================================
# Helpers
# ============================================================
def _read_vasp_energy(folder: Path) -> float:
    osz = folder / "OSZICAR"
    if not osz.exists():
        raise FileNotFoundError(f"Missing OSZICAR in {folder}")

    lines = osz.read_text().splitlines()
    for line in reversed(lines):
        if "F=" in line:
            parts = line.split()
            for i, tok in enumerate(parts):
                if tok == "F=" and i + 1 < len(parts):
                    return float(parts[i + 1])

    raise RuntimeError(f"Could not parse energy from {osz}")


def _build_old19_points(
    qA_min1: float,
    qA_max: float,
    qA_min2: float,
    qE_axis: float = 0.12,
    qE_diag: float = 0.08,
) -> list[tuple[float, float, float, str]]:
    pts: list[tuple[float, float, float, str]] = []
    qA_slices = [qA_min1, qA_max, qA_min2]

    for i, qA in enumerate(qA_slices):
        tag = f"A{i+1}"

        pts.append((qA, 0.00, 0.00, f"{tag}_center"))
        pts.append((qA, +qE_axis, 0.00, f"{tag}_Eg1_p"))
        pts.append((qA, -qE_axis, 0.00, f"{tag}_Eg1_m"))
        pts.append((qA, 0.00, +qE_axis, f"{tag}_Eg2_p"))
        pts.append((qA, 0.00, -qE_axis, f"{tag}_Eg2_m"))

        if i == 0 or i == 2:
            pts.append((qA, +qE_diag, +qE_diag, f"{tag}_diag_pp"))
            pts.append((qA, -qE_diag, -qE_diag, f"{tag}_diag_mm"))

    return pts


def _load_new62_summary(summary_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_file}")

    arr = np.loadtxt(summary_file, comments="#", usecols=(0, 1, 2, 3))
    qA = np.asarray(arr[:, 1], dtype=float)
    qE1 = np.asarray(arr[:, 2], dtype=float)
    qE2 = np.asarray(arr[:, 3], dtype=float)
    return qA, qE1, qE2


def _find_stationary_points_1d(
    qA_1d: np.ndarray,
    E_1d_eV: np.ndarray,
) -> tuple[float, float, float, InterpolatedUnivariateSpline]:
    order = np.argsort(qA_1d)
    qA_1d = np.asarray(qA_1d[order], dtype=float)
    E_1d_eV = np.asarray(E_1d_eV[order], dtype=float)

    k = min(3, len(qA_1d) - 1)
    spl = InterpolatedUnivariateSpline(qA_1d, E_1d_eV, k=k)
    dspl = spl.derivative(1)
    d2spl = spl.derivative(2)

    qscan = np.linspace(qA_1d.min(), qA_1d.max(), 4000)
    dscan = dspl(qscan)

    roots = []
    for i in range(len(qscan) - 1):
        x1, x2 = qscan[i], qscan[i + 1]
        y1, y2 = dscan[i], dscan[i + 1]

        if abs(y1) < 1e-12:
            roots.append(x1)
        elif y1 * y2 < 0:
            try:
                roots.append(brentq(dspl, x1, x2))
            except ValueError:
                pass

    roots = np.array(sorted(set(np.round(roots, 10))), dtype=float)

    minima = []
    maxima = []
    for r in roots:
        curv = float(d2spl(r))
        if curv > 0:
            minima.append((float(r), float(spl(r))))
        elif curv < 0:
            maxima.append((float(r), float(spl(r))))

    if len(minima) < 2:
        raise RuntimeError("Could not find two minima from the 1D A1g slice.")

    minima = sorted(minima, key=lambda x: x[0])
    qA_min1 = minima[0][0]
    qA_min2 = minima[-1][0]

    maxima_between = [(r, e) for (r, e) in maxima if qA_min1 < r < qA_min2]
    if len(maxima_between) == 0:
        raise RuntimeError("Could not find the central maximum from the 1D A1g slice.")

    qA_max = sorted(maxima_between, key=lambda x: x[1], reverse=True)[0][0]
    return qA_min1, qA_max, qA_min2, spl


@dataclass
class PES3D:
    rbf_internal: RBFInterpolator
    qA_saddle_raw: float
    qA_min1_shift: float
    qA_min2_shift: float
    x_scale: float
    bi_mass: float = BI_MASS
    energy_in_ev: float = ENERGY_IN_EV
    freq_in_thz: float = FREQ_IN_THZ

    def _as_points(self, q: Iterable[float] | np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        arr = np.asarray(q, dtype=float)
        if arr.shape == (3,):
            pts = arr.reshape(1, 3)
            out_shape = ()
        else:
            arr = np.asarray(arr, dtype=float)
            if arr.shape[-1] != 3:
                raise ValueError("Input q must have shape (3,) or (..., 3).")
            out_shape = arr.shape[:-1]
            pts = arr.reshape(-1, 3)
        return pts, out_shape

    def energy_internal(self, q: Iterable[float] | np.ndarray) -> np.ndarray | float:
        pts, out_shape = self._as_points(q)
        vals = self.rbf_internal(pts)
        vals = vals.reshape(out_shape) if out_shape != () else float(vals[0])
        return vals

    def energy_eV(self, q: Iterable[float] | np.ndarray) -> np.ndarray | float:
        return self.energy_internal(q) * self.energy_in_ev

    def gradient_internal(
        self,
        q: Iterable[float] | np.ndarray,
        h: float = 1.0e-2,
    ) -> np.ndarray:
        pts, out_shape = self._as_points(q)

        e = np.eye(3, dtype=float)
        g = np.empty_like(pts)

        for i in range(3):
            qp = pts + h * e[i]
            qm = pts - h * e[i]
            up = self.rbf_internal(qp)
            um = self.rbf_internal(qm)
            g[:, i] = (up - um) / (2.0 * h)

        return g.reshape((3,)) if out_shape == () else g.reshape(out_shape + (3,))

    def gradient_eV(
        self,
        q: Iterable[float] | np.ndarray,
        h: float = 1.0e-2,
    ) -> np.ndarray:
        return self.gradient_internal(q, h=h) * self.energy_in_ev

    def force_internal(
        self,
        q: Iterable[float] | np.ndarray,
        h: float = 1.0e-2,
    ) -> np.ndarray:
        return -self.gradient_internal(q, h=h)

    def force_eV_per_A(
        self,
        q: Iterable[float] | np.ndarray,
        h: float = 1.0e-2,
    ) -> np.ndarray:
        return -self.gradient_eV(q, h=h)

    @property
    def bi_xmax(self) -> float:
        return 0.0

    @property
    def bi_xmin(self) -> float:
        # left minimum in shifted qA coordinate
        return self.qA_min1_shift

    @property
    def bi_phomega(self) -> float:
        q0 = np.array([self.qA_min1_shift, 0.0, 0.0], dtype=float)
        h = 1.0e-2
        e = np.array([1.0, 0.0, 0.0], dtype=float)
        upp = (
            self.energy_internal(q0 + h * e)
            - 2.0 * self.energy_internal(q0)
            + self.energy_internal(q0 - h * e)
        ) / (h * h)
        return float(np.sqrt(max(upp, 0.0) / self.bi_mass))

    @property
    def bi_phfreq_in_thz(self) -> float:
        return self.bi_phomega * self.freq_in_thz / (2.0 * np.pi)


def build_bi_pes3d_dft(
    pes_1d_path: str | Path = "/Users/sagrawal/Box/Research/ANL-Research/ML_script/Input_files/toten.dat",
    base19: str | Path = "/Users/sagrawal/Box/Research/ANL-Research/Carbon-data/Bi/single-point/Bi_rev/run",
    base62: str | Path = "/Users/sagrawal/Box/Research/ANL-Research/Carbon-data/Bi/single-point/Bi_rev/run-2",
    summary62: str | Path = "/Users/sagrawal/Box/Research/ANL-Research/Carbon-data/Bi/single-point/finer-grid/analysis-plot-dielec/Bi_DFT_PES_81pt_new62/selected_modal_points_new62.dat",
    x_raw_min: float = -0.5,
    x_raw_max: float = 0.9,
    qE_axis: float = 0.12,
    qE_diag: float = 0.08,
    rbf_kernel: str = "thin_plate_spline",
    rbf_degree: int = 1,
) -> PES3D:
    pes_1d_path = Path(pes_1d_path)
    base19 = Path(base19)
    base62 = Path(base62)
    summary62 = Path(summary62)

    if not pes_1d_path.exists():
        raise FileNotFoundError(f"Missing 1D DFT PES file: {pes_1d_path}")

    # 1D slice
    E_1d_abs = np.loadtxt(pes_1d_path, dtype=float)
    E_1d_abs = np.asarray(E_1d_abs, dtype=float).ravel()
    qA_1d_raw = np.linspace(x_raw_min, x_raw_max, len(E_1d_abs))

    qA_min1_raw, qA_saddle_raw, qA_min2_raw, _ = _find_stationary_points_1d(qA_1d_raw, E_1d_abs)

    # old 19 points: reconstruct coordinates + read energies
    pts19 = _build_old19_points(
        qA_min1=qA_min1_raw,
        qA_max=qA_saddle_raw,
        qA_min2=qA_min2_raw,
        qE_axis=qE_axis,
        qE_diag=qE_diag,
    )
    qA_19_raw = np.array([p[0] for p in pts19], dtype=float)
    qE1_19 = np.array([p[1] for p in pts19], dtype=float)
    qE2_19 = np.array([p[2] for p in pts19], dtype=float)
    E_19_abs = np.array([_read_vasp_energy(base19 / f"pos{i:02d}") for i in range(1, 20)], dtype=float)

    # new 62 points
    qA_62_raw, qE1_62, qE2_62 = _load_new62_summary(summary62)
    E_62_abs = np.array([_read_vasp_energy(base62 / f"pos{i:02d}") for i in range(1, 63)], dtype=float)

    if len(qA_62_raw) != 62:
        raise RuntimeError(f"Expected 62 new points, got {len(qA_62_raw)}.")

    # common zero from all DFT data
    E0_abs = min(np.min(E_1d_abs), np.min(E_19_abs), np.min(E_62_abs))

    # shift qA so saddle is exactly at 0
    qA_1d = qA_1d_raw - qA_saddle_raw
    qA_19 = qA_19_raw - qA_saddle_raw
    qA_62 = qA_62_raw - qA_saddle_raw

    # convert to internal units
    # U_1d_internal = (E_1d_abs - E0_abs) / ENERGY_IN_EV
    # U_19_internal = (E_19_abs - E0_abs) / ENERGY_IN_EV
    # U_62_internal = (E_62_abs - E0_abs) / ENERGY_IN_EV

    U_1d_internal = (E_1d_abs) / ENERGY_IN_EV
    U_19_internal = (E_19_abs) / ENERGY_IN_EV
    U_62_internal = (E_62_abs) / ENERGY_IN_EV

    # combine 81 3D points + full 1D anchors
    qA_all = np.concatenate([qA_19, qA_62, qA_1d])
    qE1_all = np.concatenate([qE1_19, qE1_62, np.zeros_like(qA_1d)])
    qE2_all = np.concatenate([qE2_19, qE2_62, np.zeros_like(qA_1d)])
    U_all = np.concatenate([U_19_internal, U_62_internal, U_1d_internal])

    pts_all = np.column_stack([qA_all, qE1_all, qE2_all])

    rbf_internal = RBFInterpolator(
        pts_all,
        U_all,
        kernel=rbf_kernel,
        degree=rbf_degree,
    )

    qA_min1_shift = qA_min1_raw - qA_saddle_raw
    qA_min2_shift = qA_min2_raw - qA_saddle_raw
    x_scale = abs(qA_min1_shift)

    pes = PES3D(
        rbf_internal=rbf_internal,
        qA_saddle_raw=qA_saddle_raw,
        qA_min1_shift=qA_min1_shift,
        qA_min2_shift=qA_min2_shift,
        x_scale=x_scale,
    )

    return pes


if __name__ == "__main__":
    pes = build_bi_pes3d_dft()

    print("qA_saddle_raw (A) =", pes.qA_saddle_raw)
    print("bi_xmax (shifted) =", pes.bi_xmax)
    print("bi_xmin (shifted) =", pes.bi_xmin)
    print("qA_min2_shift (A) =", pes.qA_min2_shift)
    print("x_scale (A)       =", pes.x_scale)
    print("bi_phfreq_in_thz  =", pes.bi_phfreq_in_thz)

    q0 = np.array([0.0, 0.0, 0.0])
    print("\nAt saddle:")
    print("U_internal =", pes.energy_internal(q0))
    print("grad_int   =", pes.gradient_internal(q0))