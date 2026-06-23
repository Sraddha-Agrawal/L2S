import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline


def sample_field_on_grid(field, tfine):
    tfine = np.asarray(tfine, dtype=float)
    return np.asarray([field(float(t)) for t in tfine], dtype=float)

def build_field(net, tfine, mode="spline", spline_k=5, spline_n=400):
    t0, t1 = float(tfine[0]), float(tfine[-1])

    if mode == "spline":
        t_coarse = np.linspace(t0, t1, spline_n, dtype=float)
        A = np.asarray(net.get_output_batch(t_coarse), dtype=float).reshape(-1)

        spl = InterpolatedUnivariateSpline(t_coarse, A, k=spline_k)

        # norm computed on tfine via spline (consistent with field(t))
        A_fine = np.asarray(spl(tfine), dtype=float).reshape(-1)
        nn_norm = float(np.trapezoid(A_fine * A_fine, tfine))
        nn_norm = max(nn_norm, 1e-12)

        def field(t):
            return float(spl(float(t)))

        return field, nn_norm, (t_coarse, A)

    # fallback: evaluate on dense grid and interpolate linearly
    A_dense = np.asarray(net.get_output_batch(tfine), dtype=float).reshape(-1)
    nn_norm = float(np.trapezoid(A_dense * A_dense, tfine))
    nn_norm = max(nn_norm, 1e-12)

    def field(t):
        return float(np.interp(float(t), tfine, A_dense))

    return field, nn_norm, (tfine, A_dense)


def _initial_state(x0, v0, bi_xmin, ic_mode):
    if ic_mode not in ("absolute", "offset"):
        raise ValueError("ic_mode must be 'absolute' or 'offset'")

    if ic_mode == "absolute":
        return float(x0), float(v0)

    return float(bi_xmin) + float(x0), float(v0)


def _force_total(x, t, field, nn_norm, pes_interpolant, epsprime_interpolant, alpha):
    """
    Total deterministic force:
        F(x,t) = - dV/dx - dV_field/dx

    where
        dV_field/dx = epsprime(x) * alpha * A(t)^2 / nn_norm
    """
    x = float(x)
    t = float(t)

    A = float(field(t))
    drive = float(epsprime_interpolant(x)) * (alpha * (A * A) / nn_norm)
    dVdx = float(pes_interpolant(x, nu=1))

    return -(dVdx + drive)


def _rhs_deterministic(
    t, u,
    field, nn_norm,
    pes_interpolant, epsprime_interpolant,
    mode_mass, alpha, damping
):
    x, v = float(u[0]), float(u[1])

    force = _force_total(
        x=x,
        t=t,
        field=field,
        nn_norm=nn_norm,
        pes_interpolant=pes_interpolant,
        epsprime_interpolant=epsprime_interpolant,
        alpha=alpha,
    )

    dxdt = v
    dvdt = force / float(mode_mass) - float(damping) * v
    return np.array([dxdt, dvdt], dtype=float)


class SimpleSolution:
    """
    Lightweight solve_ivp-like container for BAOAB trajectories.
    """
    def __init__(self, t, x, v, success=True, message=""):
        self.t = np.asarray(t, dtype=float)
        self.y = np.vstack([np.asarray(x, dtype=float), np.asarray(v, dtype=float)])
        self.success = bool(success)
        self.message = str(message)


def _simulate_ic_deterministic(
    field, nn_norm,
    tspan, tfine,
    x_init, v_init,
    pes_interpolant, epsprime_interpolant,
    mode_mass,
    alpha=8.75,
    method="Radau", rtol=1e-3, atol=1e-6,
    damping=0.05,
):
    y0 = [float(x_init), float(v_init)]

    rhs = lambda t, u: _rhs_deterministic(
        t, u,
        field=field,
        nn_norm=nn_norm,
        pes_interpolant=pes_interpolant,
        epsprime_interpolant=epsprime_interpolant,
        mode_mass=mode_mass,
        alpha=alpha,
        damping=damping,
    )

    sol = solve_ivp(
        rhs,
        tspan,
        y0,
        t_eval=tfine,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    return sol


def _simulate_ic_baoab(
    field, nn_norm,
    tfine,
    x_init, v_init,
    pes_interpolant, epsprime_interpolant,
    mode_mass,
    alpha=8.75,
    gamma=0.05,
    kBT_internal=0.0,
    rng=None,
):
    """
    BAOAB Langevin integrator for 1D driven dynamics.

    Equation:
        m x'' = F_det(x,t) - m*gamma*v + thermal noise

    BAOAB splitting:
        B: half deterministic kick
        A: half drift
        O: exact Ornstein-Uhlenbeck thermostat step
        A: half drift
        B: half deterministic kick

    Notes
    -----
    - x is in Angstrom
    - v is in Angstrom / internal_time
    - mode_mass is in the same mass unit used by the deterministic dynamics
    - kBT_internal must be in the SAME internal energy units as the PES
    """
    tfine = np.asarray(tfine, dtype=float)
    if tfine.ndim != 1 or len(tfine) < 2:
        raise ValueError("tfine must be a 1D array with at least 2 points")

    if rng is None:
        rng = np.random.default_rng()

    m = float(mode_mass)
    gamma = float(gamma)
    kBT_internal = float(kBT_internal)

    if m <= 0:
        raise ValueError(f"mode_mass must be positive, got {m}")
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    if kBT_internal < 0:
        raise ValueError(f"kBT_internal must be >= 0, got {kBT_internal}")

    n = len(tfine)
    x = np.empty(n, dtype=float)
    v = np.empty(n, dtype=float)

    x[0] = float(x_init)
    v[0] = float(v_init)

    for i in range(n - 1):
        t = float(tfine[i])
        dt = float(tfine[i + 1] - tfine[i])

        if dt <= 0:
            raise ValueError("tfine must be strictly increasing")

        # ---- B: half kick using force at (x_n, t_n)
        f_n = _force_total(
            x=x[i],
            t=t,
            field=field,
            nn_norm=nn_norm,
            pes_interpolant=pes_interpolant,
            epsprime_interpolant=epsprime_interpolant,
            alpha=alpha,
        )
        v_half = v[i] + 0.5 * dt * (f_n / m)

        # ---- A: half drift
        x_half = x[i] + 0.5 * dt * v_half

        # ---- O: exact thermostat step
        if gamma > 0.0 and kBT_internal > 0.0:
            c1 = np.exp(-gamma * dt)
            c2 = np.sqrt((kBT_internal / m) * (1.0 - c1 * c1))
            v_ou = c1 * v_half + c2 * rng.normal()
        elif gamma > 0.0 and kBT_internal == 0.0:
            # pure damping, zero-temperature limit
            c1 = np.exp(-gamma * dt)
            v_ou = c1 * v_half
        else:
            # gamma == 0 => no thermostat action
            v_ou = v_half

        # ---- A: second half drift
        x_new = x_half + 0.5 * dt * v_ou

        # ---- B: second half kick using force at (x_{n+1}, t_{n+1})
        f_np1 = _force_total(
            x=x_new,
            t=tfine[i + 1],
            field=field,
            nn_norm=nn_norm,
            pes_interpolant=pes_interpolant,
            epsprime_interpolant=epsprime_interpolant,
            alpha=alpha,
        )
        v_new = v_ou + 0.5 * dt * (f_np1 / m)

        x[i + 1] = x_new
        v[i + 1] = v_new

    return SimpleSolution(
        t=tfine,
        x=x,
        v=v,
        success=True,
        message="BAOAB Langevin integration completed successfully",
    )


def simulate_ic(
    field, nn_norm,
    tspan, tfine,
    x0, v0,
    bi_xmin,
    pes_interpolant, epsprime_interpolant,
    mode_mass,
    alpha=8.75,
    method="Radau", rtol=1e-3, atol=1e-6,
    ic_mode="absolute",       # "absolute" or "offset"
    damping=0.05,             # deterministic damping / legacy friction
    use_langevin=False,
    integrator="deterministic",   # "deterministic" or "baoab"
    gamma=None,
    kBT_internal=0.0,
    rng=None,
):
    """
    Simulate dynamics for one initial condition.

    Parameters
    ----------
    x0, v0 :
        If ic_mode="absolute": x0 is absolute Q_init, v0 is initial velocity
                               in internal velocity units.
        If ic_mode="offset"  : x0 is displacement from bi_xmin.

    bi_xmin :
        Only used if ic_mode="offset".

    use_langevin :
        If True, uses BAOAB Langevin dynamics.
        If False, uses deterministic solve_ivp dynamics.

    integrator :
        "deterministic" or "baoab"
        If use_langevin=True, BAOAB is used regardless.

    gamma :
        Langevin friction coefficient. If None, falls back to damping.

    kBT_internal :
        Thermal energy in the SAME internal energy units as the PES.

    rng :
        np.random.Generator for reproducible Langevin noise.

    Returns
    -------
    sol :
        Either scipy solve_ivp solution (deterministic) or SimpleSolution (BAOAB),
        both with:
            sol.t
            sol.y[0] = x(t)
            sol.y[1] = v(t)
    """
    if nn_norm <= 0 or not np.isfinite(nn_norm):
        raise ValueError(f"nn_norm must be positive/finite, got {nn_norm}")

    x_init, v_init = _initial_state(x0=x0, v0=v0, bi_xmin=bi_xmin, ic_mode=ic_mode)

    if gamma is None:
        gamma = damping

    # If Langevin requested, force BAOAB branch
    if use_langevin or integrator == "baoab":
        return _simulate_ic_baoab(
            field=field,
            nn_norm=nn_norm,
            tfine=tfine,
            x_init=x_init,
            v_init=v_init,
            pes_interpolant=pes_interpolant,
            epsprime_interpolant=epsprime_interpolant,
            mode_mass=mode_mass,
            alpha=alpha,
            gamma=gamma,
            kBT_internal=kBT_internal,
            rng=rng,
        )

    # Otherwise deterministic branch
    return _simulate_ic_deterministic(
        field=field,
        nn_norm=nn_norm,
        tspan=tspan,
        tfine=tfine,
        x_init=x_init,
        v_init=v_init,
        pes_interpolant=pes_interpolant,
        epsprime_interpolant=epsprime_interpolant,
        mode_mass=mode_mass,
        alpha=alpha,
        method=method,
        rtol=rtol,
        atol=atol,
        damping=damping,
    )
