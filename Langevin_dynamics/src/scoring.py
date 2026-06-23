import numpy as np

def score3(all_v, all_x, tfine, t_in_ps, x_scale):
    x_saddle = x_scale
    scores = []
    tindex = []
    for result, x_values in zip(all_v, all_x):
        indices = []
        time_v0 = []
        time_x0 = []

        for i in range(1, len(result)):
            if (result[i] * result[i-1]) < 0:
                time_v0.append(tfine[i-1]*t_in_ps)
                indices.append(i-1)

        for i in range(1, len(indices)):
            min_x = np.min(x_values[indices[i-1]:indices[i]])
            if min_x < 0:
                for j in range(indices[i-1], indices[i]):
                    if (x_values[j-1] < -x_saddle and x_values[j] > -x_saddle) or (x_values[j-1] > -x_saddle and x_values[j] < -x_saddle):
                        time_x0.append(tfine[j-1]*t_in_ps)
            elif min_x > 0:
                for j in range(indices[i-1], indices[i]):
                    if (x_values[j-1] < x_saddle and x_values[j] > x_saddle) or (x_values[j-1] > x_saddle and x_values[j] < x_saddle):
                        time_x0.append(tfine[j-1]*t_in_ps)

        avg_pos = []
        traj_avg = []
        windows = []
        for i in range(1, len(time_v0) - 2, 2):
            start_index = indices[i]
            end_index = indices[i + 2]
            windows.append((start_index, end_index))

            if end_index < len(x_values):
                seg = x_values[start_index:end_index + 1]
                avg_pos.append(float(np.mean(seg)) if len(seg) else 0.0)
            else:
                avg_pos.append(0.0)

        for a in avg_pos:
            if a <= 0:
                frac = 1 + (a/x_saddle)
            else:
                frac = 1 - (a/x_saddle)
            traj_avg.append(frac)

        max_value = max(traj_avg) if traj_avg else 0.0
        max_index = traj_avg.index(max_value) if traj_avg else 0

        max_window_end_index = windows[max_index][1] if windows else 0
        max_time = time_v0[indices.index(max_window_end_index)] if (windows and max_window_end_index in indices) else 0.0

        scores.append(max_value)
        tindex.append(max_time)

    return np.array(scores, dtype=float), np.array(tindex, dtype=float)


def _clean_scores(scores_ic, nan_policy="omit"):
    s = np.asarray(scores_ic, dtype=float).reshape(-1)

    if nan_policy == "omit":
        s = s[np.isfinite(s)]
    elif nan_policy == "zero":
        s = np.where(np.isfinite(s), s, 0.0)
    else:
        raise ValueError(f"Unknown nan_policy={nan_policy}")

    if s.size == 0:
        return np.array([0.0], dtype=float)

    return np.clip(s, 0.0, 1.0)


def _empirical_cdf(samples, grid):
    samples = np.sort(np.asarray(samples, dtype=float))
    return np.searchsorted(samples, grid, side="right") / samples.size


def _fitness_cdf_l2(scores_ic, grid_n=401, nan_policy="omit"):
    s = _clean_scores(scores_ic, nan_policy=nan_policy)
    grid = np.linspace(0.0, 1.0, int(grid_n), dtype=float)
    F = _empirical_cdf(s, grid)

    dx = grid[1] - grid[0]
    dist = np.sqrt(np.sum(F**2) * dx)
    dist = float(np.clip(dist, 0.0, 1.0))
    return 1.0 - dist


def _fitness_survival_threshold(scores_ic, threshold=0.6, nan_policy="omit"):
    s = _clean_scores(scores_ic, nan_policy=nan_policy)
    return float(np.mean(s > float(threshold)))


def _fitness_lp_to_one(scores_ic, p=4.0, nan_policy="omit"):
    s = _clean_scores(scores_ic, nan_policy=nan_policy)
    p = float(p)
    if p <= 0:
        raise ValueError("p must be > 0")
    d = 1.0 - s
    val = np.mean(d**p)**(1.0/p)
    return float(1.0 - val)


def _fitness_soft_threshold(scores_ic, threshold=0.6, tau=0.05, nan_policy="omit"):
    s = _clean_scores(scores_ic, nan_policy=nan_policy)
    tau = float(tau)
    if tau <= 0:
        raise ValueError("tau must be > 0")
    z = (s - float(threshold)) / tau
    return float(np.mean(1.0 / (1.0 + np.exp(-z))))


def _fitness_hybrid_cdf_lp(scores_ic, grid_n=401, p=4.0, hybrid_w=0.5, nan_policy="omit"):
    w = float(hybrid_w)
    if not (0.0 <= w <= 1.0):
        raise ValueError("hybrid_w must be in [0, 1]")

    f_cdf = _fitness_cdf_l2(scores_ic, grid_n=grid_n, nan_policy=nan_policy)
    f_lp  = _fitness_lp_to_one(scores_ic, p=p, nan_policy=nan_policy)
    return float(w * f_cdf + (1.0 - w) * f_lp)


def fitness_from_scores(
    scores_ic,
    method="cvar",
    q=0.2,
    lam=0.5,
    threshold=0.6,
    grid_n=401,
    nan_policy="omit",
    p=4.0,
    tau=0.05,
    hybrid_w=0.5,
):
    """
    Supported methods:
      - mean
      - min
      - mean_std
      - cvar
      - cdf_l2
      - survival_thr
      - lp_to_one
      - soft_threshold
      - hybrid_cdf_lp
    """
    s = _clean_scores(scores_ic, nan_policy=nan_policy)

    if method == "mean":
        return float(np.mean(s))

    if method == "min":
        return float(np.min(s))

    if method == "mean_std":
        return float(np.mean(s) - lam * np.std(s))

    if method == "cvar":
        k = max(1, int(np.ceil(q * len(s))))
        worst = np.sort(s)[:k]
        return float(np.mean(worst))

    if method == "cdf_l2":
        return _fitness_cdf_l2(s, grid_n=grid_n, nan_policy=nan_policy)

    if method == "survival_thr":
        return _fitness_survival_threshold(s, threshold=threshold, nan_policy=nan_policy)

    if method == "lp_to_one":
        return _fitness_lp_to_one(s, p=p, nan_policy=nan_policy)

    if method == "soft_threshold":
        return _fitness_soft_threshold(s, threshold=threshold, tau=tau, nan_policy=nan_policy)

    if method == "hybrid_cdf_lp":
        return _fitness_hybrid_cdf_lp(s, grid_n=grid_n, p=p, hybrid_w=hybrid_w, nan_policy=nan_policy)

    raise ValueError(method)
