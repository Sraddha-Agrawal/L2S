import copy
import time
from pathlib import Path

import numpy as np

from .dynamics import build_field, simulate_ic
from .fnn import mutate
from .scoring import score3


def _make_ic_rng(base_seed: int, gen: int, net_idx: int, ic_idx: int, stream: int = 0):
    """
    Reproducible RNG per (generation, net, IC, stream).
    """
    seed = (
        int(base_seed)
        + 1000003 * int(gen)
        + 10007 * int(net_idx)
        + 101 * int(ic_idx)
        + int(stream)
    ) % (2**63 - 1)
    return np.random.default_rng(seed)


def _append_status(outdir: str, line: str) -> None:
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)
    status_path = outdir_p / "status.txt"
    with open(status_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")
    print(line, flush=True)


def _init_ic_usage_file(outdir: str) -> Path:
    """
    Create the IC usage log file with a header if it does not already exist.

    Columns:
      gen, ic_pass, gen_slot, pool_idx, x0, v0
    """
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)
    usage_path = outdir_p / "ic_usage.txt"

    if not usage_path.exists():
        with open(usage_path, "w", encoding="utf-8") as f:
            f.write("# gen ic_pass gen_slot pool_idx x0 v0\n")

    return usage_path


def _log_ic_usage(outdir: str, gen: int, ic_pass: int, gen_idx: np.ndarray, ic_pairs_gen: np.ndarray) -> None:
    """
    Append the generation's IC usage to outdir/ic_usage.txt.

    Each line records:
      gen, pass number, slot within generation, original pool row index, x0, v0
    """
    usage_path = _init_ic_usage_file(outdir)

    gen_idx = np.asarray(gen_idx, dtype=int).reshape(-1)
    ic_pairs_gen = np.asarray(ic_pairs_gen, dtype=float)

    with open(usage_path, "a", encoding="utf-8") as f:
        for slot, (pool_idx, pair) in enumerate(zip(gen_idx, ic_pairs_gen)):
            x0 = float(pair[0])
            v0 = float(pair[1])
            f.write(f"{int(gen)} {int(ic_pass)} {int(slot)} {int(pool_idx)} {x0:.16e} {v0:.16e}\n")


def _select_ic_pairs_for_generation(
    base_ic_pairs: np.ndarray,
    n_ic_per_gen: int,
    ic_order: np.ndarray,
    ic_cursor: int,
    ic_rng: np.random.Generator,
):
    """
    Select the IC subset for one generation.

    Rules
    -----
    - Sample without replacement within a pass.
    - If the remaining unused ICs are fewer than n_ic_per_gen,
      reshuffle the full pool and start a new pass.
    """
    n_pool = int(base_ic_pairs.shape[0])

    reshuffled = False
    if ic_cursor + n_ic_per_gen > n_pool:
        ic_order = ic_rng.permutation(n_pool)
        ic_cursor = 0
        reshuffled = True

    gen_idx = np.asarray(ic_order[ic_cursor: ic_cursor + n_ic_per_gen], dtype=int)
    ic_pairs_gen = np.asarray(base_ic_pairs[gen_idx], dtype=float).copy()
    ic_cursor += n_ic_per_gen

    return ic_pairs_gen, gen_idx, ic_order, ic_cursor, reshuffled


def make_chunk_grid(n_pop, n_ic, n_net_chunks, n_ic_chunks):
    """
    Build chunk ranges for scheduler-driven block evaluation.
    """
    if n_pop % n_net_chunks != 0:
        raise ValueError(f"n_pop={n_pop} must be divisible by n_net_chunks={n_net_chunks}")
    if n_ic % n_ic_chunks != 0:
        raise ValueError(f"n_ic={n_ic} must be divisible by n_ic_chunks={n_ic_chunks}")

    net_chunk_size = n_pop // n_net_chunks
    ic_chunk_size = n_ic // n_ic_chunks

    net_ranges = [(i, i + net_chunk_size) for i in range(0, n_pop, net_chunk_size)]
    ic_ranges = [(j, j + ic_chunk_size) for j in range(0, n_ic, ic_chunk_size)]

    return net_ranges, ic_ranges


def score_single_trajectory(
    field,
    nn_norm,
    tspan,
    tfine,
    x0,
    v0,
    bi_xmin,
    pes_interpolant,
    epsprime_interpolant,
    mode_mass,
    cfg,
    t_in_ps,
    net_idx,
    global_ic_idx,
    gen,
):
    """
    Run one trajectory and return one scalar score.
    """
    rng_ic = None
    if bool(cfg.use_langevin) or str(cfg.integrator) == "baoab":
        rng_ic = _make_ic_rng(
            base_seed=cfg.thermostat_seed,
            gen=gen,
            net_idx=net_idx,
            ic_idx=global_ic_idx,
            stream=0,
        )

    sol = simulate_ic(
        field, nn_norm,
        tspan=tspan, tfine=tfine,
        x0=float(x0), v0=float(v0),
        bi_xmin=bi_xmin,
        pes_interpolant=pes_interpolant,
        epsprime_interpolant=epsprime_interpolant,
        mode_mass=mode_mass,
        method=cfg.method,
        rtol=cfg.rtol,
        atol=cfg.atol,
        ic_mode=cfg.ic_mode,
        alpha=float(cfg.alpha),
        damping=float(cfg.damping),
        use_langevin=bool(cfg.use_langevin),
        integrator=str(cfg.integrator),
        gamma=float(cfg.gamma),
        kBT_internal=float(cfg.kBT_internal),
        rng=rng_ic,
    )

    yx = np.asarray(sol.y[0], dtype=float).reshape(-1)
    yv = np.asarray(sol.y[1], dtype=float).reshape(-1)

    if (not getattr(sol, "success", True)) or (yx.size != len(tfine)) or (yv.size != len(tfine)) \
       or (not np.all(np.isfinite(yx))) or (not np.all(np.isfinite(yv))):
        return np.nan

    score_ic, _ = score3(
        all_v=[yv],
        all_x=[yx],
        tfine=tfine,
        t_in_ps=t_in_ps,
        x_scale=abs(bi_xmin),
    )
    return float(np.asarray(score_ic).reshape(-1)[0])


def evaluate_chunk_scores(
    pop,
    ic_pairs_gen,
    ic_global_idx_gen,
    cfg,
    tfine,
    tspan,
    bi_xmin,
    bi_mass,
    bi_yinterp,
    poly_derivative_func,
    t_in_ps,
    gen,
    net_start,
    net_end,
    ic_start,
    ic_end,
):
    """
    Evaluate one (net chunk, ic chunk) block.

    Returns
    -------
    scores_block : ndarray, shape (net_end-net_start, ic_end-ic_start)
    """
    ic_pairs_gen = np.asarray(ic_pairs_gen, dtype=float)
    ic_global_idx_gen = np.asarray(ic_global_idx_gen, dtype=int)

    if ic_pairs_gen.ndim != 2 or ic_pairs_gen.shape[1] != 2:
        raise ValueError(f"ic_pairs_gen must have shape (n_ic, 2), got {ic_pairs_gen.shape}")
    if len(ic_global_idx_gen) != len(ic_pairs_gen):
        raise ValueError("ic_global_idx_gen length must match ic_pairs_gen length")

    local_nets = net_end - net_start
    local_ics = ic_end - ic_start
    scores_block = np.full((local_nets, local_ics), np.nan, dtype=np.float32)

    for row_net, global_net_idx in enumerate(range(net_start, net_end)):
        net = pop[global_net_idx]

        field, nn_norm, _ = build_field(
            net, tfine,
            mode=cfg.field_mode,
            spline_k=cfg.spline_k,
            spline_n=cfg.spline_n,
        )

        for col_ic, global_ic_pos in enumerate(range(ic_start, ic_end)):
            x0 = float(ic_pairs_gen[global_ic_pos, 0])
            v0 = float(ic_pairs_gen[global_ic_pos, 1])
            global_ic_idx = int(ic_global_idx_gen[global_ic_pos])

            score_val = score_single_trajectory(
                field=field,
                nn_norm=nn_norm,
                tspan=tspan,
                tfine=tfine,
                x0=x0,
                v0=v0,
                bi_xmin=bi_xmin,
                pes_interpolant=bi_yinterp,
                epsprime_interpolant=poly_derivative_func,
                mode_mass=bi_mass,
                cfg=cfg,
                t_in_ps=t_in_ps,
                net_idx=global_net_idx,
                global_ic_idx=global_ic_idx,
                gen=gen,
            )
            scores_block[row_net, col_ic] = np.float32(score_val)

    return scores_block


def build_next_population(pop, fitness, cfg, gen=None):
    """
    Build next generation from current population and fitness.

    Uses a generation-dependent RNG seed so evolution is reproducible and
    varies from one generation to the next.
    """
    n_pop = len(pop)
    num_parent = max(1, n_pop // 10)

    seed_offset = 0 if gen is None else int(gen)
    rng = np.random.RandomState(int(cfg.seed) + seed_offset)

    sorted_idx = np.argsort(fitness)[::-1]
    parents = [pop[k] for k in sorted_idx[:num_parent]]

    new_pop = [copy.deepcopy(p) for p in parents]
    while len(new_pop) < n_pop:
        p = parents[int(rng.randint(0, len(parents)))]
        child = copy.deepcopy(p)
        mutate(child, mutation_strength=float(cfg.mutation_strength))
        new_pop.append(child)

    return new_pop


def _recompute_champion_trajs(
    champ_net,
    champ_idx,
    ic_pairs,
    global_ic_indices,
    cfg,
    tfine,
    tspan,
    bi_xmin,
    bi_mass,
    bi_yinterp,
    poly_derivative_func,
    ic_mode,
    gen,
):
    """
    Recompute xs/vs + A(t) for the champion net only.

    Important:
    Uses the same per-IC RNG seeding scheme as evaluation so champion
    trajectories match the evaluated score under Langevin dynamics.
    """
    field, nn_norm, _ = build_field(
        champ_net, tfine,
        mode=cfg.field_mode,
        spline_k=cfg.spline_k,
        spline_n=cfg.spline_n,
    )

    ic_pairs = np.asarray(ic_pairs, dtype=float)
    global_ic_indices = np.asarray(global_ic_indices, dtype=int)

    if len(global_ic_indices) != len(ic_pairs):
        raise ValueError("global_ic_indices length must match ic_pairs length")

    n_t = len(tfine)
    xs_best, vs_best = [], []
    thermostat_seed = int(cfg.thermostat_seed)

    for j, (x0, v0) in enumerate(ic_pairs):
        global_ic_idx = int(global_ic_indices[j])

        rng_ic = None
        if bool(getattr(cfg, "use_langevin", False)) or str(getattr(cfg, "integrator", "deterministic")) == "baoab":
            rng_ic = _make_ic_rng(
                base_seed=thermostat_seed,
                gen=gen,
                net_idx=champ_idx,
                ic_idx=global_ic_idx,
                stream=0,
            )

        sol = simulate_ic(
            field, nn_norm,
            tspan=tspan, tfine=tfine,
            x0=float(x0), v0=float(v0),
            bi_xmin=bi_xmin,
            pes_interpolant=bi_yinterp,
            epsprime_interpolant=poly_derivative_func,
            mode_mass=bi_mass,
            method=cfg.method,
            rtol=cfg.rtol,
            atol=cfg.atol,
            ic_mode=ic_mode,
            alpha=float(cfg.alpha),
            damping=float(cfg.damping),
            use_langevin=bool(cfg.use_langevin),
            integrator=str(cfg.integrator),
            gamma=float(cfg.gamma),
            kBT_internal=float(cfg.kBT_internal),
            rng=rng_ic,
        )

        ok = bool(getattr(sol, "success", True))
        yx = np.asarray(sol.y[0], dtype=float).reshape(-1)
        yv = np.asarray(sol.y[1], dtype=float).reshape(-1)

        if (not ok) or (yx.size != n_t) or (yv.size != n_t) or (not np.all(np.isfinite(yx))) or (not np.all(np.isfinite(yv))):
            xs_best.append(np.full(n_t, np.nan, dtype=float))
            vs_best.append(np.full(n_t, np.nan, dtype=float))
        else:
            xs_best.append(yx)
            vs_best.append(yv)

    best_net_traj_x = np.asarray(xs_best, dtype=np.float32)
    best_net_traj_v = np.asarray(vs_best, dtype=np.float32)
    best_net_A = champ_net.get_output_batch(tfine).astype(np.float32)

    return best_net_A, best_net_traj_x, best_net_traj_v


def assemble_scores_from_chunks(chunk_payloads, n_pop, n_ic):
    """
    Assemble a full score matrix from chunk payloads.

    Each payload must be a dict-like object containing:
      net_start, net_end, ic_start, ic_end, scores_block
    """
    scores = np.full((n_pop, n_ic), np.nan, dtype=np.float32)

    for payload in chunk_payloads:
        net_start = int(payload["net_start"])
        net_end = int(payload["net_end"])
        ic_start = int(payload["ic_start"])
        ic_end = int(payload["ic_end"])
        block = np.asarray(payload["scores_block"], dtype=np.float32)

        expected_shape = (net_end - net_start, ic_end - ic_start)
        if block.shape != expected_shape:
            raise ValueError(
                f"Chunk block shape mismatch: got {block.shape}, expected {expected_shape}"
            )

        scores[net_start:net_end, ic_start:ic_end] = block

    return scores


def cleanup_generation_chunk_files(gen_dir):
    """
    Delete temporary chunk files under gen_dir/chunks.
    """
    gen_dir = Path(gen_dir)
    chunks_dir = gen_dir / "chunks"
    if not chunks_dir.exists():
        return

    for path in chunks_dir.glob("*.npz"):
        path.unlink()

    try:
        chunks_dir.rmdir()
    except OSError:
        # directory not empty or cannot be removed; ignore safely
        pass


def expected_chunk_file_map(gen_dir, n_net_chunks, n_ic_chunks):
    """
    Return a mapping of expected chunk IDs to file paths.
    """
    gen_dir = Path(gen_dir)
    chunks_dir = gen_dir / "chunks"
    mapping = {}
    for net_chunk_id in range(int(n_net_chunks)):
        for ic_chunk_id in range(int(n_ic_chunks)):
            name = f"chunk_net{net_chunk_id:03d}_ic{ic_chunk_id:03d}.npz"
            mapping[(net_chunk_id, ic_chunk_id)] = chunks_dir / name
    return mapping


def wait_for_expected_chunk_files(gen_dir, n_net_chunks, n_ic_chunks, poll_s=5.0, timeout_s=None):
    """
    Wait until all expected chunk files exist.

    Useful in a reducer script if scheduler dependencies are not strict.
    """
    t0 = time.perf_counter()
    mapping = expected_chunk_file_map(gen_dir, n_net_chunks, n_ic_chunks)

    while True:
        missing = [p for p in mapping.values() if not p.exists()]
        if not missing:
            return mapping

        if timeout_s is not None and (time.perf_counter() - t0) > float(timeout_s):
            raise TimeoutError(
                f"Timed out waiting for chunk files. Missing {len(missing)} files."
            )

        time.sleep(float(poll_s))
