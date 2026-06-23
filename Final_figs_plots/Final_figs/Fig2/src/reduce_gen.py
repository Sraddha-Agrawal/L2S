#!/usr/bin/env python3
"""
src/reduce_gen.py

Reducer for one GA generation.

What it does
------------
1. Wait for all expected chunk files
2. Assemble full score matrix
3. Compute fitness
4. Recompute champion trajectories
5. Save generation outputs
6. Check stop condition
7. If continuing:
   - build next population
   - select next generation IC subset
   - save generation state
   - submit next generation chunk jobs + reducer
8. Optionally clean up current generation chunk files
"""

import argparse
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_bi import load_bi_pes, load_epsprime_poly
from src.ga import (
    _append_status,
    _log_ic_usage,
    _recompute_champion_trajs,
    _select_ic_pairs_for_generation,
    assemble_scores_from_chunks,
    build_next_population,
    cleanup_generation_chunk_files,
    wait_for_expected_chunk_files,
)
from src.io_utils import (
    load_json,
    load_pickle,
    save_gen_best,
    save_gen_scores,
    save_json,
    save_pickle,
)
from src.scoring import fitness_from_scores

# ----------------------------
# Bismuth constants
# ----------------------------
BI_MASS = 208.9804
ENERGY_IN_EV = 0.004180159285619251
FREQ_IN_THZ = 6.3507799295888985
T_IN_PS = 1.0 / FREQ_IN_THZ


def build_parser():
    ap = argparse.ArgumentParser(description="Reduce one GA generation from chunk results.")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--gen", type=int, required=True)
    return ap


def gen_dir(outdir: Path, gen: int) -> Path:
    return outdir / f"gen_{gen:04d}"


def load_ic_schedule_state(outdir: Path):
    """
    Load persistent IC scheduling state.

    Stored fields:
      - ic_order   : current shuffled pool order
      - ic_cursor  : next unread cursor in the current pass
      - ic_pass    : pass count
      - rng_state  : numpy bit_generator state (stored as 0-d object array)
    """
    data = np.load(outdir / "ic_schedule_state.npz", allow_pickle=True)
    ic_order = np.asarray(data["ic_order"], dtype=int)
    ic_cursor = int(data["ic_cursor"])
    ic_pass = int(data["ic_pass"])

    rng_state_arr = data["rng_state"]
    if isinstance(rng_state_arr, np.ndarray) and rng_state_arr.shape == ():
        rng_state = rng_state_arr.item()
    else:
        rng_state = rng_state_arr

    return ic_order, ic_cursor, ic_pass, rng_state


def save_ic_schedule_state(outdir: Path, ic_order: np.ndarray, ic_cursor: int, ic_pass: int, rng_state: dict) -> None:
    """
    Save persistent IC scheduling state for the next generation.
    """
    np.savez_compressed(
        outdir / "ic_schedule_state.npz",
        ic_order=np.asarray(ic_order, dtype=np.int32),
        ic_cursor=np.int32(ic_cursor),
        ic_pass=np.int32(ic_pass),
        rng_state=np.array(rng_state, dtype=object),
    )


def submit_pbs_job(pbs_script: Path, env_vars: dict, dependency_ids=None) -> str:
    env_string = ",".join(f"{k}={v}" for k, v in env_vars.items())

    cmd = ["qsub"]
    if dependency_ids:
        dep = "afterok:" + ":".join(dependency_ids)
        cmd.extend(["-W", f"depend={dep}"])
    cmd.extend(["-v", env_string, str(pbs_script)])

    out = subprocess.check_output(cmd, text=True).strip()
    job_id = out.split()[0]
    return job_id


def submit_generation_jobs(outdir: Path, gen: int, cfg, pbs_script: Path) -> None:
    """
    Submit all chunk jobs for one generation, then a reducer job dependent on them.
    """
    chunk_job_ids = []

    for net_chunk_id in range(int(cfg.n_net_chunks)):
        for ic_chunk_id in range(int(cfg.n_ic_chunks)):
            env_vars = {
                "MODE": "chunk",
                "OUTDIR": str(outdir),
                "GEN": str(gen),
                "NET_CHUNK_ID": str(net_chunk_id),
                "IC_CHUNK_ID": str(ic_chunk_id),
            }
            jid = submit_pbs_job(
                pbs_script=pbs_script,
                env_vars=env_vars,
                dependency_ids=None,
            )
            chunk_job_ids.append(jid)

    reduce_env = {
        "MODE": "reduce",
        "OUTDIR": str(outdir),
        "GEN": str(gen),
    }
    reduce_jid = submit_pbs_job(
        pbs_script=pbs_script,
        env_vars=reduce_env,
        dependency_ids=None,
    )

    _append_status(
        str(outdir),
        f"# submitted generation {gen}: {len(chunk_job_ids)} chunk jobs + reducer job {reduce_jid}"
    )


def load_chunk_payload(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _resolve_outdir(outdir_arg: str) -> Path:
    outdir = Path(outdir_arg)
    if not outdir.is_absolute():
        outdir = (PROJECT_ROOT / outdir).resolve()
    return outdir


def main():
    ap = build_parser()
    args = ap.parse_args()

    outdir = _resolve_outdir(args.outdir)
    gen = int(args.gen)

    _append_status(str(outdir), f"[reduce-start] gen={gen}")

    cfg = load_pickle(outdir / "config.pkl")
    launcher = load_json(outdir / "launcher.json")
    _append_status(str(outdir), "[reduce] loaded config.pkl and launcher.json")

    input_dir = Path(launcher["input_dir"])
    if not input_dir.is_absolute():
        input_dir = (PROJECT_ROOT / input_dir).resolve()

    pbs_script = Path(launcher["pbs_script"])
    if not pbs_script.is_absolute():
        pbs_script = (PROJECT_ROOT / pbs_script).resolve()

    # --------------------------------------------------
    # Fitness settings
    # --------------------------------------------------
    fitness_method = str(launcher.get("fitness_method", "cvar"))
    fitness_q = float(launcher.get("fitness_q", 0.2))
    fitness_lam = float(launcher.get("fitness_lam", 0.5))
    fitness_threshold = float(launcher.get("fitness_threshold", 0.6))
    fitness_grid_n = int(launcher.get("fitness_grid_n", 401))
    fitness_nan_policy = str(launcher.get("fitness_nan_policy", "omit"))
    fitness_p = float(launcher.get("fitness_p", 4.0))
    fitness_tau = float(launcher.get("fitness_tau", 0.05))
    fitness_hybrid_w = float(launcher.get("fitness_hybrid_w", 0.5))

    _append_status(
        str(outdir),
        f"[reduce] fitness config: method={fitness_method} q={fitness_q} "
        f"lam={fitness_lam} threshold={fitness_threshold} "
        f"grid_n={fitness_grid_n} nan_policy={fitness_nan_policy}"
    )

    tfine = np.load(outdir / "tfine.npy").astype(float)
    base_ic_pairs = np.load(outdir / "ic_pool.npy").astype(float)

    gdir = gen_dir(outdir, gen)
    pop = load_pickle(gdir / "pop.pkl")
    ic_pairs_gen = np.load(gdir / "ic_pairs.npy").astype(float)
    ic_global_idx_gen = np.load(gdir / "ic_pool_idx.npy").astype(int)

    _append_status(
        str(outdir),
        f"[reduce] loaded gen state: gdir={gdir.name} n_pop={len(pop)} "
        f"n_ic={len(ic_pairs_gen)}"
    )

    # --------------------------------------------------
    # Wait for all expected chunk files
    # --------------------------------------------------
    _append_status(str(outdir), "[reduce] waiting for expected chunk files")
    wait_for_expected_chunk_files(
        gen_dir=gdir,
        n_net_chunks=cfg.n_net_chunks,
        n_ic_chunks=cfg.n_ic_chunks,
        poll_s=5.0,
        timeout_s=3600.0,
    )
    _append_status(str(outdir), "[reduce] all expected chunk files detected")

    # --------------------------------------------------
    # Load all chunk payloads and assemble full score matrix
    # --------------------------------------------------
    chunk_payloads = []
    for net_chunk_id in range(int(cfg.n_net_chunks)):
        for ic_chunk_id in range(int(cfg.n_ic_chunks)):
            cpath = gdir / "chunks" / f"chunk_net{net_chunk_id:03d}_ic{ic_chunk_id:03d}.npz"
            if not cpath.exists():
                raise FileNotFoundError(f"Missing chunk file: {cpath}")
            chunk_payloads.append(load_chunk_payload(cpath))

    _append_status(
        str(outdir),
        f"[reduce] loaded {len(chunk_payloads)} chunk payloads; assembling score matrix"
    )

    scores = assemble_scores_from_chunks(
        chunk_payloads=chunk_payloads,
        n_pop=cfg.n_pop,
        n_ic=cfg.n_ic_per_gen,
    )

    _append_status(
        str(outdir),
        f"[reduce] assembled scores matrix with shape={scores.shape}"
    )

    # --------------------------------------------------
    # Compute per-net fitness
    # --------------------------------------------------
    fitness = np.full(cfg.n_pop, -np.inf, dtype=float)
    for i in range(cfg.n_pop):
        fitness[i] = float(
            fitness_from_scores(
                scores[i, :],
                method=fitness_method,
                q=fitness_q,
                lam=fitness_lam,
                threshold=fitness_threshold,
                grid_n=fitness_grid_n,
                nan_policy=fitness_nan_policy,
                p=fitness_p,
                tau=fitness_tau,
                hybrid_w=fitness_hybrid_w,
            )
        )

    best_idx = int(np.nanargmax(fitness))
    best_fit = float(fitness[best_idx])
    mean_fit = float(np.nanmean(fitness))
    std_fit = float(np.nanstd(fitness))

    _append_status(
        str(outdir),
        f"[reduce] computed fitness: best_fit={best_fit:.6f} "
        f"mean_fit={mean_fit:.6f} std_fit={std_fit:.6f} champion={best_idx}"
    )

    # --------------------------------------------------
    # Reload PES / epsprime and recompute champion
    # --------------------------------------------------
    _append_status(str(outdir), "[reduce] loading PES and epsprime")
    x_grid, bi_yinterp, bi_xmin, bi_xmax = load_bi_pes(
        input_dir=str(input_dir),
        energy_in_ev=ENERGY_IN_EV,
        x_range=(-0.5, 0.9),
        saddle_bracket=(0.05, 0.45),
    )
    poly_derivative_func = load_epsprime_poly(
        str(input_dir / "chi_files"),
        x_grid=x_grid,
    )

    tspan = [0.0, cfg.total_time_ps / T_IN_PS]

    _append_status(str(outdir), "[reduce] recomputing champion trajectories")
    best_net_A, best_net_traj_x, best_net_traj_v = _recompute_champion_trajs(
        champ_net=pop[best_idx],
        champ_idx=best_idx,
        ic_pairs=ic_pairs_gen,
        global_ic_indices=ic_global_idx_gen,
        cfg=cfg,
        tfine=tfine,
        tspan=tspan,
        bi_xmin=bi_xmin,
        bi_mass=BI_MASS,
        bi_yinterp=bi_yinterp,
        poly_derivative_func=poly_derivative_func,
        ic_mode=cfg.ic_mode,
        gen=gen,
    )
    _append_status(str(outdir), "[reduce] champion trajectories recomputed")

    current_meta_path = gdir / "meta.json"
    current_meta = load_json(current_meta_path) if current_meta_path.exists() else {}
    ic_pass = int(current_meta.get("ic_pass", 0))

    gen_out_meta = dict(
        gen=int(gen),
        ic_pass=int(ic_pass),
        n_pop=int(cfg.n_pop),
        n_ic=int(cfg.n_ic_per_gen),
        n_net_chunks=int(cfg.n_net_chunks),
        n_ic_chunks=int(cfg.n_ic_chunks),
        fitness_method=str(fitness_method),
        fitness_q=float(fitness_q),
        fitness_lam=float(fitness_lam),
        fitness_threshold=float(fitness_threshold),
        fitness_grid_n=int(fitness_grid_n),
        fitness_nan_policy=str(fitness_nan_policy),
        fitness_p=float(fitness_p),
        fitness_tau=float(fitness_tau),
        fitness_hybrid_w=float(fitness_hybrid_w),
        ic_mode=str(cfg.ic_mode),
        field_mode=str(cfg.field_mode),
        spline_k=int(cfg.spline_k),
        spline_n=int(cfg.spline_n),
        ode_method=str(cfg.method),
        integrator=str(cfg.integrator),
        use_langevin=bool(cfg.use_langevin),
        thermostat_type=str(cfg.thermostat_type),
        temperature_K=float(cfg.temperature_K),
        kBT_internal=float(cfg.kBT_internal),
        rtol=float(cfg.rtol),
        atol=float(cfg.atol),
        alpha=float(cfg.alpha),
        damping=float(cfg.damping),
        gamma=float(cfg.gamma),
        best_fitness=best_fit,
        mean_fitness=mean_fit,
        std_fitness=std_fit,
        champion_index=int(best_idx),
    )

    _append_status(str(outdir), "[reduce] saving generation outputs")
    save_gen_scores(
        outdir=str(gdir),
        gen=gen,
        scores_matrix=scores,
        fitness=fitness,
        meta=gen_out_meta,
    )
    save_gen_best(
        outdir=str(gdir),
        gen=gen,
        ic_pairs=ic_pairs_gen,
        tfine=tfine,
        A_t=best_net_A,
        x_t=best_net_traj_x,
        v_t=best_net_traj_v,
        best_idx=best_idx,
        ic_global_idx=ic_global_idx_gen,
    )

    _append_status(
        str(outdir),
        f"[reduce] gen={gen} outputs saved successfully"
    )

    # --------------------------------------------------
    # Stop condition
    # --------------------------------------------------
    stop_now = False
    stop_reason = None

    if cfg.stop_mode == "fixed":
        if gen + 1 >= int(cfg.n_gen):
            stop_now = True
            stop_reason = f"fixed reached n_gen={cfg.n_gen}"
    elif cfg.stop_mode == "threshold":
        if best_fit >= float(cfg.stop_threshold):
            stop_now = True
            stop_reason = f"threshold reached: best_fit={best_fit:.6f} >= {cfg.stop_threshold}"
        elif cfg.max_gen is not None and (gen + 1) >= int(cfg.max_gen):
            stop_now = True
            stop_reason = f"max_gen reached: {cfg.max_gen}"
    else:
        raise ValueError(f"Unknown stop_mode={cfg.stop_mode}")

    if stop_now:
        _append_status(str(outdir), f"# STOP {stop_reason}")
        if not cfg.keep_chunk_files:
            cleanup_generation_chunk_files(gdir)
        _append_status(str(outdir), "# GA finished")
        return

    # --------------------------------------------------
    # Build next generation population
    # --------------------------------------------------
    _append_status(str(outdir), "[reduce] building next population")
    next_pop = build_next_population(pop=pop, fitness=fitness, cfg=cfg, gen=gen)

    # --------------------------------------------------
    # Continue IC scheduling state exactly from saved RNG state
    # --------------------------------------------------
    _append_status(str(outdir), "[reduce] loading IC schedule state")
    ic_order, ic_cursor, ic_pass_state, rng_state = load_ic_schedule_state(outdir)

    ic_rng = np.random.default_rng()
    ic_rng.bit_generator.state = rng_state

    next_ic_pairs, next_gen_idx, next_ic_order, next_ic_cursor, reshuffled = _select_ic_pairs_for_generation(
        base_ic_pairs=base_ic_pairs,
        n_ic_per_gen=cfg.n_ic_per_gen,
        ic_order=ic_order,
        ic_cursor=ic_cursor,
        ic_rng=ic_rng,
    )

    if reshuffled:
        ic_pass_state += 1
        _append_status(
            str(outdir),
            f"# IC pool exhausted -> reshuffled and started pass {ic_pass_state}"
        )

    save_ic_schedule_state(
        outdir=outdir,
        ic_order=next_ic_order,
        ic_cursor=next_ic_cursor,
        ic_pass=ic_pass_state,
        rng_state=ic_rng.bit_generator.state,
    )

    # --------------------------------------------------
    # Save next generation state
    # --------------------------------------------------
    next_gen = gen + 1
    ngdir = gen_dir(outdir, next_gen)
    (ngdir / "chunks").mkdir(parents=True, exist_ok=True)

    _append_status(str(outdir), f"[reduce] saving next generation state for gen={next_gen}")
    save_pickle(ngdir / "pop.pkl", next_pop)
    np.save(ngdir / "ic_pairs.npy", next_ic_pairs.astype(np.float32))
    np.save(ngdir / "ic_pool_idx.npy", np.asarray(next_gen_idx, dtype=np.int32))
    save_json(
        ngdir / "meta.json",
        {
            "gen": int(next_gen),
            "ic_pass": int(ic_pass_state),
            "ic_cursor_after_select": int(next_ic_cursor),
            "source_gen": int(gen),
        },
    )

    _log_ic_usage(
        outdir=str(outdir),
        gen=next_gen,
        ic_pass=ic_pass_state,
        gen_idx=next_gen_idx,
        ic_pairs_gen=next_ic_pairs,
    )

    # --------------------------------------------------
    # Submit next generation jobs automatically
    # --------------------------------------------------
    _append_status(str(outdir), f"[reduce] submitting generation {next_gen} jobs")
    submit_generation_jobs(outdir=outdir, gen=next_gen, cfg=cfg, pbs_script=pbs_script)

    # --------------------------------------------------
    # Cleanup current generation chunk files if requested
    # --------------------------------------------------
    if not cfg.keep_chunk_files:
        cleanup_generation_chunk_files(gdir)

    _append_status(str(outdir), f"# reducer finished for gen={gen}; scheduled gen={next_gen}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        outdir_fallback = None
        gen_fallback = "unknown"

        argv = sys.argv[1:]
        for i, tok in enumerate(argv):
            if tok == "--outdir" and i + 1 < len(argv):
                outdir_fallback = argv[i + 1]
            elif tok == "--gen" and i + 1 < len(argv):
                gen_fallback = argv[i + 1]

        try:
            if outdir_fallback is not None:
                outdir_path = _resolve_outdir(outdir_fallback)
                _append_status(
                    str(outdir_path),
                    f"[reduce-error] gen={gen_fallback} {type(e).__name__}: {e}"
                )
                tb = traceback.format_exc()
                for line in tb.rstrip().splitlines():
                    _append_status(str(outdir_path), f"[reduce-traceback] {line}")
        except Exception:
            pass

        raise
