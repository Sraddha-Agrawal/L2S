#!/usr/bin/env python3
"""
run_ga.py

Bootstrap script for scheduler-driven GA.

What it does
------------
1. Parse CLI
2. Load IC pool
3. Build config
4. Load PES / epsprime / tfine
5. Initialize population for generation 0
6. Select generation-0 IC subset
7. Save generation-0 state to disk
8. Submit all chunk jobs for generation 0
9. Submit one dependent reducer job for generation 0
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.config import GAConfig
from src.data_bi import load_bi_pes, load_epsprime_poly, compute_bi_phfreq_in_thz
from src.fnn import init_pop
from src.ga import (
    _append_status,
    _init_ic_usage_file,
    _log_ic_usage,
    _select_ic_pairs_for_generation,
    make_chunk_grid,
)
from src.io_utils import (
    read_ic_pairs,
    save_json,
    save_pickle,
)

BI_MASS = 208.9804
ENERGY_IN_EV = 0.004180159285619251
FREQ_IN_THZ = 6.3507799295888985
T_IN_PS = 1.0 / FREQ_IN_THZ


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Bootstrap scheduler-driven GA optimization for Bi protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--outdir", type=str, default="ga_out", help="Output directory.")
    ap.add_argument(
        "--input_dir",
        type=str,
        default=str(SCRIPT_DIR / "Input_files"),
        help="Directory containing toten.dat and chi_files/.",
    )
    ap.add_argument(
        "--ic_file",
        type=str,
        default=str(SCRIPT_DIR / "Input_files" / "ic_pairs_5000.txt"),
        help="IC pairs file with lines: Q_init V_init in internal units if ic_mode=absolute; else x0_offset V_init",
    )
    ap.add_argument("--n_ic_use", type=int, default=None, help="Use only first N IC pairs from ic_file.")
    ap.add_argument(
        "--pbs_script",
        type=str,
        default=str(SCRIPT_DIR / "python-sub-langevin"),
        help="PBS wrapper script used by qsub for chunk and reducer jobs.",
    )

    ap.add_argument(
        "--ic_mode",
        type=str,
        default="absolute",
        choices=["absolute", "offset"],
        help="Interpret IC file columns as absolute (Q_init,V_init) or offset (x0 from bi_xmin, V_init).",
    )
    ap.add_argument("--n_ic_per_gen", type=int, default=50)
    ap.add_argument("--ic_seed", type=int, default=None)

    ap.add_argument("--n_net_chunks", type=int, default=50)
    ap.add_argument("--n_ic_chunks", type=int, default=5)
    ap.add_argument("--keep_chunk_files", action="store_true")

    ap.add_argument("--stop_mode", type=str, default="fixed", choices=["fixed", "threshold"])
    ap.add_argument("--n_gen", type=int, default=50, help="Used when stop_mode=fixed.")
    ap.add_argument("--stop_threshold", type=float, default=0.995, help="Used when stop_mode=threshold.")
    ap.add_argument("--max_gen", type=int, default=150, help="Optional safety cap for threshold mode.")

    ap.add_argument("--n_pop", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--mutation_strength", type=float, default=0.2)

    ap.add_argument("--field_mode", type=str, default="spline", choices=["spline", "nn"])
    ap.add_argument("--spline_k", type=int, default=5)
    ap.add_argument("--spline_n", type=int, default=400)

    ap.add_argument("--total_time_ps", type=float, default=20.0)
    ap.add_argument("--dense_steps", type=int, default=25000)
    ap.add_argument("--integrator", type=str, default="deterministic", choices=["deterministic", "baoab"])
    ap.add_argument("--method", type=str, default="Radau")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--alpha", type=float, default=8.75)
    ap.add_argument("--damping", type=float, default=0.05)

    ap.add_argument("--use_langevin", action="store_true")
    ap.add_argument("--temperature_K", type=float, default=300.0)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--langevin_seed", type=int, default=None)
    ap.add_argument("--energy_conv_eV", type=float, default=ENERGY_IN_EV)

    ap.add_argument(
    "--fitness_method",
    type=str,
    default="cvar",
    choices=["mean", "min", "mean_std", "cvar", "cdf_l2", "survival_thr","lp_to_one","soft_threshold","hybrid_cdf_lp"],
)
    ap.add_argument("--fitness_q", type=float, default=0.2)
    ap.add_argument("--fitness_lam", type=float, default=0.5)
    # new
    ap.add_argument("--fitness_threshold", type=float, default=0.99)
    ap.add_argument("--fitness_grid_n", type=int, default=401)
    ap.add_argument("--fitness_nan_policy", type=str, default="omit", choices=["omit", "zero"])
    ap.add_argument("--fitness_p", type=float, default=4.0)
    ap.add_argument("--fitness_tau", type=float, default=0.05)
    ap.add_argument("--fitness_hybrid_w", type=float, default=0.5)
    return ap


def parse_args_allow_unknown(ap: argparse.ArgumentParser) -> Tuple[argparse.Namespace, List[str]]:
    args, unknown = ap.parse_known_args()
    return args, unknown


def gen_dir(outdir: Path, gen: int) -> Path:
    return outdir / f"gen_{gen:04d}"


def save_ic_schedule_state(
    outdir: Path,
    ic_order: np.ndarray,
    ic_cursor: int,
    ic_pass: int,
    rng_state: dict,
) -> None:
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


def submit_generation_jobs(outdir: Path, gen: int, cfg: GAConfig, pbs_script: Path) -> None:
    net_ranges, ic_ranges = make_chunk_grid(
        n_pop=cfg.n_pop,
        n_ic=cfg.n_ic_per_gen,
        n_net_chunks=cfg.n_net_chunks,
        n_ic_chunks=cfg.n_ic_chunks,
    )

    chunk_job_ids = []
    for net_chunk_id, _ in enumerate(net_ranges):
        for ic_chunk_id, _ in enumerate(ic_ranges):
            env_vars = {
                "MODE": "chunk",
                "OUTDIR": str(outdir),
                "GEN": str(gen),
                "NET_CHUNK_ID": str(net_chunk_id),
                "IC_CHUNK_ID": str(ic_chunk_id),
            }
            jid = submit_pbs_job(pbs_script=pbs_script, env_vars=env_vars, dependency_ids=None)
            chunk_job_ids.append(jid)

    _append_status(
        str(outdir),
        f"[submit] generation {gen}: submitted {len(chunk_job_ids)} chunk jobs"
    )

    reduce_env = {
        "MODE": "reduce",
        "OUTDIR": str(outdir),
        "GEN": str(gen),
    }

    try:
        reduce_jid = submit_pbs_job(
            pbs_script=pbs_script,
            env_vars=reduce_env,
            dependency_ids=None,   # important change
        )
    except Exception as e:
        _append_status(
            str(outdir),
            f"[submit-error] reducer submission failed for gen={gen}: {type(e).__name__}: {e}"
        )
        raise

    _append_status(
        str(outdir),
        f"# submitted generation {gen}: {len(chunk_job_ids)} chunk jobs + reducer job {reduce_jid}"
    )

def main() -> None:
    ap = build_parser()
    args, unknown = parse_args_allow_unknown(ap)

    if unknown:
        print("[WARN] Ignoring unknown CLI args:", unknown, flush=True)

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = (SCRIPT_DIR / input_dir).resolve()

    ic_file = Path(args.ic_file)
    if not ic_file.is_absolute():
        ic_file = (SCRIPT_DIR / ic_file).resolve()

    pbs_script = Path(args.pbs_script)
    if not pbs_script.is_absolute():
        pbs_script = (SCRIPT_DIR / pbs_script).resolve()

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (SCRIPT_DIR / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ic_pairs = read_ic_pairs(str(ic_file))
    if len(ic_pairs) == 0:
        raise RuntimeError(f"No IC pairs found in: {ic_file}")

    if args.n_ic_use is not None:
        ic_pairs = ic_pairs[: int(args.n_ic_use)]

    if int(args.n_ic_per_gen) > len(ic_pairs):
        raise ValueError(
            f"--n_ic_per_gen={args.n_ic_per_gen} exceeds available IC pool size={len(ic_pairs)}"
        )

    use_langevin = bool(args.use_langevin) or (str(args.integrator) == "baoab")

    cfg = GAConfig(
        n_pop=int(args.n_pop),
        n_gen=int(args.n_gen),
        seed=int(args.seed),
        mutation_strength=float(args.mutation_strength),
        stop_mode=str(args.stop_mode),
        stop_threshold=float(args.stop_threshold),
        ic_mode=str(args.ic_mode),
        max_gen=args.max_gen if args.max_gen is None else int(args.max_gen),
        n_net_chunks=int(args.n_net_chunks),
        n_ic_chunks=int(args.n_ic_chunks),
        n_ic_per_gen=int(args.n_ic_per_gen),
        ic_seed=args.ic_seed if args.ic_seed is None else int(args.ic_seed),
        keep_chunk_files=bool(args.keep_chunk_files),
        field_mode=str(args.field_mode),
        spline_k=int(args.spline_k),
        spline_n=int(args.spline_n),
        total_time_ps=float(args.total_time_ps),
        dense_steps=int(args.dense_steps),
        integrator=str(args.integrator),
        method=str(args.method),
        rtol=float(args.rtol),
        atol=float(args.atol),
        alpha=float(args.alpha),
        damping=float(args.damping),
        gamma=None if args.gamma is None else float(args.gamma),
        use_langevin=use_langevin,
        temperature_K=float(args.temperature_K),
        energy_conv_eV=float(args.energy_conv_eV),
        langevin_seed=args.langevin_seed if args.langevin_seed is None else int(args.langevin_seed),
    )

    base_ic_pairs = np.asarray(ic_pairs, dtype=float)
    n_ic_pool = len(base_ic_pairs)

    _append_status(
        str(outdir),
        f"# bootstrap start | n_pop={cfg.n_pop} n_ic_pool={n_ic_pool} n_ic_per_gen={cfg.n_ic_per_gen} "
        f"n_net_chunks={cfg.n_net_chunks} n_ic_chunks={cfg.n_ic_chunks} "
        f"integrator={cfg.integrator} use_langevin={cfg.use_langevin}"
    )

    save_pickle(outdir / "config.pkl", cfg)
    np.save(outdir / "ic_pool.npy", base_ic_pairs.astype(np.float32))

    launcher_meta = {
        "project_root": str(SCRIPT_DIR),
        "input_dir": str(input_dir),
        "pbs_script": str(pbs_script),
        "fitness_method": str(args.fitness_method),
        "fitness_q": float(args.fitness_q),
        "fitness_lam": float(args.fitness_lam),
        "fitness_p": float(args.fitness_p),
        "fitness_tau": float(args.fitness_tau),
        "fitness_hybrid_w": float(args.fitness_hybrid_w),
        "fitness_threshold": float(args.fitness_threshold),
        "fitness_grid_n": int(args.fitness_grid_n),
        "fitness_nan_policy": str(args.fitness_nan_policy),
        "bi_mass": float(BI_MASS),
        "energy_in_ev": float(ENERGY_IN_EV),
        "freq_in_thz": float(FREQ_IN_THZ),
        "t_in_ps": float(T_IN_PS),
    }
    save_json(outdir / "launcher.json", launcher_meta)

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

    bi_phfreq_in_thz = compute_bi_phfreq_in_thz(
        bi_yinterp, bi_xmin, BI_MASS, FREQ_IN_THZ
    )

    tspan = [0.0, cfg.total_time_ps / T_IN_PS]
    tfine = np.linspace(tspan[0], tspan[1], cfg.dense_steps, dtype=float)
    np.save(outdir / "tfine.npy", tfine.astype(np.float32))

    pop = init_pop(
        cfg.n_pop,
        cfg.input_size,
        cfg.hidden_size,
        cfg.output_size,
        bi_phfreq_in_thz,
        FREQ_IN_THZ,
        seed=cfg.seed,
    )

    ic_rng = np.random.default_rng(cfg.effective_ic_seed)
    ic_order = ic_rng.permutation(n_ic_pool)
    ic_cursor = 0
    ic_pass = 0

    ic_pairs_gen, gen_idx, ic_order, ic_cursor, reshuffled = _select_ic_pairs_for_generation(
        base_ic_pairs=base_ic_pairs,
        n_ic_per_gen=cfg.n_ic_per_gen,
        ic_order=ic_order,
        ic_cursor=ic_cursor,
        ic_rng=ic_rng,
    )
    if reshuffled:
        ic_pass += 1

    save_ic_schedule_state(
        outdir,
        ic_order=ic_order,
        ic_cursor=ic_cursor,
        ic_pass=ic_pass,
        rng_state=ic_rng.bit_generator.state,
    )

    _init_ic_usage_file(str(outdir))
    _log_ic_usage(
        outdir=str(outdir),
        gen=0,
        ic_pass=ic_pass,
        gen_idx=gen_idx,
        ic_pairs_gen=ic_pairs_gen,
    )

    g0_dir = gen_dir(outdir, 0)
    (g0_dir / "chunks").mkdir(parents=True, exist_ok=True)

    save_pickle(g0_dir / "pop.pkl", pop)
    np.save(g0_dir / "ic_pairs.npy", ic_pairs_gen.astype(np.float32))
    np.save(g0_dir / "ic_pool_idx.npy", np.asarray(gen_idx, dtype=np.int32))
    save_json(
        g0_dir / "meta.json",
        {
            "gen": 0,
            "ic_pass": int(ic_pass),
            "ic_cursor_after_select": int(ic_cursor),
            "bi_xmin": float(bi_xmin),
            "bi_xmax": float(bi_xmax),
            "t_final_internal": float(tspan[1]),
        },
    )

    _append_status(
        str(outdir),
        f"# generation 0 prepared | ic_pass={ic_pass} | saved to {g0_dir}"
    )

    submit_generation_jobs(outdir=outdir, gen=0, cfg=cfg, pbs_script=pbs_script)

    _append_status(str(outdir), "# bootstrap finished")


if __name__ == "__main__":
    main()
