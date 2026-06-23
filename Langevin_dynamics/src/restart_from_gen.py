#!/usr/bin/env python3
"""
src/restart_from_gen.py

Restart helper for scheduler-driven GA.

Supports two restart modes:

1) resubmit_existing
   - Use when target generation directory already exists
   - Re-submits chunk jobs + reducer for that generation
   - Does NOT modify IC schedule state

2) recreate_next
   - Use when target generation directory does NOT exist
   - Reconstructs target generation from previous completed generation
   - Advances IC schedule exactly once
   - Saves generation state and submits jobs

Typical use:
    python src/restart_from_gen.py --outdir <OUTDIR> --gen 31 --mode resubmit_existing
    python src/restart_from_gen.py --outdir <OUTDIR> --gen 31 --mode recreate_next
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ga import (
    _append_status,
    _log_ic_usage,
    _select_ic_pairs_for_generation,
    build_next_population,
)
from src.io_utils import (
    load_json,
    load_pickle,
    save_json,
    save_pickle,
)

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="Restart scheduler-driven GA from a chosen generation.")
    ap.add_argument("--outdir", type=str, required=True, help="GA output directory")
    ap.add_argument("--gen", type=int, required=True, help="Generation to restart from (target generation)")
    ap.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["resubmit_existing", "recreate_next"],
        help="Restart mode",
    )
    return ap


def resolve_outdir(outdir_arg: str) -> Path:
    outdir = Path(outdir_arg)
    if not outdir.is_absolute():
        outdir = (PROJECT_ROOT / outdir).resolve()
    return outdir


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
    Submit all chunk jobs for one generation, then a reducer job.

    IMPORTANT:
    Reducer is submitted WITHOUT PBS dependencies.
    The reducer itself waits for all expected chunk files.
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

    _append_status(
        str(outdir),
        f"[restart-submit] generation {gen}: submitted {len(chunk_job_ids)} chunk jobs; submitting reducer without PBS dependencies"
    )

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
        f"[restart-submit] generation {gen}: reducer job {reduce_jid}"
    )


def generation_exists(gdir: Path) -> bool:
    return (
        (gdir / "pop.pkl").exists()
        and (gdir / "ic_pairs.npy").exists()
        and (gdir / "ic_pool_idx.npy").exists()
    )


def main():
    args = build_parser().parse_args()

    outdir = resolve_outdir(args.outdir)
    gen = int(args.gen)
    mode = str(args.mode)

    cfg = load_pickle(outdir / "config.pkl")
    launcher = load_json(outdir / "launcher.json")

    pbs_script = Path(launcher["pbs_script"])
    if not pbs_script.is_absolute():
        pbs_script = (PROJECT_ROOT / pbs_script).resolve()

    _append_status(str(outdir), f"[restart] requested mode={mode} target_gen={gen}")

    target_gdir = gen_dir(outdir, gen)

    if mode == "resubmit_existing":
        if not generation_exists(target_gdir):
            raise FileNotFoundError(
                f"Target generation {gen} does not exist or is incomplete: {target_gdir}"
            )

        _append_status(
            str(outdir),
            f"[restart] resubmitting existing generation {gen} from {target_gdir}"
        )
        submit_generation_jobs(outdir=outdir, gen=gen, cfg=cfg, pbs_script=pbs_script)
        _append_status(str(outdir), f"[restart] done: resubmitted existing generation {gen}")
        return

    if mode == "recreate_next":
        prev_gen = gen - 1
        if prev_gen < 0:
            raise ValueError("Cannot recreate generation 0 from a previous generation.")

        prev_gdir = gen_dir(outdir, prev_gen)

        if generation_exists(target_gdir):
            raise FileExistsError(
                f"Target generation {gen} already exists at {target_gdir}. "
                f"Use --mode resubmit_existing instead."
            )

        # Need a completed previous generation
        prev_pop_path = prev_gdir / "pop.pkl"
        prev_fit_path = prev_gdir / f"fitness_gen_{prev_gen:04d}.npy"
        if not prev_pop_path.exists():
            raise FileNotFoundError(f"Missing previous generation population: {prev_pop_path}")
        if not prev_fit_path.exists():
            raise FileNotFoundError(
                f"Missing previous generation reduced fitness file: {prev_fit_path}\n"
                f"Cannot recreate gen {gen} unless gen {prev_gen} has been reduced successfully."
            )

        pop_prev = load_pickle(prev_pop_path)
        fitness_prev = np.load(prev_fit_path).astype(float)

        _append_status(
            str(outdir),
            f"[restart] recreating generation {gen} from previous completed generation {prev_gen}"
        )

        # Build next population exactly as reducer would
        next_pop = build_next_population(pop=pop_prev, fitness=fitness_prev, cfg=cfg, gen=prev_gen)

        # Continue IC scheduling exactly from current saved state
        base_ic_pairs = np.load(outdir / "ic_pool.npy").astype(float)
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
                f"[restart] IC pool exhausted -> reshuffled and started pass {ic_pass_state}"
            )

        save_ic_schedule_state(
            outdir=outdir,
            ic_order=next_ic_order,
            ic_cursor=next_ic_cursor,
            ic_pass=ic_pass_state,
            rng_state=ic_rng.bit_generator.state,
        )

        # Save target generation state
        (target_gdir / "chunks").mkdir(parents=True, exist_ok=True)

        save_pickle(target_gdir / "pop.pkl", next_pop)
        np.save(target_gdir / "ic_pairs.npy", next_ic_pairs.astype(np.float32))
        np.save(target_gdir / "ic_pool_idx.npy", np.asarray(next_gen_idx, dtype=np.int32))
        save_json(
            target_gdir / "meta.json",
            {
                "gen": int(gen),
                "ic_pass": int(ic_pass_state),
                "ic_cursor_after_select": int(next_ic_cursor),
                "source_gen": int(prev_gen),
                "restart_created": True,
            },
        )

        _log_ic_usage(
            outdir=str(outdir),
            gen=gen,
            ic_pass=ic_pass_state,
            gen_idx=next_gen_idx,
            ic_pairs_gen=next_ic_pairs,
        )

        _append_status(
            str(outdir),
            f"[restart] generation {gen} recreated successfully; submitting jobs"
        )
        submit_generation_jobs(outdir=outdir, gen=gen, cfg=cfg, pbs_script=pbs_script)
        _append_status(str(outdir), f"[restart] done: recreated and submitted generation {gen}")
        return

    raise ValueError(f"Unknown mode={mode}")


if __name__ == "__main__":
    main()
