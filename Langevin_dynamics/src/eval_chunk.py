#!/usr/bin/env python3
"""
src/eval_chunk.py

Evaluate one (net_chunk, ic_chunk) block for one generation.

Intended to be launched by the PBS wrapper script with environment or CLI args:
  --outdir
  --gen
  --net_chunk_id
  --ic_chunk_id
  --progress_every
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_bi import load_bi_pes, load_epsprime_poly
from src.dynamics import build_field
from src.ga import (
    _append_status,
    make_chunk_grid,
    score_single_trajectory,
)
from src.io_utils import load_json, load_pickle

# ----------------------------
# Bismuth constants
# ----------------------------
BI_MASS = 208.9804
ENERGY_IN_EV = 0.004180159285619251
FREQ_IN_THZ = 6.3507799295888985
T_IN_PS = 1.0 / FREQ_IN_THZ


def build_parser():
    ap = argparse.ArgumentParser(description="Evaluate one scheduler chunk for one GA generation.")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--net_chunk_id", type=int, required=True)
    ap.add_argument("--ic_chunk_id", type=int, required=True)
    ap.add_argument("--progress_every", type=int, default=10,
                    help="Log progress every N population networks within this chunk.")
    return ap


def gen_dir(outdir: Path, gen: int) -> Path:
    return outdir / f"gen_{gen:04d}"


def main():
    ap = build_parser()
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (PROJECT_ROOT / outdir).resolve()

    gen = int(args.gen)
    net_chunk_id = int(args.net_chunk_id)
    ic_chunk_id = int(args.ic_chunk_id)
    progress_every = max(1, int(args.progress_every))

    cfg = load_pickle(outdir / "config.pkl")
    launcher = load_json(outdir / "launcher.json")

    input_dir = Path(launcher["input_dir"])
    if not input_dir.is_absolute():
        input_dir = (PROJECT_ROOT / input_dir).resolve()

    tfine = np.load(outdir / "tfine.npy").astype(float)

    gdir = gen_dir(outdir, gen)
    pop = load_pickle(gdir / "pop.pkl")
    ic_pairs_gen = np.load(gdir / "ic_pairs.npy").astype(float)
    ic_global_idx_gen = np.load(gdir / "ic_pool_idx.npy").astype(int)

    net_ranges, ic_ranges = make_chunk_grid(
        n_pop=cfg.n_pop,
        n_ic=cfg.n_ic_per_gen,
        n_net_chunks=cfg.n_net_chunks,
        n_ic_chunks=cfg.n_ic_chunks,
    )

    if net_chunk_id < 0 or net_chunk_id >= len(net_ranges):
        raise IndexError(f"net_chunk_id={net_chunk_id} out of range")
    if ic_chunk_id < 0 or ic_chunk_id >= len(ic_ranges):
        raise IndexError(f"ic_chunk_id={ic_chunk_id} out of range")

    net_start, net_end = net_ranges[net_chunk_id]
    ic_start, ic_end = ic_ranges[ic_chunk_id]

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

    n_local_nets = net_end - net_start
    n_local_ics = ic_end - ic_start
    scores_block = np.full((n_local_nets, n_local_ics), np.nan, dtype=np.float32)

    t0_chunk = time.perf_counter()

    _append_status(
        str(outdir),
        f"[chunk-start] gen={gen} net_chunk={net_chunk_id} ic_chunk={ic_chunk_id} "
        f"nets={net_start}:{net_end} (count={n_local_nets}) "
        f"ics={ic_start}:{ic_end} (count={n_local_ics}) "
        f"progress_every={progress_every}"
    )

    for local_net_idx, global_net_idx in enumerate(range(net_start, net_end)):
        net = pop[global_net_idx]

        # Build field once per network
        field, nn_norm, _ = build_field(
            net, tfine,
            mode=cfg.field_mode,
            spline_k=cfg.spline_k,
            spline_n=cfg.spline_n,
        )

        for local_ic_idx, global_ic_pos in enumerate(range(ic_start, ic_end)):
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
                mode_mass=BI_MASS,
                cfg=cfg,
                t_in_ps=T_IN_PS,
                net_idx=global_net_idx,
                global_ic_idx=global_ic_idx,
                gen=gen,
            )
            scores_block[local_net_idx, local_ic_idx] = np.float32(score_val)

        done = local_net_idx + 1
        if (done % progress_every == 0) or (done == n_local_nets):
            elapsed = time.perf_counter() - t0_chunk
            rate = elapsed / max(done, 1)
            remaining = rate * max(n_local_nets - done, 0)

            _append_status(
                str(outdir),
                f"[chunk-progress] gen={gen} net_chunk={net_chunk_id} ic_chunk={ic_chunk_id} "
                f"done={done}/{n_local_nets} "
                f"elapsed={elapsed:.1f}s eta={remaining:.1f}s"
            )

    chunk_path = gdir / "chunks" / f"chunk_net{net_chunk_id:03d}_ic{ic_chunk_id:03d}.npz"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        chunk_path,
        gen=np.int32(gen),
        net_chunk_id=np.int32(net_chunk_id),
        ic_chunk_id=np.int32(ic_chunk_id),
        net_start=np.int32(net_start),
        net_end=np.int32(net_end),
        ic_start=np.int32(ic_start),
        ic_end=np.int32(ic_end),
        scores_block=np.asarray(scores_block, dtype=np.float32),
    )

    elapsed_total = time.perf_counter() - t0_chunk
    _append_status(
        str(outdir),
        f"[chunk-done] gen={gen} net_chunk={net_chunk_id} ic_chunk={ic_chunk_id} "
        f"nets={n_local_nets} ics={n_local_ics} elapsed={elapsed_total:.1f}s "
        f"saved={chunk_path.name}"
    )


if __name__ == "__main__":
    main()
