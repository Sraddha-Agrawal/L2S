import json
import pickle
from pathlib import Path

import numpy as np


def read_ic_pairs(path):
    """
    File format: one pair per line:  x0  v0
    Comments allowed with '#'
    """
    pairs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        pairs.append((float(parts[0]), float(parts[1])))
    return pairs


def _sanitize_meta(meta):
    """
    Convert numpy types to standard Python types so JSON can serialize them.
    """
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (np.floating,)):
            clean[k] = float(v)
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.ndarray,)):
            clean[k] = v.tolist()
        else:
            clean[k] = v
    return clean


def save_gen_scores(outdir, gen, scores_matrix, fitness, meta=None):
    """
    Save GA scores + optional metadata.

    metadata may include thermostat info such as:
        integrator
        use_langevin
        thermostat_type
        temperature_K
        gamma
        kBT_internal
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.save(outdir / f"scores_gen_{gen:04d}.npy", scores_matrix.astype(np.float32))
    np.save(outdir / f"fitness_gen_{gen:04d}.npy", np.asarray(fitness, dtype=np.float32))

    if meta is not None:
        meta_clean = _sanitize_meta(meta)

        if meta_clean.get("use_langevin", False):
            meta_clean.setdefault("thermostat_type", "langevin")

        with open(outdir / f"meta_gen_{gen:04d}.json", "w") as f:
            json.dump(meta_clean, f, indent=2)


def save_gen_best(outdir, gen, ic_pairs, tfine, A_t, x_t, v_t, best_idx, ic_global_idx=None):
    """
    Store champion protocol trajectories for all ICs.

    x_t, v_t shape: (n_ic, n_t)
    A_t shape: (n_t,)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    payload = dict(
        gen=int(gen),
        best_idx=int(best_idx),
        ic_pairs=np.array(ic_pairs, dtype=np.float32),
        tfine=np.array(tfine, dtype=np.float32),
        A_t=np.array(A_t, dtype=np.float32),
        x_t=np.array(x_t, dtype=np.float32),
        v_t=np.array(v_t, dtype=np.float32),
    )
    if ic_global_idx is not None:
        payload["ic_global_idx"] = np.asarray(ic_global_idx, dtype=np.int32)

    np.savez_compressed(
        outdir / f"best_gen_{gen:04d}.npz",
        **payload,
    )


def save_pickle(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_sanitize_meta(obj), f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
