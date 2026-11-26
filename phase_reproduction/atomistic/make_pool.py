"""Make a deterministic, independently reconstructed Ti-Al CE data design."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from hcp_orbits import ORBIT_NAMES, VOLUME_RATIOS, build_orbits, features, hcp_geometry, sublattices
from mtp_cfg import Frame, write_cfg


def decoration(shape, n_al, ordered, rng):
    n = 2*math.prod(shape)
    if ordered:
        preferred = sublattices(shape) == 0
        order = np.concatenate([rng.permutation(np.flatnonzero(preferred)),
                                rng.permutation(np.flatnonzero(~preferred))])
    else:
        order = rng.permutation(n)
    spins = -np.ones(n, dtype=np.int8)
    spins[order[:n_al]] = 1
    return spins


def generate(seed=20261124):
    rng = np.random.default_rng(seed)
    cache, seen = {}, set()
    def entry(shape, n_al, ordered=False):
        if shape not in cache:
            cache[shape] = build_orbits(shape)
        for _ in range(100):
            s = decoration(shape, n_al, ordered, rng)
            x = features(s, cache[shape])
            key = tuple(np.round(x, 10))
            if key not in seen:
                seen.add(key)
                return (shape, s, x, ordered)
        return None

    train = [entry((1, 1, 1), 0), entry((2, 2, 1), 2, True)]
    control = []
    for al, split in [(17, train), (20, train), (22, control)]:
        for ordering in (False, True):
            split.append(entry((4, 4, 2), al, ordering))
    for al, order in [(3, False), (5, True), (7, False), (9, True), (10, False)]:
        control.append(entry((4, 2, 2), al, order))
    # Choose the additional design rows by information gain after reserving
    # controls; no E or F labels enter the choice.
    shapes = [(2,2,1), (2,2,2), (3,2,1), (3,2,2), (4,2,2), (4,4,2)]
    candidates = []
    for _ in range(900):
        shape = shapes[rng.integers(len(shapes))]
        n_al = int(rng.integers(1, math.prod(shape)+1))
        ordered = shape[0] % 2 == 0 and rng.random() < .32
        candidate = entry(shape, n_al, ordered)
        if candidate is not None:
            candidates.append(candidate)
    x = np.stack([c[2] for c in candidates])
    scale = np.maximum(np.linalg.norm(x, axis=0)/np.sqrt(len(x)), 1e-8)
    x /= scale
    used = set()
    while len(train) < 39:
        a = np.stack([c[2] for c in train]) / scale
        inv = np.linalg.inv(a.T @ a + 1e-6*np.eye(len(ORBIT_NAMES)))
        leverage = np.einsum("ij,jk,ik->i", x, inv, x)
        leverage[list(used)] = -np.inf
        idx = int(np.argmax(leverage))
        train.append(candidates[idx])
        used.add(idx)
    return train, control


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--seed", type=int, default=20261124)
    args = p.parse_args()
    if args.output.exists():
        p.error("choose a fresh output directory")
    train, control = generate(args.seed)
    if any(row is None for row in train + control):
        raise RuntimeError("Failed to generate a distinct correlation vector for a required decoration")
    args.output.mkdir(parents=True)
    (args.output/"cells").mkdir()
    with (args.output/"design.csv").open("w", newline="") as table, (args.output/"jobs.csv").open("w", newline="") as jobs:
        w, j = csv.writer(table), csv.writer(jobs)
        w.writerow(["id", "split", "cell", "cell_sha256", "shape", "N", "N_Al", "D019_preferred", *ORBIT_NAMES])
        j.writerow(["id", "ratio", "cell", "repeat_a", "repeat_b", "repeat_c", "mesh", "label_subdir"])
        for split, entries in [("train", train), ("control", control)]:
            for index, (shape, spins, corr, ordered) in enumerate(entries):
                ident = f"{split}_{index:02d}"
                n = len(spins)
                rel = f"cells/{ident}.cfg"
                path = args.output/rel
                cell, pos = hcp_geometry(shape)
                write_cfg(path, [Frame(cell, pos, (spins > 0).astype(int))])
                w.writerow([ident, split, rel, hashlib.sha256(path.read_bytes()).hexdigest(),
                            "x".join(map(str, shape)), n, np.sum(spins > 0), int(ordered), *corr])
                repeats = [math.ceil(target/size) for target, size in zip([8,8,6], shape)]
                for ratio in VOLUME_RATIOS:
                    j.writerow([ident, ratio, rel, *repeats, 16 if n <= 32 else 6, f"{ident}_v{ratio:.2f}"])
    x = np.stack([t[2] for t in train])
    info = {"source":"independent reconstruction, not the authors' 46 decorations",
            "seed":args.seed, "n_train":len(train), "n_control":len(control),
            "rank":int(np.linalg.matrix_rank(x)), "condition_number":float(np.linalg.cond(x)),
            "orbit_names":ORBIT_NAMES}
    if info["rank"] != len(ORBIT_NAMES):
        raise RuntimeError(f"Singular design: {info}")
    (args.output/"design.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
