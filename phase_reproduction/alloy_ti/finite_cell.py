"""Enumerate Ti-Al decorations at several Al counts using the parent HCP space group."""
import argparse
from hashlib import sha256
from itertools import combinations, product
import json
from math import comb
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "atomistic"))
from hcp_orbits import ORBIT_NAMES, VREF, build_orbits, features, sublattices

KB = 8.617333262145e-5


def parent_permutations(shape):
    """Exact affine operations of an HCP supercell with equal basal repeats."""
    nx, ny, nz = shape
    if nx != ny or min(shape) < 1:
        raise ValueError("Use positive repeats with equal basal periods")
    lattice = product(range(nx), range(ny), range(nz), range(2))
    sites = np.array([(6*i+2*b, 6*j+2*b, 6*k+3*b) for i, j, k, b in lattice])
    index = {tuple(p): j for j, p in enumerate(sites)}
    metric = np.array([[2, 1], [1, 2]])  # twice the 60-degree basal metric
    periods = 6*np.array(shape)
    permutations = set()
    for entries in product((-1, 0, 1), repeat=4):
        rotation = np.reshape(entries, (2, 2))
        if not np.array_equal(rotation.T @ metric @ rotation, metric):
            continue
        basal = sites[:, :2] @ rotation.T
        for sign in (-1, 1):
            transformed = np.column_stack((basal, sign * sites[:, 2]))
            for origin in sites:
                mapped = tuple(index.get(tuple(p), -1) for p in (transformed + origin) % periods)
                if -1 not in mapped and len(set(mapped)) == len(sites):
                    permutations.add(mapped)
    return np.array(sorted(permutations))


def decoration_orbits(permutations, n_al):
    """Yield sorted representatives and members (Al indices) at fixed composition."""
    n = permutations.shape[1]
    remaining = set(combinations(range(n), n_al))
    while remaining:
        representative = min(remaining)
        members = set(map(tuple, np.sort(permutations[:, representative], axis=1).tolist()))
        if not members <= remaining or representative not in members:
            raise RuntimeError("The parent operations did not produce disjoint complete orbits")
        remaining.difference_update(members)
        yield representative, sorted(members)


def canonical_statistics(entries, model, n, kt):
    energy = np.array([entry[f"{model}_eV_atom"] for entry in entries])
    counts = np.array([entry["degeneracy"] for entry in entries])
    lowest = np.min(energy)
    weights = counts*np.exp(-n*(energy-lowest)/kt)
    probability = weights/np.sum(weights)
    for entry, e, p in zip(entries, energy, probability):
        entry[f"{model}_orbit_probability"] = float(p)
        entry[f"{model}_above_lowest_meV_atom"] = float(1000*(e-lowest))
    sorted_p = np.argsort(-probability, kind="stable")
    covered = np.cumsum(probability[sorted_p])
    return {
        "canonical_eV_atom": float(lowest - kt/n*np.log(np.sum(weights))),
        "ranked_orbit_ids": [entries[i]["id"] for i in sorted_p],
        "top_orbits_probability": {str(i): float(covered[i-1]) for i in (1, 3, 5, 10) if i <= len(entries)},
        "orbits_for_cumulative_probability": {str(p): int(np.searchsorted(covered, p)+1)
                                             for p in (.90, .95, .99, .999)},
    }


def calculate(grid, shape, al_counts, temperature):
    with np.load(grid, allow_pickle=False) as data:
        if tuple(data["orbit_names"]) != ORBIT_NAMES:
            raise ValueError("ECI orbit names do not match the project basis")
        iv, = np.flatnonzero(np.isclose(data["volume_A3_atom"], VREF, rtol=0, atol=1e-8))
        it, = np.flatnonzero(np.isclose(data["temperature_K"], temperature, rtol=0, atol=1e-8))
        static = data["J_static"][iv]
        ha = static + data["J_vib"][iv, it]
    permutations = parent_permutations(shape)
    n = permutations.shape[1]
    if temperature <= 0 or any(k < 0 or k > n for k in al_counts):
        raise ValueError("Check temperature and Al count")
    clusters = build_orbits(shape)
    colors = sublattices(shape)
    result = {
        "temperature_K": temperature, "V_A3_atom": VREF, "shape": shape, "N": n,
        "source_grid_sha256": sha256(grid.read_bytes()).hexdigest(),
        "unique_site_permutations": len(permutations), "compositions": {},
    }
    for k in al_counts:
        entries = []
        for representative, members in decoration_orbits(permutations, k):
            initial = np.full(n, -1)
            initial[list(representative)] = 1
            x = features(initial, clusters)
            # Cross-check every periodic decoration, including small-cell self images.
            for member in members[1:]:
                spins = np.full(n, -1)
                spins[list(member)] = 1
                if not np.allclose(features(spins, clusters), x, rtol=0, atol=1e-12):
                    raise RuntimeError("Point-group operation changed CE correlations")
            occupancies = np.bincount(colors[list(representative)], minlength=4)
            entries.append({
                "id": f"k{k:02d}_" + "-".join(map(str, representative)),
                "representative_Al_indices": list(representative), "degeneracy": len(members),
                "Al_by_D019_sublattice": occupancies.tolist(),
                "static_eV_atom": float(x @ static), "quantum_HA_eV_atom": float(x @ ha),
            })
        total_count = sum(e["degeneracy"] for e in entries)
        if total_count != comb(n, k):
            raise RuntimeError("Decoration multiplicities do not sum to the binomial")
        result["compositions"][str(k)] = {
            "n_al": k, "number_of_decorations": total_count, "number_of_orbits": len(entries),
            "orbits": entries,
            **{model: canonical_statistics(entries, model, n, KB*temperature)
               for model in ("static", "quantum_HA")},
        }
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--grid", type=Path, default=PROJECT / "author_comparison/eci/author_refit_grid.npz")
    cli.add_argument("--shape", nargs=3, type=int, default=[2, 2, 2])
    cli.add_argument("--al-counts", nargs="+", type=int, default=[2, 3, 4])
    cli.add_argument("--temperature", type=float, default=1000.)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    result = calculate(args.grid, tuple(args.shape), args.al_counts, args.temperature)
    encoded = json.dumps(result, allow_nan=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
