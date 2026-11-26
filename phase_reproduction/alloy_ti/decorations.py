"""Exact enumeration of the 28 Ti6Al2 decorations in a 2x2x1 HCP cell."""
import argparse
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "atomistic"))
from hcp_orbits import ORBIT_NAMES, VREF, build_orbits, features, sublattices

SHAPE = (2, 2, 1)
N = 8
KB = 8.617333262145e-5


def representatives():
    """Al site indices in the order returned by hcp_orbits.hcp_geometry(SHAPE)."""
    return {"d019": (0, 7), "interlayer": (0, 1), "basal": (0, 2)}


def symmetry_permutations():
    """Find affine HCP operations preserving the supercell and its two-site basis.

    Use integer coordinates in sixths of the primitive basis vectors. In a
    60-degree basal basis, the 12 planar isometries have entries 0, +1 or -1.
    Each operation maps the origin to a site of the parent crystal. Test every
    such translation and both signs of z, retaining distinct site permutations.
    """
    sites = np.array([(6*i + 2*b, 6*j + 2*b, 3*b)
                      for i, j, b in product(range(2), range(2), range(2))])
    index = {tuple(s): i for i, s in enumerate(sites)}
    metric = np.array([[2, 1], [1, 2]])
    periods = np.array([12, 12, 6])
    permutations = set()
    for entries in product((-1, 0, 1), repeat=4):
        R = np.reshape(entries, (2, 2))
        if not np.array_equal(R.T @ metric @ R, metric):
            continue
        planar = sites[:, :2] @ R.T
        for sign in (-1, 1):
            transformed = np.column_stack((planar, sign * sites[:, 2]))
            for origin in sites:
                ids = tuple(index.get(tuple(p), -1) for p in (transformed + origin) % periods)
                if len(set(ids)) == N and -1 not in ids:
                    permutations.add(ids)
    return permutations


def decoration_orbits():
    operations = symmetry_permutations()
    groups = {name: sorted({tuple(sorted((p[i], p[j]))) for p in operations})
              for name, (i, j) in representatives().items()}
    pairs = [pair for orbit in groups.values() for pair in orbit]
    if len(pairs) != len(set(pairs)) or set(pairs) != set(combinations(range(N), 2)):
        raise RuntimeError("The representative orbits must partition all 28 decorations")
    colors = sublattices(SHAPE)
    ordered = {pair for pair in pairs if colors[pair[0]] == colors[pair[1]]}
    if ordered != set(groups["d019"]):
        raise RuntimeError("The D019 orbit does not match the four D019 sublattice choices")
    return groups


def calculate(grid, temperature=1000.):
    groups = decoration_orbits()
    geometric_orbits = build_orbits(SHAPE)
    with np.load(grid, allow_pickle=False) as data:
        if tuple(data["orbit_names"]) != ORBIT_NAMES:
            raise ValueError("ECI orbit names do not match the TiAl project basis")
        iv, = np.flatnonzero(np.isclose(data["volume_A3_atom"], VREF, rtol=0, atol=1e-8))
        it, = np.flatnonzero(np.isclose(data["temperature_K"], temperature, rtol=0, atol=1e-8))
        static = data["J_static"][iv]
        ha = static + data["J_vib"][iv, it]
    result = {"temperature_K": temperature, "V_A3_atom": VREF, "shape": SHAPE,
              "N": N, "N_Al": 2, "source_grid_sha256": hashlib.sha256(grid.read_bytes()).hexdigest(),
              "unique_site_permutations": len(symmetry_permutations()),
              "number_of_decorations": sum(map(len, groups.values())), "orbits": {}}
    for name, members in groups.items():
        vectors = []
        for pair in members:
            spins = np.full(N, -1)
            spins[list(pair)] = 1
            vectors.append(features(spins, geometric_orbits))
        if not np.allclose(vectors, vectors[0], rtol=0, atol=1e-12):
            raise RuntimeError("Symmetry-equivalent decorations have different CE correlations")
        result["orbits"][name] = {
            "representative_Al_indices": representatives()[name], "degeneracy": len(members),
            "member_Al_indices": members, "static_eV_atom": float(vectors[0] @ static),
            "quantum_HA_eV_atom": float(vectors[0] @ ha)}
    for model in ("static", "quantum_HA"):
        entries = result["orbits"].values()
        root = result["orbits"]["d019"][f"{model}_eV_atom"]
        energies = np.array([r[f"{model}_eV_atom"] for r in entries])
        counts = np.array([r["degeneracy"] for r in entries])
        weights = counts * np.exp(-N*(energies - min(energies)) / (KB*temperature))
        result[f"canonical_{model}_eV_atom"] = float(min(energies) - KB*temperature/N*np.log(sum(weights)))
        for entry, energy, prob in zip(entries, energies, weights / sum(weights)):
            entry[f"{model}_relative_to_d019_meV_atom"] = float((energy-root)*1000)
            entry[f"{model}_orbit_probability"] = float(prob)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=PROJECT / "author_comparison/eci/author_refit_grid.npz")
    parser.add_argument("--temperature", type=float, default=1000.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(calculate(args.grid, args.temperature), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
