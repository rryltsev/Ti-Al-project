"""Static and classical Gamma-point MTP statistics on all small-cell decorations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from math import comb
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import null_space
from scipy.special import logsumexp

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "atomistic"))
from hcp_orbits import VREF, hcp_geometry

KB = 8.617333262145e-5
N = 16
HASH = "704f7fd900c7446286066b741084c30ebc7fcf26e5b9b8fd2f29a18ba4ffd661"


def sample(task):
    from label_cell import evaluate, relax_fixed_cell
    from mtp_cfg import Frame
    from pilot import force_constants, site_planes
    entry, binary, model, kt = task
    cell, sites = hcp_geometry((2, 2, 2))
    indices = entry["representative_Al_indices"]
    types = np.zeros(N, dtype=int)
    types[indices] = 1
    def calc(frames):
        return evaluate(frames, binary, model, 3)
    domain = site_planes(cell, sites)
    basis = null_space(np.tile(np.eye(3), (1, N)))
    def optimize(positions):
        relaxed = relax_fixed_cell(Frame(cell, positions, types), calc)
        # Fix the arithmetic displacement center; translation costs zero energy.
        relaxed.positions = relaxed.positions - np.mean(relaxed.positions-sites, axis=0)
        max_wall = float(np.einsum("ni,nki->nk", relaxed.positions-sites, domain).max())
        return relaxed, max_wall
    def curvature(frame, step):
        hessian = force_constants(calc, frame, step)
        return np.linalg.eigh(basis.T @ hessian @ basis)
    root, max_wall = optimize(sites.copy())
    vals, eig = curvature(root, .0025)
    attempts = []
    # The ideal decoration can begin at a stationary saddle. Follow its most
    # unstable direction in both senses, checking that each atom keeps its site.
    if vals[0] < 0 and max_wall < 1:
        vec = (basis @ eig[:, 0]).reshape(N, 3)
        vec *= .10 / np.linalg.norm(vec, axis=1).max()
        candidates = []
        for sign in (-1, 1):
            shifted, wall = optimize(root.positions + sign*vec)
            sv, _ = curvature(shifted, .0025)
            attempts.append(dict(sign=sign, energy_eV=shifted.energy, min_curvature=float(sv[0]),
                                 max_wall_fraction=wall))
            if wall < 1 and sv[0] > 0 and shifted.energy <= root.energy:
                candidates.append((shifted, wall, sv))
        if candidates:
            root, max_wall, vals = min(candidates, key=lambda item: item[0].energy)
    coarse, _ = curvature(root, .005)
    stable = bool(vals[0] > 0 and coarse[0] > 0 and max_wall < 1)
    harmonic = None
    if stable:
        # Integral in 3N-3 orthonormal Cartesian displacement coordinates.
        # Measure dr/(1 Angstrom) by each coordinate. Kinetic terms omitted
        # for every composition; the chosen coordinate gauge is common to all.
        harmonic = float(.5*kt*np.log(vals/(2*np.pi*kt)).sum())
    return {
        "id": entry["id"], "N_Al": len(indices), "representative_Al_indices": indices,
        "degeneracy": entry["degeneracy"], "relaxed_positions_A": root.positions.tolist(),
        "energy_eV_cell": root.energy, "ce_static_eV_cell": N*entry["static_eV_atom"],
        "static_minus_ce_meV_atom": float(1000*(root.energy/N-entry["static_eV_atom"])),
        "gamma_stable_in_site_domain": stable, "max_wall_fraction": max_wall,
        "min_curvature_eV_A2": float(vals[0]), "coarse_min_curvature_eV_A2": float(coarse[0]),
        "gamma_curvatures_eV_A2": vals.tolist(), "symmetry_breaking_trials": attempts,
        "fmax_eV_A": float(np.linalg.norm(root.forces, axis=1).max()),
        "harmonic_config_eV_cell": harmonic,
        "harmonic_step_change_meV_atom": float(500*kt/N*np.log(vals/coarse).sum()) if stable else None,
    }


def aggregate(results, kt):
    groups = {}
    for k in (2, 3, 4):
        rows = [r for r in results if r["N_Al"] == k]
        counts = np.array([r["degeneracy"] for r in rows])
        if sum(counts) != comb(N, k):
            raise ValueError(f"Incomplete enumeration at k={k}")
        def free_energy(values):
            return float(-kt*logsumexp(np.log(counts)-np.asarray(values)/kt))
        unresolved = [r["id"] for r in rows if not r["gamma_stable_in_site_domain"]]
        ce = free_energy([r["ce_static_eV_cell"] for r in rows])
        minimum_energy_sum = None
        harmonic_sum = None
        if not unresolved:
            minimum_energy_sum = free_energy([r["energy_eV_cell"] for r in rows])
            harmonic_sum = free_energy([r["energy_eV_cell"]+r["harmonic_config_eV_cell"] for r in rows])
        groups[k] = dict(number_of_orbits=len(rows), number_of_decorations=int(sum(counts)),
                         unresolved_orbits=unresolved, ce_static_eV_cell=ce,
                         direct_static_eV_cell=minimum_energy_sum,
                         direct_static_plus_classical_gamma_config_eV_cell=harmonic_sum)
    derivatives = {}
    for field in ("ce_static_eV_cell", "direct_static_eV_cell",
                  "direct_static_plus_classical_gamma_config_eV_cell"):
        f2,f3,f4 = [groups[k][field] for k in (2,3,4)]
        mu_left = f3-f2 if f2 is not None and f3 is not None else None
        mu_right = f4-f3 if f3 is not None and f4 is not None else None
        curvature = mu_right-mu_left if mu_right is not None and mu_left is not None else None
        derivatives[field] = dict(mu_2_to_3_eV=mu_left, mu_3_to_4_eV=mu_right,
                                  central_second_difference_eV=curvature,
                                  f_double_prime_eV_atom_per_fraction2=N*curvature if curvature is not None else None)
    delta = {}
    for key, direct in derivatives["direct_static_eV_cell"].items():
        delta[key] = direct-derivatives["ce_static_eV_cell"][key] if direct is not None else None
    derivatives["direct_static_minus_ce"] = delta
    return groups, derivatives


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--mlp", required=True)
    cli.add_argument("--baseline", type=Path, default=Path("c46_N16_1000K.json"))
    cli.add_argument("--workers", type=int, default=3)
    cli.add_argument("--output", type=Path, default=Path("n16_static.json"))
    args = cli.parse_args()
    model = args.model.resolve(strict=True)
    if sha256(model.read_bytes()).hexdigest() != HASH:
        raise ValueError("MTP checksum mismatch")
    cb = json.loads(args.baseline.read_text())
    if (cb["shape"] != [2,2,2] or cb["N"] != N or cb["temperature_K"] != 1000 or
        not np.isclose(cb["V_A3_atom"], VREF) or sorted(cb["compositions"]) != ["2","3","4"]):
        raise ValueError("Unexpected C46 baseline geometry, compositions or temperature")
    kt = KB*cb["temperature_K"]
    entries = [e for c in cb["compositions"].values() for e in c["orbits"]]
    if len(entries) != 43 or len(set(e["id"] for e in entries)) != len(entries):
        raise ValueError("Expected 43 distinct symmetry representatives")
    tasks = [(e,args.mlp,str(model),kt) for e in entries]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(sample,tasks))
    groups, derivatives = aggregate(results,kt)
    for k, row in groups.items():
        expected = N*cb["compositions"][str(k)]["static"]["canonical_eV_atom"]
        if not np.isclose(row["ce_static_eV_cell"],expected,rtol=0,atol=1e-10):
            raise RuntimeError("Inconsistent C46 normalization")
    output = dict(temperature_K=cb["temperature_K"], model_sha256=HASH,
                  V_A3_atom=VREF, N=N, shape=cb["shape"], results=results, compositions=groups,
                  chemical_derivatives=derivatives,
                  harmonic_convention="Gamma only; arithmetic-mean displacement fixed; Cartesian coordinate integral in Angstrom units, kinetic factors omitted")
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
