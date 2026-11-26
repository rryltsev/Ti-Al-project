"""Classical, fixed-cell alloy TI by harmonic importance sampling of each decoration.

The arithmetic mean displacement is fixed. Atom labels refer to parent-site
Voronoi cells. At fixed composition all translation and kinetic factors cancel
between decorations. Independent mixture draws also estimate the harmonic
reference probability of staying inside the specified domain.
"""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.special import logsumexp

from decorations import representatives
from hcp_orbits import hcp_geometry
from label_cell import KB, evaluate, relax_fixed_cell
from mtp_cfg import Frame


HASH = "704f7fd900c7446286066b741084c30ebc7fcf26e5b9b8fd2f29a18ba4ffd661"
HBAR_EV = 0.0646541513  # hbar*sqrt((eV/Angstrom^2)/amu), eV
SCALES = np.array([1., 1.35])


def site_planes(cell, sites):
    shifts = np.array(list(product(range(-1, 2), repeat=3))) @ cell
    images = (sites[:, None] + shifts).reshape(-1, 3)
    planes = []
    for site in sites:
        vector = images - site
        distance = np.linalg.norm(vector, axis=1)
        # For this HCP cell the 12 nearest neighbours bound the Voronoi cell.
        use = (distance > 1e-6) & (distance < 3.5)
        if use.sum() != 12:
            raise ValueError("Verify the Voronoi coordination for this geometry")
        planes.append(2 * vector[use] / distance[use, None]**2)
    return np.array(planes)


def force_constants(calc, frame, step):
    ndim = frame.positions.size
    shifts = np.eye(ndim).reshape(ndim, -1, 3) * step
    frames = [Frame(frame.cell, frame.positions + sign * d, frame.types)
              for d in shifts for sign in (1, -1)]
    forces = np.array([r.forces.ravel() for r in calc(frames)])
    h = -(forces[::2] - forces[1::2]).T / (2 * step)
    return (h + h.T) / 2


def average(delta, log_ratio, lambdas, kt):
    """log_ratio is log(reference Gaussian / generating mixture), -inf outside."""
    logw = log_ratio[None, :] - lambdas[:, None] * delta[None, :] / kt
    logmean = logsumexp(logw, axis=1) - np.log(delta.size)
    w = np.exp(logw - logsumexp(logw, axis=1)[:, None])
    return logmean, w @ delta, 1 / np.sum(w*w, axis=1), np.max(w, axis=1)


def estimate(delta, log_ratio, kt):
    grid = np.linspace(0, 1, 65)
    logs, integrand, ess, max_w = average(delta, log_ratio, grid, kt)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return dict(fep_eV=float(-kt*logs[-1]),
                ti_eV=float(-kt*logs[0] + trapezoid(integrand, grid)),
                coarse_ti_eV=float(-kt*logs[0] + trapezoid(integrand[::4], grid[::4])),
                harmonic_domain_probability_IS=float(np.exp(logs[0])),
                ess_min=float(ess.min()), ess_endpoint=float(ess[-1]),
                max_weight_endpoint=float(max_w[-1]), lambda_grid=grid.tolist(),
                integrand_eV_cell=integrand.tolist())


def sample_case(args):
    name, indices, model, binary, nsamples, replicas, temperature = args
    cell, sites = hcp_geometry((2, 2, 1))
    n = len(sites)
    types = np.zeros(n, dtype=int)
    types[list(indices)] = 1
    def calc(frames):
        return evaluate(frames, binary, model, 3)
    frame = relax_fixed_cell(Frame(cell, sites.copy(), types), calc)
    h = force_constants(calc, frame, .005)
    h_fine = force_constants(calc, frame, .0025)
    basis = null_space(np.tile(np.eye(3), (1, n)))
    vals, vecs = np.linalg.eigh(basis.T @ h_fine @ basis)
    coarse = np.linalg.eigvalsh(basis.T @ h @ basis)
    if np.min(vals) <= 0 or np.min(coarse) <= 0:
        raise ValueError(f"Unstable reference in {name}")
    kt = KB * temperature
    # Only free-energy differences between decorations at the same N,x,T,V.
    # Use the same kinetic convention as the finite-cell quantum HA diagnostic.
    inv_mass = np.repeat(1/np.sqrt(np.where(types == 1, 26.9815385, 47.867)), 3)
    h_projected = basis @ (basis.T @ h_fine @ basis) @ basis.T
    mass_eigenvalues = np.linalg.eigvalsh(h_projected * np.outer(inv_mass, inv_mass))
    if np.max(abs(mass_eigenvalues[:3])) > 1e-7 or mass_eigenvalues[3] <= 0:
        raise ValueError("Expected three translations")
    eps = HBAR_EV * np.sqrt(mass_eigenvalues[3:])
    f_classical = kt * np.log(eps/kt).sum()
    f_quantum = (.5*eps + kt*np.log(-np.expm1(-eps/kt))).sum()
    planes = site_planes(cell, sites)
    transform = (basis @ vecs * np.sqrt(kt/vals)).T
    log_ratio, delta, wall = [], [], []
    for seed in range(replicas):
        rng = np.random.default_rng([25931, sum(1 << i for i in indices), seed])
        scale = SCALES[np.arange(nsamples) % len(SCALES)]
        z = rng.normal(size=(nsamples, len(vals))) * scale[:, None]
        r2 = np.sum(z*z, axis=1)
        log_components = (-len(vals)*np.log(SCALES) -
                          r2[:, None]/(2*SCALES**2))
        weights = -.5*r2 - logsumexp(log_components, axis=1) + np.log(len(SCALES))
        positions = frame.positions + (z @ transform).reshape(-1, n, 3)
        bounds = np.einsum("sni,nki->snk", positions-sites, planes).max(axis=(1,2))
        inside = np.flatnonzero(bounds < 1)
        weights[bounds >= 1] = -np.inf
        differences = np.zeros(nsamples)
        for start in range(0, len(inside), 256):
            ids = inside[start:start+256]
            energies = [r.energy for r in calc([Frame(cell, positions[i], types) for i in ids])]
            differences[ids] = np.array(energies) - frame.energy - .5*kt*r2[ids]
        log_ratio.append(weights)
        delta.append(differences)
        wall.append(bounds)
    log_ratio, delta, wall = map(np.asarray, (log_ratio, delta, wall))
    analysis = estimate(delta.ravel(), log_ratio.ravel(), kt)
    endlog = (log_ratio - delta/kt).ravel()
    endweights = np.exp(endlog-logsumexp(endlog))
    analysis.update(replica_fep_eV=[estimate(d, w, kt)["fep_eV"] for d, w in zip(delta, log_ratio)],
                    draws=int(delta.size), excluded_draws=int(np.sum(wall>=1)),
                    target_p_within_ten_pct_of_wall=float(endweights[wall.ravel()>.9].sum()),
                    static_eV_cell=frame.energy, harmonic_classical_eV_cell=float(f_classical),
                    harmonic_quantum_eV_cell=float(f_quantum), name=name, al_indices=list(indices),
                    hessian_step_change_meV_atom=float(500*kt/n*np.sum(np.log(vals/coarse))),
                    minimum_curvature_eV_A2=float(vals.min()),
                    relaxed_fmax_eV_A=float(np.linalg.norm(frame.forces, axis=1).max()))
    np.savez_compressed(f"samples_{name}.npz", delta_eV_cell=delta,
                        log_harmonic_to_proposal=log_ratio, max_wall_fraction=wall,
                        seed_prefix=np.array([25931, sum(1 << i for i in indices)]),
                        normal_scales=SCALES, eigenvalues_eV_A2=vals, relaxed_positions=frame.positions)
    return analysis


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--mlp", required=True)
    cli.add_argument("--samples", type=int, default=2048, help="balanced mixture draws per block")
    cli.add_argument("--replicas", type=int, default=8)
    cli.add_argument("--workers", type=int, default=3)
    args = cli.parse_args()
    if args.samples <= 0 or args.samples % 2 or args.replicas < 2:
        cli.error("need balanced proposal samples and independent replicates")
    model = args.model.resolve()
    if hashlib.sha256(model.read_bytes()).hexdigest() != HASH:
        raise ValueError("MTP checksum mismatch")
    reps = representatives()
    tasks = [(name, pairs, str(model), args.mlp, args.samples, args.replicas, 1000.)
             for name, pairs in reps.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        result = list(pool.map(sample_case, tasks))
    Path("pilot.json").write_text(json.dumps(dict(model_sha256=HASH, temperature_K=1000.,
        shape=[2,2,1], n_al=2, replicas=args.replicas, results=result), indent=2))


if __name__ == "__main__":
    main()
