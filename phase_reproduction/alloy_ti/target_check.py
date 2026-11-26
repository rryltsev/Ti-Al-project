"""Independently sample each N=8 MTP Voronoi basin at the physical endpoint.

Use pCN proposals reversible for the projected harmonic Gaussian. The
Metropolis correction then contains just U_MTP - E_min - U_harmonic.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np


def mean_and_se(independent):
    independent = np.asarray(independent)
    return {"mean": float(independent.mean()),
            "se": float(independent.std(ddof=1) / np.sqrt(independent.size))}


def sample_chains(z, excess_energy, kt, rng, burn, steps, rho):
    """Positions are whitened harmonic coordinates. Independent chains occupy rows."""
    delta, wall = excess_energy(z)
    if not np.isfinite(delta).all():
        raise ValueError("Every initial chain must be in the specified basin")
    nch = len(z)
    trace = np.empty((steps, nch))
    walls = np.empty_like(trace)
    accepted = np.zeros(nch, dtype=int)
    attempted_outside = np.zeros(nch, dtype=int)
    for k in range(burn + steps):
        proposal = rho*z + np.sqrt(1-rho**2)*rng.normal(size=z.shape)
        proposed_delta, proposed_wall = excess_energy(proposal)
        move = np.log(rng.random(nch)) < (delta - proposed_delta) / kt
        z[move] = proposal[move]
        delta[move] = proposed_delta[move]
        wall[move] = proposed_wall[move]
        if k >= burn:
            trace[k-burn] = delta
            walls[k-burn] = wall
            accepted += move
            attempted_outside += proposed_wall >= 1
    return trace, walls, accepted/steps, attempted_outside/steps


def diagnostics(trace, walls, accepted, attempted_outside):
    cut = len(trace)//2
    per_chain = trace.mean(axis=0)
    half_a = trace[:cut].mean(axis=0)
    half_b = trace[cut:].mean(axis=0)
    ncenter = trace.shape[1]//2
    center = mean_and_se(per_chain[:ncenter])
    broad = mean_and_se(per_chain[ncenter:])
    halves = np.concatenate((trace[:cut], trace[cut:]), axis=1)
    within = halves.var(axis=0, ddof=1).mean()
    rhat = (np.sqrt((cut-1)/cut + halves.mean(axis=0).var(ddof=1)/within)
            if within > 0 else None)
    return {
        "target_integrand_eV_cell": mean_and_se(per_chain),
        "production_first_half_eV_cell": mean_and_se(half_a),
        "production_second_half_eV_cell": mean_and_se(half_b),
        "paired_second_minus_first_eV_cell": mean_and_se(half_b-half_a),
        "center_start_eV_cell": center, "broad_start_eV_cell": broad,
        "broad_minus_center_eV_cell": {
            "mean": broad["mean"]-center["mean"], "se": float(np.hypot(broad["se"], center["se"]))},
        "split_rhat_integrand": None if rhat is None else float(rhat),
        "acceptance_fraction": mean_and_se(accepted),
        "min_chain_acceptance": float(np.min(accepted)),
        "proposal_fraction_outside": mean_and_se(attempted_outside),
        "target_p_within_ten_pct_of_wall": mean_and_se((walls > .9).mean(axis=0)),
    }


def run_case(case, model, mlp, chains, burn, steps, rho):
    from scipy.linalg import null_space
    from decorations import representatives
    from pilot import KB, Frame, evaluate, relax_fixed_cell, hcp_geometry, force_constants, site_planes

    pair = representatives()[case]
    cell, sites = hcp_geometry((2, 2, 1))
    n = len(sites)
    types = np.zeros(n, dtype=int)
    types[list(pair)] = 1
    kt = 1000.*KB
    calc = partial(evaluate, binary=mlp, model=model, version=3)
    frame = relax_fixed_cell(Frame(cell, sites.copy(), types), calc)
    basis = null_space(np.tile(np.eye(3), (1, n)))
    h = force_constants(calc, frame, .0025)
    vals, vecs = np.linalg.eigh(basis.T @ h @ basis)
    if vals.min() <= 0:
        raise ValueError(f"Unstable harmonic basis for {case}")
    transform = (basis @ vecs * np.sqrt(kt/vals)).T
    planes = site_planes(cell, sites)

    def geometry(z):
        positions = frame.positions + (z @ transform).reshape(-1,n,3)
        wall = np.einsum("sni,nki->snk", positions-sites, planes).max(axis=(1,2))
        return positions, wall

    def excess_energy(z):
        positions, wall = geometry(z)
        delta = np.full(len(z), np.inf)
        inside = np.flatnonzero(wall < 1.)
        if inside.size:
            energies = np.array([r.energy for r in calc(
                [Frame(cell, positions[i], types) for i in inside])])
            delta[inside] = energies-frame.energy-.5*kt*np.sum(z[inside]**2, axis=1)
        return delta, wall

    # The first half starts at the relaxed minimum. The other half starts from
    # distinct overdispersed states so that convergence from the tails is visible.
    seed = [731921, sum(1 << i for i in pair)]
    rng = np.random.default_rng(seed)
    z = np.zeros((chains, len(vals)))
    pending = np.arange(chains//2, chains)
    for attempt in range(100):
        if not pending.size:
            break
        z[pending] = 1.35*rng.normal(size=(pending.size, len(vals)))
        pending = pending[geometry(z[pending])[1] >= 1.]
    if pending.size:
        raise RuntimeError("Could not initialize the broad-start group in the basin")
    trace, walls, acceptance, out = sample_chains(z, excess_energy, kt, rng, burn, steps, rho)
    np.savez_compressed(f"target_trace_{case}.npz", delta_eV_cell=trace,
                        max_wall_fraction=walls, acceptance=acceptance,
                        outside_fraction=out, seed=np.array(seed))
    return case, diagnostics(trace, walls, acceptance, out)


def comparison(direct, sample_file, kt):
    # Leave one independent proposal block out, preserving normalization of the
    # pooled self-normalized importance estimator of the physical expectation.
    with np.load(sample_file) as f:
        delta = f["delta_eV_cell"]
        lw = f["log_harmonic_to_proposal"]-delta/kt
    weights = np.exp(lw-lw.max())
    num = (weights*delta).sum(axis=1)
    den = weights.sum(axis=1)
    estimate = float(num.sum()/den.sum())
    jk = (num.sum()-num)/(den.sum()-den)
    se = float(np.sqrt((len(jk)-1)*np.mean((jk-jk.mean())**2)))
    diff = direct["mean"]-estimate
    combined_se = float(np.hypot(direct["se"],se))
    return {"importance_integrand_eV_cell": {"mean": estimate, "se_jackknife": se},
            "direct_minus_importance_eV_cell": diff, "combined_se_eV_cell": combined_se,
            "difference_in_combined_se": diff/combined_se if combined_se else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True)
    run = cmd.add_parser("run")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--mlp", required=True)
    run.add_argument("--chains", type=int, default=48)
    run.add_argument("--burn", type=int, default=128)
    run.add_argument("--steps", type=int, default=384)
    run.add_argument("--rho", type=float, default=.70, help="pCN correlation with current coordinates")
    run.add_argument("--workers", type=int, default=3)
    compare = cmd.add_parser("compare")
    compare.add_argument("--target", type=Path, default=Path("target_check.json"))
    compare.add_argument("--samples", type=Path, required=True)
    compare.add_argument("--output", type=Path, default=Path("target_comparison.json"))
    args = parser.parse_args()
    if args.command == "compare":
        obj = json.loads(args.target.read_text())
        kt = obj["kBT_eV"]
        results = {name: comparison(case["target_integrand_eV_cell"],
                                   args.samples/f"samples_{name}.npz", kt)
                   for name, case in obj["orbits"].items()}
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
        return
    from decorations import representatives
    from pilot import HASH, KB
    if args.chains < 4 or args.chains % 2 or args.burn < 1 or args.steps < 4 or args.steps % 2:
        parser.error("Use an even chain count >= 4, positive burn-in and an even step count >= 4")
    if not 0 <= args.rho < 1 or args.workers < 1:
        parser.error("Check rho and worker count")
    model = args.model.resolve()
    if hashlib.sha256(model.read_bytes()).hexdigest() != HASH:
        raise ValueError("MTP checksum mismatch")
    fn = partial(run_case, model=str(model), mlp=args.mlp, chains=args.chains,
                 burn=args.burn, steps=args.steps, rho=args.rho)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = dict(pool.map(fn, representatives()))
    output = dict(model_sha256=HASH, T_K=1000., kBT_eV=1000.*KB, chains=args.chains,
                  burn=args.burn, production_steps=args.steps, pcn_rho=args.rho, orbits=results)
    Path("target_check.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
