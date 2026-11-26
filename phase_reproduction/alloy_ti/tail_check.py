"""Interlayer Ti6Al2 basin at 1000 K: independent Gaussian / Student-t importance sampling."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from math import lgamma, log, pi
from pathlib import Path

import numpy as np


DF = 5
SCALE = 1.20
SEED = 8261937
KB = 8.617333262145e-5
KT = 1000.*KB


def proposal(count, dim, rng):
    """Balanced stratification. A shared chi-square variate per vector gives elliptical t."""
    if count % 2:
        raise ValueError("Balanced proposal needs an even number of draws")
    z = rng.normal(size=(count, dim))
    z[1::2] *= SCALE*np.sqrt(DF/rng.chisquare(DF, count//2))[:, None]
    r2 = np.sum(z*z, axis=1)
    log_normal = -.5*(r2 + dim*log(2*pi))
    log_t = (lgamma((DF+dim)/2)-lgamma(DF/2)-dim/2*log(DF*pi)-dim*log(SCALE)
             -(DF+dim)/2*np.log1p(r2/(DF*SCALE**2)))
    log_ratio = log(2.) + log_normal - np.logaddexp(log_normal, log_t)
    return z, r2, log_ratio


def summary(delta, log_ratio, kt=KT):
    """Pool target normalization and first moment, jackknife entire independent blocks."""
    if delta.ndim != 2 or delta.shape != log_ratio.shape or len(delta) < 2:
        raise ValueError("Need matching two-dimensional arrays with multiple equal-sized blocks")
    logs = log_ratio-delta/kt
    shift = np.max(logs, axis=1)
    if not np.all(np.isfinite(shift)):
        raise ValueError("No finite target weight in at least one block")
    relative = np.exp(logs-shift[:, None])
    block_sum = relative.sum(axis=1)
    log_block = shift+np.log(block_sum)
    block_moment = (relative*delta).sum(axis=1)/block_sum
    draws_per_block = delta.shape[1]
    def pool(log_norm, moment):
        total = np.logaddexp.reduce(log_norm)
        prob = np.exp(log_norm-total)
        return (-kt*(total-log(len(moment)*draws_per_block)), prob@moment, prob)
    value_f, value_i, block_probability = pool(log_block, block_moment)
    jk_f, jk_i = np.array([pool(np.delete(log_block,i), np.delete(block_moment,i))[:2]
                           for i in range(len(delta))]).T
    def result(value, jk):
        se = np.sqrt((len(jk)-1)*np.mean((jk-jk.mean())**2))
        return {"value_eV_cell": float(value), "jackknife_se_eV_cell": float(se)}
    fep = result(value_f, jk_f)
    integrand = result(value_i, jk_i)
    w = (relative/block_sum[:,None])*block_probability[:,None]
    stats = {
        "fep": fep, "endpoint_excess_energy": integrand, "draws": delta.size,
        "effective_samples": float(1./np.sum(w*w)), "largest_normalized_weight": float(w.max()),
        "target_weight_top_10_draws": float(np.sort(w.ravel())[-10:].sum()),
        "block_target_probability": block_probability.tolist(),
        "block_fep_eV_cell": (-kt*(log_block-np.log(draws_per_block))).tolist(),
        "block_endpoint_excess_eV_cell": block_moment.tolist(),
    }
    return stats, w


def subset(weights, mask):
    vals = np.where(mask, weights, 0.)
    prob = vals.sum()
    sumsq = np.sum(vals*vals)
    return {"proposal_draws": int(np.count_nonzero(mask)), "target_probability": float(prob),
            "effective_weighted_draws": float(prob**2/sumsq) if sumsq else 0.,
            "blocks_with_positive_target_weight": int(np.count_nonzero(vals.sum(axis=1) > 0))}


def sample(block, nsamples, mlp, model, cell, types, sites, positions0, e0, transform, planes):
    from pilot import Frame, evaluate
    dim = transform.shape[0]
    z, r2, ratio = proposal(nsamples, dim, np.random.default_rng([SEED, block]))
    positions = positions0 + (z @ transform).reshape(nsamples,len(sites),3)
    wall = np.einsum("sni,nki->snk", positions-sites, planes).max(axis=(1, 2))
    ratio[wall >= 1] = -np.inf
    delta = np.zeros(nsamples)
    inside = np.flatnonzero(wall < 1.)
    for start in range(0, len(inside), 256):
        ids = inside[start:start+256]
        frames = [Frame(cell, positions[i], types) for i in ids]
        energies = [r.energy for r in evaluate(frames, mlp, model, 3)]
        delta[ids] = np.asarray(energies)-e0-.5*KT*r2[ids]
    return delta, ratio, wall, r2


def run(args):
    from scipy.linalg import null_space
    from decorations import representatives
    from pilot import (HASH, Frame, evaluate, force_constants, hcp_geometry,
                       relax_fixed_cell, site_planes)
    model = args.model.resolve()
    if hashlib.sha256(model.read_bytes()).hexdigest() != HASH:
        raise ValueError("MTP checksum mismatch")
    cell, sites = hcp_geometry((2,2,1))
    types = np.zeros(len(sites), dtype=int)
    types[list(representatives()["interlayer"])] = 1
    calc = partial(evaluate, binary=args.mlp, model=str(model), version=3)
    frame = relax_fixed_cell(Frame(cell, sites.copy(), types), calc)
    h = force_constants(calc, frame, .0025)
    basis = null_space(np.tile(np.eye(3), (1, len(sites))))
    vals, vectors = np.linalg.eigh(basis.T @ h @ basis)
    if vals.min() <= 0:
        raise ValueError("Unstable harmonic reference")
    transform = (basis @ vectors * np.sqrt(KT/vals)).T
    fn = partial(sample, nsamples=args.samples, mlp=args.mlp, model=str(model), cell=cell,
                 types=types, sites=sites, positions0=frame.positions, e0=frame.energy,
                 transform=transform, planes=site_planes(cell, sites))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        blocks = list(pool.map(fn, range(args.blocks)))
    delta, ratio, wall, r2 = (np.stack(k) for k in zip(*blocks))
    np.savez_compressed("tail_samples.npz", delta_eV_cell=delta, log_harmonic_to_proposal=ratio,
                        max_wall_fraction=wall, whitened_radius_squared=r2, seed_prefix=SEED,
                        t_df=DF, t_scale=SCALE, eigenvalues_eV_A2=vals,
                        relaxed_positions=frame.positions)
    stats, w = summary(delta, ratio)
    dim = len(vals)
    t_draws = np.broadcast_to(np.arange(args.samples)%2==1, delta.shape)
    masks = {"student_component": t_draws, "gaussian_component": ~t_draws,
             "r2_gt_2d": r2 > 2*dim, "r2_gt_3d": r2 > 3*dim,
             "near_wall_inside": (wall > .9)&(wall < 1.), "outside": wall >= 1.,
             "excess_below_minus_5kBT": (delta < -5*KT)&(wall < 1.)}
    subsets = {label: subset(w, mask) for label,mask in masks.items()}
    # Nested prefixes preserve the balanced deterministic mixture in each block.
    prefixes = {}
    for end in (args.samples//4, args.samples//2, args.samples):
        pre, _ = summary(delta[:, :end], ratio[:, :end])
        prefixes[str(end)] = {k: pre[k] for k in ("fep", "endpoint_excess_energy",
                                                   "effective_samples", "largest_normalized_weight")}
    harmonic_probability = np.exp(ratio)
    hblocks = harmonic_probability.mean(axis=1)
    output = dict(model_sha256=HASH, T_K=1000., kBT_eV=KT, name="interlayer", dim=dim,
                  n_blocks=args.blocks, balanced_draws_per_block=args.samples, t_df=DF, t_scale=SCALE,
                  seed_prefix=SEED, harmonic_probability_inside=float(hblocks.mean()),
                  harmonic_probability_inside_se=float(hblocks.std(ddof=1)/np.sqrt(len(hblocks))),
                  pooled=stats, tail_diagnostics=subsets, convergence_by_draws_per_block=prefixes)
    Path("tail_check.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")


def compare(args):
    old_new = []
    for f in (args.original, args.new):
        with np.load(f) as data:
            res, _ = summary(data["delta_eV_cell"], data["log_harmonic_to_proposal"])
            old_new.append(res)
    comparison = {"original_sha256": hashlib.sha256(args.original.read_bytes()).hexdigest(),
                  "new_sha256": hashlib.sha256(args.new.read_bytes()).hexdigest()}
    for name in ("fep", "endpoint_excess_energy"):
        old, new = (res[name] for res in old_new)
        diff = new["value_eV_cell"]-old["value_eV_cell"]
        se = float(np.hypot(new["jackknife_se_eV_cell"],old["jackknife_se_eV_cell"]))
        comparison[name] = dict(original=old, independent_student_mixture=new,
                                new_minus_original_eV_cell=diff, combined_se_eV_cell=se,
                                new_minus_original_meV_atom=125.*diff, combined_se_meV_atom=125.*se,
                                difference_in_combined_se=diff/se if se else None)
    args.output.write_text(json.dumps(comparison, indent=2, allow_nan=False)+"\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    r = commands.add_parser("run")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--mlp", required=True)
    r.add_argument("--blocks", type=int, default=8)
    r.add_argument("--samples", type=int, default=8192)
    r.add_argument("--workers", type=int, default=8)
    c = commands.add_parser("compare")
    c.add_argument("--original", type=Path, required=True)
    c.add_argument("--new", type=Path, default=Path("tail_samples.npz"))
    c.add_argument("--output", type=Path, default=Path("tail_comparison.json"))
    args = cli.parse_args()
    if args.command == "compare":
        compare(args)
        return
    if args.blocks < 2 or args.samples < 8 or args.samples%8 or args.workers < 1:
        cli.error("Use >=2 blocks, >=1 worker and a positive multiple of 8 draws per block")
    run(args)


if __name__ == "__main__":
    main()
