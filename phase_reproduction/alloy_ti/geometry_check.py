"""Replay extreme N=8 importance draws and verify their Qi-MTP energies."""
import argparse
from functools import partial
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np


def nearest_pair(cell, positions, types):
    shifts = np.array(list(product(range(-2,3), repeat=3)), dtype=int)
    cart = shifts @ cell
    best = np.inf
    symbols = ("Ti", "Al")
    for i in range(len(positions)):
        for j in range(i, len(positions)):
            distances = np.linalg.norm(positions[j]-positions[i]+cart, axis=1)
            if i == j:
                distances[np.all(shifts==0, axis=1)] = np.inf
            k = int(np.argmin(distances))
            if distances[k] < best:
                best = float(distances[k])
                pair = {"distance_A": best, "sites_zero_based": [i,j],
                        "types": [symbols[types[i]], symbols[types[j]]],
                        "image_of_second_atom": shifts[k].tolist()}
    return pair


def repeat_cell(frame):
    from mtp_cfg import Frame
    shifts = np.array(list(product(range(2), repeat=3))) @ frame.cell
    positions = (shifts[:,None,:]+frame.positions[None,:,:]).reshape(-1, 3)
    return Frame(2*frame.cell, positions, np.tile(frame.types, len(shifts)))


def minimum_distances(cell, coords):
    """All unordered pairs for this row-vector hexagonal cell, including periodicity."""
    metric = cell@cell.T
    if not np.allclose(metric[:2,2],0) or not np.isclose(metric[0,0],metric[1,1]) or not np.isclose(2*metric[0,1],metric[0,0]):
        raise ValueError("Nearest-image enumeration assumes a 60-degree equal-axis basal plane")
    i,j = np.triu_indices(coords.shape[1],1)
    diff = (coords[:,j]-coords[:,i])@np.linalg.inv(cell)
    diff -= np.rint(diff)
    x,y,z = diff.transpose(2,0,1)
    basal = np.full_like(x,np.inf)
    for a,b in product(range(-1,2),repeat=2):
        dx,dy = x+a,y+b
        np.minimum(basal,dx*dx+dy*dy+dx*dy,out=basal)
    full = metric[0,0]*basal + metric[2,2]*z*z
    return np.sqrt(np.minimum(full.min(axis=1), min(metric[0,0],metric[2,2])))


def main():
    from scipy.linalg import null_space
    from decorations import representatives
    from pilot import (HASH, Frame, evaluate, relax_fixed_cell, hcp_geometry,
                       force_constants, site_planes)
    from mtp_cfg import write_cfg
    from tail_check import SEED, KT, proposal, summary

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--mlp", required=True)
    cli.add_argument("--samples", type=Path, default=Path("tail_samples.npz"))
    args = cli.parse_args()
    model = args.model.resolve()
    if hashlib.sha256(model.read_bytes()).hexdigest() != HASH:
        raise ValueError("MTP checksum mismatch")
    cell, sites = hcp_geometry((2,2,1))
    types = np.zeros(len(sites), dtype=int)
    types[list(representatives()["interlayer"])] = 1
    calc = partial(evaluate, binary=args.mlp, model=str(model), version=3)
    frame = relax_fixed_cell(Frame(cell, sites.copy(), types), calc)
    h = force_constants(calc, frame, .0025)
    basis = null_space(np.tile(np.eye(3), (1,len(sites))))
    vals, vecs = np.linalg.eigh(basis.T @ h @ basis)
    transform = (basis @ vecs*np.sqrt(KT/vals)).T
    planes = site_planes(cell, sites)
    def wallof(p):
        return float(np.einsum("ni,nki->nk", p-sites, planes).max())
    with np.load(args.samples) as data:
        delta = data["delta_eV_cell"]
        saved_wall = data["max_wall_fraction"]
        saved_r2 = data["whitened_radius_squared"]
        seed = int(data["seed_prefix"])
        if seed != SEED or not np.allclose(vals, data["eigenvalues_eV_A2"], atol=1e-9, rtol=1e-9):
            raise RuntimeError("Replay seed/Hessian deviates from that of the archived run")
        if not np.allclose(frame.positions, data["relaxed_positions"], atol=1e-9, rtol=0):
            raise RuntimeError("Reconstructed equilibrium coordinates differ from archive")
        ratio = data["log_harmonic_to_proposal"]
        archived_logw = ratio-delta/KT
    kmax = np.unravel_index(np.argmax(archived_logw), delta.shape)
    locations = {(ib,int(np.argmax(row))) for ib,row in enumerate(archived_logw)}
    locations.update(np.unravel_index(idx, delta.shape) for idx in np.argsort(archived_logw,axis=None)[-3:])
    ordered = sorted(locations, key=lambda k: archived_logw[k], reverse=True)
    replay = {}
    records = []
    for ib, j in ordered:
        if ib not in replay:
            z, radial, _ = proposal(delta.shape[1], len(vals), np.random.default_rng([SEED,int(ib)]))
            # Use the original block-sized matrix multiplication to replay its geometry.
            all_pos = frame.positions+(z@transform).reshape(len(z),len(sites),3)
            replay[ib] = radial, all_pos
        radial, all_pos = replay[ib]
        orig = Frame(cell, all_pos[j], types)
        single = calc([orig])[0]
        d = single.energy-frame.energy-.5*KT*radial[j]
        big = calc([repeat_cell(orig)])[0]
        obs_wall = wallof(orig.positions)
        err_wall = obs_wall-saved_wall[ib,j]
        err_r2 = radial[j]-saved_r2[ib,j]
        err_energy = d-delta[ib,j]
        record = {"block_zero_based": int(ib), "index_zero_based": int(j),
                  "proposal": "t5" if j%2 else "gaussian", "wall_fraction": obs_wall,
                  "archived_wall_fraction": float(saved_wall[ib,j]), "inside": obs_wall < 1.,
                  "r2": float(radial[j]), "archived_r2": float(saved_r2[ib,j]),
                  "archived_excess_eV_cell": float(delta[ib,j]), "replayed_excess_eV_cell": float(d),
                  "replayed_minus_archived_eV_cell": float(err_energy),
                  "replay_matches": bool(abs(err_wall)<1e-8 and abs(err_r2)<1e-8 and abs(err_energy)<1e-4),
                  "standalone_MTP_eV_cell": single.energy,
                  "repeated_64_atom_MTP_eV_cell": big.energy,
                  "repeat_minus_original_eV_atom": float(big.energy/64-single.energy/8),
                  "fmax_eV_A": float(np.linalg.norm(single.forces,axis=1).max()),
                  "nearest_pair": nearest_pair(cell,orig.positions,types)}
        records.append(record)
    worst = Frame(cell, replay[kmax[0]][1][kmax[1]],types)
    write_cfg("worst.cfg", [worst])
    grid = np.linspace(0.,1.,101)
    configs = [Frame(cell,frame.positions+s*(worst.positions-frame.positions),types) for s in grid]
    path = []
    path_displacement = worst.positions-frame.positions
    for s,r in zip(grid,calc(configs)):
        path.append({"fraction": float(s), "energy_eV_cell": r.energy,
                     "relative_to_minimum_eV_cell": float(r.energy-frame.energy),
                     "directional_derivative_eV_cell": float(-np.sum(r.forces*path_displacement)),
                     "wall_fraction": wallof(r.positions),
                     "nearest_pair": nearest_pair(cell,r.positions,types)})
    peak = max(path, key=lambda x:x["energy_eV_cell"])
    distances = np.stack([minimum_distances(cell,replay[ib][1]) for ib in range(len(delta))])
    if not np.isclose(distances[kmax],records[0]["nearest_pair"]["distance_A"],rtol=0,atol=1e-10):
        raise RuntimeError("Vectorized and explicit minimum-image enumeration disagree")
    np.savez_compressed("tail_distances.npz", minimum_pair_distance_A=distances)
    cutoffs = {}
    for cutoff in (0.,1.68052,1.8,2.,2.2):
        keep = (distances>=cutoff)&np.isfinite(ratio)
        lr = np.where(keep,ratio,-np.inf)
        stats,_ = summary(delta,lr)
        hblock = np.exp(lr).mean(axis=1)
        fep = stats["fep"]["value_eV_cell"]
        cutoffs[str(cutoff)] = {"proposal_draws_kept": int(keep.sum()),
            "harmonic_probability_within_wall_and_cut": float(hblock.mean()),
            "harmonic_probability_block_se": float(hblock.std(ddof=1)/np.sqrt(len(delta))),
            "restricted_MTP_minus_unrestricted_harmonic_eV_cell": stats["fep"],
            "restricted_MTP_minus_equally_restricted_harmonic_eV_cell": float(fep+KT*np.log(hblock.mean())),
            "endpoint_excess_eV_cell": stats["endpoint_excess_energy"],
            "effective_samples": stats["effective_samples"], "largest_weight": stats["largest_normalized_weight"],
            "minimum_retained_MTP_eV_cell": float(np.min((delta+frame.energy+.5*KT*saved_r2)[keep]))}
    output = {"model_sha256": HASH, "archive_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
              "equilibrium_energy_eV_cell": frame.energy, "atom_types": types.tolist(),
              "cell_A": cell.tolist(), "relaxed_positions_A": frame.positions.tolist(),
              "worst_positions_A": worst.positions.tolist(), "checks": records,
              "worst_replay_matches": records[0]["replay_matches"], "path": path,
              "maximum_sampled_along_linear_path": peak,
              "max_path_wall_fraction": max(p["wall_fraction"] for p in path),
              "restricted_by_pair_distance_A": cutoffs}
    Path("tail_geometry.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
