"""Fixed-volume classical configurational TI for the complete N=16 HCP enumeration."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.special import logsumexp

from n16_static import HASH, KB, N, VREF, hcp_geometry


SEED = 47399
SCALES = np.array([1.0, 1.35])
COUNTS = (2, 3, 4)


def draw_block(rng, count, transform, frame, sites, planes, kt, calc):
    from mtp_cfg import Frame

    ndof = transform.shape[0]
    scales = SCALES[np.arange(count) % len(SCALES)]
    z = rng.normal(size=(count, ndof)) * scales[:, None]
    r2 = np.sum(z*z, axis=1)
    mixture = -ndof*np.log(SCALES) - r2[:, None]/(2*SCALES**2)
    log_ratio = -.5*r2 - logsumexp(mixture, axis=1) + np.log(len(SCALES))
    positions = frame.positions + (z @ transform).reshape(count, N, 3)
    wall = np.einsum("bni,nki->bnk", positions-sites, planes).max(axis=(1, 2))
    log_ratio[wall >= 1] = -np.inf
    delta = np.zeros(count)
    valid = np.flatnonzero(wall < 1)
    for start in range(0, len(valid), 256):
        indices = valid[start:start+256]
        result = calc([Frame(frame.cell, positions[i], frame.types) for i in indices])
        delta[indices] = [f.energy for f in result] - (frame.energy + .5*kt*r2[indices])
    return delta, log_ratio, wall


def sample(task):
    from label_cell import evaluate
    from mtp_cfg import Frame
    from pilot import estimate, force_constants, site_planes

    row, binary, model, kt, blocks, samples, output_dir = task
    indices = row["representative_Al_indices"]
    result = dict(id=row["id"], N_Al=len(indices), degeneracy=row["degeneracy"],
                  sampling_issues=[], harmonic_config_total_eV_cell=None)
    issues = result["sampling_issues"]
    if not row["gamma_stable_in_site_domain"]:
        issues.append("no stable in-domain harmonic minimum")
        return result
    cell, sites = hcp_geometry((2, 2, 2))
    positions = np.asarray(row["relaxed_positions_A"])
    if np.max(np.abs(np.mean(positions-sites, axis=0))) > 1e-8:
        raise ValueError(f"Wrong arithmetic displacement gauge for {row['id']}")
    types = np.zeros(N, dtype=int)
    types[indices] = 1

    def calc(frames):
        return evaluate(frames, binary, model, 3)

    frame = calc([Frame(cell, positions, types)])[0]
    if abs(frame.energy-row["energy_eV_cell"]) > 2e-6:
        raise ValueError(f"Wrong static energy for {row['id']}")
    basis = null_space(np.tile(np.eye(3), (1, N)))
    values, vectors = np.linalg.eigh(basis.T @ force_constants(calc, frame, .0025) @ basis)
    if np.min(values) <= 0:
        issues.append("nonpositive curvature when rebuilding reference")
        return result
    harmonic = .5*kt*np.log(values/(2*np.pi*kt)).sum()
    if abs(harmonic-row["harmonic_config_eV_cell"]) > 2e-5:
        raise ValueError(f"Cartesian harmonic normalization mismatch for {row['id']}")
    result["harmonic_config_total_eV_cell"] = row["energy_eV_cell"] + row["harmonic_config_eV_cell"]
    transform = (basis @ vectors * np.sqrt(kt/values)).T
    planes = site_planes(cell, sites)
    mask = sum(1 << i for i in indices)
    draws = [draw_block(np.random.default_rng([SEED, mask, b]), samples, transform,
                        frame, sites, planes, kt, calc) for b in range(blocks)]
    delta, log_ratio, wall = map(np.array, zip(*draws))
    np.savez_compressed(output_dir/f"raw_{row['id']}.npz", delta_eV_cell=delta,
                        log_harmonic_to_proposal=log_ratio, max_wall_fraction=wall.astype("f4"),
                        mixture_scales=SCALES, seed_prefix=np.array([SEED, mask]))
    if not np.isfinite(log_ratio).any(axis=1).all():
        issues.append("empty in-domain reference block")
        return result
    def summarize(keep):
        return estimate(delta[keep].ravel(), log_ratio[keep].ravel(), kt)

    full = summarize(slice(None))
    delete_one = [summarize(np.arange(blocks) != i)["ti_eV"] for i in range(blocks)]
    endlog = (log_ratio - delta/kt).ravel()
    end_weight = np.exp(endlog-logsumexp(endlog))
    result.update(ti_correction_eV_cell=full["ti_eV"], fep_correction_eV_cell=full["fep_eV"],
                  coarse_ti_correction_eV_cell=full["coarse_ti_eV"],
                  delete_one_ti_corrections_eV_cell=delete_one,
                  replica_ti_corrections_eV_cell=[summarize(i)["ti_eV"] for i in range(blocks)],
                  correction_jackknife_se_eV_cell=jackknife_se(delete_one),
                  ess_min=full["ess_min"], ess_endpoint=full["ess_endpoint"],
                  largest_endpoint_weight=full["max_weight_endpoint"],
                  harmonic_domain_probability=full["harmonic_domain_probability_IS"],
                  target_probability_near_wall=float(end_weight[wall.ravel()>.9].sum()),
                  excluded_draws=int((wall >= 1).sum()))
    # Thresholds label overlap in this pilot; they do not bound unsampled tails.
    if full["ess_min"] < 50 or full["max_weight_endpoint"] > .1:
        issues.append("importance overlap outside pilot screening thresholds")
    if any(t["min_curvature"] > 0 and t["max_wall_fraction"] < 1
           for t in row["symmetry_breaking_trials"]):
        issues.append("saddle escape: coverage of all in-domain minima unverified")
    return result


def jackknife_se(delete_one):
    values = np.asarray(delete_one)
    return float(np.sqrt((len(values)-1)*np.mean((values-values.mean())**2)))


def summarize_composition(rows, kt):
    result = dict(number_of_orbits=len(rows), number_of_decorations=sum(r["degeneracy"] for r in rows),
                  flagged_orbits=[r["id"] for r in rows if r["sampling_issues"]])
    result["status"] = "unresolved" if result["flagged_orbits"] else "pilot_estimate"
    if any("ti_correction_eV_cell" not in r for r in rows):
        return result
    log_base = np.log([r["degeneracy"] for r in rows]) - np.array([
        r["harmonic_config_total_eV_cell"] for r in rows])/kt
    correction = np.array([r["ti_correction_eV_cell"] for r in rows])
    log_weights = log_base - correction/kt
    probs = np.exp(log_weights-logsumexp(log_weights))
    ess = np.array([r["ess_endpoint"] for r in rows])
    deleted = np.array([r["delete_one_ti_corrections_eV_cell"] for r in rows])
    free = -kt*logsumexp(log_weights)
    omit = -kt*logsumexp(log_base[:, None]-deleted/kt, axis=0)
    largest_weight = [r["largest_endpoint_weight"] for r in rows]
    result.update(harmonic_config_eV_cell=float(-kt*logsumexp(log_base)),
                  trial_ti_config_eV_cell=float(free),
                  trial_fep_config_eV_cell=float(-kt*logsumexp(log_base - np.array([
                      r["fep_correction_eV_cell"] for r in rows])/kt)),
                  delete_one_config_eV_cell=omit.tolist(),
                  jackknife_se_eV_cell=jackknife_se(omit),
                  effective_sample_size_joint_endpoint=float(1/np.sum(probs**2/ess)),
                  largest_joint_endpoint_weight=float(max(probs*largest_weight)),
                  min_per_orbit_ess=float(min(r["ess_min"] for r in rows)),
                  trial_probability_of_flagged_orbits=float(sum(p for p, r in zip(probs, rows)
                                                              if r["sampling_issues"])),
                  trial_orbit_probabilities=dict(zip([r["id"] for r in rows],probs.tolist())))
    return result


def summarize_curvature(groups):
    ordered = [groups[k] for k in COUNTS]
    result = {"status": "unresolved" if any(g["status"] == "unresolved" for g in ordered)
              else "pilot_estimate"}
    if any("trial_ti_config_eV_cell" not in g for g in ordered):
        return result
    harmonic = np.array([g["harmonic_config_eV_cell"] for g in ordered])
    total = np.array([g["trial_ti_config_eV_cell"] for g in ordered])
    fep = np.array([g["trial_fep_config_eV_cell"] for g in ordered])
    deleted = np.array([g["delete_one_config_eV_cell"] for g in ordered])
    second = np.array([1., -2., 1.])
    error = jackknife_se(second @ deleted)
    result.update(trial_mu_2_to_3_eV=float(total[1]-total[0]),
                  trial_mu_3_to_4_eV=float(total[2]-total[1]),
                  harmonic_second_difference_eV=float(second @ harmonic),
                  trial_second_difference_eV=float(second @ total),
                  trial_anharmonic_shift_second_difference_eV=float(second @ (total-harmonic)),
                  trial_fep_second_difference_eV=float(second @ fep),
                  second_difference_jackknife_se_eV=error,
                  delete_one_second_difference_eV=(second @ deleted).tolist(),
                  trial_f_double_prime_eV_atom_per_fraction2=float(N*(second @ total)),
                  f_double_prime_jackknife_se_eV_atom_per_fraction2=N*error)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--mlp", required=True)
    cli.add_argument("--static", type=Path, default=Path("n16_static.json"))
    cli.add_argument("--workers", type=int, default=6)
    cli.add_argument("--replicas", type=int, default=4)
    cli.add_argument("--samples", type=int, default=1024)
    cli.add_argument("--output-dir", type=Path, default=Path("."))
    args = cli.parse_args()
    if not 1 <= args.workers <= 6 or args.replicas < 2 or args.samples < 2 or args.samples % 2:
        cli.error("use 1–6 workers, two or more replicas and positive even draws per replica")
    model = args.model.resolve(strict=True)
    static = json.loads(args.static.read_text())
    if sha256(model.read_bytes()).hexdigest() != HASH or static["model_sha256"] != HASH:
        raise ValueError("Unexpected MTP checksum")
    if (static["N"] != N or static["shape"] != [2, 2, 2] or static["temperature_K"] != 1000
            or not np.isclose(static["V_A3_atom"], VREF, rtol=0, atol=1e-8)):
        raise ValueError("Unexpected static reference cell or temperature")
    rows = static["results"]
    if len(rows) != 43 or len(set(r["id"] for r in rows)) != len(rows):
        raise ValueError("Expected 43 distinct representatives")
    for k in COUNTS:
        if sum(r["degeneracy"] for r in rows if r["N_Al"] == k) != comb(N,k):
            raise ValueError(f"Incomplete chemical enumeration at k={k}")
    kt = KB*static["temperature_K"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(r, args.mlp, str(model), kt, args.replicas, args.samples, args.output_dir) for r in rows]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(sample,tasks))
    groups = {k: summarize_composition([r for r in results if r["N_Al"]==k],kt) for k in COUNTS}
    for k, group in groups.items():
        if "harmonic_config_eV_cell" in group:
            expected = static["compositions"][str(k)]["direct_static_plus_classical_gamma_config_eV_cell"]
            if abs(expected-group["harmonic_config_eV_cell"]) > 1e-8:
                raise ValueError("Input and rebuilt harmonic canonical sums differ")
    output = dict(model_sha256=HASH, static_sha256=sha256(args.static.read_bytes()).hexdigest(),
                  N=N, shape=static["shape"], V_A3_atom=VREF, temperature_K=static["temperature_K"],
                  replicas=args.replicas, draws_per_orbit_per_replica=args.samples,
                  mixture_scales=SCALES.tolist(), seed=SEED,
                  convention="Common Cartesian gauge, arithmetic mean displacement fixed, site Voronoi domain. Kinetic terms omitted for every composition.",
                  overlap_screen="min ESS across lambda >= 50 and largest endpoint weight <= 0.1; separately flag saddle escape",
                  results=results, compositions=groups, curvature=summarize_curvature(groups))
    (args.output_dir/"n16_ti.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
