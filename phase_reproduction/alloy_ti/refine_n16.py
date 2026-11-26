"""Replace insufficient-overlap N=16 orbit samples, then recompute chemical curvature."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.special import logsumexp
import n16_ti


TARGETS = ("k04_0-1-4-5", "k04_0-2-4-6")
BASE_SEED = 529701
OVERLAP = "importance overlap outside pilot screening thresholds"


def generate(task):
    row, binary, model, kt, samples, group, base_seed, directory = task
    # sample() builds a full Hessian in the saved Cartesian gauge and writes
    # the input-to-mixture weights; each worker uses an independent seed prefix.
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        n16_ti.SEED = base_seed + group
        prefix = Path(temp)
        check = n16_ti.sample((row, binary, model, kt, 2, samples//2, prefix))
        if check["harmonic_config_total_eV_cell"] is None or "ti_correction_eV_cell" not in check:
            raise ValueError(f"Missing stable proposal or valid group for {row['id']}")
        with np.load(prefix/f"raw_{row['id']}.npz") as data:
            return (row["id"], group, *(data[key].ravel() for key in
                    ("delta_eV_cell", "log_harmonic_to_proposal", "max_wall_fraction")))


def replace_estimate(old, data, kt, base_seed, output):
    from pilot import estimate

    delta, log_ratio, wall = data
    groups, samples = delta.shape
    def correction(keep):
        return estimate(delta[keep].ravel(), log_ratio[keep].ravel(), kt)
    full = correction(slice(None))
    deleted = [correction(np.arange(groups) != g)["ti_eV"] for g in range(groups)]
    pooled_log = (log_ratio-delta/kt).ravel()
    weights = np.exp(pooled_log-logsumexp(pooled_log))
    issues = [s for s in old["sampling_issues"] if s != OVERLAP]
    if full["ess_min"] < 50 or full["max_weight_endpoint"] > .1:
        issues.append(OVERLAP)
    result = dict(old, ti_correction_eV_cell=full["ti_eV"], fep_correction_eV_cell=full["fep_eV"],
                  coarse_ti_correction_eV_cell=full["coarse_ti_eV"],
                  replica_ti_corrections_eV_cell=[correction(i)["ti_eV"] for i in range(groups)],
                  delete_one_ti_corrections_eV_cell=deleted,
                  correction_jackknife_se_eV_cell=n16_ti.jackknife_se(deleted),
                  ess_min=full["ess_min"], ess_endpoint=full["ess_endpoint"],
                  largest_endpoint_weight=full["max_weight_endpoint"],
                  harmonic_domain_probability=full["harmonic_domain_probability_IS"],
                  target_probability_near_wall=float(weights[wall.ravel()>.9].sum()),
                  excluded_draws=int(np.sum(~np.isfinite(log_ratio))), sampling_issues=issues)
    raw = output/f"raw_{old['id']}.npz"
    np.savez_compressed(raw, delta_eV_cell=delta, log_harmonic_to_proposal=log_ratio,
                        max_wall_fraction=wall, mixture_scales=n16_ti.SCALES,
                        group_seed_prefix=np.arange(base_seed,base_seed+groups),
                        half_draws_per_group=samples//2)
    return result


def load_inputs(original_path, static_path, model_path):
    original = json.loads(original_path.read_text())
    static = json.loads(static_path.read_text())
    if sha256(model_path.read_bytes()).hexdigest() != n16_ti.HASH or any(
            r["model_sha256"] != n16_ti.HASH for r in (static,original)):
        raise ValueError("Unexpected model checksum")
    if (sha256(static_path.read_bytes()).hexdigest() != original["static_sha256"]
            or original["N"] != 16 or original["temperature_K"] != 1000
            or original["shape"] != [2,2,2] or original["replicas"] != 4
            or original["mixture_scales"] != n16_ti.SCALES.tolist()):
        raise ValueError("The original and static references must match the N16 run")
    rows = original["results"]
    ids = {r["id"] for r in rows}
    if len(ids) != len(rows) or len(ids) != 43 or ids != {r["id"] for r in static["results"]}:
        raise ValueError("Expected matching complete 43-orbit datasets")
    flagged = {r["id"] for r in rows if r["sampling_issues"]}
    if flagged != set(TARGETS) or any(r["sampling_issues"] != [OVERLAP] for r in rows if r["id"] in flagged):
        raise ValueError("Unexpected flagged orbit(s) or source diagnosis")
    return original, static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("n16_ti.json"))
    parser.add_argument("--static", type=Path, default=Path("n16_static.json"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mlp", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--draws-per-group", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.draws_per_group < 4 or args.draws_per_group % 4:
        parser.error("Use 1–8 workers and a positive group size divisible by four")
    model = args.model.resolve(strict=True)
    original, static = load_inputs(args.original,args.static,model)
    count = original["replicas"]
    if args.seed < 0 or original["seed"] in range(args.seed,args.seed+count):
        parser.error("Use independent nonnegative seed prefixes")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    kt = n16_ti.KB*original["temperature_K"]
    static_rows = {r["id"]: r for r in static["results"]}
    tasks = [(static_rows[orbit],args.mlp,str(model),kt,args.draws_per_group,g,args.seed,directory)
             for orbit in TARGETS for g in range(count)]
    grouped = {orbit:{} for orbit in TARGETS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for orbit, group, *data in pool.map(generate, tasks):
            grouped[orbit][group] = data
    rows = []
    sampling = {}
    for row in original["results"]:
        orbit = row["id"]
        if orbit in TARGETS:
            raw = np.asarray([grouped[orbit][g] for g in range(count)])
            replacement = replace_estimate(row, (raw[:,0], raw[:,1], raw[:,2].astype("f4")),
                                           kt, args.seed, directory)
            rows.append(replacement)
            sampling[orbit] = dict(source="replacement",replicas=count,draws_per_replica=args.draws_per_group)
        else:
            rows.append(row)
            sampling[orbit] = dict(source="original",replicas=count,
                                   draws_per_replica=original["draws_per_orbit_per_replica"])
    if any(r != o for r,o in zip(rows,original["results"]) if o["id"] not in TARGETS):
        raise RuntimeError("An unflagged orbit changed")
    groups = {k:n16_ti.summarize_composition([r for r in rows if r["N_Al"] == k],kt)
              for k in n16_ti.COUNTS}
    for k,group in groups.items():
        previous = original["compositions"][str(k)]
        if abs(group["harmonic_config_eV_cell"]-previous["harmonic_config_eV_cell"]) > 1e-10:
            raise RuntimeError("Harmonic reference changed")
        if k != 4 and group != previous:
            raise RuntimeError(f"Unchanged composition k={k} changed")
    result = dict(original, results=rows, compositions=groups, curvature=n16_ti.summarize_curvature(groups),
                  draws_per_orbit_per_replica=None, seed=None, sampling_by_orbit=sampling,
                  refinement=dict(original_sha256=sha256(args.original.read_bytes()).hexdigest(),
                                  original_seed=original["seed"], replacements=list(TARGETS),
                                  replacement_group_seeds=list(range(args.seed,args.seed+count)),
                                  rng="SeedSequence([group_seed, orbit_bitmask, half_index])",
                                  retained_original_orbits=len(rows)-len(TARGETS),
                                  previous_curvature=original["curvature"]))
    (directory/"n16_refined.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
