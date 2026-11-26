"""Fit equal-structure-weight S1-order ECIs by volume with a shared stability mask."""
import argparse
import csv
import json
import io
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np

from hcp_orbits import ORBIT_NAMES, VREF


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


class Labels:
    """Read the tiny numerical labels from a results tree or a Slurm archive."""
    def __init__(self, path, additional=()):
        self.roots = []
        self.files = {}
        for path in (path, *additional):
            if path.is_dir():
                self.roots.append(path)
                continue
            with tarfile.open(path, "r:gz") as tar:
                for member in tar:
                    p = PurePosixPath(member.name)
                    if p.name in ("label.json", "vibrations.csv") and member.isfile():
                        key = (p.parent.name, p.name)
                        if key in self.files:
                            raise ValueError(f"Duplicate label record {key}")
                        self.files[key] = tar.extractfile(member).read().decode()

    def read(self, subdir, filename):
        if (subdir, filename) in self.files:
            return self.files[(subdir, filename)]
        for parent in self.roots:
            f = parent/subdir/filename
            if f.is_file():
                return f.read_text()
        raise FileNotFoundError(f"Missing label {subdir}/{filename}")


def fit_volume(x, y, is_train):
    a, b = x[is_train], y[is_train]
    coeffs, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    if rank != x.shape[1] or len(a) <= rank or not np.any(~is_train):
        raise ValueError(f"Fit needs a full-rank overdetermined train split and controls (rank {rank}, n {len(a)})")
    residuals = x @ coeffs - y
    q, _ = np.linalg.qr(a, mode="reduced")
    denom = 1 - np.sum(q*q, axis=1)
    loo = np.sqrt(np.mean((residuals[is_train]/denom[:, None])**2, axis=0)) if min(denom) > 1e-10 else None
    rms = np.sqrt(np.mean(residuals[~is_train]**2, axis=0))
    return coeffs, residuals, {"condition_number":float(singular[0]/singular[-1]),
                               "train":int(is_train.sum()), "control":int((~is_train).sum()),
                               "loo_rmse":None if loo is None else loo.tolist(),
                               "control_rmse":rms.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("design", type=Path, help="directory with design.csv and jobs.csv")
    p.add_argument("labels", type=Path, help="parent of label_subdir directories or CPU results.tar.gz")
    p.add_argument("output", type=Path, help="fresh output directory")
    p.add_argument("--ratios", nargs="+", type=float)
    p.add_argument("--additional-labels", type=Path, nargs="*", default=[],
                   help="additional ratio-level archives or label directories")
    p.add_argument("--static-only", action="store_true", help="diagnostic static fit without a phonon stability mask")
    args = p.parse_args()
    design = {r["id"]:r for r in read_rows(args.design/"design.csv")}
    jobs = read_rows(args.design/"jobs.csv")
    labels = Labels(args.labels, args.additional_labels)
    ratios = sorted(set(args.ratios or [float(j["ratio"]) for j in jobs]))
    report, residual_rows, pair_rows, coeffs, expected_t, model_sha = [], [], [], [], None, None
    for ratio in ratios:
        x, y, names, split, unstable = [], [], [], [], []
        batch = [j for j in jobs if float(j["ratio"]) == ratio]
        if {j["id"] for j in batch} != design.keys() or len(batch) != len(design):
            raise ValueError(f"Missing/duplicate jobs at volume ratio {ratio}")
        for j in batch:
            ident = j["id"]
            record = design[ident]
            path = args.labels / j["label_subdir"]
            label = json.loads(labels.read(j["label_subdir"], "label.json"))
            if (label.get("source_cell_sha256") != record["cell_sha256"] or
                not np.isclose(label["volume_A3_atom"], VREF*ratio, rtol=1e-7) or
                label["n_atoms"] != int(record["N"]) or
                label["fmax_eV_A"] > 1e-5):
                raise ValueError(f"Structure/volume/relaxation mismatch at {path}")
            if model_sha is None:
                model_sha = label["model_sha256"]
            if label["model_sha256"] != model_sha:
                raise ValueError("Mixed MTP weights")
            if not args.static_only and label["stable_mesh"] is None:
                raise ValueError(f"Phonons were not computed at {path}")
            if not args.static_only and not label["stable_mesh"]:
                unstable.append(ident)
                continue
            thermal = [] if args.static_only else list(csv.DictReader(io.StringIO(labels.read(j["label_subdir"], "vibrations.csv"))))
            t = [float(r["temperature_K"]) for r in thermal]
            if expected_t is None:
                expected_t = t
            if t != expected_t:
                raise ValueError(f"Temperature grid mismatch at {path}")
            y.append([label["static_eV_atom"], *[float(r["Fvib_eV_atom"]) for r in thermal]])
            x.append([float(record[key]) for key in ORBIT_NAMES])
            split.append(record["split"] == "train")
            names.append(ident)
        y = np.asarray(y)
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite source label")
        c, residuals, score = fit_volume(np.asarray(x), y, np.asarray(split))
        mask = ~np.array(split)
        big = np.array([int(design[name]["N"])==64 and int(design[name]["N_Al"])==22 for name in names])
        for key, subset in [("seven",mask),("small_controls",mask & ~big),("22Al_64",mask & big)]:
            if not np.any(subset):
                continue
            r = residuals[subset]
            stats = dict(count=len(r), rmse_static_eV_atom=float(np.sqrt((r[:,0]**2).mean())))
            if expected_t:
                stats["rmse_vib_eV_atom"] = np.sqrt((r[:,1:]**2).mean(axis=0)).tolist()
                stats["rmse_total_eV_atom"] = np.sqrt(((r[:,1:]+r[:,:1])**2).mean(axis=0)).tolist()
                stats["max_abs_total_eV_atom"] = np.max(abs(r[:,1:]+r[:,:1]),axis=0).tolist()
            score.setdefault("groups",{})[key]=stats
        # For the predeclared independent 64-site pairs compare equal-composition
        # differences before rounding. ECI constant and point cancel identically.
        strata = {}
        for i, ident in enumerate(names):
            r = design[ident]
            if int(r["N"]) != 64:
                continue
            key = (r["split"], int(r["N_Al"]))
            strata.setdefault(key, {0:[], 1:[]})[int(r["D019_preferred"])].append(i)
        for (group, n_al), indices in strata.items():
            if not indices[0] or not indices[1]:
                continue
            random, ordered = indices[0], indices[1]
            target = y[random].mean(axis=0) - y[ordered].mean(axis=0)
            bias = residuals[random].mean(axis=0) - residuals[ordered].mean(axis=0)
            observables=["static", *[f"Fvib_{t:g}K" for t in expected_t], *[f"E_plus_Fvib_{t:g}K" for t in expected_t]]
            for obs, direct, delta in zip(observables, np.r_[target,target[0]+target[1:]],
                                          np.r_[bias,bias[0]+bias[1:]]):
                pair_rows.append([ratio, group, n_al, "|".join(names[i] for i in random),
                                  "|".join(names[i] for i in ordered), obs, direct, direct+delta, delta])
        report.append(dict(ratio=ratio, excluded_unstable=unstable, **score))
        coeffs.append(c)
        residual_rows.extend([ratio, name, *row] for name, row in zip(names, residuals))
    coefficients = np.stack(coeffs)
    args.output.mkdir(parents=True, exist_ok=False)
    grid = dict(J_static=coefficients[:, :, 0], volume_A3_atom=np.array(ratios)*VREF, orbit_names=ORBIT_NAMES)
    if args.static_only:
        np.savez(args.output/"static_grid.npz", **grid)
    else:
        np.savez(args.output/"grid.npz", J_vib=coefficients[:, :, 1:].transpose(0, 2, 1),
                 temperature_K=expected_t, **grid)
        with (args.output/"interactions.csv").open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["volume_ratio","volume_A3_atom","temperature_K","orbit_S1","J_static_eV",
                        "J_vib_eV","J_total_eV"])
            for ratio,c in zip(ratios,coefficients):
                for t,vib in zip(expected_t,c[:,1:].T):
                    w.writerows((ratio,ratio*VREF,t,name,static,d,static+d)
                                for name,static,d in zip(ORBIT_NAMES,c[:,0],vib))
    (args.output/"validation.json").write_text(json.dumps({"model_sha256":model_sha,
                     "observables":["static", *expected_t], "phonon_mask_applied":not args.static_only,
                     "volumes":report}, indent=2))
    with (args.output/"residuals_eV_atom.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["volume_ratio", "id", "static", *expected_t])
        w.writerows(residual_rows)
    with (args.output/"ordered_random_eV_atom.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["volume_ratio", "split", "N_Al_per_64", "random_ids", "preferred_ids", "observable",
                    "MTP_random_minus_preferred", "CE_random_minus_preferred", "CE_minus_MTP_difference"])
        w.writerows(pair_rows)


if __name__ == "__main__":
    main()
