"""Compare direct static and quantum QHA states from CPU MTP runs to v16/SI."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
from scipy.interpolate import CubicSpline

from hcp_orbits import VOLUME_RATIOS, STATIC_S1, VREF, build_orbits, features, sublattices
from mtp_cfg import read_cfg


def minimum(volume, values):
    spline = CubicSpline(volume, values)
    critical = spline.derivative().roots(extrapolate=False)
    candidates = np.r_[volume[0], critical[(critical >= volume[0]) & (critical <= volume[-1])], volume[-1]]
    index = np.argmin(spline(candidates))
    v = candidates[index]
    return float(v), float(spline(v)), bool(v == volume[0] or v == volume[-1])


def load(archives):
    data = {}
    model = None
    for path in archives:
        with tarfile.open(path, "r:gz") as tar:
            for member in tar:
                key = PurePosixPath(member.name)
                if key.name != "label.json":
                    continue
                label = json.load(tar.extractfile(member))
                if model is None:
                    model = label["model_sha256"]
                if label["model_sha256"] != model or not label["stable_mesh"] or label["fmax_eV_A"] > 1e-5:
                    raise ValueError("Mixed models, unstable spectrum or unconverged relaxation")
                handle = tar.extractfile(str(key.with_name("vibrations.csv")))
                rows = list(csv.DictReader(io.TextIOWrapper(handle)))
                t = np.array([float(r["temperature_K"]) for r in rows])
                vib = np.array([float(r["Fvib_eV_atom"]) for r in rows])
                signature = (label["structure"], label["volume_ratio"])
                if signature in data:
                    raise ValueError(f"Duplicate volume state {signature}")
                data[signature] = (label, t, vib)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archives", type=Path, nargs="+", required=True)
    p.add_argument("--efs", type=Path, required=True, help="fresh 18-frame pilot EFS result")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = load(args.archives)
    frames = read_cfg(args.efs)
    with args.manifest.open(newline="") as f:
        manifest = list(csv.DictReader(f))
    if len(data) != len(frames) or len(frames) != len(manifest):
        raise ValueError("Mismatch of pilot and phonon records")
    args.out.mkdir(parents=True, exist_ok=True)
    reference = {}
    with (args.out/"ordered_states.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["structure", "ratio", "v_A3_atom", "ideal_E_eV_atom", "relaxed_E_eV_atom",
                    "ideal_fmax_eV_A", "relaxed_fmax_eV_A", "lowest_internal_THz", "n_displacements"])
        for row, frame in zip(manifest, frames):
            ident, ratio = row["structure"], float(row["volume_ratio"])
            lab, t, vib = data[ident, ratio]
            if not np.isclose(np.linalg.det(frame.cell)/len(frame.types), ratio*VREF, rtol=1e-6) or np.sum(frame.types) != int(row["N_Al"]):
                raise ValueError("Pilot result or species order mismatch")
            w.writerow([ident, ratio, lab["volume_A3_atom"], frame.energy/len(frame.types), lab["static_eV_atom"],
                        np.linalg.norm(frame.forces, axis=1).max(), lab["fmax_eV_A"], lab["minimum_internal_THz"], lab["n_displacements"]])
            if ratio == 1:
                shape = (2,2,1)
                spins = np.where(sublattices(shape) == 0, 1, -1) if ident == "Ti3Al_D019" else -np.ones(8)
                prediction = features(spins, build_orbits(shape)) @ STATIC_S1
                reference[ident] = dict(E_MTP_eV_atom=lab["static_eV_atom"], E_rounded_CE_eV_atom=prediction,
                                        CE_minus_MTP_eV_atom=prediction-lab["static_eV_atom"])
    with (args.out/"ordered_qha.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["structure", "temperature_K", "vopt_A3_atom", "HA_eV_atom", "QHA_eV_atom",
                    "QHA_minus_HA_eV_atom", "minimum_at_edge"])
        for ident in sorted({k[0] for k in data}):
            series = [data[ident, r] for r in VOLUME_RATIOS]
            v = np.array([lab["volume_A3_atom"] for lab, _, _ in series])
            if not all(np.array_equal(t, series[0][1]) for _,t,_ in series):
                raise ValueError("Unequal temperature grids")
            totals = np.array([lab["static_eV_atom"]+vib for lab, _, vib in series])
            for t, row in zip(series[0][1], totals.T):
                vopt, qha, edge = minimum(v, row)
                ha = row[VOLUME_RATIOS.index(1.)]
                w.writerow([ident, t, vopt, ha, qha, qha-ha, edge])
                if t == 600 and ident == "Ti3Al_D019":
                    reference["D019_S2_600K"] = dict(v_computed=vopt, v_printed=16.97,
                        delta_eV_atom_computed=qha-ha, delta_eV_atom_printed=-.00216,
                        v_discrepancy_A3_atom=vopt-16.97, delta_discrepancy_eV_atom=qha-ha+.00216)
    reference["sha256"] = {str(f):hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in [*args.archives, args.efs, args.manifest]}
    (args.out/"comparison.json").write_text(json.dumps(reference, indent=2))


if __name__ == "__main__":
    main()
