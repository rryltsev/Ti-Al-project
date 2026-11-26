"""Audit the shape guard separately from the displacement wall on saved trajectories."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
PROJECT = STUDY.parent
CORE_BOUND = 1.68052


def nearest(volume, axial):
    basal = np.cbrt(4.0 * volume / (np.sqrt(3.) * axial))
    return basal * np.minimum(1., np.sqrt(1. / 3. + axial**2 / 4.))


def clearance(distance, cap):
    return (1. - cap) * distance - CORE_BOUND


def selected_manifests(history):
    if history:
        yield from (PROJECT / "solvus_1000/series").glob("*/windows.csv")
        yield from (PROJECT / "solvus_temperatures/atomistic/series").glob("*/windows.csv")
        base = PROJECT / "solvus_closure/atomistic/series"
        for series in ("zc4523c", "zc4523s"):
            yield base / series / "windows.csv"
    else:
        yield from (STUDY / "atomistic/series").glob("*/windows.csv")


def analyze_group(chains, cap):
    recorded = []
    envelope = []
    labels = []
    counts = []
    unread = []
    for manifest, spec in chains:
        file = manifest.parent / (spec["run"] + ".csv")
        if not file.is_file():
            unread.append({"run": spec["run"], "reason": "missing"})
            continue
        with file.open(encoding="utf8") as stream:
            names = stream.readline().strip().split(",")
            cols = [names.index(c) for c in ("sweep", "volume_A3_atom", "c_a")]
            sample = np.loadtxt(stream, delimiter=",", usecols=cols, ndmin=2)
        steps, volume, axial = sample.T
        burn, prod, stride = (int(spec[c]) for c in ("burn", "sweeps", "stride"))
        expected = np.arange(stride, burn + prod + 1, stride)
        if not np.array_equal(steps, expected) or burn < stride:
            unread.append({"run": spec["run"], "reason": "incomplete trace or burn"})
            continue
        active = steps > burn
        recorded.append(nearest(volume[active], axial[active]))
        # Start at the last saved state at or before burn. A sweep contains one
        # volume proposal followed, every third sweep, by one axial proposal.
        # Include the last interval even if the final sweep was not saved.
        end = np.minimum(steps + stride, burn + prod)
        active_interval = (end > burn) & (end > steps)
        dv = .22 * np.sqrt(128. / int(spec["N"]))
        da = .008 * np.sqrt(128. / int(spec["N"]))
        vmin = np.maximum(15., volume - (end - steps) * dv)
        nax = np.floor_divide(end, 3) - np.floor_divide(steps, 3)
        amin = np.maximum(1.45, axial * np.exp(-da * nax))
        amax = np.minimum(1.72, axial * np.exp(da * nax))
        # At fixed V, d_nn increases up to ideal HCP c/a then decreases.
        bound = np.minimum(nearest(vmin, amin), nearest(vmin, amax))
        envelope.append(bound[active_interval])
        labels.append(spec["run"])
        counts.append(int(active.sum()))
    if not recorded:
        return {"analyzed": 0, "omitted": unread}
    r = np.concatenate(recorded)
    b = np.concatenate(envelope)
    tests = {}
    for tested in (cap, .40):
        key = f"{tested:.2f}"
        if key in tests:
            continue
        h, lower = clearance(r, tested), clearance(b, tested)
        tests[key] = {
            "recorded_min_A": float(h.min()), "recorded_01pct_A": float(np.quantile(h, .001)),
            "recorded_median_A": float(np.median(h)), "recorded_below_zero": int((h < 0.).sum()),
            "recorded_le_1_milliangstrom": int((h <= .001).sum()),
            "min_reachable_interval_bound_A": float(lower.min()),
            "intervals_with_possible_guard_contact": int((lower <= 1e-9).sum()),
        }
    coverage = {"runs": labels, "frames": counts}
    return {"analyzed": len(labels), "omitted": unread, "frames": len(r), "intervals": len(b),
            "min_nearest_A":float(r.min()), "guard_by_cap": tests, "coverage": coverage}


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--new", action="store_true", help="Use completed windows in the new N=2304 study.")
    arg.add_argument("--output", type=Path, required=True)
    args = arg.parse_args()
    groups = defaultdict(list)
    for manifest in sorted(selected_manifests(not args.new)):
        with manifest.open(encoding="utf8", newline="") as f:
            for row in csv.DictReader(f):
                if not int(row["axial_free"]):
                    continue
                key = (float(row.get("T_K") or 1000.), int(row["N"]), float(row["cap"]))
                groups[key].append((manifest, row))
    records = []
    for (temp, size, cap), group in sorted(groups.items()):
        records.append(dict(T_K=temp, N=size, W=cap, **analyze_group(group, cap)))
    if not records:
        raise ValueError("No matching manifests")
    result = {
        "note": "Raw production frames including biased umbrellas. Gap h=(1-cap)*d_nn-1.68052. "
                "An interval lower boundabove zero rules out a geometric rejection in that "
                "interval; a non-positive bound only means one cannot exclude it. The log "
                "records cumulative acceptance, not the reason for rejection. Fields, "
                "temperatures and sizes are not combined into a thermodynamic estimate.",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
