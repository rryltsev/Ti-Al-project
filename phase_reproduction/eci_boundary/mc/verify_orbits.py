"""Verify per-frame integer cluster sums and reconstruct fixed-volume CE energies."""
import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from prepare import HERE, ORBIT_NAMES


MULT = (3, 3, 3, 1, 6, 3, 6, 6, 3, 6, 12, 12, 6, 1, 1, 6, 12, 3)


def verify(samples, coefficients):
    counts = Counter()
    max_error = 0.
    opener = gzip.open if samples.suffix == ".gz" else open
    with opener(samples, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        names = ["S_"+name for name in ORBIT_NAMES[2:]]
        if not all(name in reader.fieldnames for name in names):
            raise ValueError(f"Missing orbit sums in {samples}")
        for row in reader:
            n, n_al = int(row["N"]), int(row["n_Al"])
            name = row["model"]
            sums = [int(row[k]) for k in names]
            if not 0 <= n_al <= n or any(abs(s)>n*m or (s-n*m)%2 for s,m in zip(sums, MULT)):
                raise ValueError(f"Invalid composition or orbit sum at {name}/{row['replica']}/{row['sweep']}")
            products = [n/2, 2*n_al-n, *sums]
            computed = math.fsum(j*s for j,s in zip(coefficients[name], products)) / n
            error = abs(computed-float(row["Heff_eV_atom"]))
            if not math.isfinite(error) or error>1e-7:
                raise ValueError(f"Orbit reconstruction fails at {name}/{row['replica']}/{row['sweep']}")
            max_error = max(max_error, error)
            counts[name] += 1
    if not counts:
        raise ValueError("Empty archive")
    return dict(samples_by_model=dict(counts), samples=sum(counts.values()), max_error_eV_atom=max_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--inputs", type=Path, default=HERE/"hybrids")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    meta = json.loads((args.inputs/"metadata.json").read_text())
    if tuple(meta["orbit_names"]) != ORBIT_NAMES:
        raise ValueError("Unexpected orbit ordering")
    result = verify(args.samples, {v["name"]:v["J_eV"] for v in meta["variants"]})
    output = json.dumps(result, indent=2)+"\n"
    if args.output:
        args.output.write_text(output,encoding="utf8")
    print(output,end="")
