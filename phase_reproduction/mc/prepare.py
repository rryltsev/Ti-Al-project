"""Export the translational motifs and per-temperature coefficient rows for vcs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "atomistic"))
from hcp_orbits import ORBIT_NAMES, STATIC_S1, VREF, templates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--coefficients", type=Path, help="JSON {orbit_names, volumes_A3, coefficients_eV}, one temperature")
    source.add_argument("--grid", type=Path, help="NPZ produced by the atomistic CE fit")
    p.add_argument("--model", choices=["static", "HA", "QHA"], default="static")
    p.add_argument("--temperature", type=float, help="tabulated temperature in kelvin")
    args = p.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    groups = templates()
    with (args.directory / "motifs.txt").open("w", encoding="ascii", newline="\n") as f:
        for j, name in enumerate(ORBIT_NAMES[2:], 2):
            for cluster in groups[name]:
                fields = [j, len(cluster), *(x for node in cluster for x in node)]
                print(*fields, file=f)
    volumes, coeff = [VREF], [STATIC_S1]
    if args.coefficients:
        data = json.loads(args.coefficients.read_text(encoding="utf-8"))
        if tuple(data["orbit_names"]) != ORBIT_NAMES:
            raise ValueError("Unexpected orbit order")
        volumes, coeff = data["volumes_A3"], data["coefficients_eV"]
    if args.grid:
        with np.load(args.grid, allow_pickle=False) as grid:
            if tuple(grid["orbit_names"].astype(str)) != ORBIT_NAMES:
                raise ValueError("Unexpected orbit order in fit")
            volumes = np.asarray(grid["volume_A3_atom"])
            coeff = grid["J_static"].copy()
            if args.model != "static":
                idx = np.flatnonzero(grid["temperature_K"] == args.temperature)
                if len(idx) != 1:
                    raise ValueError("choose an explicitly fitted temperature using --temperature")
                coeff += grid["J_vib"][:,idx[0],:]
            if args.model != "QHA":
                idx = np.flatnonzero(np.isclose(volumes, VREF, atol=1e-6, rtol=0))
                if len(idx) != 1:
                    raise ValueError("Vref is missing or ambiguous")
                volumes, coeff = volumes[idx], coeff[idx]
    elif not args.coefficients and (args.model != "static" or args.temperature is not None):
        p.error("HA and QHA require fitted coefficients")
    values = np.asarray(coeff)
    if not (len(volumes) and values.shape == (len(volumes), 20) and np.isfinite(values).all()
            and np.isfinite(volumes).all() and np.min(volumes) > 0 and np.all(np.diff(volumes) > 0)):
        raise ValueError("Invalid volume/coefficient grid")
    np.savetxt(args.directory / "coefficients.txt", np.column_stack([volumes, values]), fmt="%.17g")


if __name__ == "__main__":
    main()
