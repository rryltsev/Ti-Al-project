"""Prepare the two uniquely specified ordered structures for a CPU MTP EOS check."""
import argparse
import csv
from pathlib import Path

import numpy as np

from hcp_orbits import VOLUME_RATIOS, VREF, hcp_geometry, sublattices
from mtp_cfg import Frame, write_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = []
    with (args.out/"pilot_manifest.csv").open("w", newline="") as f:
        table = csv.writer(f)
        table.writerow(["frame", "structure", "volume_ratio", "volume_A3_per_atom",
                        "N", "N_Ti", "N_Al", "type0", "type1"])
        # Both cells (96 sites) extend beyond the pair cutoff in every direction.
        shape = (4, 4, 3)
        for structure in ("Ti_hcp", "Ti3Al_D019"):
            types = (sublattices(shape) == 0).astype(int)
            if structure == "Ti_hcp":
                types[:] = 0
            for ratio in VOLUME_RATIOS:
                cell, positions = hcp_geometry(shape, ratio)
                frames.append(Frame(cell, positions, types))
                table.writerow([len(frames), structure, ratio, ratio*VREF, len(types),
                                int(np.sum(types == 0)), int(np.sum(types == 1)), "Ti", "Al"])
    write_cfg(args.out/"pilot.cfg", frames)


if __name__ == "__main__":
    main()
