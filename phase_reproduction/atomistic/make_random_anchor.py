"""Generate an explicitly reconstructed Ti48Al16 random anchor for SI S2."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from hcp_orbits import hcp_geometry
from mtp_cfg import Frame, write_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    shape = (4, 4, 2)
    cell, pos = hcp_geometry(shape)
    types = np.zeros(len(pos), dtype=int)
    rng = np.random.Generator(np.random.PCG64(20261013))
    indices = rng.choice(len(pos), 16, replace=False)
    types[indices] = 1
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output/"Ti48Al16_candidate.cfg"
    write_cfg(path, [Frame(cell, pos, types)])
    info = dict(status="independent decoration; SI S2 reports composition and seed only",
                shape=shape, seed=20261013, bit_generator="PCG64", selection="choice without replacement",
                Al_site_indices_zero_based=sorted(map(int, indices)),
                cell_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), supercell=[2,2,3], mesh=[6,6,6])
    (args.output/"source.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
