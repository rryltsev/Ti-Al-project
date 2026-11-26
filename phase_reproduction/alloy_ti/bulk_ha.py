"""Measure the bulk harmonic C46 fit error on the three periodic pilot decorations."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from decorations import representatives
from hcp_orbits import hcp_geometry
from label_cell import H_THZ, KB
from mtp_cfg import Frame, write_cfg


def run(task):
    name, sites, args = task
    cell, positions = hcp_geometry((2, 2, 1))
    types = np.zeros(len(positions), dtype=int)
    types[list(sites)] = 1
    source = Path(name + ".cfg")
    write_cfg(source, [Frame(cell, positions, types)])
    subprocess.run([sys.executable, "label_cell.py", "--cell", str(source), "--model", args.model,
                    "--mlp", args.mlp, "--output", name, "--supercell", "5", "5", "6",
                    "--mesh", "16"], check=True)
    label = json.loads((Path(name) / "label.json").read_text())
    if label["stable_mesh"]:
        data = np.load(Path(name)/"mesh.npz")
        nu, q, w = (data[key] for key in ("frequencies_THz", "qpoints", "weights"))
        gamma, = np.flatnonzero(np.max(abs(q - np.rint(q)), axis=1) < 1e-8)
        keep = np.ones(nu.shape, dtype=bool)
        keep[gamma, np.argsort(abs(nu[gamma]))[:3]] = False
        degeneracy = np.broadcast_to(w[:,None]/w.sum(), nu.shape)[keep] / len(types)
        eps = H_THZ * nu[keep]
        kt = KB * 1000
        label["classical_vib_eV_atom"] = float(degeneracy @ (kt*np.log(eps/kt)))
        label["quantum_vib_eV_atom"] = float(degeneracy @ (.5*eps+kt*np.log(-np.expm1(-eps/kt))))
        from_csv = np.loadtxt(Path(name)/"vibrations.csv", delimiter=",", skiprows=1)
        if not np.isclose(from_csv[from_csv[:,0] == 1000,1].item(),
                          label["quantum_vib_eV_atom"], atol=1e-9, rtol=0):
            raise ValueError("Inconsistent quantum phonon sum")
    return name, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mlp", required=True)
    args = parser.parse_args()
    tasks = [(name, sites, args) for name, sites in representatives().items()]
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = dict(pool.map(run, tasks))
    Path("bulk_ha.json").write_text(json.dumps(results, indent=2))
