"""Subset of MLIP text CFG used by the Ti-Al force/energy workflow."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Frame:
    cell: np.ndarray
    positions: np.ndarray
    types: np.ndarray
    energy: float | None = None
    forces: np.ndarray | None = None


def write_cfg(path, frames):
    with Path(path).open("w", encoding="ascii", newline="\n") as out:
        for f in frames:
            out.write(f"BEGIN_CFG\n Size\n    {len(f.types)}\n Supercell\n")
            np.savetxt(out, f.cell, fmt=" %.15g")
            out.write(" AtomData: id type cartes_x cartes_y cartes_z\n")
            for i, (typ, xyz) in enumerate(zip(f.types, f.positions), 1):
                out.write(f"    {i} {typ} " + " ".join(f"{v:.15g}" for v in xyz) + "\n")
            out.write("END_CFG\n\n")


def read_cfg(path):
    frames = []
    for raw in Path(path).read_text().split("BEGIN_CFG")[1:]:
        if "END_CFG" not in raw:
            raise ValueError(f"Incomplete frame in {path}")
        lines = [s.strip() for s in raw.split("END_CFG", 1)[0].splitlines() if s.strip()]
        n = int(lines[lines.index("Size")+1])
        start = lines.index("Supercell") + 1
        cell = np.array([v.split() for v in lines[start:start+3]], dtype=float)
        at = next(i for i, s in enumerate(lines) if s.startswith("AtomData:"))
        keys = lines[at].split()[1:]
        values = np.array([v.split() for v in lines[at+1:at+1+n]], dtype=float)
        order = np.argsort(values[:, keys.index("id")])
        if not np.array_equal(values[order, keys.index("id")], np.arange(1, n+1)):
            raise ValueError(f"Invalid atom IDs in {path}")
        def columns(names):
            return values[order][:, [keys.index(s) for s in names]]
        typ = columns(["type"]).ravel()
        if not np.isin(typ, [0, 1]).all():
            raise ValueError(f"Expected Ti=0, Al=1 in {path}")
        energy = float(lines[lines.index("Energy")+1]) if "Energy" in lines else None
        forces = columns(["fx", "fy", "fz"]) if "fx" in keys else None
        frames.append(Frame(cell, columns(["cartes_x", "cartes_y", "cartes_z"]),
                            typ.astype(int), energy, forces))
    return frames
