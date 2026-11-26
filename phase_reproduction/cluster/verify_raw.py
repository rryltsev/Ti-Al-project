"""Audit all downloaded ATAT-MTP raw phonon archives and their numerical labels."""
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np


def verify(run):
    dest = run / "outputs"
    archive = dest / "results.tar.gz"
    with archive.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    if (dest / "raw.sha256").read_text().strip() != sha + "  results.tar.gz":
        raise ValueError(f"Local transfer checksum mismatch in {run}")
    summary = {}
    with tarfile.open(archive, "r|gz") as tar:
        for m in tar:
            path = Path(m.name)
            if not m.isfile() or path.parts[0] != "results":
                continue
            entry = summary.setdefault(path.parent.name, {})
            key = path.name
            if key in entry:
                raise ValueError(f"Duplicate {m.name}")
            if key == "label.json":
                entry[key] = json.load(tar.extractfile(m))
            elif key in ("mesh.npz", "raw_forces.npz"):
                with np.load(io.BytesIO(tar.extractfile(m).read())) as data:
                    if not all(np.isfinite(data[k]).all() for k in data):
                        raise ValueError(f"Nonfinite values in {m.name}")
                    if key == "mesh.npz":
                        freq = data["frequencies_THz"]
                        w = data["weights"]
                        if np.any(w <= 0) or data["qpoints"].shape != (len(w), 3):
                            raise ValueError(f"Invalid q integration grid in {m.name}")
                        entry[key] = (freq.shape, len(w), int(w.sum()))
                    else:
                        entry[key] = data["forces"].shape
            elif key == "vibrations.csv":
                text = tar.extractfile(m).read().decode()
                data = list(csv.DictReader(io.StringIO(text)))
                t = [float(row["temperature_K"]) for row in data]
                if t != list(range(600, 1801, 25)) or not all(np.isfinite(float(row["Fvib_eV_atom"])) for row in data):
                    raise ValueError(f"Invalid free energies in {m.name}")
                entry[key] = len(data)
            else:
                entry[key] = m.size
    rows = []
    for key, item in sorted(summary.items()):
        lab = item["label.json"]
        n = lab["n_atoms"]
        ns = int(np.prod(lab["supercell"])) * n
        shape, nq, weight = item["mesh.npz"]
        if (item["raw_forces.npz"] != (1 + lab["n_displacements"], ns, 3) or
                shape != (nq, 3*n) or weight != lab["mesh"]**3 or
                lab["fmax_eV_A"] > 1e-5 or not np.isfinite(lab["static_eV_atom"]) or
                min(item["ideal.cfg"], item["relaxed.cfg"], item["phonopy.yaml.gz"]) <= 0 or
                lab["stable_mesh"] != ("vibrations.csv" in item) or
                lab["stable_mesh"] != (lab["minimum_internal_THz"] > 0)):
            raise ValueError(f"Raw output is inconsistent for {run.name}/{key}")
        rows.append(dict(run=run.name, label=key, ratio=lab["volume_ratio"], atoms=n,
            force_cell_atoms=ns, n_displacements=lab["n_displacements"], mesh=lab["mesh"],
            min_internal_THz=lab["minimum_internal_THz"], stable=lab["stable_mesh"],
            E_static_eV_atom=lab["static_eV_atom"], fmax_eV_A=lab["fmax_eV_A"],
            model_sha256=lab["model_sha256"], source_cell_sha256=lab["source_cell_sha256"]))
    if len(rows) != 46:
        raise ValueError(f"Missing labels from {archive}")
    return rows


def main():
    root = Path(__file__).resolve().parent
    runs = [p for p in (root / "runs").glob("atat-v[0-9]*") if (p/"outputs/raw.sha256").exists()]
    if len(runs) != 9:
        raise ValueError(f"Expected all nine retrieved volume campaigns, found {len(runs)}")
    rows = []
    for p in sorted(runs):
        rows.extend(verify(p))
    with (root / "raw_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x:(x["ratio"],x["label"])))
    print(f"{len(rows)} raw states verified; {sum(r['stable'] for r in rows)} stable")


if __name__ == "__main__":
    main()
