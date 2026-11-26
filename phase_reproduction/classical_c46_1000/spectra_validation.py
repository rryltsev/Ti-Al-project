"""Compare recovered classical labels with direct mode sums on saved independent meshes."""
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np

from fit_classical import HERE, KB, T0, classical_labels


PROJECT = HERE.parent
H_THZ = 0.004135667696923859


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def direct(mesh, n):
    """Use the same gamma/cutoff convention as atomistic/label_cell.py."""
    nu, qpoints, weights = (mesh[key] for key in ("frequencies_THz", "qpoints", "weights"))
    if nu.shape != (len(qpoints), 3*n) or np.any(weights <= 0):
        raise ValueError("Unexpected mesh shape or q weights")
    gamma, = np.flatnonzero(np.all(np.isclose(qpoints, np.rint(qpoints), atol=1e-8), axis=1))
    zero = np.argsort(np.abs(nu[gamma]))[:3]
    if np.max(np.abs(nu[gamma, zero])) > 1e-5:
        raise ValueError("Three common translations must be present")
    keep = np.ones(nu.shape, dtype=bool)
    keep[gamma, zero] = False
    if np.min(nu[keep]) <= 1e-5:
        raise ValueError("Unexpected extra soft or unstable mode in a stable structure")
    wn = np.broadcast_to((weights / weights.sum() / n)[:, None], nu.shape)[keep]
    z = H_THZ*nu[keep]/(KB*T0)
    fclass = KB*T0*float(np.dot(wn, np.log(z)))
    fquant = KB*T0*float(np.dot(wn, np.log(2*np.sinh(z/2))))
    return fclass, fquant


def inspect_archive(path, expected):
    """Stream each gzip archive once after checksum verification."""
    states = {}
    sha256 = checksum(path)
    if sha256.lower() != expected.lower():
        raise ValueError(f"Transfer checksum disagrees for {path}")
    with tarfile.open(path, "r|gz") as f:
        for member in f:
            name = Path(member.name)
            if not member.isfile() or name.name not in ("label.json", "mesh.npz", "vibrations.csv"):
                continue
            state = states.setdefault(name.parent.name, {})
            if name.name in state:
                raise ValueError(f"Duplicate {name}")
            stream = f.extractfile(member)
            if name.name == "label.json":
                state[name.name] = json.load(stream)
            elif name.name == "mesh.npz":
                with np.load(io.BytesIO(stream.read())) as z:
                    state[name.name] = {k: z[k] for k in z}
            else:
                state[name.name] = list(csv.DictReader(io.StringIO(stream.read().decode())))
    return sha256, states


def summary(values):
    values = np.asarray(values)
    return dict(n=len(values), rms=float(np.sqrt(np.mean(values**2))),
                min=float(values.min()), max=float(values.max()), max_abs=float(abs(values).max()))


def run():
    mapping_file = PROJECT/"author_comparison/cells/shared_decorations.csv"
    labels_file = HERE/"structure_labels.csv"
    mapping = {r["ours"]: r["author"] for r in rows(mapping_file)}
    author = {(r["configuration"], float(r["volume_A3_atom"])): r for r in rows(labels_file)}
    ordered_volumes = sorted({key[1] for key in author})
    vref, = [v for v in ordered_volumes if abs(v-17.28833) < 1e-5]
    archives = sorted((PROJECT/"cluster/runs").glob("atat-v*/outputs/results.tar.gz"))
    pilot = PROJECT/"cluster/runs/alloy-ti-bulkha-01/outputs/phonons.tar.gz"
    sources, shared, pilot_records, all_atat = {}, [], [], []
    for path in [*archives, pilot]:
        is_pilot = path == pilot
        if is_pilot:
            verification = (path.parent/"verified.sha256").read_text().splitlines()
            sha, = [line.split()[0] for line in verification if line.split()[-1].endswith(path.name)]
        else:
            sha = (path.parent/"raw.sha256").read_text().split()[0]
        digest, states = inspect_archive(path, sha)
        archive_key = path.relative_to(PROJECT).as_posix()
        sources[archive_key] = dict(sha256=digest, structures=len(states))
        for tag, state in sorted(states.items()):
            meta = state["label.json"]
            if not meta["stable_mesh"]:
                assert "vibrations.csv" not in state
                continue
            klass, qsum = direct(state["mesh.npz"], meta["n_atoms"])
            decimals = state["vibrations.csv"]
            temps = np.array([float(r["temperature_K"]) for r in decimals])
            quantum = np.array([float(r["Fvib_eV_atom"]) for r in decimals])
            fq, = quantum[temps == T0]
            recovered = classical_labels(temps, quantum[None, :], 1000, 4)[0].item()
            row = dict(structure=tag, Fcl_direct_eV_atom=klass,
                       Fcl_recovered_minus_direct_eV_atom=recovered-klass,
                       Fq_archived_minus_direct_eV_atom=float(fq-qsum))
            if is_pilot:
                pilot_records.append(row)
                continue
            all_atat.append(row)
            own = tag.split("_v")[0]
            if own in mapping:
                ratio = meta["volume_ratio"]
                v, = [v for v in ordered_volumes if abs(v/vref-ratio) < 1e-7]
                target = author.get((mapping[own], v))
                if target is not None:
                    shared.append(dict(author=mapping[own], atat=own, volume_ratio=ratio,
                                       direct_minus_author_classical_meV_atom=1000*(klass-float(target["Fcl_eV_atom"])),
                                       direct_minus_author_quantum_meV_atom=1000*(qsum-float(target["Fq_eV_atom"]))))
    if len(archives) != 9 or len(all_atat) != 396 or len(pilot_records) != 3:
        raise ValueError("Incomplete ATAT or pilot archives")
    groups = {}
    for title, group in [("ATAT", all_atat), ("eight_atom_pilots", pilot_records)]:
        groups[title] = {key: summary([r[key] for r in group]) for key in
                         ["Fcl_recovered_minus_direct_eV_atom", "Fq_archived_minus_direct_eV_atom"]}
        if any(value["max_abs"] > 5e-10 for value in groups[title].values()):
            raise ValueError(f"Spectral recovery disagrees for {title}")
    comparisons = {}
    for name, subset in [("all_volumes", shared), ("reference_volume", [r for r in shared if r["volume_ratio"] == 1])]:
        comparisons[name] = {key: summary([r[key] for r in subset]) for key in
                             ["direct_minus_author_classical_meV_atom", "direct_minus_author_quantum_meV_atom"]}
    out = dict(temperature_K=T0, energy_convention="Per atom with three translations at Gamma removed; "
               "q weights normalized over the Brillouin zone. Input frequencies are ordinary frequencies in THz.",
               numerical_check="Independent ATAT or pilot archive: fit the actual stored quantum free-energy "
               "table (1000 to 1800 K, 25 K increments, order four) and compare to the direct mode sum at 1000 K.",
               shared_check="Compare each available matched decoration and volume from two separate phonon "
               "calculations. Subtract the author's recovered classical label from our direct spectral sum; "
               "repeat for quantum labels to quantify preparation and mesh differences.",
               local_archives=sources, shared_map_sha256=checksum(mapping_file),
               author_labels_sha256=checksum(labels_file), same_spectrum_validation=groups,
               pilot_details=pilot_records, matched_author_validation=comparisons, matched_author_details=shared)
    (HERE/"spectra_validation.json").write_text(json.dumps(out, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"spectral": groups, "matched": comparisons}, indent=2))


if __name__ == "__main__":
    run()
