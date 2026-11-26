"""Reproduce the author C46 QHA tangent and prepare optional atomistic starting cells.

python c46_reference.py extract PATH_TO_AUTHOR_PACKAGE
python c46_reference.py seed alpha2 4 4 4 --rng 123 --output initial.cfg
"""
import argparse
import csv
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REFERENCE = HERE / "c46_reference.json"
GRID = PROJECT / "author_comparison/eci/author_refit_grid.npz"
sys.path.insert(0, str(PROJECT / "thermo"))
sys.path.insert(0, str(PROJECT / "atomistic"))
from hcp_orbits import VREF, hcp_geometry, sublattices
from mfa import KB, VolumeIsotherm, sublattice_weights
from mtp_cfg import Frame, write_cfg


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(file):
    with file.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def extract(package):
    windows = "phase_check/MC_window_means.csv"
    contacts = "phase_check/C46_phase_boundary_points.csv"
    replay = "phase_check/reproduce_C46_phase_contacts.py"
    manifest = json.loads((package / "manifest.json").read_text())
    hashes = {name: digest(package / name) for name in (windows, contacts, replay)}
    if any(value != manifest["files_sha256"][key] for key, value in hashes.items()):
        raise ValueError("Author manifest checksum mismatch")
    points = [r for r in rows(package / windows)
              if (r["model"], r["T_K"], r["N"]) == ("QHA", "1000", "5120")]
    line, = [r for r in rows(package / contacts)
             if (r["statistics"], r["energetic_model"], r["T_K"], r["MC_sites"])
             == ("MC", "QHA", "1000", "5120")]
    xy = sorted((float(r["mean_c_Al"]), float(r["mean_delta_mu_eV"])) for r in points)
    tangent = runpy.run_path(str(package / replay))["tangent"](*np.array(xy).T)
    if not tangent["resolved"]:
        raise ValueError("C46 tangent unresolved")
    with np.load(GRID) as grid:
        t, = np.flatnonzero(grid["temperature_K"] == 1000)
        model = VolumeIsotherm(sublattice_weights(), grid["volume_A3_atom"],
                               grid["J_static"] + grid["J_vib"][:, t, :],
                               1000, volume_spline="not-a-knot")
    branches = {}
    for name in ("alpha", "alpha2"):
        c = tangent[f"c_{name}"]
        if abs(100*c - float(line[f"c_{name}_Al_at_percent"])) > 1e-9:
            raise ValueError(f"Published and replayed C46 {name} disagree")
        q = c if name == "alpha" else float(model.equilibrium(np.array([c]))[1][0])
        _, volume = model.minimize_volume(np.array([c]), np.array([q]))
        branches[name] = {
            "Al_at_percent": 100*c,
            "Al_at_percent_conditional_95": [
                float(line[f"{name}_conditional_95_{bound}_at_percent"])
                for bound in ("low", "high")],
            "start_mfa_q": q, "start_mfa_r": (4*c-q)/3,
            "start_mfa_volume_A3_atom": float(volume[0])}
    mass_shift = 1.5*KB*1000*math.log(26.9815385 / 47.867)
    return {
        "reference": "Author v16 C46, VCSGC Monte Carlo, QHA, fixed c/a",
        "temperature_K": 1000, "pressure_GPa": 0, "N_sites": 5120,
        "model_sha256": manifest["MTP_sha256"], "model_DOI": manifest["MTP_DOI"],
        "package_files_sha256": hashes, "refitted_eci_grid_sha256": digest(GRID),
        "phase": branches, "delta_mu_Al_minus_Ti_eV": tangent["delta_mu_eV"],
        "delta_mu_spin_eV": tangent["delta_mu_eV"]/2,
        "cartesian_config_field_start_estimate_eV": tangent["delta_mu_eV"] + mass_shift,
        "cartesian_config_minus_classical_physical_field_eV": mass_shift,
        "barrier_eV_per_atom": tangent["finite_cell_barrier_eV_atom"],
        "tangent_residual_eV": tangent["root_residual_eV"],
        "bootstrap": {k: int(line["bootstrap_" + k]) for k in ("resolved", "attempts")},
        "windows_c_Al_and_mu_occ_eV": xy,
        "initialization": "Volumes and preferred-site probabilities are fixed-c/a C46 MFA starts evaluated at MC endpoint compositions. The coordinate-only starting field includes the classical momentum correction to C46. All starts require direct atomistic equilibration.",
        "uncertainty": "Reported 95% intervals are conditional author bootstrap results from the 5120-site MC analysis; they omit model and finite-size uncertainty."}


def seed(phase, shape, rng_seed, output):
    data = json.loads(REFERENCE.read_text())["phase"][phase]
    labels = sublattices(shape)
    rng = np.random.default_rng(rng_seed)
    total = round(len(labels)*data["Al_at_percent"]/100)
    types = np.zeros(len(labels), dtype=int)
    if phase == "alpha":
        types[rng.choice(len(labels), total, replace=False)] = 1
    else:
        preferred = np.flatnonzero(labels == 0)
        other = np.flatnonzero(labels != 0)
        first = min(total, round(len(preferred)*data["start_mfa_q"]))
        types[rng.choice(preferred, first, replace=False)] = 1
        types[rng.choice(other, total-first, replace=False)] = 1
    cell, positions = hcp_geometry(shape, data["start_mfa_volume_A3_atom"]/VREF)
    write_cfg(output, [Frame(cell, positions, types)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("extract").add_argument("package", type=Path)
    make = sub.add_parser("seed")
    make.add_argument("phase", choices=["alpha", "alpha2"])
    make.add_argument("shape", type=int, nargs=3, metavar=("NX", "NY", "NZ"))
    make.add_argument("--rng", type=int, required=True)
    make.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "extract":
        REFERENCE.write_text(json.dumps(extract(args.package), indent=2)+"\n", encoding="utf-8")
    else:
        seed(args.phase, tuple(args.shape), args.rng, args.output)
