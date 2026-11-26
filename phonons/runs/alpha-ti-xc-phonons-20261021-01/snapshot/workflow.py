"""Bounded alpha-Ti XC comparison; run VASP only within a Slurm allocation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

import ase
from ase.build import bulk
from ase.io import read, write
import numpy as np
import phonopy
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp, write_vasp

ROOT = Path(__file__).resolve().parent
C = json.loads((ROOT / "protocol.json").read_text())
LIB = Path(C["paw_library"])

def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def incar(xc, relax=False, encut=None, sigma=None):
    p = dict(SYSTEM="alpha-Ti " + xc, PREC="Accurate", ENCUT=encut or C["encut_eV"],
             EDIFF=C["ediff_eV"], NELM=200, ALGO="Normal", ISPIN=1, ISMEAR=C["ismear"],
             SIGMA=C["sigma_eV"] if sigma is None else sigma, LASPH=".TRUE.",
             LREAL=".FALSE.", LWAVE=".FALSE.", LCHARG=".FALSE.", LMAXMIX=4,
             ISYM=2 if relax else 0, NCORE=4, KPAR=1, ISTART=0, ICHARG=2,
             NSW=100 if relax else 0, IBRION=2 if relax else -1, ISIF=3 if relax else 2,
             PSTRESS=0, POTIM=0.3, EDIFFG=1e-7)
    if xc == "r2SCAN": p["METAGGA"] = "R2SCAN"
    else: p["GGA"] = {"PBE": "PE", "PBEsol": "PS"}[xc]
    return "\n".join(f"{k} = {v}" for k, v in p.items()) + "\n"

def inputs(folder, atoms, xc, mesh, relax=False, **kwargs):
    folder.mkdir()
    write(folder / "POSCAR", atoms, format="vasp", direct=True, vasp5=True)
    (folder / "INCAR").write_text(incar(xc, relax, **kwargs))
    (folder / "KPOINTS").write_text("Gamma-centered common protocol\n0\nGamma\n" +
                                     " ".join(map(str, mesh)) + "\n0 0 0\n")
    shutil.copyfile(ROOT / "POTCAR", folder / "POTCAR")
    shutil.copyfile(ROOT / "potcar-provenance.json", folder / "potcar-provenance.json")
    save(folder / "input-hashes.json", {n: digest(folder / n) for n in
                                        ["POSCAR", "INCAR", "KPOINTS", "POTCAR"]})

def validate(folder, relax=False):
    text = (folder / "OUTCAR").read_text(errors="replace")
    if "General timing and accounting informations" not in text:
        raise RuntimeError(f"{folder}: missing normal VASP termination")
    xml = ET.parse(folder / "vasprun.xml").getroot()
    steps = xml.findall("calculation")
    if not steps or any(len(x.findall("scstep")) >= 200 for x in steps):
        raise RuntimeError(f"{folder}: electronic convergence not established")
    if relax and "reached required accuracy" not in text:
        raise RuntimeError(f"{folder}: ionic convergence not established")
    atoms = read(folder / "vasprun.xml", index=-1)
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=False) * 1602.1766208
    if not np.isfinite(forces).all() or not np.isfinite(stress).all():
        raise RuntimeError("Non-finite VASP result")
    result = {"energy_eV": float(atoms.get_potential_energy()),
              "free_energy_eV": float(atoms.calc.results["free_energy"]),
              "max_force_eV_A": float(np.linalg.norm(forces, axis=1).max()),
              "stress_kbar_ASE_tension_positive": stress.tolist(),
              "max_stress_kbar": float(np.abs(stress).max()),
              "force_drift_eV_A": forces.mean(axis=0).tolist(),
              "ionic_steps": len(steps), "scf_steps": [len(x.findall("scstep")) for x in steps],
              "volume_A3": float(atoms.get_volume()),
              "outputs": {n: digest(folder / n) for n in ["OUTCAR", "vasprun.xml", "CONTCAR"]}}
    save(folder / "validation.json", result)
    np.save(folder / "forces.npy", forces)
    return atoms, result

def vasp(folder, relax=False):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("VASP must run under Slurm")
    with (folder / "vasp.stdout").open("w") as output:
        subprocess.run(["srun", "--ntasks=" + str(C["resources"]["ranks"]), "vasp_std"],
                       cwd=folder, stdout=output, stderr=subprocess.STDOUT, check=True)
    return validate(folder, relax)

def phonon_object(poscar):
    cell = read_vasp(str(poscar))
    cell.masses = [C["mass_amu"]] * len(cell)
    ph = Phonopy(cell, np.diag(C["supercell"]), primitive_matrix=np.eye(3),
                 symprec=1e-5, calculator="vasp")
    ph.generate_displacements(distance=C["displacement_A"], is_plusminus=True)
    return ph

def prepare():
    spec = importlib.util.spec_from_file_location("paw", LIB / "vasp_potentials.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.assemble(LIB / "registry.json", C["paw_family"], [C["paw_variant"]], ROOT / "POTCAR")
    if digest(ROOT / "POTCAR") != C["paw_sha256"]: raise RuntimeError("Wrong Ti potential")
    atoms = bulk("Ti", "hcp", a=C["initial_a_A"], c=C["initial_c_A"])
    for xc in C["functionals"]:
        folder = ROOT / xc
        folder.mkdir()
        inputs(folder / "relax-01", atoms, xc, C["relax_kmesh"], relax=True)
    ph = phonon_object(ROOT / "PBE/relax-01/POSCAR")
    assert len(ph.supercell) == 96
    assert len(ph.supercells_with_displacements) % 2 == 0
    # Exercise the exact API and file writers before production submission.
    ph.save(filename=str(ROOT / "smoke-phonopy.yaml"))
    write_vasp(str(ROOT / "smoke-SPOSCAR"), ph.supercell)
    save(ROOT / "software.json", {"python": sys.version, "phonopy": phonopy.__version__,
         "ase": ase.__version__, "numpy": np.__version__, "atoms": len(ph.supercell),
         "displacements": len(ph.supercells_with_displacements), "spacegroup": str(ph.symmetry.dataset.international),
         "vasp_sha256": digest(Path("/home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/src/vasp.6.6.1/bin/vasp_std"))})
    print((ROOT / "software.json").read_text(), flush=True)

def run(xc):
    if xc not in C["functionals"]: raise ValueError(xc)
    folder = ROOT / xc
    state = {"xc": xc, "job_id": os.environ.get("SLURM_JOB_ID"), "state": "running"}
    def progress(step):
        state["step"] = step
        save(folder / "status.json", state)
        print(xc, step, flush=True)
    try:
        progress("relax-01")
        atoms, _ = vasp(folder / "relax-01", relax=True)
        # Restart from CONTCAR with a fresh plane-wave basis to reduce cell-relaxation Pulay bias.
        for attempt in range(2, 4):
            name = f"relax-{attempt:02d}"
            inputs(folder / name, atoms, xc, C["relax_kmesh"], relax=True)
            progress(name)
            atoms, _ = vasp(folder / name, relax=True)
            name = f"equilibrium-{attempt:02d}"
            inputs(folder / name, atoms, xc, C["relax_kmesh"])
            progress(name)
            atoms, quality = vasp(folder / name)
            if quality["max_force_eV_A"] <= C["relax_max_force_eV_A"] and quality["max_stress_kbar"] <= C["relax_max_stress_kbar"]:
                break
        else: raise RuntimeError("Equilibrium force/stress gate failed after bounded relaxations")
        write(folder / "POSCAR-relaxed", atoms, format="vasp", direct=True)
        ph = phonon_object(folder / "POSCAR-relaxed")
        if ph.symmetry.dataset.number != 194: raise RuntimeError("Relaxed structure is not hcp P63/mmc")
        ph.save(filename=str(folder / "phonopy_disp.yaml"))
        write_vasp(str(folder / "SPOSCAR"), ph.supercell)
        save(folder / "geometry.json", {"cell_A": atoms.cell.tolist(), "volume_A3": atoms.get_volume(),
             "spacegroup": int(ph.symmetry.dataset.number), "n_displacements": len(ph.supercells_with_displacements)})
        def to_ase(cell):
            return ase.Atoms(symbols=cell.symbols, cell=cell.cell, scaled_positions=cell.scaled_positions, pbc=True)
        reference = folder / "force-000"
        inputs(reference, to_ase(ph.supercell), xc, C["force_kmesh"])
        progress(reference.name)
        vasp(reference)
        forces = []
        for i, cell in enumerate(ph.supercells_with_displacements, 1):
            target = folder / f"force-{i:03d}"
            inputs(target, to_ase(cell), xc, C["force_kmesh"])
            progress(target.name)
            result, _ = vasp(target)
            forces.append(result.get_forces())
        np.save(folder / "displaced-forces-raw.npy", np.array(forces))
        # Keep uncorrected forces; residual subtraction and acoustic sum rule are later analysis choices.
        first = to_ase(ph.supercells_with_displacements[0])
        diagnostics = []
        for name, mesh, kw in [
            ("check-cutoff", C["force_kmesh"], {"encut": C["diagnostics"]["encut_eV"]}),
            ("check-kmesh", C["diagnostics"]["kmesh"], {}),
            ("check-smearing", C["force_kmesh"], {"sigma": C["diagnostics"]["sigma_eV"]})]:
            target = folder / name
            inputs(target, first, xc, mesh, **kw)
            progress(name)
            result, _ = vasp(target)
            delta = float(np.linalg.norm(result.get_forces() - forces[0], axis=1).max())
            diagnostics.append({"name": name, "max_force_difference_eV_A": delta,
                                "within_target": delta <= C["diagnostic_force_tolerance_eV_A"]})
        save(folder / "electronic-diagnostics.json", diagnostics)
        state["state"] = "completed_raw_forces"
        state["electronic_diagnostics_pass"] = all(d["within_target"] for d in diagnostics)
        state["publication_qualified"] = False
        progress("ready_for_analysis_and_supercell_convergence_review")
        files = {str(p.relative_to(folder)): digest(p) for p in folder.rglob("*") if p.is_file()}
        save(folder / "output-manifest.json", files)
    except Exception:
        state["state"] = "failed"
        state["error"] = traceback.format_exc()
        save(folder / "status.json", state)
        raise

if __name__ == "__main__":
    if sys.argv[1] == "prepare": prepare()
    else: run(sys.argv[2])
