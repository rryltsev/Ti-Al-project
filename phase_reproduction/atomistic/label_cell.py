"""Label a fixed chemical configuration using CPU MLIP and quantum harmonic phonons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.optimize import minimize
import phonopy
from phonopy.structure.atoms import PhonopyAtoms

from hcp_orbits import VREF, hcp_geometry, sublattices
from mtp_cfg import Frame, read_cfg, write_cfg


KB = 8.617333262145e-5  # eV/K
H_THZ = .004135667696923859  # eV/THz (ordinary frequency)
CUTOFF_THZ = 1e-5
MASSES = {"Ti": 47.867, "Al": 26.9815385}


def evaluate(frames, binary, model, version):
    with tempfile.TemporaryDirectory() as work:
        base = Path(work)
        write_cfg(base/"in.cfg", frames)
        if version == 3:
            command = [binary, "calculate_efs", model, "in.cfg", "--output_filename=out.cfg"]
        else:
            command = [binary, "calc-efs", model, "in.cfg", "out.cfg"]
        subprocess.run(command, cwd=base, check=True, stdout=subprocess.DEVNULL)
        files = list(base.glob("out.cfg*"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one MLIP result file, found {files}")
        results = read_cfg(files[0])
    if len(results) != len(frames):
        raise ValueError("MLIP frame count mismatch")
    for expected, result in zip(frames, results):
        if not np.array_equal(expected.types, result.types) or not np.allclose(expected.cell, result.cell, atol=1e-6, rtol=0):
            raise ValueError("Cell or species mismatch in MLIP output")
        delta = (expected.positions - result.positions) @ np.linalg.inv(expected.cell)
        error = (delta - np.rint(delta)) @ expected.cell
        if np.max(abs(error)) > 1.1e-6:
            raise ValueError("Position or frame order mismatch in MLIP output")
        if result.energy is None or result.forces is None or not np.isfinite(result.energy) or not np.isfinite(result.forces).all():
            raise ValueError("Missing/nonfinite MLIP energy or forces")
        # MLIP text EFS prints geometry to six places. Keep the evaluated input
        # coordinates at full precision for symmetry analysis and displacements.
        result.cell = expected.cell
        result.positions = expected.positions
    return results


def relax_fixed_cell(frame, calc):
    """Fixed-cell optimization, verified against the largest Cartesian force norm."""
    def objective(x):
        f = calc([Frame(frame.cell, x.reshape(-1, 3), frame.types)])[0]
        return f.energy, -f.forces.ravel()
    initial = calc([frame])[0]
    if np.linalg.norm(initial.forces, axis=1).max() <= 1e-5:
        return initial
    result = minimize(objective, frame.positions.ravel(), jac=True, method="L-BFGS-B",
                      options={"gtol": 3e-6, "ftol": 1e-15, "maxiter": 2000, "maxls": 40})
    relaxed = calc([Frame(frame.cell, result.x.reshape(-1, 3), frame.types)])[0]
    fmax = np.linalg.norm(relaxed.forces, axis=1).max()
    if fmax > 1e-5:
        raise RuntimeError(f"Fixed-cell relaxation stopped at fmax={fmax}: {result.message}")
    return relaxed


def phonon_frame(atoms):
    types = np.array([{"Ti": 0, "Al": 1}[s] for s in atoms.symbols])
    return Frame(np.asarray(atoms.cell), np.asarray(atoms.positions), types)


def harmonic_labels(frequencies, qpoints, weights, temperatures, n):
    """Return per-atom quantum Fvib and stability on this mesh."""
    nu = np.asarray(frequencies)
    q = np.asarray(qpoints)
    if not np.isfinite(nu).all() or nu.shape != (len(q), 3*n):
        raise ValueError("Invalid phonon array")
    gamma = np.flatnonzero(np.all(np.isclose(q, np.rint(q), atol=1e-8), axis=1))
    if len(gamma) != 1:
        raise ValueError("Gamma-centered mesh required")
    keep = np.ones(nu.shape, dtype=bool)
    translations = np.argsort(abs(nu[gamma[0]]))[:3]
    if np.max(abs(nu[gamma[0], translations])) > CUTOFF_THZ:
        raise ValueError("Gamma translational modes exceed 1e-5 THz after ASR")
    keep[gamma[0], translations] = False
    minimum = float(nu[keep].min())
    if minimum <= 0:
        return None, minimum
    w = np.broadcast_to((weights / np.sum(weights))[:, None], nu.shape)
    eps = H_THZ*nu[keep]
    weight = w[keep] / n
    zpe = .5 * np.dot(weight, eps)
    selected = eps > H_THZ*CUTOFF_THZ
    result = []
    for temp in temperatures:
        thermal = np.log(-np.expm1(-eps[selected] / (KB*temp)))
        result.append(float(zpe + KB*temp * np.dot(weight[selected], thermal)))
    return result, minimum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--mlp", required=True, help="serial MLIP binary (absolute path or command on PATH)")
    p.add_argument("--mlip-version", type=int, choices=[2, 3], default=3)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--prototype", choices=["Ti_hcp", "Ti3Al_D019"])
    source.add_argument("--cell", type=Path, help="one ideal chemical CFG at vref, Ti=0 and Al=1")
    p.add_argument("--ratio", type=float, default=1.)
    p.add_argument("--static-only", action="store_true", help="write fixed-cell E without phonon sampling")
    p.add_argument("--supercell", type=int, nargs=3,
                   help="phonon repeats of the chemical cell")
    p.add_argument("--mesh", type=int, help="isotropic Gamma-centered mesh size")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not args.static_only and args.supercell is None:
        p.error("--supercell is required for phonons")
    if not np.isfinite(args.ratio) or args.ratio <= 0 or (args.supercell and min(args.supercell) < 1) or (args.mesh is not None and args.mesh < 1):
        p.error("positive volume ratio, mesh and repeats required")
    if args.output.exists():
        p.error("output directory already exists, choose a fresh directory")
    model = args.model.resolve(strict=True)
    if args.cell:
        frames = read_cfg(args.cell)
        if len(frames) != 1:
            p.error("supply a single chemical configuration")
        initial = frames[0]
        v = abs(np.linalg.det(initial.cell)) / len(initial.types)
        if not np.isclose(v, VREF, rtol=5e-5):
            p.error(f"input volume {v} A^3/atom does not match vref")
        scale = (args.ratio*VREF/v)**(1/3)
        frame = Frame(scale*initial.cell, scale*initial.positions, initial.types)
    else:
        shape = (2, 2, 1) if args.prototype == "Ti3Al_D019" else (1, 1, 1)
        cell, pos = hcp_geometry(shape, args.ratio)
        types = (sublattices(shape) == 0).astype(int) if args.prototype == "Ti3Al_D019" else np.zeros(len(pos), dtype=int)
        frame = Frame(cell, pos, types)
    def calc(frames):
        return evaluate(frames, args.mlp, model, args.mlip_version)
    args.output.mkdir(parents=True)
    write_cfg(args.output/"ideal.cfg", [frame])
    relaxed = relax_fixed_cell(frame, calc)
    write_cfg(args.output/"relaxed.cfg", [relaxed])
    symbols = np.where(frame.types, "Al", "Ti").tolist()
    n = len(symbols)
    metadata = dict(model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    source_cell_sha256=hashlib.sha256(args.cell.read_bytes()).hexdigest() if args.cell else None,
                    ideal_cfg_sha256=hashlib.sha256((args.output/"ideal.cfg").read_bytes()).hexdigest(),
                    structure=args.prototype or args.cell.name, n_atoms=n, species=symbols,
                    volume_A3_atom=abs(np.linalg.det(frame.cell))/n, volume_ratio=args.ratio,
                    static_eV_atom=relaxed.energy/n, fmax_eV_A=np.linalg.norm(relaxed.forces, axis=1).max())
    if args.static_only:
        metadata["stable_mesh"] = None
        (args.output/"label.json").write_text(json.dumps(metadata, indent=2))
        return
    atoms = PhonopyAtoms(symbols=symbols, masses=[MASSES[s] for s in symbols],
                         cell=relaxed.cell, positions=relaxed.positions)
    ph = phonopy.Phonopy(atoms, np.diag(args.supercell), primitive_matrix=np.eye(3), symprec=1e-5)
    ph.generate_displacements(distance=.01, is_plusminus=True)
    cells = [ph.supercell, *ph.supercells_with_displacements]
    raw = calc([phonon_frame(a) for a in cells])
    forces = np.stack([f.forces for f in raw])
    np.savez_compressed(args.output/"raw_forces.npz", forces=forces)
    ph.forces = forces[1:] - forces[0]
    ph.produce_force_constants(fc_calculator="traditional", calculate_full_force_constants=False)
    ph.symmetrize_force_constants()
    ph.save(filename=str(args.output/"phonopy.yaml"), settings={"force_constants": True})
    mesh = args.mesh or (16 if n <= 32 else 6)
    ph.run_mesh([mesh]*3, is_gamma_center=True)
    spectrum = ph.mesh
    temperatures = np.arange(600., 1801., 25.)
    free, minimum = harmonic_labels(spectrum.frequencies, spectrum.qpoints, spectrum.weights, temperatures, n)
    np.savez_compressed(args.output/"mesh.npz", frequencies_THz=spectrum.frequencies,
                        qpoints=spectrum.qpoints, weights=spectrum.weights)
    metadata.update(minimum_internal_THz=minimum, stable_mesh=free is not None, supercell=args.supercell,
                    mesh=mesh, n_displacements=len(raw)-1, phonopy_version=phonopy.__version__)
    if free is not None:
        with (args.output/"vibrations.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["temperature_K", "Fvib_eV_atom"])
            w.writerows(zip(temperatures, free))
    (args.output/"label.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
