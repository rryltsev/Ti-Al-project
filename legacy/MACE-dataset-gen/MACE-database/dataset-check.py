#!/usr/bin/env python3
"""
QC script for (ext)XYZ datasets (ASE-readable).

Checks (per configuration):
  1) min interatomic distance (MIC-aware)
  2) cell volume > 0
  3) no NaN/Inf in positions/cell/energy/forces/stress
  4) max |F_i| below threshold (if forces exist)
  5) stress sanity (if stress exists)

Usage:
  python qc_xyz.py dataset.xyz
  python qc_xyz.py dataset.xyz --min-dist 1.8 --fmax 50 --smax 100 --out report.csv
"""

from __future__ import annotations


#!/usr/bin/env python3

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from ase.io import iread

try:
    from ase.io import iread
    from ase.geometry import cellpar_to_cell
except ImportError as e:
    raise SystemExit("This script requires ASE: pip install ase") from e


@dataclass
class QCResult:
    idx: int
    natoms: int
    ok: bool
    min_dist_A: float
    volume_A3: float
    has_forces: bool
    fmax_eVA: Optional[float]
    has_stress: bool
    smax_GPa: Optional[float]
    issues: str


def is_finite_array(x: np.ndarray) -> bool:
    return np.isfinite(x).all()


def cell_volume_A3(cell: np.ndarray) -> float:
    # cell is 3x3 (A). Volume = det(cell)
    return float(np.linalg.det(cell))


def min_distance_mic(atoms) -> float:
    """
    Compute minimum pair distance with Minimum Image Convention if PBC.
    O(N^2) but fine for "express QC" for ~100-300 atom configs.
    """
    pos = atoms.get_positions()  # (N,3)
    n = len(pos)
    if n < 2:
        return float("inf")

    pbc = atoms.get_pbc()
    use_mic = bool(np.any(pbc)) and atoms.cell is not None and atoms.cell.rank == 3

    if use_mic:
        cell = atoms.cell.array  # 3x3
        inv_cell = np.linalg.inv(cell.T)  # for fractional mapping (transpose convention)

        # Compute all pair deltas in a vectorized way
        # deltas[i,j,:] = pos[j]-pos[i]
        deltas = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (N,N,3)
        # Convert to fractional
        frac = deltas @ inv_cell  # (N,N,3)
        # Wrap to [-0.5,0.5)
        frac -= np.round(frac)
        # Back to Cartesian
        deltas_mic = frac @ cell.T  # (N,N,3)
        d2 = np.sum(deltas_mic**2, axis=-1)
    else:
        deltas = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        d2 = np.sum(deltas**2, axis=-1)

    # Ignore diagonal
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(np.min(d2)))


def get_forces_if_present(atoms) -> Optional[np.ndarray]:
    """
    extxyz often stores forces in atoms.arrays['forces'] or ['force'].
    We'll check several common keys.
    """
    for key in ("forces", "force", "F", "for"):
        if key in atoms.arrays:
            arr = np.asarray(atoms.arrays[key], dtype=float)
            if arr.shape == (len(atoms), 3):
                return arr
    # Sometimes stored in calculator results; but in pure xyz without calc it won't exist.
    return None


def get_stress_if_present(atoms) -> Optional[np.ndarray]:
    """
    extxyz often stores stress in atoms.info (e.g., 'stress', 'virial', 'stresses').
    We support common encodings:
      - stress as 6-vector: [xx, yy, zz, yz, xz, xy] in eV/A^3 or GPa (unknown)
      - stress as 3x3
      - virial as 6-vector in eV (cell*stress*volume) -- ambiguous
    For a simple QC, we look for a 'stress' 3x3 or 6-vector in GPa/eV/A^3.
    If units are unknown, we still compute a magnitude in given units and report.
    """
    for key in ("stress", "stresses"):
        if key in atoms.info:
            s = np.asarray(atoms.info[key], dtype=float)
            return s
    # Some workflows store 'virial' (often in eV). Hard to auto-convert reliably.
    # We'll still report it if present.
    if "virial" in atoms.info:
        return np.asarray(atoms.info["virial"], dtype=float)
    return None


def stress_to_GPa(stress: np.ndarray, volume_A3: float) -> Tuple[Optional[np.ndarray], str]:
    """
    Attempt to convert stress/virial to GPa if possible.
    Heuristics:
      - If shape is (3,3) or (6,), assume it's stress in eV/A^3 unless magnitude looks like GPa.
      - If key 'virial' used: often eV (not eV/A^3). If so, divide by volume to get eV/A^3.
    Because units are ambiguous in xyz, we return (converted_or_none, note).
    """
    # eV/A^3 -> GPa conversion:
    EV_A3_TO_GPA = 160.21766208

    s = np.array(stress, dtype=float)

    # Detect virial-like huge values: if abs values >> 1e3, likely eV or eV*something
    # But this is very heuristic. We'll keep conservative.
    note = "stress_units=unknown"

    # Convert 6-vector to 3x3 if needed (Voigt: xx, yy, zz, yz, xz, xy)
    if s.shape == (6,):
        xx, yy, zz, yz, xz, xy = s.tolist()
        s33 = np.array([[xx, xy, xz],
                        [xy, yy, yz],
                        [xz, yz, zz]], dtype=float)
        s = s33
    elif s.shape == (3, 3):
        pass
    else:
        return None, f"stress_shape={tuple(s.shape)}_unsupported"

    # Heuristic: if typical magnitude is between 0.1 and 1000, could be eV/A^3 or GPa
    mag = float(np.max(np.abs(s)))

    # If it looks like already GPa (e.g., 10..500), converting would explode.
    # If it looks like eV/A^3 (usually <~1), converting makes sense.
    if mag < 10.0:
        # assume eV/A^3
        return s * EV_A3_TO_GPA, "stress_units=assumed_eV/A3"
    else:
        # assume already GPa
        return s, "stress_units=assumed_GPa"


def qc_one(atoms, idx: int, min_dist_thresh: float, fmax_thresh: float, smax_thresh_GPa: float) -> QCResult:
    issues: List[str] = []

    natoms = len(atoms)

    # Check NaN/Inf in positions
    pos = atoms.get_positions()
    if not is_finite_array(pos):
        issues.append("nan_in_positions")

    # Cell / volume checks
    cell = atoms.cell.array if atoms.cell is not None else np.zeros((3, 3))
    if not is_finite_array(cell):
        issues.append("nan_in_cell")

    vol = cell_volume_A3(cell) if atoms.cell is not None and atoms.cell.rank == 3 else 0.0
    if not math.isfinite(vol):
        issues.append("nan_in_volume")
    if vol <= 0.0 and np.any(atoms.get_pbc()):
        issues.append("nonpositive_cell_volume")

    # Minimum distance
    try:
        dmin = min_distance_mic(atoms)
        if not math.isfinite(dmin):
            issues.append("nan_in_min_dist")
        elif dmin < min_dist_thresh:
            issues.append(f"min_dist<{min_dist_thresh:.3f}A")
    except Exception as e:
        dmin = float("nan")
        issues.append(f"min_dist_failed:{type(e).__name__}")

    # Forces (if present)
    forces = get_forces_if_present(atoms)
    has_forces = forces is not None
    fmax = None
    if has_forces:
        if not is_finite_array(forces):
            issues.append("nan_in_forces")
        else:
            fmax = float(np.max(np.linalg.norm(forces, axis=1)))
            if fmax > fmax_thresh:
                issues.append(f"fmax>{fmax_thresh:.1f}eV/A")

    # Stress (if present)
    stress_raw = get_stress_if_present(atoms)
    has_stress = stress_raw is not None
    smax = None
    if has_stress:
        if not is_finite_array(np.asarray(stress_raw, dtype=float)):
            issues.append("nan_in_stress")
        else:
            sGPa, note = stress_to_GPa(np.asarray(stress_raw, dtype=float), vol if vol > 0 else 1.0)
            if sGPa is None:
                issues.append(f"stress_unusable:{note}")
            else:
                smax = float(np.max(np.abs(sGPa)))
                if smax > smax_thresh_GPa:
                    issues.append(f"smax>{smax_thresh_GPa:.1f}GPa({note})")

    ok = len(issues) == 0
    return QCResult(
        idx=idx,
        natoms=natoms,
        ok=ok,
        min_dist_A=float(dmin),
        volume_A3=float(vol),
        has_forces=has_forces,
        fmax_eVA=fmax,
        has_stress=has_stress,
        smax_GPa=smax,
        issues=";".join(issues) if issues else "",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", help="Input XYZ/extXYZ file (ASE-readable), e.g. dataset.xyz")
    ap.add_argument("--min-dist", type=float, default=1.8, help="Reject if min pair distance < this (A)")
    ap.add_argument("--fmax", type=float, default=50.0, help="Reject if max |F| > this (eV/A), if forces exist")
    ap.add_argument("--smax", type=float, default=100.0, help="Reject if max |stress| > this (GPa), if stress exists")
    ap.add_argument("--out", type=str, default="qc_report.csv", help="Output CSV report")
    ap.add_argument("--fail-out", type=str, default="qc_failed.xyz",
                    help="Write failed configurations to this XYZ (ASE extxyz)")
    ap.add_argument("--pass-out", type=str, default="qc_passed.xyz",
                    help="Write passed configurations to this XYZ (ASE extxyz)")
    ap.add_argument("--max", type=int, default=0, help="Only process first N configs (0=all)")
    args = ap.parse_args()

    results: List[QCResult] = []
    failed_atoms = []
    passed_atoms = []

    n_total = 0
    for i, atoms in enumerate(iread(args.xyz, index=":")):
        if args.max and i >= args.max:
            break
        n_total += 1
        res = qc_one(atoms, i, args.min_dist, args.fmax, args.smax)
        results.append(res)
        if res.ok:
            passed_atoms.append(atoms)
        else:
            failed_atoms.append(atoms)

    # Write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "idx", "natoms", "ok",
            "min_dist_A", "volume_A3",
            "has_forces", "fmax_eV_A",
            "has_stress", "smax_GPa",
            "issues"
        ])
        for r in results:
            w.writerow([
                r.idx, r.natoms, int(r.ok),
                f"{r.min_dist_A:.6f}" if math.isfinite(r.min_dist_A) else "nan",
                f"{r.volume_A3:.6f}" if math.isfinite(r.volume_A3) else "nan",
                int(r.has_forces),
                f"{r.fmax_eVA:.6f}" if r.fmax_eVA is not None else "",
                int(r.has_stress),
                f"{r.smax_GPa:.6f}" if r.smax_GPa is not None else "",
                r.issues
            ])

    # Optional: write passed/failed xyz (requires ASE write; keep simple)
    try:
        from ase.io import write
        if passed_atoms:
            write(args.pass_out, passed_atoms, format="extxyz")
        if failed_atoms:
            write(args.fail_out, failed_atoms, format="extxyz")
    except Exception as e:
        print(f"[WARN] Could not write extxyz outputs: {e}", file=sys.stderr)

    n_ok = sum(1 for r in results if r.ok)
    n_fail = len(results) - n_ok
    print(f"Processed: {len(results)} configs (from {args.xyz})")
    print(f"Passed:    {n_ok}")
    print(f"Failed:    {n_fail}")
    print(f"CSV report: {args.out}")
    if n_ok:
        print(f"Passed xyz: {args.pass_out}")
    if n_fail:
        print(f"Failed xyz: {args.fail_out}")


if __name__ == "__main__":
    main()