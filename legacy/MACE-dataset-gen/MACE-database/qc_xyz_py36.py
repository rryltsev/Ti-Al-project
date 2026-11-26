#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC script for (ext)XYZ datasets (ASE-readable), compatible with Python 3.6.

Checks (per configuration):
  1) min interatomic distance (MIC-aware if PBC)
  2) cell volume > 0 (if periodic)
  3) no NaN/Inf in positions/cell/energy/forces/stress (where present)
  4) max |F_i| below threshold (if forces exist)
  5) stress sanity (if stress exists; attempts eV/Å^3 -> GPa conversion)

Usage:
  python qc_xyz_py36.py dataset.xyz
  python qc_xyz_py36.py dataset.xyz --min-dist 1.8 --fmax 50 --smax 100 --out report.csv
"""

from __future__ import print_function

import argparse
import csv
import math
import sys

import numpy as np

try:
    from ase.io import iread, write
except Exception as e:
    raise SystemExit("This script requires ASE (pip install ase). Import error: %s" % e)


EV_A3_TO_GPA = 160.21766208


def is_finite_array(arr):
    """Return True if all values are finite (no NaN/Inf)."""
    return np.isfinite(arr).all()


def cell_volume_A3(cell):
    """cell: 3x3 (Å). Volume = det(cell)."""
    return float(np.linalg.det(cell))


def min_distance_mic(atoms):
    """
    Compute minimum pair distance with Minimum Image Convention if PBC.
    O(N^2) but OK for quick QC.
    """
    pos = atoms.get_positions()
    n = len(pos)
    if n < 2:
        return float("inf")

    pbc = atoms.get_pbc()
    use_mic = bool(np.any(pbc)) and (atoms.cell is not None) and (atoms.cell.rank == 3)

    if use_mic:
        cell = atoms.cell.array
        inv_cell_T = np.linalg.inv(cell.T)

        deltas = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (N,N,3)
        frac = np.dot(deltas, inv_cell_T)                       # (N,N,3)
        frac -= np.round(frac)                                  # wrap to [-0.5,0.5)
        deltas_mic = np.dot(frac, cell.T)
        d2 = np.sum(deltas_mic ** 2, axis=-1)
    else:
        deltas = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        d2 = np.sum(deltas ** 2, axis=-1)

    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(np.min(d2)))


def get_forces_if_present(atoms):
    """Try common keys for forces in extxyz."""
    for key in ("forces", "force", "F", "for"):
        if key in atoms.arrays:
            arr = np.asarray(atoms.arrays[key], dtype=float)
            if arr.shape == (len(atoms), 3):
                return arr
    return None


def get_stress_if_present(atoms):
    """
    Try common keys for stress in extxyz.
    stress may be in atoms.info['stress'] (6-vector or 3x3).
    """
    for key in ("stress", "stresses"):
        if key in atoms.info:
            return np.asarray(atoms.info[key], dtype=float)
    if "virial" in atoms.info:
        return np.asarray(atoms.info["virial"], dtype=float)
    return None


def voigt6_to_3x3(v6):
    """Voigt: [xx, yy, zz, yz, xz, xy] -> 3x3."""
    xx, yy, zz, yz, xz, xy = v6.tolist()
    return np.array([[xx, xy, xz],
                     [xy, yy, yz],
                     [xz, yz, zz]], dtype=float)


def stress_to_GPa(stress):
    """
    Heuristic conversion:
      - if max|s| < 10 -> assume eV/Å^3 and convert to GPa
      - else assume already GPa
    Returns (sGPa, note) or (None, note) if unsupported.
    """
    s = np.array(stress, dtype=float)

    if s.shape == (6,):
        s = voigt6_to_3x3(s)
    elif s.shape == (3, 3):
        pass
    else:
        return None, "stress_shape=%s_unsupported" % (str(tuple(s.shape)))

    mag = float(np.max(np.abs(s)))
    if mag < 10.0:
        return s * EV_A3_TO_GPA, "stress_units=assumed_eV/A3"
    else:
        return s, "stress_units=assumed_GPa"


def qc_one(atoms, idx, min_dist_thresh, fmax_thresh, smax_thresh_GPa):
    """Return a dict with QC metrics and issues."""
    issues = []

    natoms = len(atoms)
    pos = atoms.get_positions()
    if not is_finite_array(pos):
        issues.append("nan_in_positions")

    cell = atoms.cell.array if atoms.cell is not None else np.zeros((3, 3))
    if not is_finite_array(cell):
        issues.append("nan_in_cell")

    vol = 0.0
    if atoms.cell is not None and atoms.cell.rank == 3:
        try:
            vol = cell_volume_A3(cell)
        except Exception:
            vol = float("nan")
            issues.append("volume_failed")

    if not math.isfinite(vol):
        issues.append("nan_in_volume")

    # Only enforce volume>0 if periodic; for non-periodic clusters volume can be 0
    if np.any(atoms.get_pbc()) and vol <= 0.0:
        issues.append("nonpositive_cell_volume")

    # Min distance
    try:
        dmin = min_distance_mic(atoms)
        if not math.isfinite(dmin):
            issues.append("nan_in_min_dist")
        elif dmin < min_dist_thresh:
            issues.append("min_dist<%.3fA" % min_dist_thresh)
    except Exception as e:
        dmin = float("nan")
        issues.append("min_dist_failed:%s" % type(e).__name__)

    # Forces
    forces = get_forces_if_present(atoms)
    has_forces = forces is not None
    fmax = None
    if has_forces:
        if not is_finite_array(forces):
            issues.append("nan_in_forces")
        else:
            fmax = float(np.max(np.linalg.norm(forces, axis=1)))
            if fmax > fmax_thresh:
                issues.append("fmax>%.1feV/A" % fmax_thresh)

    # Stress
    stress_raw = get_stress_if_present(atoms)
    has_stress = stress_raw is not None
    smax = None
    stress_note = ""
    if has_stress:
        if not is_finite_array(np.asarray(stress_raw, dtype=float)):
            issues.append("nan_in_stress")
        else:
            sGPa, note = stress_to_GPa(stress_raw)
            stress_note = note
            if sGPa is None:
                issues.append("stress_unusable:%s" % note)
            else:
                smax = float(np.max(np.abs(sGPa)))
                if smax > smax_thresh_GPa:
                    issues.append("smax>%.1fGPa(%s)" % (smax_thresh_GPa, note))

    ok = (len(issues) == 0)

    return {
        "idx": idx,
        "natoms": natoms,
        "ok": ok,
        "min_dist_A": dmin,
        "volume_A3": vol,
        "has_forces": has_forces,
        "fmax_eV_A": fmax,
        "has_stress": has_stress,
        "smax_GPa": smax,
        "stress_note": stress_note,
        "issues": ";".join(issues) if issues else ""
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", help="Input XYZ/extXYZ file (ASE-readable), e.g. dataset.xyz")
    ap.add_argument("--min-dist", type=float, default=1.8, help="Reject if min pair distance < this (Å)")
    ap.add_argument("--fmax", type=float, default=50.0, help="Reject if max |F| > this (eV/Å), if forces exist")
    ap.add_argument("--smax", type=float, default=100.0, help="Reject if max |stress| > this (GPa), if stress exists")
    ap.add_argument("--out", type=str, default="qc_report.csv", help="Output CSV report")
    ap.add_argument("--fail-out", type=str, default="qc_failed.xyz", help="Write failed configurations to this extxyz")
    ap.add_argument("--pass-out", type=str, default="qc_passed.xyz", help="Write passed configurations to this extxyz")
    ap.add_argument("--max", type=int, default=0, help="Only process first N configs (0=all)")
    args = ap.parse_args()

    results = []
    failed_atoms = []
    passed_atoms = []

    for i, atoms in enumerate(iread(args.xyz, index=":")):
        if args.max and i >= args.max:
            break
        res = qc_one(atoms, i, args.min_dist, args.fmax, args.smax)
        results.append(res)
        if res["ok"]:
            passed_atoms.append(atoms)
        else:
            failed_atoms.append(atoms)

    # Write CSV
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx", "natoms", "ok",
            "min_dist_A", "volume_A3",
            "has_forces", "fmax_eV_A",
            "has_stress", "smax_GPa", "stress_note",
            "issues"
        ])
        for r in results:
            w.writerow([
                r["idx"], r["natoms"], int(r["ok"]),
                ("%.6f" % r["min_dist_A"]) if math.isfinite(r["min_dist_A"]) else "nan",
                ("%.6f" % r["volume_A3"]) if math.isfinite(r["volume_A3"]) else "nan",
                int(r["has_forces"]),
                ("%.6f" % r["fmax_eV_A"]) if r["fmax_eV_A"] is not None else "",
                int(r["has_stress"]),
                ("%.6f" % r["smax_GPa"]) if r["smax_GPa"] is not None else "",
                r["stress_note"],
                r["issues"]
            ])

    # Write extxyz outputs (best-effort)
    try:
        if passed_atoms:
            write(args.pass_out, passed_atoms, format="extxyz")
        if failed_atoms:
            write(args.fail_out, failed_atoms, format="extxyz")
    except Exception as e:
        print("[WARN] Could not write extxyz outputs: %s" % e, file=sys.stderr)

    n_ok = sum(1 for r in results if r["ok"])
    n_fail = len(results) - n_ok
    print("Processed: %d configs (from %s)" % (len(results), args.xyz))
    print("Passed:    %d" % n_ok)
    print("Failed:    %d" % n_fail)
    print("CSV report: %s" % args.out)
    if n_ok:
        print("Passed xyz: %s" % args.pass_out)
    if n_fail:
        print("Failed xyz: %s" % args.fail_out)


if __name__ == "__main__":
    main()
