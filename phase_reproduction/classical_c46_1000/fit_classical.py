"""Recover the classical harmonic limit of each author's C46 thermal label.

At reference temperature T0, Fq(T)/(T/T0) is linear in log(T/T0)
plus an analytic series in (T0/T)^2. The intercept gives Fcl(T0).
The log coefficient also measures the mode weight actually used in each
author curve, including the convention used at Gamma and for reused cells.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

T0 = 1000.0
KB = 8.617333262145e-5
HERE = Path(__file__).resolve().parent
REPLAY = HERE.parent / "author_comparison/eci/author_refit_grid.npz"


def classical_labels(temperatures, quantum, low, order):
    use = temperatures >= low
    scale = temperatures[use] / T0
    matrix = np.column_stack([np.ones_like(scale), np.log(scale)]
                             + [scale ** (-2*k) for k in range(1, order+1)])
    rhs = quantum[:, use].T / scale[:, None]
    coefficients = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    residual = (matrix @ coefficients-rhs)*scale[:, None]
    return (coefficients[0], -coefficients[1]/(KB*T0),
            np.max(np.abs(residual), axis=0))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(src, out):
    static_path = src / "models/static/volume_temperature_coefficients.npz"
    quantum_path = src / "models/HA_QHA/volume_temperature_coefficients.npz"
    with np.load(static_path) as s, np.load(quantum_path) as q, np.load(REPLAY) as replay:
        baseline = {k: replay[k] for k in replay}
        temperatures, volumes = q["temperatures_K"], q["volumes_A3_atom"]
        stable, train, features = q["stable"], q["training"], q["X"]
        quantum = q["Y_eV_atom"] - s["E_static_eV_atom"][:, :, None]
        j_quantum = q["eci"]
        for key in ("temperatures_K", "volumes_A3_atom", "stable", "training", "X"):
            np.testing.assert_array_equal(q[key], s[key])
    with (src / "structures/index.csv").open(encoding="utf-8-sig") as f:
        structures = list(csv.DictReader(f))
    k0, = np.flatnonzero(temperatures == T0)
    np.testing.assert_array_equal(volumes, baseline["volume_A3_atom"])
    np.testing.assert_allclose(baseline["J_static"] + baseline["J_vib"][:, k0],
                               j_quantum[:, k0], atol=1e-12, rtol=0)
    labels, checks, classical_eci, sampled_records = [], [], [], []
    for v, keep in enumerate(stable):
        fit = train[keep]
        x = features[keep]
        al_fraction = np.array([int(r["Al_atoms"])/int(r["atoms"]) for r in structures])[keep]
        y = quantum[v, keep]
        f_cl, modes, resid = classical_labels(temperatures, y, 1000, 4)
        alternatives = [classical_labels(temperatures, y, t, m)[0]
                        for t, m in [(800, 5), (1200, 4), (1400, 3)]]
        spread = np.max(np.abs(np.array(alternatives)-f_cl), axis=0)
        q0 = y[:, k0]
        jc = np.linalg.lstsq(x[fit], f_cl[fit], rcond=None)[0]
        jq = np.linalg.lstsq(x[fit], q0[fit], rcond=None)[0]
        np.testing.assert_allclose(jq, baseline["J_vib"][v, k0], atol=1e-12, rtol=0)
        predicted = x @ jc - f_cl
        affine_design = np.column_stack([np.ones_like(al_fraction), al_fraction])
        shape_change = {}
        for source, changes in [("label", f_cl-q0), ("CE", x @ (jc-jq))]:
            affine = np.linalg.lstsq(affine_design, changes, rcond=None)[0]
            residual = 1000*(changes-affine_design @ affine)
            shape_change[source] = dict(
                intercept_meV_atom=float(1000*affine[0]),
                slope_meV_per_unit_x=float(1000*affine[1]),
                rms_resid_meV_atom=float(np.sqrt(np.mean(residual**2))),
                max_abs_resid_meV_atom=float(np.max(abs(residual))))
        classical_eci.append(jc[None, :])
        if np.min(q0-f_cl) <= 0 or np.max(spread) > 2e-7 or np.max(resid) > 2e-8:
            raise ValueError(f"Classical-limit diagnostics failed at v={v}: {np.max(spread)}, {np.max(resid)}")
        for i, j in enumerate(np.flatnonzero(keep)):
            labels.append(dict(volume_A3_atom=volumes[v], configuration=structures[j]["configuration"],
                               N=int(structures[j]["atoms"]), training=bool(fit[i]),
                               Fq_eV_atom=q0[i], Fcl_eV_atom=f_cl[i],
                               classical_minus_quantum_meV_atom=(f_cl[i]-q0[i])*1000,
                               effective_mode_weight=modes[i], max_range_change_eV_atom=spread[i],
                               max_thermal_residual_eV_atom=resid[i]))
        checks.append(dict(volume_A3_atom=volumes[v], n_fit=int(fit.sum()),
                           rank=int(np.linalg.matrix_rank(x[fit])),
                           class_train_rmse_meV_atom=float(1000*np.sqrt(np.mean(predicted[fit]**2))),
                           class_control_rmse_meV_atom=float(1000*np.sqrt(np.mean(predicted[~fit]**2))),
                           shape_after_affine_removal_equal_weight_all_stable=shape_change,
                           max_thermal_residual_eV_atom=float(np.max(resid)),
                           max_range_change_eV_atom=float(np.max(spread))))
        for orbit, jclass, jquant in zip(baseline["orbit_names"], jc, jq):
            sampled_records.append(dict(volume_A3_atom=volumes[v], orbit=orbit,
                                        Jv_classical_eV=jclass, Jv_quantum_eV=jquant,
                                        classical_minus_quantum_meV=(jclass-jquant)*1000))
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in [("structure_labels.csv", labels), ("eci_changes.csv", sampled_records)]:
        with (out / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    for label, grid in [("quantum", baseline["J_vib"][:, [k0]]), ("classical", classical_eci)]:
        np.savez(out / f"{label}_grid.npz", temperature_K=[T0], volume_A3_atom=volumes,
                 J_static=baseline["J_static"], J_vib=np.asarray(grid), orbit_names=baseline["orbit_names"])
    summary = dict(T_K=T0, stable_rows=len(labels), comparison_fit_windows_K_and_orders=[[800,5],[1200,4],[1400,3]],
                   primary_fit_window_K=[1000,2200], primary_order=4, volumes=checks,
                   sha256={str(p.relative_to(src)): digest(p) for p in [static_path, quantum_path]},
                   refit_grid_input_sha256=digest(REPLAY))
    (out / "fit_diagnostics.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf8")
    print(f"{len(labels)} stable rows, max label spread {max(r['max_range_change_eV_atom'] for r in labels):.3g} eV/atom")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("author_package", type=Path)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    run(args.author_package, args.output)
