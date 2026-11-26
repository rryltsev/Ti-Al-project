"""Maxwell construction from discrete CE window means; sensitivity to the composition grid."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator, PPoly
from scipy.optimize import brentq


HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "solvus_1000/c46_reference.json"


def tangent(points, method):
    x, y = np.asarray(points, dtype=float).T
    if np.any(np.diff(x) <= 0) or not np.isfinite([x,y]).all():
        raise ValueError("Input must be finite and ordered by mean Al fraction")
    if method == "PCHIP":
        curve = PchipInterpolator(x, y)
    elif method == "linear":
        curve = PPoly(np.stack([np.diff(y)/np.diff(x), y[:-1]]), x)
    else:
        raise ValueError(method)
    primitive = curve.antiderivative()
    rising, = np.where(np.diff(y) > 0)

    def crossing(i, field):
        if y[i] == field:
            return x[i]
        if y[i+1] == field:
            return x[i+1]
        return brentq(lambda z: curve(z)-field, x[i], x[i+1], xtol=2e-15)

    solutions = []
    for l in rising:
        for r in rising[rising > l+1]:
            lower = max(y[l],y[r])
            upper = min(y[l+1],y[r+1])
            if lower >= upper:
                continue
            def residual(field):
                a,b = crossing(l,field), crossing(r,field)
                return primitive(b)-primitive(a)-field*(b-a)
            if residual(lower)*residual(upper) > 0:
                continue
            field = brentq(residual,lower,upper,xtol=2e-15)
            a,b = crossing(l,field),crossing(r,field)
            stationary = curve.solve(field, extrapolate=False)
            sites = np.r_[x[0],stationary[np.isfinite(stationary)],x[-1]]
            gap = primitive(sites)-primitive(a)-field*(sites-a)
            if min(gap) < -1e-9:
                continue
            if (not x[0] < a < b < x[-1]) or b-a<.005:
                continue
            solutions.append((b-a, a,b,field,residual(field),min(gap)))
    if not solutions:
        raise ValueError("No resolved interior supporting common tangent")
    _, a,b,mu,res,support = max(solutions)
    return dict(alpha_at_pct=100*a,alpha2_at_pct=100*b,mu_eV=mu,
                maxwell_residual_eV_atom=res,min_support_gap_eV_atom=support)


def compare(q, c=None):
    if len(q)!=13 or (c is not None and len(c)!=13):
        raise ValueError("Expected the 13 matched windows centred at x = 0.13 ... 0.25")
    masks = {"all_13": np.arange(13), "even_7": np.arange(0,13,2),
             "odd_plus_ends_8": np.r_[0,np.arange(1,12,2),12]}
    result = {}
    for method in ["PCHIP", "linear"]:
        for name, indices in masks.items():
            qr = tangent(np.asarray(q)[indices],method)
            record = dict(window_centers_at_pct=(indices+13).tolist(),quantum=qr)
            if c is not None:
                cr = tangent(np.asarray(c)[indices],method)
                record.update(classical=cr,classical_minus_quantum_pp={
                    phase: cr[phase+"_at_pct"]-qr[phase+"_at_pct"] for phase in ("alpha","alpha2")})
            result[f"{method}_{name}"] = record
    return result


def author_check():
    archive = json.loads(REFERENCE.read_text())
    result = compare(archive["windows_c_Al_and_mu_occ_eV"])
    full = result["PCHIP_all_13"]["quantum"]
    for phase in ("alpha","alpha2"):
        np.testing.assert_allclose(full[phase+"_at_pct"],archive["phase"][phase]["Al_at_percent"],
                                   atol=0.002,rtol=0)
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_check(path):
    data = json.loads(path.read_text())
    means = np.asarray(data["window_means"])
    mus = np.asarray(data["window_mu_eV"])
    if data["row_order"] != ["quantum","classical"] or means.shape != (2,13) or mus.shape != (2,13):
        raise ValueError(f"Unexpected quantum/classical window arrays in {path}")
    np.testing.assert_allclose(data["window_centers"],np.arange(13,26)/100,atol=2e-15,rtol=0)
    result = compare(np.column_stack([means[0], mus[0]]),
                     np.column_stack([means[1], mus[1]]))
    for method, expected in zip(["quantum","classical"], data["estimates"],strict=True):
        full=result["PCHIP_all_13"][method]
        np.testing.assert_allclose([full["alpha_at_pct"],full["alpha2_at_pct"],full["mu_eV"]],
                                   expected,atol=2e-8,rtol=0)
    summary = {}
    for scheme, included in [("PCHIP_and_thinning", list(result)[:3]),
                             ("all_interpolants_and_thinning", list(result))]:
        summary[scheme] = {}
        for phase in ("alpha","alpha2"):
            shifts = np.array([result[k]["classical_minus_quantum_pp"][phase] for k in included])
            summary[scheme][phase] = dict(min_shift_pp=float(shifts.min()), max_shift_pp=float(shifts.max()),
                                          max_absolute_difference_from_primary_pp=float(max(abs(shifts-shifts[0]))))
    one_removed = []
    for drop in range(1,12):
        keep = np.arange(13)!=drop
        qr,cr = [tangent(np.column_stack([x[keep], y[keep]]),"PCHIP") for x,y in zip(means,mus)]
        one_removed.append(dict(omitted_window_center_at_pct=drop+13,
                                **{phase+"_shift_pp":cr[phase+"_at_pct"]-qr[phase+"_at_pct"]
                                   for phase in ("alpha","alpha2")}))
    return dict(N=data["N"], input_sha256=digest(path), tests=result, sensitivity=summary,
                pchip_one_interior_window_omitted=one_removed)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-only", action="store_true")
    parser.add_argument("--inputs",nargs="+",type=Path,default=[HERE/"paired768_final.json", HERE/"paired5120.json"])
    parser.add_argument("--dense",type=Path,default=HERE/"dense_grid.json",
                        help="Include the independent 25-center test when the file is present")
    parser.add_argument("--output",type=Path,default=HERE/"interpolation_check.json")
    args=parser.parse_args()
    historical=author_check()
    if args.author_only:
        print(json.dumps(historical,indent=2))
    else:
        paired={p.name:paired_check(p) for p in args.inputs}
        report=dict(method="PCHIP or piecewise-linear interpolation of window chemical potentials vs observed "
                    "mean Al fraction. Solve Maxwell equal-area along all increasing branches, require an "
                    "interior pair supported globally over the measured interval, and retain the widest gap. "
                    "Thinned grids keep both endpoint windows and alternate interior centers; the same subset "
                    "is applied in quantum and classical samples. The differential is computed within each "
                    "paired sample, not against a separately sampled historical bound.",
                    uncertainty="Sensitivity values compare reconstructions on fixed window means. They quantify "
                    "changes when varying the chosen windows and interpolation, not statistical confidence limits.",
                    historical_author_input_sha256=digest(REFERENCE), historical_author=historical,
                    paired_samples=paired)
        if args.dense.is_file():
            dense=json.loads(args.dense.read_text())
            dense_old=dense["interpolation_controls"]["old_13"]["PCHIP"]["classical_minus_quantum"][:2]
            large=[r for r in paired.values() if r["N"]==5120]
            if len(large)!=1:
                raise ValueError("A matching 5120-site paired file is required for the dense comparison")
            q=large[0]["tests"]["PCHIP_all_13"]["classical_minus_quantum_pp"]
            np.testing.assert_allclose([q["alpha"],q["alpha2"]],dense_old,atol=1e-7,rtol=0)
            report["dense_grid"]=dict(input_sha256=digest(args.dense), grid_centers=dense["window_centers"],
                                      tests=dense["interpolation_controls"], paired_change=dense["primary"]["full_25_minus_old_13"],
                                      paired_change_95=dense["intervals_95"]["full_25_minus_old_13"])
        args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        for name,r in paired.items():
            print(name,r["tests"]["PCHIP_all_13"]["classical_minus_quantum_pp"],r["sensitivity"])
