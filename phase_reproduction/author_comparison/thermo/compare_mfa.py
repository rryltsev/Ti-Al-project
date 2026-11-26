"""Compute Ti-Al MFA using exact C46 ECI and the independent frozen fit."""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path[:0] = [str(BASE / "thermo"), str(BASE / "eci_boundary/mfa")]
from attribute import Tangents
from mfa import (Isotherm, VolumeIsotherm, VREF, ORBIT_NAMES, sublattice_weights,
                 lower_hull_edges, mfa_contacts)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf8") as f:
        out = csv.DictWriter(f, fieldnames=rows[0])
        out.writeheader()
        out.writerows(rows)


def grid_data(package):
    author = package / "models"
    names = ["empty", "point", *ORBIT_NAMES[2:15], "t222_capped", "t222_uncapped", *ORBIT_NAMES[17:]]
    with (package / "ECI_Vref_1000K.csv").open(encoding="utf8") as f:
        small = list(csv.DictReader(f))
        if [r["name"] for r in small] != names:
            raise ValueError("Unexpected chemical orbit identities or order")
    with np.load(author / "HA_QHA/volume_temperature_coefficients.npz") as h, np.load(
            author / "static/volume_temperature_coefficients.npz") as s, np.load(
            BASE / "atomistic/atat/qha_fit/grid.npz") as ours:
        ts = np.arange(600, 1801, 25)
        vol = h["volumes_A3_atom"]
        if (not np.allclose(s["volumes_A3_atom"],vol,atol=1e-12,rtol=0) or
            not np.allclose(ours["volume_A3_atom"],vol,atol=1e-12,rtol=0)):
            raise ValueError("Mismatched volume knots")
        ia = np.searchsorted(h["temperatures_K"],ts)
        assert np.array_equal(h["temperatures_K"][ia],ts)
        assert np.array_equal(ours["temperature_K"],ts)
        assert tuple(ours["orbit_names"])==ORBIT_NAMES
        models = {"author":(s["eci"][:,0,:],h["eci"][:,ia,:]),"ours":(
                  ours["J_static"],ours["J_static"][:,None,:]+ours["J_vib"])}
        assert np.max(np.abs(s["eci"]-s["eci"][:,:1,:]))<1e-11
    iv, = np.flatnonzero(abs(vol-VREF)<1e-10)
    ti, = np.flatnonzero(ts==1000)
    a,e = models["author"]
    assert np.allclose(a[iv], [float(r["J_static_eV"]) for r in small],rtol=0,atol=1e-13)
    assert np.allclose(e[iv,ti]-a[iv], [float(r["delta_J_vibrational_1000K_eV"]) for r in small],rtol=0,atol=1e-13)
    return vol, ts, iv, models


def one_isotherm(job):
    source, label, bc, t, vol, j, w, refine = job
    c = np.linspace(0,.35,701)
    if label=="QHA":
        model = VolumeIsotherm(w,vol,j,t,volume_spline=bc)
    else:
        model = Isotherm(w,j,t)
    f,_ = model.equilibrium(c)
    # Retain every appreciable lower-hull facet. At incl. 1750 K author HA
    # contains a grid width of exactly 0.50 at.%.
    edges = [(k,l) for k,l in lower_hull_edges(c,f) if l-k>=10]
    if len(edges)>1:
        raise ValueError(f"Multiple appreciable MFA facets at {source}/{label}/{t}: {edges}")
    row = dict(source=source,model=label,volume_spline=bc,T_K=int(t),
               status="unresolved",grid_alpha_at_pct="",grid_alpha2_at_pct="",alpha_at_pct="",
               alpha2_at_pct="",volume_alpha_A3="",volume_alpha2_A3="")
    if edges:
        k,l = edges[0]
        row.update(grid_alpha_at_pct=100*c[k],grid_alpha2_at_pct=100*c[l],
                   status="window_edge" if k==0 or l==len(c)-1 else "interior_grid")
        if refine and row["status"]=="interior_grid":
            ans, = mfa_contacts(model,c,f,.00495)
            a,b,status,_ = ans
            if status!="interior_tangent":
                raise ValueError(f"Unrefined 1000-K contact {ans}")
            if label=="QHA":
                ab = np.array([a,b])
                _,q = model.equilibrium(ab)
                _,v = model.minimize_volume(ab,q)
            else:
                v = [VREF,VREF]
            row.update(status=status,alpha_at_pct=100*a,alpha2_at_pct=100*b,
                       volume_alpha_A3=v[0],volume_alpha2_A3=v[1])
    return row


def attribution(iv,ts,models,out):
    ti, = np.flatnonzero(ts==1000)
    model = Tangents()
    rows = []
    for name in ("static","HA"):
        def coeff(src):
            s,h = models[src]
            return (s[iv] if name=="static" else h[iv,ti])[2:]
        j0,j1 = coeff("author"),coeff("ours")
        a = model.solve(j0,[.08,.78,.001])
        b = model.solve(j1,a["x"])
        dj = j1-j0
        nodes,ws = np.polynomial.legendre.leggauss(12)
        scale = np.zeros((2,len(dj)))
        for node,weight in zip((nodes+1)/2,ws/2):
            p = model.solve(j0+node*dj,a["x"]*(1-node)+node*b["x"],check=False)
            scale += 100*weight*p["derivatives"]
        parts = scale*dj
        if np.max(np.abs(parts.sum(axis=1)-100*(b["c"]-a["c"])))>1e-6:
            raise ValueError("Path decomposition did not reproduce composition shifts")
        for k,orb in enumerate(ORBIT_NAMES[2:]):
            hybrid = j0.copy()
            hybrid[k] = j1[k]
            left = model.solve(hybrid,a["x"],check=False)
            hybrid = j1.copy()
            hybrid[k] = j0[k]
            right = model.solve(hybrid,b["x"],check=False)
            rows.append(dict(model=name,orbit=orb,J_author_meV=1000*j0[k],J_ours_meV=1000*j1[k],
                             path_alpha_at_pct=parts[0,k],path_alpha2_at_pct=parts[1,k],
                             swap_author_to_ours_alpha2_pp=100*(left["c"]-a["c"])[1],
                             revert_ours_to_author_alpha2_pp=100*(right["c"]-b["c"])[1]))
    write_rows(out/"exact_orbit_attribution.csv",rows)


def run(pkg,out,all_temperatures,workers):
    out.mkdir(parents=True,exist_ok=True)
    vol,ts,iv,models = grid_data(pkg)
    w = sublattice_weights()
    with (pkg/"phase_check/C46_phase_boundary_points.csv").open(encoding="utf8") as f:
        grid = {(r["energetic_model"],int(r["T_K"])):r for r in csv.DictReader(f) if r["statistics"]=="MFA"}
    exact = {r["kind"].replace("HA_C40","HA"):r
             for r in json.loads((pkg/"phase_check/C46_MFA_tangent_refinement_1000K.json").read_text())}
    jobs = []
    for src,(st,eff) in models.items():
        for it,t in enumerate(ts):
            if not all_temperatures and t!=1000:
                continue
            for label,bc,j in [("static","fixed",st[iv]),("HA","fixed",eff[iv,it]),
                               ("QHA","not-a-knot",eff[:,it]),("QHA","natural",eff[:,it])]:
                jobs.append((src,label,bc,t,vol,j,w,t==1000))
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows=list(pool.map(one_isotherm,jobs))
    else:
        rows=list(map(one_isotherm,jobs))
    for r in rows:
        m,t=r["model"],r["T_K"]
        ref=grid.get((m,t))
        r["author_grid_listed"]=ref is not None
        for branch in ("alpha","alpha2"):
            y=float(ref["c_"+branch+"_Al_at_percent"]) if ref else None
            r["author_grid_"+branch+"_at_pct"]=y
            x=r["grid_"+branch+"_at_pct"]
            r["grid_minus_author_"+branch+"_pp"]=x-y if x!="" and y is not None else ""
            y=100*exact[m]["c_"+branch] if t==1000 else None
            z=r[branch+"_at_pct"]
            r["refined_minus_author_"+branch+"_pp"]=z-y if y is not None and z!="" else ""
    write_rows(out/("mfa_comparison_grid.csv" if all_temperatures else "mfa_1000K.csv"),rows)
    if not all_temperatures:
        attribution(iv,ts,models,out)
    for r in rows:
        if r["T_K"]==1000:
            print(r["source"],r["model"],r["volume_spline"],r["alpha_at_pct"],r["alpha2_at_pct"],flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("package",type=Path)
    cli.add_argument("--out",type=Path,default=HERE)
    cli.add_argument("--all-temperatures",action="store_true")
    cli.add_argument("--workers",type=int,default=1)
    args=cli.parse_args()
    run(args.package,args.out,args.all_temperatures,args.workers)
