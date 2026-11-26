"""Matched 13- and 25-window C46 isotherms with a shared, window-wise block bootstrap."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from interpolation_check import tangent
from paired_mc import HERE, KT, contact, estimate, read_run, spline_for


def contacts(means, centers, mu0, kappa):
    fields = mu0-2*KT*kappa*(means-centers)
    pair = [contact(x, y) for x,y in zip(means,fields)]
    return fields, np.stack(pair) if all(r is not None for r in pair) else None


def quantiles(x):
    return np.quantile(x,[.025,.975],axis=0).T.tolist()


def estimates(z):
    return dict(quantum=z[0].tolist(), classical=z[1].tolist(), classical_minus_quantum=(z[1]-z[0]).tolist())


def intervals(z):
    return dict(quantum=quantiles(z[:,0]),classical=quantiles(z[:,1]),
                classical_minus_quantum=quantiles(z[:,1]-z[:,0]))


def main(args):
    q,names = spline_for("quantum")
    c,cnames = spline_for("classical")
    if names != cnames:
        raise ValueError("Different quantum and classical basis")
    slots = defaultdict(list)
    hashes, source_diagnostics = {}, {}
    conditions = None
    for folder in args.runs:
        cfg, chains = read_run(folder,[q,c],names,args.block_multiplier)
        state = tuple(cfg[k] for k in ("T_K","size","mu","kappa","coeff_interpolation"))
        if conditions is None:
            conditions = state
        if state != conditions or state[:2]!=(1000,5120):
            raise ValueError("Inconsistent mechanical or thermodynamic conditions")
        hashes[folder.name] = {name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                               for name in ["manifest.json","classical.txt","quantum.txt"]}
        diag = []
        for idx,(center,chain) in enumerate(zip(cfg["windows"],chains,strict=True),1):
            key = round(1000*center)
            if abs(key/1000-center)>1e-10:
                raise ValueError(f"Unmatched center {center}")
            slots[key].append(chain)
            diag.append(dict(window_center=center,**chain["diag"]))
            archive = folder/"samples"/f"{idx}.csv.gz"
            hashes[folder.name][f"samples/{idx}.csv.gz"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        source_diagnostics[folder.name] = diag
    keys = sorted(slots)
    if keys!=list(range(130,251,5)):
        raise ValueError(f"Incomplete regularly spaced 25-window grid, got {keys}")
    windows=[slots[k] for k in keys]
    old=np.arange(0,25,2)
    for i,win in enumerate(windows):
        origins=[chain["source"] for chain in win]
        if not (sorted(origins)==[0,1] if i%2==0 else origins==[1]):
            raise ValueError(f"Unexpected source replica(s) at center {keys[i]/1000}: {origins}")
    centers=np.array(keys)/1000
    mu0,kappa=conditions[2:4]
    def solve(curves):
        fields,z25=contacts(curves,centers,mu0,kappa)
        _,z13=contacts(curves[:,old],centers[old],mu0,kappa)
        return fields,z13,z25

    means=np.array([estimate(win,[slice(None)]*len(win)) for win in windows]).T
    fields,z13,z25=solve(means)
    if z13 is None or z25 is None:
        raise ValueError("Unresolved central 13- or 25-window estimate")
    classic_only=[list(filter(lambda r:r["source"]==1,w)) for w in windows]
    classic_means=np.array([estimate(w,[slice(None)]*len(w)) for w in classic_only]).T
    _,zs13,zs25=solve(classic_means)
    if zs13 is None or zs25 is None:
        raise ValueError("Unresolved classical-source control")
    independent={}
    sets={"full_25":np.arange(25),"old_13":old,"new_12_plus_edges":np.r_[0,np.arange(1,24,2),24]}
    for name,keep in sets.items():
        independent[name]={}
        for method in ("PCHIP","linear"):
            result = [tangent(np.column_stack([x[keep],y[keep]]), method) for x,y in zip(means,fields)]
            pair = np.array([[r["alpha_at_pct"],r["alpha2_at_pct"],r["mu_eV"]] for r in result])
            independent[name][method]=estimates(pair)
            if name!="new_12_plus_edges" and method=="PCHIP":
                np.testing.assert_allclose(pair, z13 if name=="old_13" else z25,atol=1e-7,rtol=0)
    rng=np.random.default_rng(args.seed)
    boots13,boots25=[],[]
    trials=dict(resolved_13=0,resolved_25=0,simultaneously_resolved=0)
    for _ in range(args.bootstrap):
        matrix=[]
        for win in windows:
            draws=[]
            for run in win:
                blocks=run["groups"]
                draws.append(np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))]))
            matrix.append(estimate(win,draws))
        _,b13,b25=solve(np.array(matrix).T)
        if b13 is not None:
            trials["resolved_13"]+=1
        if b25 is not None:
            trials["resolved_25"]+=1
        if b13 is not None and b25 is not None:
            boots13.append(b13)
            boots25.append(b25)
            trials["simultaneously_resolved"]+=1
    if len(boots25)<2:
        raise ValueError("Not enough simultaneously resolved bootstrap replicates")
    boots13,boots25=np.array(boots13),np.array(boots25)
    report=dict(N=5120,T_K=1000,p_GPa=0,columns=["alpha_at_pct","alpha2_at_pct","Delta_mu_eV_per_substitution"],
                row_order=["quantum","classical"],window_centers=centers.tolist(),window_means=means.tolist(),
                window_mu_eV=fields.tolist(),primary=dict(old_13=estimates(z13),full_25=estimates(z25),
                                                          full_25_minus_old_13=estimates(z25-z13)),
                intervals_95=dict(old_13=intervals(boots13),full_25=intervals(boots25),
                                  full_25_minus_old_13=intervals(boots25-boots13)),
                interpolation_controls=independent,
                classical_source_only=dict(old_13=estimates(zs13),full_25=estimates(zs25)),
                method="Every old center (0.13:0.01:0.25) uses two-state MBAR with direct quantum and classical "
                "frames. Every intervening center uses direct classical frames reweighted to quantum. Each "
                "13-grid reconstruction uses the identical original windows in the matching 25-grid reconstruction. "
                "Additional one-source controls use only classical trajectories on both grids.",
                bootstrap=dict(attempts=args.bootstrap, seed=args.seed,
                               block_multiplier_integrated_tau=args.block_multiplier,**trials,
                               method="Draw temporal blocks with replacement separately per trajectory and center. "
                               "Recompute the two ensemble means and both grid contacts for each draw. "
                               "All reported intervals use jointly resolved pairs; the grid-change interval "
                               "preserves shared blocks across the two grids."),
                diagnostics=source_diagnostics,source_sha256=hashes)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".npz"),old_13=boots13,full_25=boots25,
                        full_25_minus_old_13=boots25-boots13,columns=report["columns"],row_order=report["row_order"])
    print(json.dumps(dict(contacts=report["primary"],intervals=report["intervals_95"],bootstrap=report["bootstrap"])))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs",nargs="*",type=Path,default=[HERE/x for x in ("mc5120c_a","mc5120q_b","mc5120c_mid")])
    parser.add_argument("--bootstrap",type=int,default=512)
    parser.add_argument("--block-multiplier",type=float,default=6)
    parser.add_argument("--seed",type=int,default=9345215)
    parser.add_argument("--output",type=Path,default=HERE/"dense_grid.json")
    main(parser.parse_args())
