"""Matched quantum/classical VCSGC isotherms from full-volume, full-orbit trajectory weights."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq, root
from scipy.spatial import ConvexHull
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"thermo"))
from vcsgc import correlation_time

KT = 8.617333262145e-5 * 1000
NAMES = ("quantum","classical")


def contact(x,y):
    order=np.argsort(x)
    x,y=x[order],y[order]
    middle=np.median(y)
    slope=PchipInterpolator(x,y-middle,extrapolate=False)
    free=slope.antiderivative()
    grid=np.linspace(x[0],x[-1],3001)
    vals=free(grid)
    hull=ConvexHull(np.column_stack((grid,vals))).vertices
    end=np.roll(hull,-1)
    gap=np.argmax(end-hull)
    initial=grid[[hull[gap],end[gap]]]
    if initial[1]-initial[0]<.005:
        return None
    def equations(z):
        a,b=z
        if b<=a:
            return [1e2,1e2]
        return [slope(a)-slope(b),(free(b)-free(a))/(b-a)-slope(a)]
    fit=root(equations,initial,tol=1e-10)
    a,b=fit.x
    if (not fit.success or not x[0]<a<b<x[-1] or b-a<.005 or
            np.max(np.abs(equations(fit.x)))>1e-7 or
            (vals-free(a)-slope(a)*(grid-a)).min() < -1e-7):
        return None
    return np.array([100*a,100*b,middle+slope(a)])


def spline_for(model):
    with np.load(HERE/f"{model}_grid.npz") as data:
        assert data["temperature_K"].tolist()==[1000]
        return (CubicSpline(data["volume_A3_atom"],data["J_static"]+data["J_vib"][:,0,:],axis=0,
                            extrapolate=False),data["orbit_names"].astype(str).tolist())


def read_run(run,splines,names,block_multiplier):
    cfg=json.loads((run/"manifest.json").read_text())
    n,origin=cfg["size"],NAMES.index(cfg["source"])
    results=[]
    for i,c0 in enumerate(cfg["windows"],1):
        with gzip.open(run/"samples"/f"{i}.csv.gz","rt",newline="") as fp:
            reader=csv.DictReader(fp)
            if reader.fieldnames[12:] != ["S_"+s for s in names[2:]]:
                raise ValueError("orbit order is inconsistent with fit")
            rows=list(reader)
        if len(rows)!=cfg["prod"]//cfg["stride"]:
            raise ValueError("Unexpected number of MC records")
        if not all(int(r["N"])==n and float(r["c0"])==c0 and r["model"]==NAMES[origin]
                   and int(r["sweep"])==cfg["eq"]+(j+1)*cfg["stride"]
                   and abs(float(r["delta_mu0_eV"])-cfg["mu"])<1e-10 for j,r in enumerate(rows)):
            raise ValueError("Mismatched MC frames")
        arr=np.array([[float(r[s]) for s in ("n_Al","m","volume_A3_atom","Heff_eV_atom",*reader.fieldnames[12:])]
                       for r in rows])
        k,order,v,e=arr[:,:4].T
        S=np.column_stack([np.full(len(v),n/2),2*k-n,arr[:,4:]])
        Jq,Jc=[sp(v) for sp in splines]
        err=float(np.max(abs(np.einsum("ij,ij->i",S,(Jq,Jc)[origin])/n-e)))
        if not np.isfinite(err) or err>2e-9:
            raise ValueError(f"MC and exact spline energies differ by {err} eV/atom")
        delta=np.einsum("ij,ij->i",S,Jc-Jq)/KT
        centered=delta-delta.mean()
        x=k/n
        logw=(-1 if origin==0 else 1)*centered
        w=np.exp(logw-logw.max()); w/=w.mean()
        # This observable directly estimates the differential weighted mean.
        dx_contact=(w*(x-np.average(x,weights=w)) - (x-x.mean()))
        tau={key:float(correlation_time(observable) or .5) for key,observable in
             (("x",x),("m",order),("v",v),("delta_H",delta),("weighted_difference_x",dx_contact))}
        length=max(10,int(np.ceil(block_multiplier*max(tau.values()))))
        groups=np.array_split(np.arange(len(x)),max(2,len(x)//length))
        results.append(dict(x=x,m=order,v=v,d=delta,source=origin,groups=groups,
                            diag=dict(tau_saved_frames=tau,block_count=len(groups),energy_check_eV_atom=err,
                                      delta_H_mean_kT=float(delta.mean()),delta_H_sd_kT=float(delta.std()),
                                      target_Kish_fraction=float(1/np.mean(w*w)),
                                      mean_m=float(order.mean()),mean_volume_A3_atom=float(v.mean()),
                                      range_volume_A3_atom=[float(v.min()),float(v.max())],
                                      max_normalized_weight=float(w.max()/w.sum()))))
    return cfg,results


def estimate(runs,select):
    x=np.concatenate([r["x"][idx] for r,idx in zip(runs,select)])
    d=np.concatenate([r["d"][idx] for r,idx in zip(runs,select)])
    n=np.array([sum(len(r["x"][idx]) for r,idx in zip(runs,select) if r["source"]==s) for s in (0,1)])
    d=d-np.mean(d)
    if np.all(n):
        # Two-state MBAR/Bennett, same umbrella, conditioning on this window.
        log_ratio=np.log(n[1]/n[0])
        f=brentq(lambda val:np.sum(expit(log_ratio+val-d))-n[1],min(d)-50,max(d)+50)
        weights=[expit(-log_ratio-f+d),expit(log_ratio+f-d)]
    else:
        logw=d if n[1] else -d
        target=np.exp(logw-logw.max())
        weights=[target,np.ones_like(d)] if n[1] else [np.ones_like(d),target]
    return np.array([np.average(x,weights=w) for w in weights])


def solve(windows,cfg,select):
    means=np.array([estimate(runs,idx) for runs,idx in zip(windows,select)]).T
    mu=cfg["mu"]-2*KT*cfg["kappa"]*(means-np.array(cfg["windows"]))
    pairs=[contact(x,y) for x,y in zip(means,mu)]
    return means,mu,(np.stack(pairs) if all(p is not None for p in pairs) else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs",nargs="+",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=384)
    p.add_argument("--block-multiplier",type=float,default=6)
    p.add_argument("--seed",type=int,default=674251)
    a=p.parse_args()
    q,names=spline_for("quantum"); c,cnames=spline_for("classical")
    assert names==cnames
    configs,allruns=zip(*(read_run(run,[q,c],names,a.block_multiplier) for run in a.runs))
    cfg=configs[0]
    assert all(r[k]==cfg[k] for r in configs for k in ("windows","mu","kappa","T_K","size","coeff_interpolation"))
    reference=json.loads((HERE.parent/"solvus_1000/c46_reference.json").read_text())
    xa,ya=np.array(reference["windows_c_Al_and_mu_occ_eV"]).T
    ref=contact(xa,ya)
    assert np.max(abs(ref[:2]-[reference["phase"][s]["Al_at_percent"] for s in ("alpha","alpha2")]))<1e-8
    windows=list(zip(*allruns))
    select=[[slice(None) for r in runs] for runs in windows]
    means,mu,raw=solve(windows,cfg,select)
    if raw is None:
        raise ValueError("Unresolved full-sample coexistence")
    diag={rr.name:[r["diag"] for r in rows] for rr,rows in zip(a.runs,allruns)}
    arcs={}
    for name,k in (("early",0),("late",1)):
        halves=[[idx for r in runs for idx in [np.array_split(np.arange(len(r["x"])),2)[k]]] for runs in windows]
        v=solve(windows,cfg,halves)[2]
        arcs[name]=v.tolist() if v is not None else None
    independent=[]
    for rr,rows in zip(a.runs,allruns):
        _,_,v=solve([[r] for r in rows],cfg,[[slice(None)] for r in rows])
        independent.append(dict(name=rr.name,values=None if v is None else v.tolist()))
    rng=np.random.default_rng(a.seed)
    boot=[]
    for _ in range(a.bootstrap):
        draws=[]
        for win in windows:
            idx=[]
            for r in win:
                b=r["groups"]
                idx.append(np.concatenate([b[i] for i in rng.integers(len(b),size=len(b))]))
            draws.append(idx)
        v=solve(windows,cfg,draws)[2]
        if v is not None:
            boot.append(v)
    draws=np.array(boot)
    if len(draws)<2:
        raise ValueError("Insufficient resolved bootstrap contacts")
    intervals=lambda x:np.percentile(x,[2.5,97.5],axis=0).T.tolist()
    # Secondary diagnostic anchored to the historical quantum window means.
    anchored_x=xa+means[1]-means[0]
    anchored=contact(anchored_x,ya-2*KT*cfg["kappa"]*(means[1]-means[0]))
    out=dict(N=cfg["size"],T_K=cfg["T_K"],shape=cfg["shape"],window_centers=cfg["windows"],
             columns=["Al_at_percent_alpha","Al_at_percent_alpha2","field_eV"],
             estimates=raw.tolist(),classical_minus_quantum=(raw[1]-raw[0]).tolist(),
             historical_quantum=ref.tolist(),quantum_minus_historical=(raw[0]-ref).tolist(),
             shifted_historical_windows_classical=anchored.tolist() if anchored is not None else None,
             row_order=list(NAMES),quantum_intervals_95=intervals(draws[:,0,:]),
             classical_intervals_95=intervals(draws[:,1,:]),paired_difference_intervals_95=intervals(draws[:,1,:]-draws[:,0,:]),
             bootstrap_seed=a.seed,bootstrap_attempts=a.bootstrap,bootstrap_resolved=len(draws),
             block_length_multiplier_integrated_tau=a.block_multiplier,window_means=means.tolist(),
             window_mu_eV=mu.tolist(),diagnostics=diag,halves=arcs,individual_runs=independent)
    a.out.parent.mkdir(exist_ok=True,parents=True)
    a.out.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    np.savez_compressed(a.out.with_suffix(".npz"),quantum=draws[:,0,:],classical=draws[:,1,:],
                        classical_minus_quantum=draws[:,1,:]-draws[:,0,:],columns=out["columns"])
    print(json.dumps({k:out[k] for k in ("N","estimates","classical_minus_quantum","paired_difference_intervals_95",
                                         "bootstrap_attempts","bootstrap_resolved")}))


if __name__=="__main__":
    main()
