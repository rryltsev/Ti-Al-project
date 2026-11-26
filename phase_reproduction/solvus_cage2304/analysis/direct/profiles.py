"""Bias-free P(Q), F(Q) and the local composition at matched field, apex and direct contacts."""
from pathlib import Path
import csv
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import softmax

from isotherms import HERE,load,load_json,mbar


def rows(profile):
    rec=load_json(profile)
    t,n,_,cap,_,runs=next(load(list(map(Path,rec["manifests"])),0))
    biased=[r for r in runs if r["spec"][1]>0]
    k,q,specs,blocks,diag,_=mbar.make_data(n,biased)
    hist,_=mbar.aggregate(blocks,len(specs));h=hist.sum(axis=0)
    ref,base,_,_=mbar.fit_mbar(k,q,specs,hist,kt=t*mbar.KB)
    infl=sum(d["statistical_inefficiency"]*np.asarray(b[1].sum(axis=0)).ravel() for d,b in zip(diag,blocks))
    model=rec["fixed_cut"]["0.37"]
    targets=[("optimal",model["maximum"]["mu_eV"])]
    if 3.545 in rec["direct_fields_eV"]:targets.append(("matched_3.545",3.545))
    targets.extend((r["branch"]+"_contact",r["mu_eV"]) for r in model["crossings"])
    bins=np.clip(np.searchsorted(np.linspace(0,1,201),q,side="right")-1,0,199)
    bw=.04
    kernel=gaussian_filter1d(np.eye(200),bw/.005,axis=0,mode="reflect")
    for label,mu in targets:
        w=softmax(base+(mu-ref)*k/(t*mbar.KB))
        Q,density,_=mbar.order_profile(q,w,bw)
        vmass=np.bincount(bins,weights=infl*(w/h)**2,minlength=200)
        variance=(kernel*kernel)@vmass
        ess=np.divide((.005*density)**2,variance,out=np.zeros_like(variance),where=variance>0)
        mx=np.bincount(bins,weights=w*k/n,minlength=200)
        x=np.divide(kernel@mx,.005*density,out=np.zeros(200),where=density>0)
        if not np.isclose(.005*density.sum(),1) or not np.isclose(.005*(density@x),w@(k/n)):
            raise ValueError("Density smoothing changed total probability or composition")
        for j in range(200):
            yield dict(T_K=t,W=cap,N=n,field_kind=label,mu_eV=mu,cut_Q=.37,Q=Q[j],density_Q=density[j],
                       relative_density=density[j]/density.max(),F_kBT=-np.log(density[j]/density.max()) if density[j]>0 else None,
                       x_conditional_pct=100*x[j] if density[j]>0 else None,local_time_ESS_approx=ess[j])


def write_csv(path,items):
    tmp=path.with_suffix(".tmp")
    with tmp.open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(items[0]))
        writer.writeheader();writer.writerows(items)
    tmp.replace(path)


def generate():
    folder=HERE/"profiles";folder.mkdir(exist_ok=True)
    whole=[]
    for p in sorted(HERE.glob("final_T[0-9]*.json")):
        out=folder/(p.stem+".csv")
        if not out.exists() or out.stat().st_mtime<p.stat().st_mtime:
            write_csv(out,list(rows(p)))
        with out.open(encoding="utf8",newline="") as f:
            whole.extend(csv.DictReader(f))
    if whole:write_csv(HERE/"order_profiles.csv",whole)


if __name__=="__main__":
    generate()
