"""Temperature of zero optimized phase-weight advantage from independent direct isotherms."""
import csv
import json
from pathlib import Path
from collections import Counter,defaultdict

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.special import expit

from isotherms import HERE,load_json,atomic_json


def zero(grid,s):
    return [float(t1+y1*(t2-t1)/(y1-y2)) for t1,t2,y1,y2 in zip(grid,grid[1:],s,s[1:])
            if y1>=0>=y2 and y1>y2]


def interval(v):
    return np.quantile(v,[.025,.975]).tolist() if len(v)>1 else None


def interpolate(records,rng,ndraws):
    # Positive scores on a search edge provide positive lower bounds, but they are not measured maxima.
    usable=[r for r in records if not r["field_edge"] and len(r["bootstrap_scores"])>1]
    if not usable:
        return dict(usable_T=[],observed=records),np.full(ndraws,np.nan)
    t=np.array([r["T_K"] for r in usable]); score=np.array([r["S"] for r in usable])
    weights=1/np.maximum(np.array([np.var(r["bootstrap_scores"],ddof=1) for r in usable]),1.e-12)
    iso=isotonic_regression(score,weights=weights,increasing=False).x
    crossings=zero(t,score)
    predictive=zero(t,iso)
    ids=[rng.integers(len(r["bootstrap_scores"]),size=ndraws) for r in usable]
    direct=np.stack([np.asarray(r["bootstrap_scores"])[ix] for r,ix in zip(usable,ids)],axis=1)
    draws_mu=np.stack([np.asarray(r["bootstrap_mu"])[ix] for r,ix in zip(usable,ids)],axis=1)
    draws_x=np.stack([np.asarray(r["bootstrap_x_pct"])[ix] for r,ix in zip(usable,ids)],axis=1)
    roots=np.full(ndraws,np.nan)
    mu=np.full(ndraws,np.nan);x=np.full(ndraws,np.nan)
    raw_counts=Counter()
    for i,row in enumerate(direct):
        raw_counts[len(zero(t,row))]+=1
        zr=zero(t,isotonic_regression(row,weights=weights,increasing=False).x)
        if len(zr)==1:
            roots[i]=zr[0]
            mu[i]=np.interp(roots[i],t,draws_mu[i])
            x[i]=np.interp(roots[i],t,draws_x[i])
    ok=np.isfinite(roots)
    observed=[{k:v for k,v in r.items() if not k.startswith("bootstrap_")} for r in records]
    result=dict(usable_T=t.tolist(),observed=observed,raw_linear_downward_zeros=crossings,
        weighted_monotone_S=iso.tolist(),weighted_monotone_crossings=predictive,
        conditional_interpolated_T_interval_95=interval(roots[ok]),
        interpolated_mu_eV=float(np.interp(predictive[0],t,[r["mu_eV"] for r in usable])) if len(predictive)==1 else None,
        interpolated_x_pct=float(np.interp(predictive[0],t,[r["x_pct"] for r in usable])) if len(predictive)==1 else None,
        conditional_mu_eV_interval_95=interval(mu[ok]),conditional_x_pct_interval_95=interval(x[ok]),
        bootstrap_fraction_with_bracket=float(ok.mean()),bootstrap_fraction_no_bracket=float((~ok).mean()),
        raw_bootstrap_root_number_fraction={int(k):v/ndraws for k,v in raw_counts.items()})
    return result,roots


def generate():
    source=sorted(HERE.glob("final_T[0-9]*.json"))
    data=defaultdict(list)
    for f in source:
        r=load_json(f)
        for c in (.32,.37,.42,.47,.5):
            z=r["fixed_cut"][str(c)];boot=r["bootstrap"]["cuts"][str(c)]
            a=z["maximum"];v=np.asarray([u["apex"] for u in boot["replicates"]])
            data[r["cap"],c].append(dict(T_K=r["T_K"],field_edge=z["maximum_on_edge"],S=a["score"],
                S_interval_95=boot["apex"]["interval_95"][4],mu_eV=a["mu_eV"],x_pct=100*a["x"],
                estimated_local_linear_zero=a["T_local_linear_zero"],delta_h_meV_atom=a["delta_h_meV"],
                bootstrap_scores=v[:,4].tolist(),bootstrap_mu=v[:,0].tolist(),
                bootstrap_x_pct=(100*(v[:,1]+(v[:,2]-v[:,1])*expit(v[:,4]+np.log(4)))).tolist(),file=f.name))
    rng=np.random.default_rng(135479)
    all_records=[];pairs=defaultdict(dict)
    for (w,c),lst in sorted(data.items()):
        vals,roots=interpolate(sorted(lst,key=lambda r:r["T_K"]),rng,10000)
        vals.update(W=w,cut_Q=c)
        all_records.append(vals)
        pairs[c][w]=roots
    gaps=[]
    for c,table in pairs.items():
        if set(table)!={.35,.4}:continue
        gap=table[.35]-table[.4]
        ok=np.isfinite(gap)
        gaps.append(dict(cut_Q=c,conditional_W35_minus_W40_interval_95=interval(gap[ok]),
                         fraction_both_T_bracketed=float(ok.mean())))
    out=dict(method="Adjacent direct S(T) interpolated linearly. A second estimate applies inverse-bootstrap-variance "
        "weighted decreasing isotonic regression before interpolation. For uncertainty, an independent block resample "
        "at each temperature is drawn 10000 times. The interval is conditional on a measured sign bracket, "
        "and the missing-bracket fraction is reported. Numerical maxima on a simulated-field edge are excluded.",
        records=all_records,W35_minus_W40=gaps)
    atomic_json(HERE/"direct_temperature_zeros.json",out)


if __name__=="__main__":
    generate()
