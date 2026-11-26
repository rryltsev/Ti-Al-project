"""Smoothed order PMF and conditional composition at the directly measured maximum of S."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import softmax

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"solvus_temperatures"/"analysis"))
from isotherm import load,mbar


def profiles(source,cut):
    for z in json.loads(source.read_text(encoding="utf8")):
        if abs(z["T_K"]-1245)>1.e-9:
            continue
        apex=next(p for p in z["apex"] if abs(p["cut_Q"]-cut)<1.e-8)
        mu=apex["mu"]
        t,n,_,_,_,runs=next(load([Path(v) for v in z["manifests"]],0))
        k,q,specs,blocks,diag,_=mbar.make_data(n,runs)
        hist,_=mbar.aggregate(blocks,len(specs))
        ref,base,_,_=mbar.fit_mbar(k,q,specs,hist,kt=t*mbar.KB)
        w=softmax(base+(mu-ref)*k/(t*mbar.KB))
        bw=z["bandwidth_Q"]
        grid,pdf,_=mbar.order_profile(q,w,bw)
        edges=np.linspace(0,1,201)
        ibins=np.clip(np.searchsorted(edges,q,side="right")-1,0,199)
        totals=hist.sum(axis=0)
        n_inflated=sum(d["statistical_inefficiency"]*np.asarray(block[1].sum(axis=0)).ravel()
                       for d,block in zip(diag,blocks))
        mass_variance=np.bincount(ibins,weights=n_inflated*(w/totals)**2,minlength=200)
        kernel=gaussian_filter1d(np.eye(200),bw/.005,axis=0,mode="reflect")
        variance=(kernel*kernel)@mass_variance
        local_ess=np.divide((.005*pdf)**2,variance,out=np.zeros(len(pdf)),where=variance>0)
        mx=np.bincount(ibins,weights=w*k/n,minlength=200)
        cx=np.divide(gaussian_filter1d(mx,bw/.005,mode="reflect")/.005,pdf,
                     out=np.zeros(len(pdf)),where=pdf>0)
        if not np.isclose(pdf.sum()*.005,1) or not np.isclose(.005*pdf@cx,apex["x"]):
            raise ArithmeticError("Histogram normalization or conditional composition differs from MBAR")
        with np.errstate(divide="ignore"):
            free=-np.log(pdf/pdf.max())
        for i,Q in enumerate(grid):
            yield dict(T_K=t,N=n,mu_eV=mu,cut_Q=cut,Q=Q,p_density=pdf[i],
                       delta_F_kBT=free[i] if np.isfinite(free[i]) else None,
                       relative_density=pdf[i]/pdf.max(),local_time_ESS_approx=local_ess[i],
                       conditional_x_pct=100*cx[i] if pdf[i]>0 else None,source=source.name)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs",nargs="*",type=Path)
    p.add_argument("--cut",type=float,default=.47)
    p.add_argument("--out",type=Path,default=HERE/"critical_order_profiles.csv")
    args=p.parse_args()
    sources=args.inputs or sorted(HERE.glob("T1245_N*.json"))
    rows=[row for s in sources for row in profiles(s,args.cut)]
    if not rows: raise ValueError("No 1245 K profile")
    rows.sort(key=lambda v:(v["N"],v["Q"]))
    tmp=args.out.with_suffix(".csv.tmp")
    with tmp.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    tmp.replace(args.out)
