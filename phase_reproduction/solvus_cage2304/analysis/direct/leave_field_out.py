"""Refit MBAR after omitting every window at each distinct actual chemical potential."""
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from isotherms import HERE,Odds,load,mbar,load_json,atomic_json


def subset(t,n,runs):
    k,q,specs,blocks,diag,_=mbar.make_data(n,runs)
    hist,_=mbar.aggregate(blocks,len(specs));h=hist.sum(axis=0)
    bc=mbar.connectivity(hist,k,q,n);tree=minimum_spanning_tree(-bc).data
    if len(tree)!=len(specs)-1:return None,None
    ref,base,_,_=mbar.fit_mbar(k,q,specs,hist,kt=t*mbar.KB)
    infl=sum(d["statistical_inefficiency"]*np.asarray(b[1].sum(axis=0)).ravel() for d,b in zip(diag,blocks))
    fit=Odds(n,t,k,q,base,h,ref,np.unique(specs[:,0]),np.zeros(len(q)),infl)
    return fit,float(np.min(-tree))


def evaluate(source,dest):
    orig=load_json(source)
    if dest.is_file():
        old=load_json(dest)
        if old["count_original_windows"]==orig["direct_runs"] and old["original_fields"]==orig["direct_fields_eV"]:
            return
    t,n,_,W,_,runs=next(load(list(map(Path,orig["manifests"])),0))
    biased=[r for r in runs if r["spec"][1]>0];fields=orig["direct_fields_eV"]
    rec=[]
    for mu in fields:
        fit,bc=subset(t,n,[r for r in biased if r["spec"][0]!=mu])
        row=dict(omitted_mu_eV=mu,fit_ok=fit is not None,remaining_overlap_min=bc,cuts={})
        if fit is not None:
            for c in (.37,.47):
                key=str(c);central=orig["fixed_cut"][key]
                x=fit.at(mu,c,True);S0=next(z["score"] for z in central["sampled_field_scores"] if z["mu_eV"]==mu)
                opt=fit.fit(c,False);roots={r["branch"]:r for r in opt["crossings"]}
                row["cuts"][key]=dict(S_at_held_out_field=x["score"],S_full_at_held_out_field=S0,
                    omitted_field_within_retained_range=bool(fit.fields[0]<=mu<=fit.fields[-1]),
                    conditional_time_ESS_at_held_out_field=[v["time_ESS"] for v in x["conditional"]],
                    Smax=opt["maximum"]["score"],mu_at_Smax=opt["maximum"]["mu_eV"],
                    max_inside_retained_range=not opt["maximum_on_edge"],
                    roots={b:dict(mu_eV=v["mu_eV"],x_low_pct=100*v["x_low"],x_high_pct=100*v["x_high"],gap_pp=100*v["gap"])
                           for b,v in roots.items()})
        rec.append(row)
    out=dict(T_K=t,W=W,N=n,source=source.name,count_original_windows=orig["direct_runs"],original_fields=fields,
             leaveouts=rec)
    atomic_json(dest,out)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch",action="store_true");args=p.parse_args()
    groups=["final_T1242_W0p40","initial_T1243_W0p40","final_T1243_W0p35","final_T1243_W0p40"]
    while True:
        for name in groups:
            source=HERE/(name+".json")
            if source.is_file():evaluate(source,HERE/("leaveout_"+name+".json"))
        if not args.watch or all((HERE/("leaveout_"+s+".json")).is_file() for s in groups):break
        time.sleep(30)
