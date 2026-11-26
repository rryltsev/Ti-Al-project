"""Compare independent K=0 starts with MBAR composition in each order basin."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

from isotherms import HERE,load,load_json,atomic_json,kinetic_record


def compare():
    records=[]
    for path in sorted(HERE.glob("final_T[0-9]*.json")):
        z=load_json(path)
        if not z["K0_independent"]:continue
        free_paths=[]
        for m in map(Path,z["manifests"]):
            with m.open(encoding="utf-8-sig") as f:
                if any(float(r["K_eV"])==0 for r in csv.DictReader(f)):
                    free_paths.append(m)
        t,n,_,w,_,runs=next(load(free_paths,0))
        by=defaultdict(list)
        model={c:{s["mu_eV"]:s for s in z["fixed_cut"][c]["sampled_field_scores"]} for c in ("0.37","0.47")}
        for r in runs:
            if r["spec"][1]!=0:continue
            x=r["k"]/n;q=4*r["code"]/(3*n);mu=r["spec"][0]
            a=kinetic_record(n,r);a["basins"]={}
            for c,tab in model.items():
                if mu not in tab:continue
                reference=tab[mu]
                cond=[]
                for key,side in (("low",q<=float(c)),("high",q>float(c))):
                    target=reference["x_"+key]
                    half=[float(100*xi[mi].mean()) if mi.any() else None for xi,mi in zip(np.array_split(x,2),np.array_split(side,2))]
                    sample=float(x[side].mean()) if side.any() else None
                    cond.append(dict(side=key,raw_frames=int(side.sum()),direct_conditional_pct=100*sample if sample is not None else None,
                        direct_minus_MBAR_pp=100*(sample-target) if sample is not None else None,
                        halves_pct=half,MBAR_pct=100*target))
                a["basins"][c]=cond
            by[mu].append(a)
        summary=[]
        for mu,starts in sorted(by.items()):
            q=[s["Q"] for s in starts]
            summary.append(dict(mu_eV=mu,n_starts=len(starts),
                Q_spread_of_start_means=max(q)-min(q),total_crossings=sum(s["traversals_between_cores"] for s in starts),
                starts_with_round_trip=sum(s["round_trips"]>0 for s in starts),independent_raw_starts=starts))
        records.append(dict(T_K=t,W=w,summary=summary))
    atomic_json(HERE/"free_start_comparison.json",dict(records=records,
          note="Conditional compositions in a poorly sampled basin are imprecise. Each start's raw frequency "
               "is a kinetic diagnostic; the deliberately paired starts do not set equilibrium weights."))


if __name__=="__main__":
    compare()
