"""Independent Q-window integrations at each actual chemical field and duplicate-start controls."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import time

import numpy as np

from isotherms import HERE,STUDY,load,discover,fit_runs,mbar,atomic_json,load_json


def block_mean(v):
    ineff=mbar.tau_int(v)
    n=max(1,len(v)//max(1,int(np.ceil(2*ineff))))
    blocks=np.array([np.mean(s) for s in np.array_split(v,n)])
    return dict(mean=float(np.mean(v)),blocks=n,SE=float(np.std(blocks,ddof=1)/np.sqrt(n)) if n>1 else None)


def extract(manifest):
    t,n,shape,cap,_,runs=next(load([manifest],0))
    byfield=defaultdict(list)
    for r in runs:
        if r["spec"][1]>0:byfield[r["spec"][0]].append(r)
    fields=[]
    for mu,items in sorted(byfield.items()):
        fit=fit_runs(t,n,items,32,False,51839,(.37,.47))
        fields.append(dict(mu_eV=mu,BC=fit["least_BC_tree"],few_blocks=fit["few_blocks"],cuts={}))
        for c in ("0.37","0.47"):
            a=fit["fixed_cut"][c]["maximum"]
            b=fit["bootstrap"]["cuts"][c]
            fields[-1]["cuts"][c]=dict(S=a["score"],interval_95_S=b["apex"]["interval_95"][4],
                deltaH_meV=a["delta_h_meV"],local_slope_per_K=a["dS_dT_fixed_mu"],T_local=a["T_local_linear_zero"],
                x_low_pct=100*a["x_low"],x_high_pct=100*a["x_high"],phase_ESS=[r["time_ESS"] for r in a["conditional"]],
                structural_modes=a["profiles"]["0.04"])
    byspec=defaultdict(list)
    for r in runs:byspec[r["spec"]].append(r)
    duplicates=[]
    for (mu,K,q0),items in byspec.items():
        if len(items)<2:continue
        components=[]
        for r in items:
            q=4*r["code"]/(3*n);x=100*r["k"]/n
            components.append(dict(name=r["run"],start=r["start"],Q=block_mean(q),x_pct=block_mean(x),
                 Q_halves=[float(v.mean()) for v in np.array_split(q,2)],
                 x_pct_halves=[float(v.mean()) for v in np.array_split(x,2)]))
        duplicates.append(dict(mu_eV=mu,K=K,q0=q0,independent_starts=components))
    return dict(T_K=t,N=n,W=cap,manifest=str(manifest),fields=fields,identically_biased_starts=duplicates)


def run(base,folder):
    data,pend=discover(base)
    changed=False
    for key in sorted(data,key=lambda p:(p[0]!=1243,p[1],p[0])):
        for m in data[key]:
            output=folder/(m.parent.name+".json")
            if not output.exists():
                atomic_json(output,extract(m))
                changed=True
    if changed and list(HERE.glob("final_T[0-9]*.json")):
        from bracket_checks import assess
        assess()
    return pend


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch",action="store_true")
    args=p.parse_args()
    base=STUDY/"atomistic"/"series"
    while True:
        pend=run(base,HERE/"fields")
        if not args.watch or (not pend and len(list(base.glob("*/windows.csv")))>=52):break
        time.sleep(30)
