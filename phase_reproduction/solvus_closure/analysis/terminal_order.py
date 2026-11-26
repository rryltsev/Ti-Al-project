"""Local D019 order of terminal configurations around the 1245 K apex."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter


def local(types,sub,size):
    occ=np.stack([2*uniform_filter((types*(sub==j)).sum(axis=-1).astype(float),size=size,mode="wrap")
                  for j in range(4)])
    magnitude=4*(occ.max(axis=0)-occ.mean(axis=0))/3
    mask=magnitude>.5
    variant=np.argmax(occ,axis=0)
    proportions=np.bincount(variant[mask],minlength=4)/mask.sum() if mask.any() else np.zeros(4)
    return dict(mean_Q=float(magnitude.mean()),median_Q=float(np.median(magnitude)),
                fraction_Q_gt_half=float(mask.mean()),variant_shares_among_high=proportions.tolist())


def evaluate(paths,mu,tolerance,shuffle_count):
    out=[]
    for path in paths:
        for r in csv.DictReader(path.open(encoding="utf-8-sig")):
            if abs(float(r["mu_eV"])-mu)>tolerance+1.e-9:
                continue
            shape=tuple(int(v) for v in r["shape"].split("x"))
            state=path.parent/(r["run"]+".state")
            t=np.loadtxt(state,dtype=int,usecols=0,skiprows=1).reshape(*shape,2)
            i,j,_,b=np.indices(t.shape)
            sub=2*((i+b)%2)+(j+b)%2
            masks=[sub==c for c in range(4)]
            p=[t[s].mean() for s in masks]
            q=4*(max(p)-t.mean())/3
            blocks=[v for v in ((4,4,3),(6,6,4),(8,8,5)) if all(a<=b for a,b in zip(v,shape)) and v!=shape]
            obs={str(v):local(t,sub,v) for v in blocks}
            null={str(v):[] for v in blocks}
            rng=np.random.default_rng(int(r["seed"])+1245)
            for _ in range(shuffle_count):
                scrambled=t.copy()
                for mask in masks:
                    scrambled[mask]=rng.permutation(t[mask])
                for v in blocks:
                    null[str(v)].append(local(scrambled,sub,v)["mean_Q"])
            for v in obs:
                obs[v]["shuffle_mean_Q"]=float(np.mean(null[v]))
                obs[v]["shuffle_interval_Q"]=np.quantile(null[v],[.025,.975]).tolist()
            out.append(dict(run=r["run"],N=int(r["N"]),T_K=float(r["T_K"]),mu_eV=float(r["mu_eV"]),
                K_eV=float(r["K_eV"]),q0=float(r["q0"]),x_pct=100*float(t.mean()),global_Q=float(q),blocks=obs))
    return out


def summarize(states):
    out={}
    for label,select in (("low_Q_le_0.25",lambda q:q<=.25),("high_Q_ge_0.65",lambda q:q>=.65)):
        choice=[s for s in states if select(s["global_Q"])]
        cell={}
        for shape in sorted({key for s in choice for key in s["blocks"]}):
            blocks=[s["blocks"][shape] for s in choice if shape in s["blocks"]]
            values=[[v["mean_Q"],v["shuffle_mean_Q"],v["fraction_Q_gt_half"],
                     max(v["variant_shares_among_high"])] for v in blocks]
            cell[shape]=dict(snapshots=len(values),averages=np.mean(values,axis=0).tolist(),
                             min_max=np.asarray([np.min(values,axis=0),np.max(values,axis=0)]).T.tolist())
        out[label]=dict(snapshots=len(choice),blocks=cell)
    return dict(columns=["mean_local_Q","permuted_mean_local_Q","fraction_local_Q_above_half",
                         "majority_variant_fraction_among_high_local_Q"],groups=out)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests",nargs="+",type=Path)
    p.add_argument("--mu",type=float,default=3.545)
    p.add_argument("--tolerance",type=float,default=.0101)
    p.add_argument("--shuffles",type=int,default=64)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); states=evaluate(a.manifests,a.mu,a.tolerance,a.shuffles)
    if not states:
        raise ValueError("No state falls in the requested field interval")
    result=dict(central_mu=a.mu,field_tolerance=a.tolerance,shuffles=a.shuffles,
            note="One terminal state per window. Each permutation independently scrambles occupations within each "
            "global sublattice, retaining its occupancy. These snapshots diagnose spatial patterns at each order "
            "and have no equilibrium weights.",descriptive_groups=summarize(states),states=states)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
