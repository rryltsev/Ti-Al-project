"""Cross-check unbiased trajectories at each sampled field and between adjacent fields."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from isotherm import load, mbar


def summarize(n, runs):
    groups = defaultdict(list)
    for run in runs:
        if run["spec"][1] == 0:
            groups[run["spec"][0]].append(run)
    fields = []
    last = None
    last_mu = None
    overlap = []
    for mu, group in sorted(groups.items()):
        h = np.zeros(n+1)
        trajectories = []
        for r in group:
            x = r["k"]/n
            q = 4*r["code"]/(3*n)
            h += np.bincount(r["k"],minlength=n+1)
            row = dict(run=r["run"], start=r["start"], samples=len(x))
            for name,y in (("x",x),("Q",q)):
                split = np.array_split(y,2)
                row[name] = dict(mean=float(y.mean()), sd=float(y.std(ddof=1)),
                                 halves=[float(i.mean()) for i in split],
                                 inefficiency=float(mbar.tau_int(y)))
            trajectories.append(row)
        h /= h.sum()
        if last is not None:
            overlap.append(dict(fields_eV=[last_mu,mu],composition_BC=float(np.sqrt(last*h).sum()),
                                composition_overlap=float(np.minimum(last,h).sum())))
        last,last_mu = h,mu
        fields.append(dict(mu_eV=mu,mean_x_range=[min(r["x"]["mean"] for r in trajectories),
                                                   max(r["x"]["mean"] for r in trajectories)],
                           mean_Q_range=[min(r["Q"]["mean"] for r in trajectories),
                                         max(r["Q"]["mean"] for r in trajectories)],trajectories=trajectories))
    return dict(fields=fields,adjacent_unbiased_composition_overlap=overlap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests",nargs="+",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    out=[dict(temperature_K=t,N=n,shape=sh,cap=cap,axial_free=ax,
              **summarize(n,r)) for t,n,sh,cap,ax,r in load(args.manifests,0)]
    args.out.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf8")
