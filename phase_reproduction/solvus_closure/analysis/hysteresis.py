"""Unbiased ordered/disordered starts, occupancy and transitions between order basins."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"solvus_temperatures"/"analysis"))
from isotherm import load
from direct_pairs import summarize


def record(n,run):
    x=run["k"]/n
    q=4*run["code"]/(3*n)
    lo=q<.35; hi=q>.60
    core=np.where(hi,1,-1)[lo|hi]
    transitions=int(np.count_nonzero(np.diff(core))) if len(core) else 0
    return dict(start=run["start"],run=run["run"],mu_eV=run["spec"][0],nframes=len(q),
                x_pct=float(100*x.mean()),Q=float(q.mean()),Q_range=[float(q.min()),float(q.max())],
                fraction_Q_low=float(lo.mean()),fraction_Q_high=float(hi.mean()),
                x_pct_with_Q_low=float(100*x[lo].mean()) if lo.any() else None,
                x_pct_with_Q_high=float(100*x[hi].mean()) if hi.any() else None,
                traversals_between_cores=transitions,round_trips=transitions//2)


if __name__=="__main__":
    pr=argparse.ArgumentParser(description=__doc__)
    pr.add_argument("manifests",nargs="+",type=Path)
    pr.add_argument("--out",type=Path,required=True)
    args=pr.parse_args()
    out=[]
    for t,n,shape,cap,ax,r in load(args.manifests,0):
        rr=[v for v in r if v["spec"][1]==0]
        if not rr:
            raise ValueError("No unbiased starts for this isotherm")
        out.append(dict(T_K=t,N=n,W=cap,axial_free=ax,unbiased_pairs=summarize(n,rr),
                        excursions=[record(n,v) for v in rr],
                        note="Core-to-core traversals ignore intermediate frames (Q from 0.35 to 0.60). "
                             "Counts describe mixing on the sampled time scale. Purposeful numbers of starts "
                             "in each basin do not set phase weights."))
    args.out.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf8")
