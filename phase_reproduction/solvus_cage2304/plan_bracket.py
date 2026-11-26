"""Use the local derivatives of the measured phase odds to choose additional windows."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"solvus_closure"))
from thermal_check import load_source,reference,mbar


def measured(manifests,mu,cut):
    t,n,shape,wall,axial,runs=load_source(manifests)
    model=reference(n,runs,t)
    k=model["k"][model["idx"]]
    q=model["q"][model["idx"]]
    beta=1/(mbar.KB*t)
    e=model["energy"]-(mu-model["ref"])*k
    logw=model["log_frame"]+beta*(mu-model["ref"])*(k-k.mean())
    stats=[]
    for side in (q<=cut,q>cut):
        lw=logw[side]
        logz=logsumexp(lw)
        p=np.exp(lw-logz)
        k_avg=p@k[side]
        stats.append(np.array([logz,k_avg,p@(k[side]-k_avg)**2,p@e[side],
                               1/((p*p)@model["g"][side])]))
    diff=stats[1]-stats[0]
    s=diff[0]-np.log(4)
    first=beta*diff[1]
    curvature=beta*beta*diff[2]
    slope=diff[3]*beta/t
    optimum=mu-first/curvature if curvature<0 else None
    roots=sorted(float(mu+r.real) for r in np.roots([curvature/2,first,s]) if abs(r.imag)<1.e-8) if curvature<0 else []
    return dict(N=n,T_K=t,W=wall,field_eV=mu,cut_Q=cut,log_odds_per_variant=s,
                measured_fields=sorted(set(float(r["spec"][0]) for r in runs)),
                slope_per_K=slope,slope_per_eV=first,curvature_per_eV2=curvature,
                conditional_ESS=[v[-1] for v in stats],quadratic_apex_mu=optimum,
                quadratic_crossings_eV=roots,linear_zero_T_at_fixed_field=t-s/slope if slope<0 else None,
                latent_projection_meV=-1000*diff[3]/n,
                note="Local extrapolations for placement of direct UMT jobs; higher derivatives are omitted.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests",type=Path,nargs="+")
    p.add_argument("--mu",type=float,default=3.545)
    p.add_argument("--cut",type=float,default=.37)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    with threadpool_limits(limits=1):
        result=measured(a.manifests,a.mu,a.cut)
    msg=json.dumps(result,allow_nan=False,indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(msg+"\n",encoding="utf8")
    print(msg)
