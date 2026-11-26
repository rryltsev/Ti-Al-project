"""Estimate the same order odds using independently sampled chemical-field subsets."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import logsumexp,softmax

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"solvus_temperatures"/"analysis"))
from isotherm import load,mbar


def check(n,t,runs,mu,cut):
    k,q,s,blocks,diag,_=mbar.make_data(n,runs)
    hist,_=mbar.aggregate(blocks,len(s)); h=hist.sum(axis=0)
    ref,base,_,_=mbar.fit_mbar(k,q,s,hist,kt=t*mbar.KB)
    u=base+(mu-ref)*k/(t*mbar.KB); mask=q>cut
    inflate=sum(d["statistical_inefficiency"]*np.asarray(v[1].sum(axis=0)).ravel() for d,v in zip(diag,blocks))
    cond=[]
    for side in (~mask,mask):
        w=softmax(u[side]); pf=w/h[side]
        cond.append(dict(x_pct=float(100*w@k[side]/n),Q=float(w@q[side]),time_ESS=float(1/(inflate[side]@(pf**2)))))
    bc=mbar.connectivity(hist,k,q,n)
    tree=minimum_spanning_tree(-bc).data
    ans=dict(mu_eval=mu,direct_fields=np.unique(s[:,0]).tolist(),runs=len(runs),
             n_bias_windows=len(s),BC_tree_min=float(np.min(-tree)) if len(tree)==len(s)-1 and len(tree) else None,
             S_at_common_mu=float(logsumexp(u[mask])-logsumexp(u[~mask])-np.log(4.)),
             conditional_low_high=cond,range_Q0=[float(s[:,2].min()),float(s[:,2].max())])
    return ans


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cut",type=float,default=.47)
    args=p.parse_args()
    z=json.loads(args.profile.read_text(encoding="utf8"))[0]
    t,n,_,_,_,r=next(load(list(map(Path,z["manifests"])),0))
    apex=next(a for a in z["apex"] if abs(a["cut_Q"]-args.cut)<1.e-9)
    mu=apex["mu"]
    fields=sorted({v["spec"][0] for v in r})
    mid=min(fields,key=lambda x:abs(x-mu))
    out={str(h):check(n,t,[u for u in r if u["spec"][0]==h],mu,args.cut) for h in fields}
    if len(fields)>2:
        out["all_except_nearest_field"]=check(n,t,[u for u in r if u["spec"][0]!=mid],mu,args.cut)
    data=dict(T_K=t,N=n,cut_Q=args.cut,pooled_S=apex["log_weight_per_variant"],evaluations=out,
              note="All estimates use the same target field and cutoff. Each individual field group has independent "
                   "seeds and its own MBAR normalization. Marginally overlapping Q windows or very small effective "
                   "counts limit the evidential value of the individual reconstruction.")
    args.out.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf8")
