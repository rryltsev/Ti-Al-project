"""Reconstruct the two contacts at 1243 K jointly from the four neighboring isotherms."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from pooled_wall import HERE, pooled, source_paths


def profile(k,q,weights,g,n,kt,mu0,fields,cut):
    bins=[]
    for mask in (q<=cut,q>cut):
        mass=np.bincount(k[mask],weights[mask],minlength=n+1)
        square=np.bincount(k[mask],weights[mask]**2*g[mask],minlength=n+1)
        valid=mass>0
        bins.append((np.flatnonzero(valid),np.log(mass[valid]),square[valid]/mass[valid]**2))
    def phase(mu):
        out=[]
        for counts,logs,inverse in bins:
            lw=logs+(mu-mu0)/kt*(counts-.27*n)
            total=logsumexp(lw)
            p=np.exp(lw-total)
            out.append((total,float(p@counts)*100/n,float(1/((p*p)@inverse))))
        return out
    def odds(mu):
        a,b=phase(mu)
        return b[0]-a[0]-np.log(4)
    mesh=np.linspace(fields[0],fields[-1],51)
    s=np.asarray([odds(f) for f in mesh])
    extrema=[(s[0],mesh[0]),(s[-1],mesh[-1])]
    for i in np.flatnonzero((s[1:-1]>=s[:-2])&(s[1:-1]>=s[2:]))+1:
        res=minimize_scalar(lambda m:-odds(m),bounds=(mesh[i-1],mesh[i+1]),method="bounded",
                            options={"xatol":1e-12})
        extrema.append((-res.fun,res.x))
    maximum,mu=max(extrema)
    result=dict(Smax=float(maximum),field_at_max=float(mu),edges=[float(s[0]),float(s[-1])],
                maximum_at_edge=mu in (fields[0],fields[-1]),Ti=None,Al=None)
    if maximum<=0 or result["maximum_at_edge"]:
        return result
    for name,end in (("Ti",fields[0]),("Al",fields[-1])):
        previous=mu
        for curr in np.linspace(mu,end,32)[1:]:
            if odds(curr)<0:
                root=brentq(odds,*sorted((curr,previous)),xtol=1e-12)
                low,high=phase(root)
                result[name]=dict(field=root,x_alpha=low[1],x_alpha2=high[1],
                                  gap_pp=high[1]-low[1],ESS=[low[2],high[2]])
                break
            previous=curr
    return result


def main(a):
    paths,pending=source_paths(a.wall,a.sources,None,getattr(a,"available",False),False)
    n,shape,cap,axial,s,v,origin,blocks,g=pooled.samples(paths,a.stride)
    k_float,q,e=v.T
    k=k_float.astype(int)
    t0,mu0=a.target,3.545
    kt=pooled.mbar.KB*t0
    fields=np.unique(s[s[:,0]==t0,1])
    if len(fields)<3:
        raise ValueError("Need an actual target isotherm with at least three fields")
    en=e-mu0*k
    en-=en.mean()
    beta=1/(pooled.mbar.KB*s[:,0,None])
    u=(beta-1/kt)*en + beta*(.5*s[:,2,None]*(q-s[:,3,None])**2-(s[:,1,None]-mu0)*(k-k.mean()))
    mix=pooled.Mixture(u,origin)
    del u
    def estimate(lw):
        return profile(k,q,np.exp(lw-np.max(lw)),g,n,kt,mu0,fields,a.cut)
    central=estimate(-mix.logden)
    indices=[]; sizes=[]; offset=0
    for start,stop,nb in blocks:
        sizes.append(nb)
        lengths=[len(b) for b in np.array_split(np.arange(stop-start),nb)]
        indices.append(np.repeat(np.arange(nb)+offset,lengths))
        offset+=nb
    blockid=np.concatenate(indices)
    rng=np.random.default_rng(4583129+round(100*a.wall))
    reps=[]
    for _ in range(a.draws):
        copies=np.concatenate([np.bincount(rng.integers(nb,size=nb),minlength=nb) for nb in sizes])
        reps.append(estimate(mix.bootstrap(copies[blockid])))
    intervals={}
    for side in ("Ti","Al"):
        good=[r[side] for r in reps if r[side]]
        intervals[side]=dict(count=len(good),conditional={
            x:np.quantile([row[x] for row in good],[.025,.5,.975]).tolist()
            for x in ("field","gap_pp","x_alpha","x_alpha2") if good})
    record=dict(T_K=t0,N=n,shape=shape,W=cap,free_axial=axial,pressure_GPa=0.,cut_Q=a.cut,
                target_direct_fields=fields.tolist(),source_T_K=sorted(set(s[:,0])),stride=a.stride,
                n_trajectories=len(blocks),draws=a.draws,center=central,
                both_contacts_replicates=sum(bool(r["Al"] and r["Ti"]) for r in reps),
                Smax_025_50_975=np.quantile([r["Smax"] for r in reps],[.025,.5,.975]).tolist() if reps else None,
                intervals=intervals,resampled_profiles=reps,
                manifests=[str(p.relative_to(HERE)) for p in paths],pending_at_fit=pending,
                definition="Multi-temperature MBAR evaluated at one actually simulated target T. "
                           "Search restricted to the interval directly sampled at that T. "
                           "Intervals on contacts are conditional on existence of the branch.")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    temp=a.out.with_suffix(".partial")
    temp.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf8")
    temp.replace(a.out)
    print(a.wall,central,record["both_contacts_replicates"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wall",type=float,choices=[.35,.40],required=True)
    p.add_argument("--sources",nargs="+",type=float,default=[1242.,1243.,1244.,1245.])
    p.add_argument("--target",type=float,default=1243.)
    p.add_argument("--cut",type=float,default=.37)
    p.add_argument("--draws",type=int,default=64)
    p.add_argument("--available",action="store_true",help="Diagnostic on completed groups only")
    p.add_argument("--stride",type=int,default=12)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    with threadpool_limits(limits=1):
        main(args)
