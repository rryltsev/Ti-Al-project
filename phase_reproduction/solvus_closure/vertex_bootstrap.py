"""Block uncertainty on a finite-cell congruent temperature from raw MTP energies.

Each draw refits MBAR, then solves both per-variant phase odds and the
stationary chemical-field equation. The curvature check selects a maximum.
Report changes of order and energy as well as a composition; a single-mode
order profile cannot establish coexistence even if the algebraic root exists.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root
from scipy.special import logsumexp

from thermal_check import ROOT, load_source, reference, mbar
sys.path.insert(0,str(ROOT/"analysis"))
from resampled_mbar import ResampledMBAR


def solve(model,logweights,n,t0,cut,initial,bounds,bandwidth):
    k,q=model["k"][model["idx"]],model["q"][model["idx"]]
    ok=np.isfinite(logweights)
    e,k,q,w=model["energy"][ok],k[ok],q[ok],logweights[ok]
    side=q>cut
    kt0=mbar.KB*t0
    ref=model["ref"]
    kmean=k.mean()
    k0=k-kmean
    data=[(e[s],k0[s],w[s]) for s in (~side,side)]
    if min(len(r[0]) for r in data)==0:
        return None
    def values(params):
        a,b=params
        stats=[]
        for en,ka,lw in data:
            z=lw+a*en+b*ka
            logz=logsumexp(z)
            p=np.exp(z-logz)
            ee,kk=p@en,p@ka
            dk=ka-kk
            stats.append(np.array([logz,ee,kk,(p*dk)@(en-ee),(p*dk)@dk]))
        d=stats[1]-stats[0]
        return [d[0]-np.log(4),d[2]], np.array([[d[1],d[2]],[d[3],d[4]]])
    temp,mu=initial
    guess=np.array([1/kt0-1/(mbar.KB*temp),(mu-ref)/(mbar.KB*temp)])
    fit=root(values,guess,jac=True,tol=1e-9)
    a,b=fit.x
    if not np.isfinite(fit.x).all() or (1/kt0-a)<=0:
        return None
    temp=1/((1/kt0-a)*mbar.KB)
    mu=ref+b*mbar.KB*temp
    eq,jac=values(fit.x)
    if max(np.abs(eq))>1.e-6 or jac[1,1]>=0 or not (bounds[0]<temp<bounds[1] and bounds[2]<mu<bounds[3]):
        return None
    z=w+a*e+b*k0
    p=np.exp(z-logsumexp(z))
    gg=model["g"][ok]
    cond=[]
    for s in (~side,side):
        ph=p[s]/p[s].sum()
        cond.append(dict(x_atpct=float(100*(ph@k[s])/n),mean_Q=float(ph@q[s]),
                         semigrand_meV_atom=float(1000*(ph@(e[s]-(mu-ref)*k0[s]))/n),
                         effective_frames=float(1/((ph*ph)@gg[s]))))
    grid,density,peaks=mbar.order_profile(q,p,bandwidth)
    left,right=peaks[grid[peaks]<cut],peaks[grid[peaks]>cut]
    profile=None
    if len(left) and len(right):
        i,j=left[np.argmax(density[left])],right[np.argmax(density[right])]
        m=i+int(np.argmin(density[i:j+1]))
        if i<m<j and density[m]>0:
            profile=dict(Q_modes=grid[[i,j]].tolist(),Q_valley=float(grid[m]),
                         lower_barrier=float(np.log(min(density[i],density[j])/density[m])))
    return dict(T_K=float(temp),mu_eV=float(mu),x_atpct=cond[0]["x_atpct"],
                latent_meV_atom=cond[0]["semigrand_meV_atom"]-cond[1]["semigrand_meV_atom"],
                jump_Q=cond[1]["mean_Q"]-cond[0]["mean_Q"],phases=cond,profile=profile)


def run(a):
    t0,n,shape,cap,axial,runs=load_source(a.manifests)
    if a.part!="all":
        runs=mbar.half_runs(runs,a.part=="early")
    m=reference(n,runs,t0)
    fit=lambda lw,initial:solve(m,lw,n,t0,a.cut,initial,a.bounds,a.bandwidth)
    central=fit(m["log_frame"],a.initial)
    if not central:
        raise ValueError("No stationary maximum with equal per-variant weights in bounds")
    opts=[central["T_K"],central["mu_eV"]]
    sensitivity=[]
    for divider in sorted(set([.32,.37,.42,.47,a.cut]+([central["profile"]["Q_valley"]] if central["profile"] else []))):
        r=solve(m,m["log_frame"],n,t0,divider,opts,a.bounds,a.bandwidth)
        sensitivity.append(dict(Q_divider=divider,root=r))
    nbin=len(m["k"])
    source=np.empty(len(m["idx"]),dtype=int)
    timeblocks=np.empty_like(source)
    blocks=[]
    start=0
    for lo,hi,s,g in m["segments"]:
        source[lo:hi]=s
        nb=max(1,(hi-lo)//int(np.ceil(2*g)))
        for j,idx in enumerate(np.array_split(np.arange(lo,hi),nb)):
            timeblocks[idx]=start+j
        blocks.append(nb)
        start+=nb
    rng=np.random.default_rng(a.seed)
    states=len(m["specs"])
    values=[]
    resampler=ResampledMBAR(m["k"],m["q"],m["specs"],mbar.KB*t0,m["ref"],m["offsets"],m["hist"])
    for _ in range(a.draws):
        mult=np.concatenate([np.bincount(rng.integers(nb,size=nb),minlength=nb) for nb in blocks])[timeblocks]
        counts=np.bincount(source*nbin+m["idx"],weights=mult,minlength=states*nbin).reshape(states,nbin)
        h=counts.sum(axis=0)
        try:
            logmass=resampler.fit(counts)
        except (ArithmeticError,np.linalg.LinAlgError):
            _,logmass,_,_=mbar.fit_mbar(m["k"],m["q"],m["specs"],counts,m["offsets"],kt=mbar.KB*t0)
        lw=np.full(len(mult),-np.inf)
        good=mult>0
        gi=m["idx"][good]
        lw[good]=np.log(mult[good])+logmass[gi]-np.log(h[gi])
        estimate=fit(lw,opts)
        if estimate is not None:
            values.append(dict(T_K=estimate["T_K"],mu_eV=estimate["mu_eV"],x_atpct=estimate["x_atpct"],
                               latent_meV_atom=estimate["latent_meV_atom"],jump_Q=estimate["jump_Q"],
                               lower_barrier=None if not estimate["profile"] else estimate["profile"]["lower_barrier"]))
    intervals={}
    for key in (values[0] if values else {}):
        v=[r[key] for r in values if r[key] is not None]
        if v:
            intervals[key]=np.quantile(v,[.025,.5,.975]).tolist()
    out=dict(source_T_K=t0,N=n,shape=shape,cap=cap,axial_free=axial,time_part=a.part,seed=a.seed,
             Q_divider=a.cut,Q_bandwidth=a.bandwidth,search_T_mu=a.bounds,draws=a.draws,converged=len(values),
             min_blocks=min(blocks),one_block_runs=blocks.count(1),central=central,conditional_95=intervals,
             divider_sensitivity=sensitivity,replicates=values,
             manifests=[p.resolve().relative_to(ROOT.parent).as_posix() for p in a.manifests])
    a.out.parent.mkdir(exist_ok=True,parents=True)
    a.out.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests",type=Path,nargs="+")
    p.add_argument("--initial",type=float,nargs=2,default=[1245.,3.545])
    p.add_argument("--bounds",type=float,nargs=4,default=[1200.,1280.,3.47,3.62])
    p.add_argument("--cut",type=float,default=.37)
    p.add_argument("--bandwidth",type=float,default=.04)
    p.add_argument("--part",choices=["early","late","all"],default="all")
    p.add_argument("--draws",type=int,default=64)
    p.add_argument("--seed",type=int,default=31292026)
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args())
