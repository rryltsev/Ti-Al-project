"""Unbinned multitemperature MBAR for the maximum of the Ti–Al ordering dome.

Only positive, equilibrated Q umbrellas with identical mechanical constraints
enter by default. Retain the total energy of each frame. Thinning reduces the
size of the dense bias matrix; within-chain block resampling retains slower
correlations. Compare independent sources before interpreting a pooled estimate.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from thermal_check import ROOT,mbar
from vertex_bootstrap import solve


def samples(manifests,stride):
    frames=[]; labels=[]; blocks=[]; ineff=[]
    thermo=set(); states=[]; runs=set(); offset=0
    for f in manifests:
        with f.open(encoding="utf-8-sig",newline="") as fp:
            rows=list(csv.DictReader(fp))
        for r in rows:
            if r["run"] in runs:
                raise ValueError("Repeated trajectory "+r["run"])
            runs.add(r["run"])
            t,mu,K,q0=[float(r[s]) for s in ("T_K","mu_eV","K_eV","q0")]
            thermo.add((int(r["N"]),r["shape"],float(r["cap"]),int(r["axial_free"])))
            if K<=0:
                raise ValueError("Only positive umbrellas may supply the pooled phase odds")
            state=(t,mu,K,q0)
            if state not in states:
                states.append(state)
            # Validate the recorded temperature/length and use only retained sweeps.
            v=np.loadtxt(f.parent/(r["run"]+".csv"),delimiter=",",skiprows=1,usecols=(0,1,6,7,8,10,15))
            v=v[v[:,0]>int(r["burn"])]
            if len(v)!=int(r["sweeps"])//int(r["stride"]) or not np.allclose(v[:,-1],t):
                raise ValueError("Bad trajectory "+r["run"])
            # Use a deterministic offset across chains to avoid cadence aliasing.
            reduced=v[int(r["seed"])%stride::stride]
            if len(reduced)<4:
                raise ValueError("Too few frames after thinning")
            def autocorr(a):
                return max(mbar.tau_int(x) for x in (a[:,1],a[:,2],a[:,3],a[:,4]-mu*a[:,1],a[:,5]))
            g=max(1.,autocorr(v)/stride,autocorr(reduced))
            count=len(reduced)
            nb=max(1,count//int(np.ceil(2*g)))
            blocks.append((offset,offset+count,nb))
            offset+=count
            labels.append(np.full(count,states.index(state),dtype=int))
            ineff.append(np.full(count,g))
            frames.append(reduced[:,[1,2,4]])
    (n,shape,cap,axial),=thermo
    return n,shape,cap,axial,np.array(states),np.vstack(frames),np.concatenate(labels),blocks,np.concatenate(ineff)


class Mixture:
    def __init__(self,u,origin,precision=2e-8):
        counts=np.bincount(origin,minlength=len(u)).astype(float)
        logs=np.log(counts)[:,None]-u
        col=logs.max(axis=0)
        a=np.exp(logs-col)
        del logs
        norm=len(origin)
        def objective(x):
            f=np.r_[0.,x]; shift=f.max(); factor=np.exp(f-shift)
            d=factor@a
            if np.min(d)<=0:
                return 1e50,np.zeros_like(x)
            return (np.log(d).mean()+shift-counts@f/norm,(factor*(a@(1/d))-counts)[1:]/norm)
        opt=minimize(objective,np.zeros(len(u)-1),jac=True,method="L-BFGS-B",
                     options=dict(maxiter=3500,maxcor=25,ftol=1e-16,gtol=1e-11,maxls=60))
        f=np.r_[0.,opt.x]
        lden=logsumexp(np.log(counts)[:,None]+f[:,None]-u,axis=0)
        self.post=np.exp(np.log(counts)[:,None]+f[:,None]-u-lden)
        self.logden=lden; self.counts=counts; self.origin=origin
        self.error=float(np.max(np.abs(self.post.sum(axis=1)-counts))/norm)
        if self.error>precision:
            raise ValueError(f"Multitemperature MBAR residual {self.error:.2g}")

    def bootstrap(self,mult):
        counts=np.bincount(self.origin,weights=mult,minlength=len(self.counts))
        coeff=counts/self.counts
        total=counts.sum()
        if min(counts)<=0:
            raise ValueError("Empty bias after block resampling")
        delta=np.zeros(len(counts))
        for _ in range(32):
            shift=delta.max()
            factors=coeff*np.exp(delta-shift)
            den=factors@self.post
            p=(factors[:,None]*self.post)/den
            observed=p@mult
            grad=observed-counts
            if max(abs(grad))/total<2e-9:
                out=np.full(len(mult),-np.inf)
                ok=mult>0
                out[ok]=np.log(mult[ok])-self.logden[ok]-shift-np.log(den[ok])
                return out
            z=p[1:]*np.sqrt(mult)
            h=np.diag(observed[1:])-z@z.T
            step=np.r_[0.,np.linalg.solve(h,grad[1:])]
            value=mult@np.log(den)+total*shift-counts@delta
            for j in range(20):
                scale=.5**j
                trial=delta-scale*step
                sh=trial.max()
                d=(coeff*np.exp(trial-sh))@self.post
                objective=mult@np.log(d)+total*sh-counts@trial
                if objective<=value-1e-4*scale*(grad@step)+3e-9:
                    delta=trial
                    break
            else:
                raise ValueError("Bootstrap Newton line search failed")
        raise ValueError("Bootstrap Newton residual did not converge")


def main(a):
    n,shape,cap,axial,s,v,origin,blocks,g=samples(a.manifests,a.stride)
    k,q,e=v.T
    t0,mu0=1245.,3.545
    en=e-mu0*k
    en-=en.mean()
    b=1/(mbar.KB*s[:,0,None])
    u=(b-1/(mbar.KB*t0))*en+b*(.5*s[:,2,None]*(q-s[:,3,None])**2-(s[:,1,None]-mu0)*(k-k.mean()))
    mix=Mixture(u,origin)
    del u
    model=dict(idx=np.arange(len(k)),k=k,q=q,energy=en,g=g,ref=mu0)
    def root_at(lw,cut):
        return solve(model,lw,n,t0,cut,[t0,mu0],a.bounds,.04)
    central=root_at(-mix.logden,a.cut)
    if not central:
        raise ValueError("No physical stationary root within the supplied bounds")
    ts=np.unique(s[:,0])
    by_t=np.array([mix.post[s[:,0]==t].sum(axis=0) for t in ts])
    overlap=[by_t[:,s[:,0][origin]==t].mean(axis=1).tolist() for t in ts]
    sensitivity=[dict(cut=cut,estimate=root_at(-mix.logden,cut)) for cut in (.32,a.cut,.42,.47)]
    ids=[]; sizes=[]
    offset=0
    for lo,hi,nb in blocks:
        sizes.append(nb)
        ids.append(np.repeat(np.arange(nb)+offset,[len(b) for b in np.array_split(np.arange(hi-lo),nb)]))
        offset+=nb
    ids=np.concatenate(ids)
    rng=np.random.default_rng(a.seed)
    estimates=[]
    for _ in range(a.draws):
        mult=np.concatenate([np.bincount(rng.integers(ns,size=ns),minlength=ns) for ns in sizes])[ids]
        cur=root_at(mix.bootstrap(mult),a.cut)
        if cur:
            estimates.append([cur[k] for k in ("T_K","mu_eV","x_atpct","latent_meV_atom","jump_Q")])
    z=dict(N=n,shape=shape,W=cap,free_axial=axial,pressure_GPa=0.,source_T_K=ts.tolist(),source_umbrellas=len(s),
           raw_trajectories=len(blocks),retained_frames=len(origin),thinning_stride=a.stride,T_reference=t0,
           configuration_field_reference_eV=mu0,Q_divider=a.cut,search_bounds=a.bounds,gradient_error=mix.error,
           source_overlap_from_rows_to_columns=overlap,central=central,divider_sensitivity=sensitivity,
           bootstrap_columns=["T_K","mu_eV","x_atpct","latent_meV_atom","jump_Q"],draws=a.draws,seed=a.seed,
           resolved=len(estimates),min_blocks=min(sizes),one_block_runs=sizes.count(1),
           interval_95=np.quantile(estimates,[.025,.5,.975],axis=0).T.tolist() if estimates else None,
           replicates=estimates,manifests=[p.resolve().relative_to(ROOT.parent).as_posix() for p in a.manifests])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(z,indent=2,allow_nan=False)+"\n",encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests",type=Path,nargs="+")
    p.add_argument("--stride",type=int,default=8)
    p.add_argument("--cut",type=float,default=.37)
    p.add_argument("--bounds",type=float,nargs=4,default=[1200.,1280.,3.47,3.62])
    p.add_argument("--draws",type=int,default=32)
    p.add_argument("--seed",type=int,default=6213129)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.stride<=0 or a.draws<0:
        p.error("stride and bootstrap count do not define a valid sample")
    main(a)
