"""Locate the finite-cell maximum temperature of per-variant coexistence.

At a specified order divider the limiting state obeys log(P2/4P1)=0
and d log(P2/P1)/dmu = beta (mean(k)_2 - mean(k)_1) = 0. A split of
a unimodal distribution has no phase interpretation. Inspect the free-energy
valley, mode separation, basin ESS, and independent temperature sources.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp

from thermal_check import ROOT, load_source, reference, tilted_weights, mbar


def vertex(model,n,t0,cut,field_bounds,temperatures,bandwidth):
    k,q,ref=model["k"],model["q"],model["ref"]
    selected=q>cut
    @lru_cache(maxsize=40)
    def at(t):
        base, inverse=tilted_weights(model,t0,t)
        kt=mbar.KB*t
        kmass=[np.full(n+1,-np.inf), np.full(n+1,-np.inf)]
        for h,side in zip(kmass,(~selected,selected)):
            np.logaddexp.at(h,k[side],base[side])
        valid=np.isfinite(kmass[0])|np.isfinite(kmass[1])
        kval=np.flatnonzero(valid)
        lo,hi=[h[valid] for h in kmass]
        def odds(mu):
            b=(mu-ref)/kt*kval
            return logsumexp(hi+b)-logsumexp(lo+b)-np.log(4.)
        grid=np.linspace(*field_bounds,61)
        curve=np.array([odds(mu) for mu in grid])
        i=np.argmax(curve)
        mu=grid[i]
        edge=i==0 or i==len(grid)-1
        if not edge:
            res=minimize_scalar(lambda z:-odds(z),bounds=(grid[i-1],grid[i+1]),method="bounded",
                                options={"xatol":1e-10})
            mu=float(res.x)
        w=np.exp(base+(mu-ref)*k/kt-logsumexp(base+(mu-ref)*k/kt))
        qs,den,peaks=mbar.order_profile(q,w,bandwidth)
        lp=peaks[qs[peaks]<cut]
        hp=peaks[qs[peaks]>cut]
        barrier=None
        if len(lp) and len(hp):
            a,b=lp[np.argmax(den[lp])],hp[np.argmax(den[hp])]
            j=a+int(np.argmin(den[a:b+1]))
            if a<j<b and den[j]>0:
                barrier=dict(Q_peaks=qs[[a,b]].tolist(),Q_valley=float(qs[j]),
                             from_lower_peak_kBT=float(np.log(min(den[a],den[b])/den[j])))
        details=[]
        for side in (~selected,selected):
            p=w[side]/w[side].sum()
            details.append(dict(Al_at_pct=float(100*p@k[side]/n),Q=float(p@q[side]),
                                corrected_Kish=float(1/((p*p)@inverse[side]))))
        # Conditional enthalpy at the same field; the stored microscopic energy
        # minus mu_ref*k differs from the physical total by an irrelevant offset.
        idx=model["idx"]
        logw=(model["log_frame"]-(1/t-1/t0)*model["energy"]/mbar.KB
              +(mu-ref)*k[idx]/kt)
        e=model["energy"]-(mu-ref)*k[idx]
        means=[]
        for side in (~selected[idx],selected[idx]):
            f=logw[side]
            means.append(float(np.exp(f-logsumexp(f))@e[side]))
        return dict(temperature_K=t,field_eV=mu,maximum_on_field_boundary=bool(edge),
                    max_log_ordered_over_four_disordered=float(odds(mu)),profile=barrier,
                    phases=details,semigrand_energy_ordered_minus_disordered_meV_atom=1000*(means[1]-means[0])/n)
    first,last=[at(float(t)) for t in temperatures]
    tip=None
    if first["max_log_ordered_over_four_disordered"]>0>last["max_log_ordered_over_four_disordered"]:
        root=brentq(lambda t:at(t)["max_log_ordered_over_four_disordered"],*temperatures,xtol=2e-4)
        tip=at(float(root))
    return dict(Q_divider=cut,endpoints=[first,last],finite_cell_candidate=tip)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests",type=Path,nargs="+")
    p.add_argument("--temps",type=float,nargs=2,required=True)
    p.add_argument("--fields",type=float,nargs=2,required=True)
    p.add_argument("--cuts",type=float,nargs="+",default=[.42,.47,.52,.57])
    p.add_argument("--part",choices=["early","late","all"],default="all")
    p.add_argument("--bandwidth",type=float,default=.04)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    t0,n,shape,cap,axial,runs=load_source(a.manifests)
    if a.part!="all":
        runs=mbar.half_runs(runs,a.part=="early")
    model=reference(n,runs,t0)
    data=dict(source_T_K=t0,N=n,shape=shape,W_max=cap,axial_free=axial,pressure_GPa=0.,
              trajectories=len(runs),time_part=a.part,Q_bandwidth=a.bandwidth,
              manifests=[p.resolve().relative_to(ROOT.parent).as_posix() for p in a.manifests],
              T_search=a.temps,mu_search=a.fields,lambda2=model["lambda2"],
              estimates=[vertex(model,n,t0,c,a.fields,a.temps,a.bandwidth) for c in a.cuts])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
