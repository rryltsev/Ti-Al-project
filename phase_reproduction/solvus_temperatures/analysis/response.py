"""Find maxima of chemical and order susceptibilities over the supported field scan."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks
from scipy.special import logsumexp, softmax

from isotherm import load, mbar


def evaluate(n,k,q,base,h,ref,kt,mu_limits,check_modes=False):
    ks,idx=np.unique(k,return_inverse=True)
    local_mass=np.full(len(ks),-np.inf)
    np.logaddexp.at(local_mass,idx,base)
    finite=np.isfinite(local_mass)
    conditional=np.zeros_like(base)
    take=np.isfinite(base)
    conditional[take]=np.exp(base[take]-local_mass[idx[take]])
    qbar=np.bincount(idx,weights=conditional*q)[finite]
    q2bar=np.bincount(idx,weights=conditional*q*q)[finite]
    ess_inverse=np.bincount(idx,weights=conditional**2/np.maximum(1,h))[finite]
    ks,local_mass=ks[finite],local_mass[finite]
    x=ks/n
    def moments(mu):
        w=softmax(local_mass+(mu-ref)*ks/kt)
        xbar=float(w@x)
        qmean=float(w@qbar)
        return np.array([xbar,qmean,float(w@((x-xbar)**2))*n/kt,
                         n*max(0,float(w@q2bar)-qmean*qmean), 1/float(w**2@ess_inverse)])
    def record(mu):
        xx,qq,chi,vq,ess=moments(mu)
        w=softmax(local_mass+(mu-ref)*ks/kt)
        joint=softmax(base+(mu-ref)*k/kt)
        uq,iq=np.unique(q,return_inverse=True)
        pq=np.bincount(iq,weights=joint)
        cx=np.divide(np.bincount(iq,weights=joint*k/n),pq,
                     out=np.zeros_like(pq),where=pq>0)
        cum=np.cumsum(pq)
        def quartile(lower,upper):
            selected=np.maximum(0,np.minimum(cum,upper)-np.maximum(cum-pq,lower))
            return dict(mean_Q=float(selected@uq/selected.sum()),
                        mean_x_at_percent=float(100*(selected@cx)/selected.sum()))
        raw=np.bincount(ks,weights=w,minlength=n+1)
        # Fixed 0.25 percentage point composition bandwidth, at least 0.75 lattice step.
        std_bins=max(.0025*n,.75)
        density=gaussian_filter1d(raw,std_bins)
        peaks=find_peaks(np.r_[-1,density,-1])[0]-1
        peaks=[p for p in peaks if density[p]>.10*max(density)]
        # All order is included in these composition modes.
        return dict(mu_eV=float(mu),x_Al=xx,mean_Q=qq,dx_dmu_eV_inv=chi,Nvar_Q=vq,
                    low_Q_quartile=quartile(0,.25),high_Q_quartile=quartile(.75,1),
                    correlation_x_Q=float(joint@((k/n-xx)*(q-qq)) /
                                          max(1.e-30,np.sqrt(chi*kt*vq)/n)),
                    prob_Q_above_0p4=float(joint[q>.4].sum()),prob_Q_above_0p6=float(joint[q>.6].sum()),
                    equal_total_conditional_gap_at_most_at_percent=200*np.sqrt(kt*chi/n),
                    Kish_frames=ess,composition_smoothing_sd_at_percent=100*std_bins/n,
                    composition_modes_at_percent=[100*p/n for p in peaks],
                    composition_mode_relative_heights=[float(density[p]/density.max()) for p in peaks])
    xbounds=[moments(mu)[0] for mu in mu_limits]
    def max_on_field(bounds,coord):
        lo,hi=bounds
        grid=np.linspace(lo,hi,max(3,int(np.ceil((hi-lo)/.0005))+1))
        obs=np.array([moments(z)[coord] for z in grid])
        candidates=[(obs[0],lo),(obs[-1],hi)]
        for i in find_peaks(obs)[0]:
            opt=minimize_scalar(lambda z:-moments(z)[coord],bounds=(grid[i-1],grid[i+1]),method="bounded",
                                options=dict(xatol=2.e-10))
            candidates.append((-opt.fun,opt.x))
        fm=max(candidates)[1]
        out=record(fm)
        out["at_field_interval_edge"]=bool(np.min(np.abs(np.array(bounds)-fm))<1e-7)
        return out
    def scan_composition_modes(lo,hi):
        # Compare two distinct, appreciably populated concentration maxima at
        # every field. Ratios and barriers refer to the smoothed finite-cell PMF.
        fields=np.linspace(lo,hi,max(3,int(np.ceil((hi-lo)/.0005))+1))
        kernel=max(.0025*n,.75)
        any_multiple=0
        strongest=None
        widest=None
        for mu in fields:
            w=softmax(local_mass+(mu-ref)*ks/kt)
            density=gaussian_filter1d(np.bincount(ks,weights=w,minlength=n+1),kernel)
            peaks=find_peaks(np.r_[-1,density,-1])[0]-1
            peaks=[i for i in peaks if density[i]>.1*density.max()]
            if len(peaks)>1:
                any_multiple+=1
            for a,b in zip(peaks,peaks[1:]):
                cut=a+int(np.argmin(density[a:b+1]))
                if cut in (a,b) or b-a<2*kernel or not .05<density[:cut+1].sum()<.95:
                    continue
                barrier=float(np.log(min(density[a],density[b])/density[cut]))
                item=dict(mu_eV=float(mu),mean_x=float(w@x),modes_at_percent=[100*a/n,100*b/n],
                          delta_modes_pp=100*(b-a)/n,barrier_from_smaller_peak_kBT=barrier)
                if strongest is None or barrier>strongest["barrier_from_smaller_peak_kBT"]:
                    strongest=item
                if widest is None or b-a>n*widest["delta_modes_pp"]/100+1.e-8:
                    widest=item
        return dict(number_fields=len(fields),fields_with_multiple_modes_above_10pct=any_multiple,
                    strongest_pair=strongest,widest_pair=widest,
                    note="Adjacent peaks with >10% maximum density, separation >=2 smoothing widths, "
                    "and >5% smoothed total mass on each side of the intervening minimum.")
    output={}
    for name,limits in (("experimental_alpha_alpha2",(0.195,.230)),("complete_sampled_fields",tuple(xbounds))):
        lower=max(limits[0],xbounds[0]); upper=min(limits[1],xbounds[1])
        if lower>=upper:
            continue
        def crossing(target,anchor):
            if abs(xbounds[anchor]-target)<1e-12:
                return mu_limits[anchor]
            return brentq(lambda z:moments(z)[0]-target,*mu_limits,xtol=2e-12)
        lo=crossing(lower,0); hi=crossing(upper,1)
        output[name]=dict(x_range=[lower,upper],mu_range=[lo,hi],max_chemical_response=max_on_field((lo,hi),2),
                          max_order_response=max_on_field((lo,hi),3))
        if name=="experimental_alpha_alpha2" and lower<=.2125<=upper:
            output[name]["experimental_midpoint"]=record(crossing(.2125,0))
        if check_modes:
            output[name]["composition_mode_scan"]=scan_composition_modes(lo,hi)
    return output


def main(args):
    result=[]
    for t,n,shape,cap,axial,runs in load(args.manifests,0):
        if args.half:
            runs=mbar.half_runs(runs,args.half=="first")
        k,q,specs,blocks,diag,_=mbar.make_data(n,runs)
        hist,_=mbar.aggregate(blocks,len(specs))
        kt=mbar.KB*t
        ref,base,f,_=mbar.fit_mbar(k,q,specs,hist,kt=kt)
        h=hist.sum(axis=0)
        # Limit scientific interpretation to the fields sampled in the merged set.
        bounds=args.field_bounds or [float(specs[:,0].min()),float(specs[:,0].max())]
        stat=evaluate(n,k,q,base,h,ref,kt,bounds,check_modes=True)
        boot_diag={}
        if args.bootstrap:
            rng=np.random.default_rng(args.seed)
            for trial in range(args.bootstrap):
                bh,_=mbar.aggregate(blocks,len(specs),rng)
                br,bb,_,_=mbar.fit_mbar(k,q,specs,bh,f,kt=kt)
                fit=evaluate(n,k,q,bb,bh.sum(axis=0),br,kt,bounds)
                for region,data in fit.items():
                    boot_diag.setdefault(region,[]).append([data["max_chemical_response"]["dx_dmu_eV_inv"],
                                                             data["max_order_response"]["Nvar_Q"]])
        result.append(dict(temperature_K=t,N=n,cap=cap,axial_free=axial,production_half=args.half,statistics=stat,
                           diagnostic_dx_dmu_for_3p5pp_equal_total=n*.035**2/(4*kt),
                           finite_size_mu_scale_for_3p5pp_eV=kt/(n*.035),bootstrap_count=args.bootstrap,
                           bootstrap_max_chi_x_and_NvarQ_95={reg:np.quantile(vals,[.025,.975],axis=0).T.tolist()
                                                           for reg,vals in boot_diag.items()},
                           note="Resolved maxima at fixed reweighted fields on a <=0.0005 eV search mesh with local "
                           "refinement. Composition modes are Gaussian-smoothed peaks above 10% of the dominant peak. "
                           "The susceptibility scale N(0.035)^2/(4kBT) assumes equally populated phases, and gives "
                           "their between-phase contribution only; a variance bound on conditional means applies to "
                           "the recovered finite-size distribution. Neither is a confidence limit on a bulk gap."))
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests",nargs="+",type=Path)
    p.add_argument("--field-bounds",nargs=2,type=float)
    p.add_argument("--bootstrap",type=int,default=0)
    p.add_argument("--half",choices=("first","last"))
    p.add_argument("--seed",type=int,default=31735)
    p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
