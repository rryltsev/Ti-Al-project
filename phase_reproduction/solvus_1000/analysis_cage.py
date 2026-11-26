"""Cross-check nested displacement cages with a direct equilibrium probability ratio."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from analysis import KT, aggregate, fit_mbar, height_separator, individual_fit, load_runs, make_data


def conditioned(n, runs, cap, inner, outer):
    k,q,specs,blocks,_,_=make_data(n,runs)
    hist,vs=aggregate(blocks,len(specs))
    h=hist.sum(axis=0)
    v=vs/h
    ref,base,_,_=fit_mbar(k,q,specs,hist)
    sep=height_separator(k,q,base,ref,.04)
    if not sep:
        return {"status":"unresolved_source"}
    fit=individual_fit(n,k,q,base,ref,h,v,.04,4,fixed_cut=sep["order_cut"])
    if not fit:
        return {"status":"unresolved_source"}
    states=k*(3*n+1)+np.rint(q*3*n/4).astype(int)
    counts=np.zeros(len(k))
    vsums=np.zeros(len(k))
    retained=[]
    for r in runs:
        indices=np.searchsorted(states,r["k"]*(3*n+1)+r["code"])
        ar=r["ca"]
        nearest=(4*r["v"]/(np.sqrt(3)*ar))**(1/3)*np.minimum(1,np.sqrt(1/3+ar**2/4))
        # Both ensembles must satisfy the outer cage's stronger minimum-volume rule.
        keep=(r["wall"]<inner) & (nearest*(1-outer)>=1.68052)
        counts+=np.bincount(indices[keep],minlength=len(k))
        vsums+=np.bincount(indices[keep],weights=r["v"][keep],minlength=len(k))
        retained.append(dict(run=r["run"],kept=int(keep.sum()),total=len(keep)))
    active=counts>0
    dest=np.full(len(k),-np.inf)
    dest[active]=base[active]+np.log(counts[active]/h[active])
    cv=np.divide(vsums,counts,out=np.zeros_like(vsums),where=active)
    dm=(fit["mu_eV"]-ref)/KT
    masks=[q<=sep["order_cut"],q>sep["order_cut"]]
    fractions=[]
    ess=[]
    for mask in masks:
        t=dest[mask]+dm*k[mask]
        lnz=logsumexp(t)
        frac=np.exp(lnz-logsumexp(base[mask]+dm*k[mask]))
        fractions.append(float(frac))
        if np.isfinite(lnz):
            p=np.exp(t-lnz)
            ess.append(float(1/np.sum(p**2/np.maximum(counts[mask],1))))
        else:
            ess.append(0.)
    result=dict(source_cap=cap,target_wall=inner,volume_guard_cap=outer,source_mu_eV=fit["mu_eV"],
                source_order_cut=sep["order_cut"],fractions_alpha_alpha2=fractions,
                phase_Kish_ess_retained_frames=ess,selection_by_run=retained)
    if min(fractions)>0:
        delta=-KT*np.log(fractions[1]/fractions[0])
        result["delta_semigrand_alpha2_minus_alpha_eV_cell"]=float(delta)
        result["linearized_delta_mu_eV"]=float(delta/(n*fit["width"]))
        csep=height_separator(k,q,dest,ref,.04,sep)
        result["conditioned_equal_height_separator"]=csep
        result["conditioned_fit"]=(individual_fit(n,k,q,dest,ref,counts,cv,.04,4,
                                                   fixed_cut=csep["order_cut"]) if csep else None)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path,nargs="+")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--inner",type=float,default=.35)
    parser.add_argument("--outer",type=float,default=.40)
    args=parser.parse_args()
    data=load_runs(args.manifest,None,0)
    result={}
    for (n,shape,cap,ax),runs in data.items():
        if cap in (args.inner,args.outer):
            result[f"N{n}_{shape}_cap{cap:g}_axial{ax}"]=conditioned(n,runs,cap,args.inner,args.outer)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
