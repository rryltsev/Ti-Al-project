"""Composition widths of the Q=.37 conditional contacts from the 1245 K N2304 sample."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from thermal_check import load_source, reference, tilted_weights, mbar
from resampled_mbar import ResampledMBAR

TEMPERATURES = (1240., 1242., 1243., 1244., 1245.)
MU_DATA = (3.535, 3.555)
MU_SEARCH = (3.49, 3.60)
CUT = .37


def contacts(model, n, t, base, inverse=None):
    k, q, ref = model["k"], model["q"], model["ref"]
    high = q > CUT
    binned = []
    for group in (~high, high):
        mass = np.full(n+1, -np.inf)
        np.logaddexp.at(mass, k[group], base[group])
        valid = np.isfinite(mass)
        binned.append((np.flatnonzero(valid), mass[valid]))
    kt = mbar.KB*t
    def S(mu):
        logz = [logsumexp(mass+(mu-ref)/kt*(key-n*.27)) for key,mass in binned]
        return float(logz[1]-logz[0]-np.log(4.))
    # Track the local maximum nearest the central field even when the estimated
    # maximum moves beyond the sampled interval in a bootstrap draw.
    probe=np.linspace(*MU_SEARCH,45)
    curve=np.array([S(m) for m in probe])
    maxima=np.flatnonzero((curve[1:-1]>=curve[:-2]) & (curve[1:-1]>=curve[2:]))+1
    if not len(maxima):
        return dict(T_K=t,peak_mu_eV=None,peak_log_odds_over_four=None,
                    peak_within_simulated_fields=False,branches={"Ti_rich":None,"Al_rich":None})
    nearest=min(maxima,key=lambda j:abs(probe[j]-ref))
    peak = minimize_scalar(lambda mu: -S(mu),bounds=probe[[nearest-1,nearest+1]],method="bounded",options={"xatol":1e-12})
    mu0 = float(peak.x)
    out = {"T_K":t,"peak_mu_eV":mu0,"peak_log_odds_over_four":S(mu0),
           "peak_within_simulated_fields":MU_DATA[0]<=mu0<=MU_DATA[1],"branches":{}}
    for side,end in [("Ti_rich",MU_SEARCH[0]),("Al_rich",MU_SEARCH[1])]:
        out["branches"][side] = None
        if out["peak_log_odds_over_four"] <= 0:
            continue
        # Moving outwards from the measured maximum selects the nearest root.
        prev = mu0
        for mu in np.linspace(mu0,end,33)[1:]:
            if S(mu) <= 0:
                r = brentq(S,*sorted((prev,mu)),xtol=2.e-11)
                x,ess = [],[]
                for group in (~high,high):
                    z = base[group]+(r-ref)/kt*(k[group]-n*.27)
                    w = np.exp(z-logsumexp(z))
                    x.append(float(100*(w@k[group])/n))
                    if inverse is not None:
                        ess.append(float(1/((w*w)@inverse[group])))
                gap=x[1]-x[0]
                dist=max(MU_DATA[0]-r,r-MU_DATA[1],0.)
                out["branches"][side]=dict(mu_eV=r,x_alpha_atpct=x[0],x_alpha2_atpct=x[1],
                    signed_width_pp=gap,width_pp=abs(gap),inside_measured_mu=dist==0,
                    extrapolation_meV=1000*dist,phase_ESS_alpha_alpha2=ess or None)
                break
            prev=mu
    return out


def resamples(model,n,t0,draws,seed):
    idx=model["idx"]
    source=np.empty_like(idx)
    block_index=np.empty_like(idx)
    counts_by_run=[]
    blocks_total=0
    for lo,hi,state,g in model["segments"]:
        source[lo:hi]=state
        nb=max(1,(hi-lo)//int(np.ceil(2*g)))
        for j,ix in enumerate(np.array_split(np.arange(lo,hi),nb)):
            block_index[ix]=blocks_total+j
        blocks_total+=nb
        counts_by_run.append(nb)
    count=model["hist"]
    k,q,ref,specs=model["k"],model["q"],model["ref"],model["specs"]
    solver=ResampledMBAR(k,q,specs,mbar.KB*t0,ref,model["offsets"],count)
    rng=np.random.default_rng(seed)
    rows={t:[] for t in TEMPERATURES}
    for _ in range(draws):
        copies=np.concatenate([np.bincount(rng.integers(nb,size=nb),minlength=nb) for nb in counts_by_run])
        multiplicity=copies[block_index]
        hist=np.bincount(source*len(k)+idx,weights=multiplicity,minlength=count.size).reshape(count.shape)
        try:
            logmass=solver.fit(hist)
        except (ArithmeticError,np.linalg.LinAlgError):
            _,logmass,_,_=mbar.fit_mbar(k,q,specs,hist,model["offsets"],kt=mbar.KB*t0)
        present=multiplicity>0
        group=idx[present]
        logframe=logmass[group]-np.log(hist.sum(axis=0)[group])+np.log(multiplicity[present])
        for t in TEMPERATURES:
            base=np.full(len(k),-np.inf)
            np.logaddexp.at(base,group,logframe-(1/t-1/t0)/mbar.KB*model["energy"][present])
            rows[t].append(contacts(model,n,t,base))
    results={}
    for t,values in rows.items():
        summary={"draws":len(values),"both_roots_found":sum(all(x["branches"].values()) for x in values),
                 "both_roots_inside_sampled_fields":sum(all(v and v["inside_measured_mu"] for v in x["branches"].values()) for x in values),
                 "peak_inside_sampled_fields":sum(x["peak_within_simulated_fields"] for x in values)}
        for side in ("Ti_rich","Al_rich"):
            found=[r["branches"][side] for r in values if r["branches"][side] is not None]
            summary[side]=dict(root_found=len(found),inside_sampled_fields=sum(r["inside_measured_mu"] for r in found),
                conditional_intervals_2p5_50_97p5={k:np.quantile([r[k] for r in found],[.025,.5,.975]).tolist()
                                                      for k in ("mu_eV","width_pp","x_alpha_atpct","x_alpha2_atpct")}
                                                    if found else None)
        results[t]=summary
    return dict(draws=draws,seed=seed,blocks_per_trajectory=counts_by_run,intervals=results,
                method="Sample contiguous time blocks with replacement separately within each chain; target block length >=2g. Each replica refits the source MBAR offsets before temperature reweighting.")


def main(draws):
    manifests=[HERE.parent/"atomistic"/"series"/name/"windows.csv" for name in ["zc4523c","zc4523s"]]
    t0,n,_,_,_,runs=load_source(manifests)
    model=reference(n,runs,t0)
    assert n==2304 and t0==1245 and len(model["idx"])==38400
    result=dict(source_T_K=t0,N=n,pressure_GPa=0,W_max=.40,cut_Q=CUT,production_frames=len(model["idx"]),
        source_mu_eV=sorted(set(float(s[0]) for s in model["specs"])),search_bounds_eV=MU_SEARCH,
        scope="Finite-cell conditional projected contacts. Interior refers to chemical field only; all T other than 1245 K are reweighted. Width denotes the absolute difference of the two conditional Al compositions on one flank, in percentage points.",
        temperatures=[contacts(model,n,t,*tilted_weights(model,t0,t)) for t in TEMPERATURES])
    outfile=HERE/"n2304_widths.json"
    if draws:
        result["bootstrap"]=resamples(model,n,t0,draws,3129243)
    temp=outfile.with_suffix(".json.part")
    temp.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(outfile)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draws",type=int,default=64)
    with threadpool_limits(limits=1):
        main(p.parse_args().draws)
