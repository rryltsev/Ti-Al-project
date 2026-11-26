"""Direct same-temperature MBAR of the matched 2304-site Ti–Al cage study."""
import argparse
import csv
from pathlib import Path
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np
from scipy.optimize import brentq,minimize_scalar
from scipy.signal import find_peaks
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import logsumexp,softmax

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
sys.path.insert(0,str(STUDY.parent/"solvus_closure"/"analysis"))
from closure import load,mbar,order_moments,energy_blocks,draw_blocks,separated_peaks
from resampled_mbar import ResampledMBAR
from hysteresis import record as kinetic_record

CUTS=(.32,.37,.42,.47,.50)
PRIMARY=.37
CHANNELS=("mu_eV","x_low","x_high","gap","score","delta_h_meV","Q_low","Q_high")


class DisconnectedWindows(ValueError):
    """The observed joint (x,Q) window graph cannot normalize all sampled groups."""


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value,allow_nan=False,indent=2)+"\n",encoding="utf8")
    tmp.replace(path)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def discover(base):
    grouped=defaultdict(list)
    incomplete=[]
    for manifest in sorted(base.glob("*/windows.csv")):
        with manifest.open(encoding="utf-8-sig",newline="") as stream:
            records=list(csv.DictReader(stream))
        statusfile=manifest.parent/"window_status.json"
        if not statusfile.is_file():
            incomplete.append(manifest.parent.name)
            continue
        status=load_json(statusfile)["windows"]
        names={r["run"] for r in records}
        if names!=set(status) or not all(status[n].get("status")=="ok" and status[n].get("exit_code")==0 and
                                         (manifest.parent/(n+".csv")).is_file() for n in names):
            incomplete.append(manifest.parent.name)
            continue
        settings={(float(r["T_K"]),float(r["cap"]),int(r["N"]),int(r["axial_free"])) for r in records}
        if len(settings)!=1:
            raise ValueError(f"Mixed settings in {manifest}")
        t,cap,n,axial=settings.pop()
        if n!=2304 or axial!=1:
            raise ValueError(f"Unexpected geometry in {manifest}")
        grouped[t,cap].append(manifest)
    return grouped,incomplete


class Odds:
    def __init__(self,n,t,k,q,base,h,ref,fields,energy,ineff=None):
        self.n,self.kt=n,mbar.KB*t
        self.k,self.q,self.base,self.h,self.ref=k,q,base,h,ref
        self.fields=np.asarray(fields)
        self.energy,self.ineff=energy,ineff
        self.ks,self.inverse=np.unique(k,return_inverse=True)
        self.mesh=np.unique(np.r_[np.linspace(self.fields[0],self.fields[-1],43),self.fields])

    def at(self,mu,cut,detail):
        w=softmax(self.base+(mu-self.ref)*self.k/self.kt)
        hi=self.q>cut
        statistics=[]
        for mask in (~hi,hi):
            mass=w[mask].sum()
            v=w[mask]/mass
            entry=dict(x=float(v@self.k[mask]/self.n),Q=float(v@self.q[mask]),u=float(v@self.energy[mask]))
            if detail and self.ineff is not None:
                ww=np.divide(v,self.h[mask],out=np.zeros_like(v),where=self.h[mask]>0)
                entry["time_ESS"]=float(1/(self.ineff[mask]@(ww*ww)))
            statistics.append(entry)
        lo,up=statistics
        gap=up["x"]-lo["x"]
        row=dict(mu_eV=float(mu),x=float(w@self.k/self.n),x_low=lo["x"],x_high=up["x"],gap=gap,
                 score=float(np.log(w[hi].sum()/w[~hi].sum()/4)),delta_h_meV=1000*(up["u"]-lo["u"]-mu*gap),
                 Q_low=lo["Q"],Q_high=up["Q"])
        temperature=self.kt/mbar.KB
        row["dS_dmu_eV_inv"]=self.n*gap/self.kt
        row["mu_local_linear_zero"]=mu-row["score"]/row["dS_dmu_eV_inv"] if abs(row["dS_dmu_eV_inv"])>1.e-6 else None
        row["dS_dT_fixed_mu"]=self.n*.001*row["delta_h_meV"]/(mbar.KB*temperature**2)
        row["T_local_linear_zero"]=temperature-row["score"]/row["dS_dT_fixed_mu"] if row["dS_dT_fixed_mu"] else None
        if detail:
            row["conditional"]=statistics
            row["profiles"]={str(bw):separated_peaks(self.q,w,bw,cut) for bw in (.03,.04,.05)}
            v=np.divide(w,self.h,out=np.zeros_like(w),where=self.h>0)
            row["total_time_ESS"]=float(1/(self.ineff@(v*v))) if self.ineff is not None else None
        return row

    def fit(self,cut,detail=False):
        mask=self.q>cut
        spectra=[]
        for side in (~mask,mask):
            a=np.full(len(self.ks),-np.inf)
            np.logaddexp.at(a,self.inverse[side],self.base[side])
            spectra.append(a)
        def score(mu):
            z=(mu-self.ref)*self.ks/self.kt
            return float(logsumexp(spectra[1]+z)-logsumexp(spectra[0]+z)-np.log(4))
        s=np.asarray([score(m) for m in self.mesh])
        maxima=[(s[0],self.mesh[0]),(s[-1],self.mesh[-1])]
        for i in find_peaks(s)[0]:
            opt=minimize_scalar(lambda m:-score(m),method="bounded",
                                bounds=self.mesh[[i-1,i+1]],options={"xatol":1.e-10})
            maxima.append((-opt.fun,opt.x))
        _,mu=max(maxima)
        apex=self.at(mu,cut,detail)
        # Include every extremum in the search to resolve close roots in one mesh interval.
        probes=np.unique(np.r_[self.mesh,[v for _,v in maxima]])
        scores=np.array([score(v) for v in probes])
        crossings=[]
        for i in np.flatnonzero(scores[:-1]*scores[1:]<0):
            m=brentq(score,probes[i],probes[i+1],xtol=1.e-12)
            row=self.at(m,cut,detail)
            row["branch"]="Ti" if row["gap"]>0 else "Al"
            row["direct_field_bracket_eV"]=[float(self.fields[self.fields<=m].max()),float(self.fields[self.fields>=m].min())]
            crossings.append(row)
        samples=[self.at(m,cut,True) for m in self.fields] if detail else [
                    dict(mu_eV=float(m),score=score(m)) for m in self.fields]
        return dict(cut=cut,maximum=apex,maximum_on_edge=bool(np.isclose(mu,self.fields[0],atol=2.e-7,rtol=0) or
                                                               np.isclose(mu,self.fields[-1],atol=2.e-7,rtol=0)),
                    sampled_field_scores=samples,
                    crossings=crossings)


def compact(z,sampler,cut):
    a=z["maximum"]
    def heights(row):
        w=softmax(sampler.base+(row["mu_eV"]-sampler.ref)*sampler.k/sampler.kt)
        pair=separated_peaks(sampler.q,w,.04,cut)
        return [*pair["modes_Q"],pair["cut"],pair["barrier"]] if pair else None
    return dict(apex=[a[k] for k in CHANNELS],apex_profile=heights(a),maximum_on_edge=z["maximum_on_edge"],
                edge_scores=[z["sampled_field_scores"][i]["score"] for i in (0,-1)],
                actual_field_scores=[s["score"] for s in z["sampled_field_scores"]],
                root_profiles={r["branch"]:heights(r) for r in z["crossings"]},
                roots={r["branch"]:[r[k] for k in CHANNELS] for r in z["crossings"]})


def summarize(reps):
    def stats(a):
        if not a: return None
        return {"count":len(a),"interval_95":np.quantile(a,[.025,.975],axis=0).T.tolist()}
    return dict(columns=list(CHANNELS),replicates=reps,apex=stats([r["apex"] for r in reps]),
                profile_columns=["Q_mode_low","Q_mode_high","Q_valley","B_from_smaller_peak_kBT"],
                apex_profile=stats([r["apex_profile"] for r in reps if r["apex_profile"]]),
                root_profiles={b:stats([r["root_profiles"][b] for r in reps if r["root_profiles"].get(b)])
                               for b in ("Ti","Al")},
                fractional_positive_S=float(np.mean([r["apex"][4]>0 for r in reps])),
                fractional_interior_maximum=float(np.mean([not r["maximum_on_edge"] for r in reps])),
                fractional_both_roots_inside=float(np.mean([len(r["roots"])==2 for r in reps])),
                edge_scores=stats([r["edge_scores"] for r in reps]),
                actual_field_scores=stats([r["actual_field_scores"] for r in reps]),
                branches={b:stats([r["roots"][b] for r in reps if b in r["roots"]]) for b in ("Ti","Al")})


def fit_runs(t,n,runs,boot,halves,seed,cuts=CUTS):
    k,q,spec,blocks,diag,aux=mbar.make_data(n,runs)
    hist,_=mbar.aggregate(blocks,len(spec))
    h=hist.sum(axis=0)
    kt=t*mbar.KB
    bc=mbar.connectivity(hist,k,q,n)
    tree=minimum_spanning_tree(-bc).data
    if len(hist)>1 and len(tree)!=len(hist)-1:
        raise DisconnectedWindows("Disjoint observed (x,Q) window groups; evaluate independent fields until a connecting group arrives.")
    ref,base,offset,overlap=mbar.fit_mbar(k,q,spec,hist,kt=kt)
    _,_,energy=order_moments(n,runs,k,q,h)
    g=sum(d["statistical_inefficiency"]*np.asarray(v[1].sum(axis=0)).ravel() for d,v in zip(diag,blocks))
    fields=np.unique(spec[:,0])
    solver=Odds(n,t,k,q,base,h,ref,fields,energy,g)
    results={str(c):solver.fit(c,True) for c in cuts}
    # Determine a data-dependent divider from the local density valley near the primary maximum.
    profile=results.get(str(PRIMARY),{}).get("maximum",{}).get("profiles",{}).get("0.04")
    if profile:
        valley=profile["cut"]
        if str(valley) not in results:
            results[str(valley)]=solver.fit(valley,True)
    out=dict(T_K=t,N=n,direct_fields_eV=fields.tolist(),direct_runs=len(runs),frames=int(h.sum()),
             least_BC_tree=float(min(-tree)) if len(tree)==len(hist)-1 and len(tree) else None,
             second_overlap_eigenvalue=float(np.sort(np.linalg.eigvals(overlap).real)[-2]) if len(hist)>1 else None,
             few_blocks=[r["run"] for r in diag if r["blocks"]<3],physical_valley_Q=profile["cut"] if profile else None,
             fixed_cut=results)
    if halves:
        out["halves"]={name:fit_runs(t,n,mbar.half_runs(runs,first),0,False,seed,(PRIMARY,.47))["fixed_cut"]
                       for name,first in (("first",True),("last",False))}
    if boot:
        eblocks=energy_blocks(n,k,q,runs,blocks)
        msolver=ResampledMBAR(k,q,spec,kt,ref,offset,hist)
        rng=np.random.default_rng(seed)
        reps={key:[] for key in results}
        failures=0
        for _ in range(boot):
            bh,_,be=draw_blocks(blocks,eblocks,len(spec),rng)
            hh=bh.sum(axis=0);bref=ref
            try:
                bb=msolver.fit(bh)
            except (ArithmeticError,np.linalg.LinAlgError):
                bref,bb,_,_=mbar.fit_mbar(k,q,spec,bh,offset,kt=kt)
                failures+=1
            sampler=Odds(n,t,k,q,bb,hh,bref,fields,np.divide(be,hh,out=np.zeros_like(be),where=hh>0))
            for key in results:
                cut=float(key)
                reps[key].append(compact(sampler.fit(cut),sampler,cut))
        out["bootstrap"]=dict(count=boot,seed=seed,raw_solver_fallbacks=failures,cuts={k:summarize(r) for k,r in reps.items()})
    return out


def manual(manifests,out,boot,halves,seed):
    groups=list(load(manifests,0))
    if len(groups)!=1:raise ValueError("Use one T, W, N and cell-shape model per fit.")
    t,n,shape,cap,axial,runs=groups[0]
    biased=[r for r in runs if r["spec"][1]>0]
    free=[r for r in runs if r["spec"][1]==0]
    rec=fit_runs(t,n,biased,boot,halves,seed) if biased else {"T_K":t,"N":n}
    rec.update(cap=cap,shape=shape,axial_free=axial,manifests=[str(p.resolve()) for p in manifests],
               K0_independent=[kinetic_record(n,r) for r in free])
    atomic_json(out,rec)
    return rec


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests",nargs="+",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=64)
    p.add_argument("--halves",action="store_true")
    p.add_argument("--seed",type=int,default=341247)
    args=p.parse_args()
    manual(args.manifests,args.out,args.bootstrap,args.halves,args.seed)
