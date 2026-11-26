"""Direct isotherms on the MTP. Search both sides of the ordered region in field."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq,minimize_scalar
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import logsumexp

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"solvus_temperatures"/"analysis"))
from isotherm import load,mbar
from resampled_mbar import ResampledMBAR


def separated_peaks(q,w,bw,cut):
    grid,density,peaks=mbar.order_profile(q,w,bw)
    left=peaks[grid[peaks]<cut]; right=peaks[grid[peaks]>cut]
    if not len(left) or not len(right):
        return None
    a=left[np.argmax(density[left])]; b=right[np.argmax(density[right])]
    j=a+np.argmin(density[a:b+1])
    if not a<j<b or grid[b]-grid[a]<2*bw or density[j]<=0:
        return None
    barriers=np.log(density[[a,b]]/density[j])
    return dict(modes_Q=grid[[a,b]].tolist(),cut=float(grid[j]),barrier=float(min(barriers)),
                barriers_from_peaks=barriers.tolist())


class Isotherm:
    def __init__(self,n,t,k,q,base,h,ref,volume,bounds,bw,frame_ineff,mom,aux=None,energy=None):
        self.n,self.kt=n,mbar.KB*t
        self.k,self.q,self.base,self.h,self.ref=k,q,base,h,ref
        self.volume,self.bounds,self.bw=volume,bounds,bw
        self.frame_ineff,self.mom=frame_ineff,mom
        self.aux=aux
        self.energy=energy if energy is not None else (mom[2] if mom is not None else None)

    def weights(self,mu):
        u=self.base+(mu-self.ref)*self.k/self.kt
        return np.exp(u-logsumexp(u))

    def point(self,mu):
        w=self.weights(mu)
        k,q,n=self.k,self.q,self.n
        x=float(w@k/n); qm=float(w@q)
        profile,density,peaks=mbar.order_profile(q,w,self.bw)
        pair=None
        for i,a in enumerate(peaks):
            for b in peaks[i+1:]:
                if profile[b]-profile[a]<2*self.bw:
                    continue
                j=a+np.argmin(density[a:b+1])
                if j in (a,b) or density[j]<=0:
                    continue
                high=q>profile[j]
                ph=float(w[high].sum())
                if not .05<ph<.95:
                    continue
                barrier=float(np.log(min(density[a],density[b])/density[j]))
                if pair is None or barrier>pair["barrier"]:
                    pair=dict(modes_Q=profile[[a,b]].tolist(),cut=float(profile[j]),barrier=barrier,
                              prob_high=ph,x_low=float(w[~high]@k[~high]/(1-ph)/n),
                              x_high=float(w[high]@k[high]/ph/n))
        dxsd=float(np.sqrt(w@((k/n-x)**2)))
        xpdf=gaussian_filter1d(np.bincount(k,weights=w,minlength=n+1),max(.75,.0025*n))
        xp=find_peaks(np.r_[-1,xpdf,-1])[0]-1
        xp=xp[xpdf[xp]>.1*xpdf.max()]
        perframe=np.divide(w,self.h,out=np.zeros_like(w),where=self.h>0)
        out=dict(mu=float(mu),x=x,Q=qm,NvarQ=float(n*(w@(q-qm)**2)),chi_x=n*dxsd**2/self.kt,
                 x_modes=(xp/n).tolist(),x_sd=dxsd,pair=pair,
                 ess_frames=float(1/(self.h@perframe**2)))
        if self.frame_ineff is not None:
            out["ess_time_approx"]=float(1/(self.frame_ineff@(perframe**2)))
        if self.mom is not None:
            s2,s4,_=self.mom
            m2=float(w@s2)
            out["vector_order_rms"]=np.sqrt(m2)
            out["vector_order_moment_ratio"]=float((w@s4)/(m2*m2))
        return out

    def branch(self,direction,anchor=None):
        k,q,b,ref=self.k,self.q,self.base,self.ref
        opts=dict(kt=self.kt,field_bounds=self.bounds,direction=direction)
        peak=mbar.height_separator(k,q,b,ref,self.bw,anchor,**opts)
        out=dict(equal_height=peak)
        if peak is None:
            return out
        for g,label in ((4,"equal_variant"),(1,"equal_total")):
            fit=mbar.individual_fit(self.n,k,q,b,ref,self.h,self.volume,self.bw,g,
                                    fixed_cut=peak["order_cut"],**opts)
            if fit and self.frame_ineff is not None:
                w=self.weights(fit["mu_eV"])
                hi=q>fit["order_cut"]
                ess=[]
                for mask in (~hi,hi):
                    ww=np.divide(w[mask]/w[mask].sum(),self.h[mask],out=np.zeros(mask.sum()),
                                 where=self.h[mask]>0)
                    ess.append(float(1/(self.frame_ineff[mask]@(ww**2))))
                fit["phase_time_ess_approx"]=ess
            out[label]=fit
        return out

    def apex(self,cut,fields):
        # At fixed order separator d ln(P_hi/P_lo)/dmu=N(x_hi-x_lo)/kBT.
        # Its interior maximum can be meaningful even if both compositions agree.
        mask=self.q>cut
        if not (np.isfinite(self.base[mask]).any() and np.isfinite(self.base[~mask]).any()):
            return None
        ks,ki=np.unique(self.k,return_inverse=True)
        spectra=[]
        for sel in (~mask,mask):
            spec=np.full(len(ks),-np.inf)
            np.logaddexp.at(spec,ki[sel],self.base[sel])
            spectra.append(spec)
        def score(mu):
            tilt=(mu-self.ref)*ks/self.kt
            return float(logsumexp(spectra[1]+tilt)-logsumexp(spectra[0]+tilt)-np.log(4.))
        y=np.array([score(mu) for mu in fields])
        candidates=[(y[0],fields[0]),(y[-1],fields[-1])]
        for i in find_peaks(y)[0]:
            fit=minimize_scalar(lambda v:-score(v),bounds=(fields[i-1],fields[i+1]),
                                method="bounded",options={"xatol":1.e-9})
            candidates.append((-fit.fun,fit.x))
        val,mu=max(candidates)
        crossings=[]
        mesh=np.unique(np.r_[fields,[v for _,v in candidates]])
        scores=np.interp(mesh,fields,y)
        for value,pos in candidates:
            scores[np.argmin(abs(mesh-pos))]=value
        for index in np.flatnonzero(scores[:-1]*scores[1:]<0):
            at=brentq(score,mesh[index],mesh[index+1],xtol=1.e-11)
            mass=self.weights(at)
            x0=float(mass[~mask]@self.k[~mask]/mass[~mask].sum()/self.n)
            x1=float(mass[mask]@self.k[mask]/mass[mask].sum()/self.n)
            row=dict(mu_eV=float(at),x_lowQ=x0,x_highQ=x1,signed_gap=x1-x0,
                     branch="Ti_rich" if x1>x0 else "Al_rich")
            if self.mom is not None:
                ptroot=self.point(at)
                pair=ptroot["pair"]
                row["pair"]=pair
                row["cut_between_modes"]=bool(pair and pair["modes_Q"][0]<cut<pair["modes_Q"][1])
                row["time_ESS"]=ptroot.get("ess_time_approx")
            crossings.append(row)
        pt=self.point(mu)
        w=self.weights(mu)
        cond=[]
        for sel in (~mask,mask):
            ws=w[sel]/w[sel].sum()
            row=dict(x=float(ws@self.k[sel]/self.n),Q=float(ws@self.q[sel]))
            row["chi_x_eV_inv"]=float(self.n*(ws@((self.k[sel]/self.n-row["x"])**2))/self.kt)
            if self.energy is not None:
                row["U_eV_atom"]=float(ws@self.energy[sel])
            if self.mom is not None:
                row["volume_A3_atom"]=float(ws@self.volume[sel])
            if self.aux is not None:
                row["c_a"]=float(ws@self.aux["mean_c_a"][sel])
                row["ranked_Al_sublattice_fractions"]=[float(ws@self.aux[f"Al_fraction_sublattice_rank_{i}"][sel])
                                                      for i in range(1,5)]
            if self.frame_ineff is not None:
                frame=np.divide(ws,self.h[sel],out=np.zeros_like(ws),where=self.h[sel]>0)
                row["approx_time_ESS"]=float(1/(self.frame_ineff[sel]@(frame**2)))
            cond.append(row)
        out=dict(cut_Q=float(cut),mu=float(mu),x=pt["x"],at_field_edge=bool(np.min(abs(np.array(self.bounds)-mu))<1.e-6),
                    log_weight_per_variant=float(val),delta_omega_meV_atom=-1000*self.kt*val/self.n,
                    conditional_low_high=cond,pair_at_max=pt["pair"],fixed_cut_crossings=crossings)
        out["d2_log_odds_dmu2_eV_inv2"]=self.n*(cond[1]["chi_x_eV_inv"]-cond[0]["chi_x_eV_inv"])/self.kt
        if self.energy is not None:
            lo,hi=cond
            out["delta_U_minus_mu_x_meV_atom"]=1000*(hi["U_eV_atom"]-lo["U_eV_atom"]-mu*(hi["x"]-lo["x"]))
        bandwidths=sorted(set([.03,self.bw,.05])) if self.mom is not None else [self.bw]
        out["basin_profiles_by_bandwidth_Q"]={str(bw):separated_peaks(self.q,w,bw,cut) for bw in bandwidths}
        return out


def order_moments(n,runs,k,q,h):
    # Sorting loses orientation only, leaving the norm of the 3-component order intact.
    code=k*(3*n+1)+np.rint(q*(3*n)/4).astype(int)
    s2sum=np.zeros(len(k)); s4sum=np.zeros(len(k)); esum=np.zeros(len(k))
    for r in runs:
        idx=np.searchsorted(code,r["k"]*(3*n+1)+r["code"])
        deviations=4*r["occ_sorted"]/n-r["k"][:,None]/n
        z=np.sum(deviations**2,axis=1)
        s2sum+=np.bincount(idx,weights=z,minlength=len(k))
        s4sum+=np.bincount(idx,weights=z*z,minlength=len(k))
        if "energy_per_atom" not in r:
            with open(r["path"],encoding="utf8") as fp:
                header=next(csv.reader(fp))
            raw=np.loadtxt(r["path"],delimiter=",",skiprows=1,usecols=[header.index("sweep"),header.index("E_eV")],ndmin=2)
            which=np.searchsorted(raw[:,0],r["sweep"])
            if not np.array_equal(raw[which,0],r["sweep"]):
                raise ValueError("Energy and occupation recorded at different sweeps")
            r["energy_per_atom"]=raw[which,1]/n
        esum+=np.bincount(idx,weights=r["energy_per_atom"],minlength=len(k))
    return s2sum/h,s4sum/h,esum/h


def energy_blocks(n,k,q,runs,blocks):
    code=k*(3*n+1)+np.rint(q*(3*n)/4).astype(int)
    energy=[]
    for run,(_,hist,_) in zip(runs,blocks):
        idx=np.searchsorted(code,run["k"]*(3*n+1)+run["code"])
        sizes=np.asarray(hist.sum(axis=1)).ravel().astype(int)
        row=np.repeat(np.arange(len(sizes)),sizes)
        energy.append(csr_matrix((run["energy_per_atom"],(row,idx)),shape=hist.shape))
    return energy


def draw_blocks(blocks,energy,nspec,rng):
    hist=np.zeros((nspec,blocks[0][1].shape[1]))
    vtotal=np.zeros(hist.shape[1]); etotal=np.zeros(hist.shape[1])
    for (i,h,v),e in zip(blocks,energy):
        b=h.shape[0]
        copies=np.bincount(rng.integers(b,size=b),minlength=b)
        hist[i]+=h.T@copies
        vtotal+=v.T@copies
        etotal+=e.T@copies
    return hist,vtotal,etotal


def analyze(t,n,runs,bw,bounds,bootstrap,seed,halves):
    k,q,specs,blocks,diag,moments=mbar.make_data(n,runs)
    hist,vsum=mbar.aggregate(blocks,len(specs))
    h=hist.sum(axis=0); kt=mbar.KB*t
    ref,base,offset,overlap=mbar.fit_mbar(k,q,specs,hist,kt=kt)
    ineff=np.array([d["statistical_inefficiency"] for d in diag])
    per_run=np.array([np.asarray(block[1].sum(axis=0)).ravel() for block in blocks])
    isoth=Isotherm(n,t,k,q,base,h,ref,vsum/h,bounds,bw,ineff@per_run,order_moments(n,runs,k,q,h),
                   {name:arr/h for name,arr in moments.items()})
    fields=np.unique(np.r_[np.linspace(*bounds,max(101,int(np.ceil((bounds[1]-bounds[0])/.001))+1)),specs[:,0]])
    fields=fields[(fields>=bounds[0])&(fields<=bounds[1])]
    trace=[isoth.point(x) for x in fields]
    candidates=[r for r in trace if r["pair"] is not None]
    strongest=max(candidates,key=lambda z:z["pair"]["barrier"],default=None)
    bc=mbar.connectivity(hist,k,q,n)
    tree=minimum_spanning_tree(-bc).data
    branches={k:isoth.branch(d) for k,d in (("Ti_rich",1),("Al_rich",-1))}
    cuts=[.42,.47,.5]
    if strongest:
        pair=strongest["pair"]
        cuts.extend(c for c in (pair["cut"]-.04,pair["cut"],pair["cut"]+.04)
                    if pair["modes_Q"][0]<c<pair["modes_Q"][1])
    out=dict(T_K=t,N=n,mu_range_eV=list(bounds),direct_fields_eV=np.unique(specs[:,0]).tolist(),
             x_range=[trace[0]["x"],trace[-1]["x"]],
             n_runs=len(runs),frames=int(h.sum()),bandwidth_Q=bw,
             series=sorted({r["run"].split("-h")[0] for r in runs}),
             BC_tree_min=float(np.min(-tree)) if len(tree)==len(specs)-1 and len(tree) else None,
             second_MBAR_eigenvalue=float(np.sort(np.linalg.eigvals(overlap).real)[-2]) if len(specs)>1 else None,
             few_blocks=[d["run"] for d in diag if d["blocks"]<3],branches=branches,
             max_chemical_response=max(trace,key=lambda z:z["chi_x"]),
             max_order_response=max(trace,key=lambda z:z["NvarQ"]),strongest_Q_pair=strongest,
             apex=[isoth.apex(c,fields) for c in sorted(set(cuts))],grid=trace)
    out["geometry_controls"]=dict(min_distance_to_lower_volume_A3_atom=min(d["min_distance_to_lower_volume_A3_atom"] for d in diag),
        max_fraction_near_bounds={field:max(d["observable_means"][field] for d in diag)
                                  for field in ("fraction_within_0p25_A3_lower_volume","fraction_within_0p25_A3_upper_volume",
                                                "fraction_within_0p01_axial_bounds")})
    if halves:
        out["halves"]={}
        for name,which in (("first",True),("last",False)):
            rr=mbar.half_runs(runs,which)
            local=analyze(t,n,rr,bw,bounds,0,seed,False)
            out["halves"][name]={a:local[a] for a in ("branches","max_chemical_response","max_order_response","strongest_Q_pair","apex")}
    if bootstrap:
        rng=np.random.default_rng(seed)
        reps={tag:[] for tag in branches}
        apex_cuts=sorted(set([.42,.47,.50,strongest["pair"]["cut"] if strongest else .50]))
        apex_reps={float(c):[] for c in apex_cuts}
        peak_reps={float(c):[] for c in apex_cuts}
        fixed_contacts={c:{"Ti_rich":[],"Al_rich":[]} for c in apex_cuts}
        solver=ResampledMBAR(k,q,specs,kt,ref,offset,hist)
        eblocks=energy_blocks(n,k,q,runs,blocks)
        fallbacks=0
        for _ in range(bootstrap):
            bh,bv,be=draw_blocks(blocks,eblocks,len(specs),rng)
            hh=bh.sum(axis=0)
            br=ref
            try:
                bb=solver.fit(bh)
            except (ArithmeticError,np.linalg.LinAlgError):
                br,bb,_,_=mbar.fit_mbar(k,q,specs,bh,offset,kt=kt)
                fallbacks+=1
            ff=Isotherm(n,t,k,q,bb,hh,br,np.divide(bv,hh,out=np.zeros_like(bv),where=hh>0),
                        bounds,bw,None,None,energy=np.divide(be,hh,out=np.zeros_like(be),where=hh>0))
            for c in apex_cuts:
                ap=ff.apex(c,np.linspace(*bounds,81))
                if ap:
                    deltae=ap["delta_U_minus_mu_x_meV_atom"]
                    slope=n*.001*deltae/(mbar.KB*t*t)
                    apex_reps[c].append([ap["mu"],ap["x"],ap["log_weight_per_variant"],
                                         ap["delta_omega_meV_atom"],deltae,slope])
                    peak=ap["basin_profiles_by_bandwidth_Q"][str(bw)]
                    if peak:
                        peak_reps[c].append([*peak["modes_Q"],peak["cut"],peak["barrier"],*peak["barriers_from_peaks"]])
                    for point in ap["fixed_cut_crossings"]:
                        fixed_contacts[c][point["branch"]].append([point[key] for key in ("mu_eV","x_lowQ","x_highQ","signed_gap")])
            for key,d in (("Ti_rich",1),("Al_rich",-1)):
                a=branches[key]["equal_height"]
                if a is None:
                    continue
                ans=ff.branch(d,a)
                eq=ans.get("equal_variant")
                if eq:
                    reps[key].append([eq["mu_eV"],eq["x_alpha"],eq["x_alpha2"],
                                      ans["equal_height"]["barrier_kBT"],eq["x_alpha2"]-eq["x_alpha"]])
        out["bootstrap"]=dict(count=bootstrap,seed=seed,gradient_solver_fallbacks=fallbacks,
            columns=["mu_eV","x_lowQ","x_highQ","barrier_equal_height_kBT","signed_gap"],
            branches={tag:dict(resolved=len(a),replicates=a,interval_95=np.quantile(a,[.025,.975],axis=0).T.tolist() if len(a)>=2 else None)
                      for tag,a in reps.items()})
        out["bootstrap"]["apex_fixed_cuts"]={str(c):dict(cut_Q=c,columns=["mu_eV","x","log_weight_per_variant","delta_omega_meV_atom",
                                                                       "delta_U_minus_mu_x_meV_atom","local_dS_dT"],
            replicates=a,interval_95=np.quantile(a,[.025,.975],axis=0).T.tolist() if len(a)>=2 else None)
            for c,a in apex_reps.items()}
        out["bootstrap"]["apex_profiles"]={str(c):dict(pairs_found=len(a),columns=["Q_low_mode","Q_high_mode","Q_valley",
            "barrier_from_smaller_peak_kBT","barrier_from_low_mode_kBT","barrier_from_high_mode_kBT"],
            interval_95_conditional_two_modes=np.quantile(a,[.025,.975],axis=0).T.tolist() if len(a)>=2 else None)
            for c,a in peak_reps.items()}
        out["bootstrap"]["fixed_cut_crossings"]={str(c):{tag:dict(n_roots=len(a),columns=["mu_eV","x_lowQ","x_highQ","signed_gap"],
            interval_95=np.quantile(a,[.025,.975],axis=0).T.tolist() if len(a)>=2 else None) for tag,a in roots.items()}
            for c,roots in fixed_contacts.items()}
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests",nargs="+",type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--field",nargs=2,type=float)
    parser.add_argument("--bandwidth",type=float,default=.04)
    parser.add_argument("--bootstrap",type=int,default=0)
    parser.add_argument("--seed",type=int,default=56823)
    parser.add_argument("--halves",action="store_true")
    args=parser.parse_args()
    records=[]
    for t,n,shape,cap,ax,r in load(args.manifests,0):
        fields=sorted({v["spec"][0] for v in r})
        bounds=args.field or [fields[0],fields[-1]]
        if bounds[1]<=bounds[0]:
            parser.error("Provide --field MIN MAX for a single direct field")
        z=analyze(t,n,r,args.bandwidth,bounds,args.bootstrap,args.seed,args.halves)
        z.update(shape=shape,cap=cap,axial_free=ax,manifests=[str(path.resolve()) for path in args.manifests])
        records.append(z)
        print(t,n,[(tag,((b.get("equal_variant") or {}).get("mu_eV"))) for tag,b in z["branches"].items()],
              "strongest barrier",(z["strongest_Q_pair"]["pair"]["barrier"] if z["strongest_Q_pair"] else None),flush=True)
    tmp=args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(records,indent=2,allow_nan=False)+"\n",encoding="utf8")
    tmp.replace(args.out)
