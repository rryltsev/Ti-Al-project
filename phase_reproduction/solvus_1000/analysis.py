"""Multistate reweighting of Ti–Al semi-grand NPT windows at fixed temperature.

The sampled bias is K*(Q-q0)^2/2 - mu*nAl in eV per cell. Combining frames
with the same composition and folded order is exact: the microscopic energy,
volume measure and cage constraints cancel between windows at the same N,cap.
"""
import argparse
import csv
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq, minimize
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.special import logsumexp


KB = 8.617333262145e-5
KT = 1000 * KB


def load_runs(manifests, samples_root, discard, temperature_K=1000.):
    groups = defaultdict(list)
    for manifest in manifests:
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                observed_temperature = float(row.get("T_K") or row.get("temperature_K") or 1000.)
                if abs(observed_temperature-temperature_K) > 1e-8:
                    raise ValueError(f"Manifest {manifest} has temperature {observed_temperature} K, expected {temperature_K} K")
                run = row["run"]
                names = [row["file"]] if row.get("file") else [run+".csv", run+"/samples.csv", "samples/"+run+".csv", run]
                choices = [base/name for base in (samples_root or manifest.parent, manifest.parent)
                           for name in names]
                path = next((p for p in choices if p.is_file()), None)
                if path is None:
                    raise FileNotFoundError(f"Missing samples for {run} at {choices}")
                columns = np.atleast_1d(np.genfromtxt(path, names=True, delimiter=",", encoding="utf-8"))
                for key in ("T_K", "temperature_K"):
                    if key in columns.dtype.names and not np.allclose(columns[key],temperature_K,atol=1e-8,rtol=0):
                        raise ValueError(f"Trajectory temperature differs from manifest for {run}")
                columns = columns[columns["sweep"] > int(row["burn"])]
                start = int(len(columns)*discard)
                columns = columns[start:]
                if not len(columns):
                    raise ValueError("Empty production record " + str(path))
                n = int(row["N"])
                occ = np.stack([columns["n"+str(j)] for j in range(4)], axis=1)
                k = columns["n_Al"].astype(int)
                if not np.array_equal(occ, np.rint(occ)) or not np.array_equal(occ.sum(axis=1), k):
                    raise ValueError("Inconsistent sublattice occupancies in " + str(path))
                code = (4*occ.max(axis=1)-k).astype(int)
                q = 4*code/(3*n)
                if np.max(abs(q-columns["Q"])) > 1.e-5:
                    raise ValueError("Order convention differs in " + str(path))
                v = columns["volume_A3_atom"]
                if not np.all(np.isfinite(v)) or np.any(v <= 0):
                    raise ValueError("Invalid volume in " + str(path))
                spec = float(row["mu_eV"]), float(row["K_eV"]), float(row["q0"])
                cap=float(row["cap"])
                ca=columns["c_a"]
                if not np.all(np.isfinite(ca)) or np.any(ca <= 0):
                    raise ValueError("Invalid axial ratio in " + str(path))
                groups[n,row["shape"],cap,int(row["axial_free"])].append(dict(
                    run=run, path=str(path.resolve()), seed=int(row["seed"]), start=row["start"],
                    spec=spec, k=k, code=code, v=v, variant=occ.argmax(axis=1),
                    occ_sorted=np.sort(occ,axis=1)[:,::-1],cap=cap, ca=ca,
                    sweep=columns["sweep"], wall=columns["wall"]))
    return groups


def tau_int(values):
    y = np.asarray(values, dtype=float)
    n = len(y)
    if n < 4 or np.ptp(y) == 0:
        return 1.
    a = y - np.mean(y)
    ft = np.fft.rfft(a, n=1 << (2*n-1).bit_length())
    ac = np.fft.irfft(ft*ft.conj())[:n] / np.arange(n, 0, -1)
    ac /= ac[0]
    pair = ac[:n//2*2].reshape(-1,2).sum(axis=1)
    stop = next((i for i,p in enumerate(pair) if p <= 0), len(pair))
    # Positive, monotone paired autocovariance sequence.
    return max(1., float(2*np.minimum.accumulate(pair[:stop]).sum()-1))


def make_data(n, runs):
    specs = sorted(set(r["spec"] for r in runs), key=lambda s:(s[2],s[0],s[1]))
    ids = {spec:i for i,spec in enumerate(specs)}
    codes = np.concatenate([r["k"]*(3*n+1)+r["code"] for r in runs])
    unique, inverse = np.unique(codes, return_inverse=True)
    k = unique//(3*n+1)
    q = 4*(unique%(3*n+1))/(3*n)
    block_data, diagnostics = [], []
    state_moments = defaultdict(lambda: np.zeros(len(k)))
    pos = 0
    for r in runs:
        length = len(r["k"])
        idx = inverse[pos:pos+length]
        pos += length
        gi = {name:tau_int(r[name]) for name in ("k","code","v","ca")}
        ta = max(gi.values())
        block_length = max(1, int(np.ceil(2*ta)))
        num_blocks = max(1, length//block_length)
        # Equal contiguous blocks, retaining the complete record.
        size,extra=divmod(length,num_blocks)
        block_sizes=np.full(num_blocks,size)
        block_sizes[:extra] += 1
        block_id=np.repeat(np.arange(num_blocks),block_sizes)
        coord=(block_id,idx)
        hist=csr_matrix((np.ones(length),coord),shape=(num_blocks,len(k)))
        vsum=csr_matrix((r["v"],coord),shape=hist.shape)
        block_data.append((ids[r["spec"]], hist, vsum))
        # The native sampler's lower density bound is joint in V and c/a.
        # Use the same 1.68052-A minimum-separation guard here to monitor it.
        ca=r["ca"]
        vguard=np.sqrt(3)*ca/4*(1.68052/(1-r["cap"])/np.minimum(1,np.sqrt(1/3+ca*ca/4)))**3
        vlo=np.maximum(15,vguard)
        observables={"mean_c_a":ca,"mean_wall":r["wall"],
                     "fraction_within_0p25_A3_lower_volume":(r["v"]-vlo < .25),
                     "fraction_within_0p25_A3_upper_volume":(21-r["v"] < .25),
                     "fraction_within_0p01_axial_bounds":np.minimum(ca-1.45,1.72-ca)<.01}
        for j in range(4):
            observables[f"Al_fraction_sublattice_rank_{j+1}"]=4*r["occ_sorted"][:,j]/n
        for name,obs in observables.items():
            state_moments[name] += np.bincount(idx,weights=obs,minlength=len(k))
        n_half=max(1,length//2)
        halves=lambda y:[float(np.mean(y[:n_half])),float(np.mean(y[-n_half:]))]
        diagnostics.append(dict(run=r["run"], start=r["start"], seed=r["seed"], spec=r["spec"],
                                frames=length, blocks=num_blocks, statistical_inefficiency=ta,
                                inefficiency_by_observable=gi,
                                mean_x=float(np.mean(r["k"])/n),mean_Q=float(np.mean(4*r["code"]/(3*n))),
                                mean_volume_A3_atom=float(np.mean(r["v"])), range_volume_A3_atom=[float(r["v"].min()),float(r["v"].max())],
                                range_c_a=[float(ca.min()),float(ca.max())],halves_c_a=halves(ca),
                                halves_volume=halves(r["v"]),halves_x=halves(r["k"]/n),halves_q=halves(4*r["code"]/(3*n)),
                                min_distance_to_lower_volume_A3_atom=float((r["v"]-vlo).min()),
                                observable_means={name:float(np.mean(obs)) for name,obs in observables.items()},
                                first_sweep=float(r["sweep"][0]),last_sweep=float(r["sweep"][-1]),
                                dominant_variants=np.bincount(r["variant"],minlength=4).tolist()))
    return k, q, np.array(specs), block_data, diagnostics, state_moments


def aggregate(blocks, num_windows, rng=None):
    hist = np.zeros((num_windows,blocks[0][1].shape[1]))
    vsum = np.zeros(hist.shape[1])
    for i, h, v in blocks:
        if rng is None:
            hist[i] += np.asarray(h.sum(axis=0)).ravel()
            vsum += np.asarray(v.sum(axis=0)).ravel()
        else:
            n_blocks=h.shape[0]
            copies = np.bincount(rng.integers(n_blocks,size=n_blocks),minlength=n_blocks)
            hist[i] += h.T @ copies
            vsum += v.T @ copies
    return hist, vsum


def fit_mbar(k,q,specs,hist,initial=None,*,kt=KT):
    h = hist.sum(axis=0)
    nk = hist.sum(axis=1)
    active = nk > 0
    if not np.all(active):
        raise ValueError("At least one MBAR state has no samples")
    ref = float(np.median(specs[:,0]))
    potential = (specs[:,1,None]*(q-specs[:,2,None])**2/2
                 -(specs[:,0,None]-ref)*k)/kt
    if specs.shape[1]==5:
        # A zero-bias trajectory confined to one pure basin has its own
        # partition function. Its arbitrary initial basin carries no phase odds.
        basin=specs[:,3,None]
        cut=specs[:,4,None]
        outside=((basin<0)&(q>cut)) | ((basin>0)&(q<=cut))
        if np.any((hist>0)&outside):
            raise ValueError("Restricted trajectory sampled outside its assigned basin")
        potential=np.where(outside,np.inf,potential)
    log_n_minus_u = np.log(nk)[:,None] - potential
    colmax=log_n_minus_u.max(axis=0)
    scaled=np.exp(log_n_minus_u-colmax)
    total=h.sum()
    def evaluate(f):
        offset=np.r_[0,f]
        shift=offset.max()
        scale=np.exp(offset-shift)
        denom=scale @ scaled
        if np.min(denom)>1e-280:
            logd=shift+np.log(denom)
            grad=scale*(scaled @ (h/denom))-nk
        else:
            # Extremely separated trial offsets demand full log-space arithmetic.
            p=offset[:,None]+log_n_minus_u
            norm=logsumexp(p,axis=0)
            logd=norm-colmax
            grad=np.exp(p-norm) @ h-nk
        return (h @ logd - nk[1:] @ f)/total, grad[1:]/total
    f0 = np.zeros(len(nk)-1) if initial is None else initial[1:]
    if len(f0):
        fit = minimize(evaluate,f0,jac=True,method="L-BFGS-B",
                       options=dict(ftol=2e-15,gtol=1e-10,maxiter=3000,maxls=80))
        f = np.r_[0,fit.x]
        error = float(np.max(abs(evaluate(fit.x)[1])))
        if error > 2e-6:
            raise ValueError(f"MBAR failed: gradient {error:.3g}")
    else:
        f = np.array([0.])
    logd = logsumexp(f[:,None]+log_n_minus_u,axis=0)
    post = np.exp(f[:,None]+log_n_minus_u-logd)
    overlap = (post*h) @ post.T / nk[:,None]
    logmass = np.full(len(h),-np.inf)
    take = h>0
    logmass[take] = np.log(h[take])-logd[take]
    return ref, logmass, f, overlap


def order_profile(q,weights,bandwidth):
    edges = np.linspace(0,1,201)
    centers = (edges[1:]+edges[:-1])/2
    bins = np.clip(np.searchsorted(edges,q,side="right")-1,0,len(centers)-1)
    mass = np.bincount(bins,weights=weights,minlength=len(centers))
    density = gaussian_filter1d(mass,bandwidth/(edges[1]-edges[0]),mode="reflect")/(edges[1]-edges[0])
    peaks = find_peaks(np.r_[-1,density,-1])[0]-1
    return centers,density,peaks


def height_separator(k,q,base,ref,bandwidth,anchor=None,*,kt=KT,field_bounds=None,direction=1):
    """Define a common basin boundary at the valley between equal-height Q modes."""
    @lru_cache(maxsize=160)
    def profile(dm):
        t=base+dm*k/kt
        return order_profile(q,np.exp(t-logsumexp(t)),bandwidth)

    def peak_pair(dm,cut):
        grid,density,peaks=profile(dm)
        lo=peaks[grid[peaks]<cut]
        hi=peaks[grid[peaks]>cut]
        if not len(lo) or not len(hi):
            return None
        a,b=lo[np.argmax(density[lo])],hi[np.argmax(density[hi])]
        if density[a]<=0 or density[b]<=0 or grid[b]-grid[a]<2*bandwidth:
            return None
        return np.log(density[b]/density[a]),a,b

    def find_cut(cuts,fields):
        candidates=[]
        for cut in cuts:
            prev=None
            for dm in fields:
                current=peak_pair(dm,cut)
                if current is not None and prev is not None and direction*prev[1]<=0<=direction*current[0]:
                    def residual(x):
                        result=peak_pair(x,cut)
                        if result is None:
                            raise ValueError("A Q peak disappeared")
                        return result[0]
                    try:
                        at=brentq(residual,prev[0],dm,xtol=3e-10)
                    except ValueError:
                        prev=(dm,current[0])
                        continue
                    ratio,lo,hi=peak_pair(at,cut)
                    grid,density,_=profile(at)
                    j=lo+np.argmin(density[lo:hi+1])
                    if abs(ratio)<1.e-4 and lo<j<hi and density[j]>0:
                        candidates.append(dict(mu_eV=float(ref+at),order_cut=float(grid[j]),
                                           basin_mode_q=[float(grid[lo]),float(grid[hi])],
                                           barrier_kBT=float(np.log(min(density[lo],density[hi])/density[j]))))
                prev=None if current is None else (dm,current[0])
        return max(candidates,key=lambda a:a["barrier_kBT"]) if candidates else None

    if anchor:
        dm=anchor["mu_eV"]-ref
        trial=dm+np.array([-.07,-.03,-.012,0,.012,.03,.07])
        if field_bounds is not None:
            trial=np.unique(np.clip(trial,field_bounds[0]-ref,field_bounds[1]-ref))
        result=find_cut([np.mean(anchor["basin_mode_q"])],trial)
        if result:
            return result
    if field_bounds is not None:
        fields=np.linspace(field_bounds[0]-ref,field_bounds[1]-ref,81)
        return find_cut(np.arange(.075,1.,.05),fields)
    return find_cut([.20,.35,.50,.65,.80],np.linspace(-.16,.16,65))


def individual_fit(n,k,q,base,ref,h,v,bandwidth,g,return_profile=False,guess=None,fixed_cut=None,*,kt=KT,field_bounds=None,direction=1):
    """Find the semigrand field with P_hi/P_lo=g, optionally using a fixed valley."""
    def solve(cut):
        high = q>cut
        low = ~high
        if not (np.isfinite(base[high]).any() and np.isfinite(base[low]).any()):
            return None
        def residual(dm):
            t = base + (dm/kt)*k
            return logsumexp(t[high])-logsumexp(t[low])-np.log(g)
        if field_bounds is None:
            a,b=-.5,.5
            if direction*residual(a)>=0 or direction*residual(b)<=0:
                return None
        else:
            a,b=np.asarray(field_bounds)-ref
            if direction*residual(a)>0 or direction*residual(b)<0:
                # A folded order can decrease on the Al-rich side. Locate the
                # increasing crossing inside the sampled field window.
                grid=np.linspace(a,b,41)
                values=direction*np.array([residual(x) for x in grid])
                crossings=np.flatnonzero((values[:-1]<=0)&(values[1:]>=0))
                if not len(crossings):
                    return None
                j=crossings[0]
                a,b=grid[j:j+2]
        dm = brentq(residual,a,b,xtol=1e-11)
        t = base+(dm/kt)*k
        w = np.exp(t-logsumexp(t))
        grid,density,peaks=order_profile(q,w,bandwidth)
        lo = peaks[grid[peaks]<cut]
        hi = peaks[grid[peaks]>cut]
        if len(lo)==0 or len(hi)==0:
            return None
        pl = lo[np.argmax(density[lo])]
        pr = hi[np.argmax(density[hi])]
        if grid[pr]-grid[pl] < 2*bandwidth:
            return None
        j = pl+np.argmin(density[pl:pr+1])
        if j==pl or j==pr or density[j]<=0:
            return None
        return dm,w,grid,density,pl,pr,j
    candidates = []
    attempts = [fixed_cut] if fixed_cut is not None else ([.22,.34,.46,.58,.70] if guess is None else [guess,.28,.50,.68])
    for first in attempts:
        cut = first
        seen = set()
        for _ in range(16):
            trial = solve(cut)
            if trial is None:
                break
            dm,w,grid,density,pl,pr,j = trial
            valley=grid[j]
            high=q>cut
            if fixed_cut is not None or np.array_equal(high,q>valley):
                probs=[w[sel]/w[sel].sum() for sel in (~high,high)]
                x=[float(p@k[sel]/n) for sel,p in zip((~high,high),probs)]
                if direction*(x[1]-x[0])<=0:
                    break
                vols=[float(p@v[sel]) for sel,p in zip((~high,high),probs)]
                ess=[float(1/np.sum(p*p/np.maximum(h[sel],1))) for sel,p in zip((~high,high),probs)]
                barrier=np.log(np.array([density[pl],density[pr]])/density[j])
                item=dict(mu_eV=float(ref+dm),order_cut=float(cut if fixed_cut is not None else valley),
                          valley_at_target=float(valley),basin_mode_q=[float(grid[pl]),float(grid[pr])],
                          x_alpha=x[0],x_alpha2=x[1],width=abs(x[1]-x[0]),phase_volumes_A3_atom=vols,
                          barrier_kBT=barrier.tolist(),phase_Kish_ess_frames=ess,
                          phase_weight_ratio=float(w[high].sum()/w[~high].sum()))
                if return_profile:
                    item["order_profile"]={"Q":grid.tolist(),"density":density.tolist()}
                candidates.append(item)
                break
            if j in seen:
                break
            seen.add(j)
            cut=valley
    if not candidates:
        return None
    return max(candidates,key=lambda r:min(r["barrier_kBT"]))


def connectivity(hist,k,q,n):
    # Independent observed overlap on coarse (x,Q) cells. The unbinned MBAR
    # remains the free-energy estimator. The bins follow the width of a
    # single-phase density fluctuation when comparing different system sizes.
    dx,dq=overlap_bins(n)
    xi=np.floor((k/n)/dx).astype(int)
    qi=np.floor(q/dq).astype(int)
    _,key=np.unique(xi*(qi.max()+1)+qi,return_inverse=True)
    h=np.array([np.bincount(key,weights=a,minlength=key.max()+1) for a in hist])
    sqrtp=np.sqrt(h/h.sum(axis=1)[:,None])
    return sqrtp @ sqrtp.T


def overlap_bins(n):
    return max(2/n,.025*np.sqrt(96/n)),max(8/(3*n),.05*np.sqrt(96/n))


def central_subset(n,runs,bandwidth,*,kt=KT,field_bounds=None,direction=1):
    k,q,specs,blocks,_,_=make_data(n,runs)
    hist,vsum=aggregate(blocks,len(specs))
    h=hist.sum(axis=0)
    ref,base,_,overlap=fit_mbar(k,q,specs,hist,kt=kt)
    anchor=height_separator(k,q,base,ref,bandwidth,kt=kt,field_bounds=field_bounds,direction=direction)
    if anchor is None:
        return None
    fit=individual_fit(n,k,q,base,ref,h,vsum/h,bandwidth,4,fixed_cut=anchor["order_cut"],kt=kt,field_bounds=field_bounds,direction=direction)
    if fit:
        eig=np.sort(np.linalg.eigvals(overlap).real)
        fit.update(frames=int(h.sum()),n_bias_states=len(specs),
                   second_MBAR_eigenvalue=float(eig[-2]) if len(eig)>1 else None)
    return fit


def half_runs(runs,early):
    subset=[]
    for run in runs:
        half=len(run["k"])//2
        if half<2:
            return []
        sl=slice(0,half) if early else slice(-half,None)
        subset.append({name:value[sl] if isinstance(value,np.ndarray) else value for name,value in run.items()})
    return subset


def analysis(n, runs, boot, seed, bandwidth=.04,*,kt=KT,field_bounds=None,direction=1):
    k,q,specs,blocks,diag,moments=make_data(n,runs)
    hist,vsum=aggregate(blocks,len(specs))
    h=hist.sum(axis=0)
    v=np.divide(vsum,h,out=np.zeros_like(vsum),where=h>0)
    ref,base,f,o=fit_mbar(k,q,specs,hist,kt=kt)
    root_options=dict(kt=kt,field_bounds=field_bounds,direction=direction)
    anchor=height_separator(k,q,base,ref,bandwidth,**root_options)
    def fixed(g,bw=bandwidth,dq=0,separator=anchor,return_profile=False):
        if separator is None:
            return None
        return individual_fit(n,k,q,base,ref,h,v,bw,g,return_profile,fixed_cut=separator["order_cut"]+dq,**root_options)
    corrected=fixed(4,return_profile=True)
    raw=fixed(1)
    if corrected:
        weights=base+((corrected["mu_eV"]-ref)/kt)*k
        high=q>corrected["order_cut"]
        corrected["phase_observables"]={}
        for name,counts in moments.items():
            means=counts/h
            corrected["phase_observables"][name]=[float(
                np.exp(weights[s]-logsumexp(weights[s])) @ means[s]) for s in (~high,high)]
        corrected["phase_origin_fraction_by_mu"]={str(mu):[] for mu in np.unique(specs[:,0])}
        corrected["phase_Kish_approx_with_run_inefficiency"]=[]
        per_run=np.array([np.asarray(block[1].sum(axis=0)).ravel() for block in blocks])
        ineff=np.array([d["statistical_inefficiency"] for d in diag])
        for s in (~high,high):
            w=np.exp(weights[s]-logsumexp(weights[s]))/h[s]
            for mu in np.unique(specs[:,0]):
                rows=specs[:,0]==mu
                corrected["phase_origin_fraction_by_mu"][str(mu)].append(float(hist[rows][:,s].sum(axis=0) @ w))
            corrected["phase_Kish_approx_with_run_inefficiency"].append(float(1/(ineff @ per_run[:,s] @ (w*w))))
    qcenter,density,peaks=order_profile(q,np.exp(base-logsumexp(base)),bandwidth)
    result=dict(N=n,mu_reference_eV=ref,n_runs=len(runs),frames=int(h.sum()),n_unique_states=len(k),
                bandwidth_q=bandwidth,bootstrap_seed=seed,bias_specs_mu_K_q0=specs.tolist(),
                mbar_offsets=f.tolist(),mbar_overlap=o.tolist(),joint_histogram_BC=connectivity(hist,k,q,n).tolist(),
                joint_histogram_bin_width_x_Q=overlap_bins(n),
                diagnostics=diag,equal_height_separator=anchor,equal_single_variant=corrected,equal_total_weight=raw,
                reference_profile=dict(q=qcenter.tolist(),density=density.tolist(),modes=qcenter[peaks].tolist()),
                bandwidth_sensitivity={str(b):fixed(4,bw=b,separator=height_separator(k,q,base,ref,b,anchor,**root_options))
                                       for b in (bandwidth*.75,bandwidth*1.25)},
                divider_sensitivity={str(dq):fixed(4,dq=dq) for dq in (-.05,.05)},
                self_consistent_valley={str(g):individual_fit(n,k,q,base,ref,h,v,bandwidth,g,**root_options) for g in (1,4)},
                status="candidate" if corrected else "unresolved")
    if corrected and boot:
        rng=np.random.default_rng(seed)
        good=[]
        for _ in range(boot):
            bh,bv=aggregate(blocks,len(specs),rng)
            bn=bh.sum(axis=0)
            bv=np.divide(bv,bn,out=np.zeros_like(bv),where=bn>0)
            r,b,_,_=fit_mbar(k,q,specs,bh,f,kt=kt)
            separator=height_separator(k,q,b,r,bandwidth,anchor,**root_options)
            fit=(individual_fit(n,k,q,b,r,bn,bv,bandwidth,4,fixed_cut=separator["order_cut"],**root_options)
                 if separator else None)
            if fit is not None:
                good.append(fit)
        result["bootstrap"]={"count":boot,"resolved":len(good),"resolved_fraction":len(good)/boot,
                             "insufficient_blocks":[r["run"] for r in diag if r["blocks"]<3]}
        if len(good)>=2:
            result["bootstrap"]["joint_mu_xalpha_xalpha2"]=[[x["mu_eV"],x["x_alpha"],x["x_alpha2"]] for x in good]
            result["bootstrap"]["intervals"]={name:dict(se=float(np.std(arr,ddof=1)),
                quantile_025_50_975=np.percentile(arr,[2.5,50,97.5]).tolist()) for name in
                ("mu_eV","x_alpha","x_alpha2","width","order_cut")
                for arr in [np.array([g[name] for g in good])]}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest",nargs="+",type=Path)
    p.add_argument("--samples-root",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=96)
    p.add_argument("--discard",type=float,default=0.,help="additional fraction of production to discard")
    p.add_argument("--bandwidth",type=float,default=.04)
    p.add_argument("--seed",type=int,default=26713)
    args=p.parse_args()
    if not 0<=args.discard<1 or not 0<args.bandwidth<.25 or args.bootstrap<0:
        p.error("Invalid discard, bandwidth or bootstrap count")
    results=load_runs(args.manifest,args.samples_root,args.discard)
    output={}
    for (n,shape,cap,axial_free),runs in results.items():
        if len({r["path"] for r in runs})!=len(runs):
            raise ValueError("Duplicate trajectories in manifests")
        full=analysis(n,runs,args.bootstrap,args.seed,args.bandwidth)
        full.update(shape=shape,cap=cap,axial_free=axial_free)
        centers={r["spec"][2] for r in runs}
        subsets={}
        comparisons=[("by_initialization",lambda r:r["start"]),("by_field",lambda r:r["spec"][0])]
        fields=np.unique([r["spec"][0] for r in runs])
        if len(fields)>2:
            gap=np.argmax(np.diff(fields))
            midpoint=float(np.mean(fields[gap:gap+2]))
            full["field_cohort_split_eV"]=midpoint
            comparisons.append(("by_field_cohort",lambda r:"lower_fields" if r["spec"][0]<midpoint else "upper_fields"))
        for label,get_value in comparisons:
            for tag in sorted({get_value(r) for r in runs}):
                selected=[r for r in runs if get_value(r)==tag]
                centers_subset={r["spec"][2] for r in selected}
                spans_q=(min(centers_subset)==min(centers) and max(centers_subset)==max(centers))
                if spans_q and len(centers_subset)>1 and len(selected)<len(runs):
                    paths=tuple(r["path"] for r in selected)
                    if paths not in subsets:
                        subsets[paths]=central_subset(n,selected,args.bandwidth)
                    full.setdefault(label,{})[str(tag)]=subsets[paths]
        for label,early in [("early",True),("late",False)]:
            half=half_runs(runs,early)
            if half:
                full.setdefault("production_halves",{})[label]=central_subset(n,half,args.bandwidth)
        output[f"N{n}_{shape}_cap{cap:g}_axial{axial_free}"]=full
    tmp=args.output.with_name(args.output.name+".tmp")
    tmp.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(args.output)


if __name__=="__main__":
    main()
