"""Composition susceptibility, sensitivity to the order cut and terminal spatial order."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from analysis import KT, aggregate, fit_mbar, load_runs, make_data


def weights(base, delta, k):
    logw = base + delta * k / KT
    return np.exp(logw - logsumexp(logw))


def concentration_profile(n,ref,k,base):
    mass = np.bincount(k,weights=np.exp(base-logsumexp(base)))
    ks = np.flatnonzero(mass)
    return n,ref,ks,np.log(mass[ks])


def size_crossing(left, right):
    """Equal <x> for two sizes, without a Q cut.

    This is the chemical-field analogue of the conjugate-energy crossing of
    Borgs and Janke (Phys. Rev. Lett. 68, 1738, doi:10.1103/PhysRevLett.68.1738).
    """
    def moments(state,mu):
        n,ref,k,logmass = state
        p = weights(logmass,mu-ref,k)
        x = k/n
        mean = p@x
        return np.array([mean,p@(x-mean)**2,p@(x-mean)**3])

    def diff(mu):
        return moments(right,mu)[0]-moments(left,mu)[0]

    nsmall,nbig = left[0],right[0]
    ref = .5*(left[1]+right[1])
    grid = np.linspace(ref-.065,ref+.065,131)
    z = [diff(mu) for mu in grid]
    candidates = []
    for i in range(1,len(grid)):
        if z[i-1] >= 0 or z[i] <= 0:
            continue
        mu = brentq(diff,grid[i-1],grid[i])
        small,big = moments(left,mu),moments(right,mu)
        var,third = ((nbig*big-nsmall*small)/(nbig-nsmall))[1:]
        slope = (nbig*big[1]-nsmall*small[1])/KT
        if var<=0:
            continue
        # In a two-delta distribution m3/var=(1-2*w_ord)*gap.
        split = third/var
        gap = np.sqrt(split*split+4*var)
        midpoint = big[0]+split/2
        candidates.append(dict(mu_eV=mu,x_cross=big[0],slope_difference_eV_inv=slope,
                               central_moments_small=small[1:].tolist(),central_moments_large=big[1:].tolist(),
                               apparent_moment_extrapolated_ends=[midpoint-gap/2,midpoint+gap/2],
                               implied_ordered_weight=(1-split/gap)/2))
    # A branch-internal crossing has a small difference in susceptibility.
    return sorted(candidates,key=lambda r:r["slope_difference_eV_inv"],reverse=True)


def susceptibility(n, k, q, base, ref, counts):
    # Integrate all order, energy and volume coordinates before choosing a field.
    min_k = int(k.min())
    ks = np.arange(min_k, int(k.max())+1)
    masses = np.bincount(k-min_k, weights=np.exp(base-logsumexp(base)), minlength=len(ks))
    active = masses > 0
    logp, ks = np.log(masses[active]), ks[active]
    xs = ks / n

    def third(delta):
        p = weights(logp, delta, ks)
        return p @ (xs - p @ xs)**3

    grid = np.linspace(-.10, .10, 121)
    signs = [third(d) for d in grid]
    peaks = []
    for a,b,fa,fb in zip(grid,grid[1:],signs,signs[1:]):
        if fa <= 0 or fb >= 0:
            continue
        delta = brentq(third,a,b)
        p = weights(base,delta,k)
        x = k/n
        midpoint = p @ x
        var = p @ (x-midpoint)**2
        scale = np.sqrt(var)
        fourth = p @ (x-midpoint)**4
        cumulant4 = fourth - 3*var**2
        # Equal-weight delta peaks give kappa4=-gap**4/8. Convolution of both
        # phases with the same Gaussian leaves kappa4 unchanged, unlike var.
        fourth_gap = float((-8*cumulant4)**.25) if cumulant4 < 0 else None
        peaks.append(dict(mu_eV=ref+delta, mean_x=midpoint, four_var_x=4*var,
                          Nxvar_x=n*var, two_sigma_x=2*scale,
                          kurtosis=fourth/var**2, kappa4_x=cumulant4,
                          fourth_cumulant_apparent_gap=fourth_gap,
                          mean_q=p @ q, probability_q_ge_half=p[q>=.5].sum(),
                          apparent_ends=[midpoint-scale,midpoint+scale],
                          Kish_frames=1/np.sum(p**2/np.maximum(counts,1))))
    return sorted(peaks,key=lambda peak:peak["four_var_x"],reverse=True)


def cut_sensitivity(n, k, q, base, ref, counts, volume):
    table = []
    for cut in np.arange(.24,.691,.05):
        high = q>cut
        for g in (1,4):
            def resid(d):
                t = base + d*k/KT
                return logsumexp(t[high])-logsumexp(t[~high])-np.log(g)
            if not (resid(-.15)<0<resid(.15)):
                continue
            delta = brentq(resid,-.15,.15)
            w = weights(base,delta,k)
            m = dict(cut=round(cut,2), ordered_to_disordered_weight=g, mu_eV=ref+delta)
            for name,sel in (("low",~high),("high",high)):
                p = w[sel]/w[sel].sum()
                m[name] = dict(x=p @ k[sel]/n, q=p @ q[sel], v=p @ volume[sel],
                               Kish_frames=1/np.sum(p**2/np.maximum(counts[sel],1)))
            table.append(m)
    return table


def order_bins(n,k,q,base,ref,mu,counts,volume,sums):
    p = weights(base,mu-ref,k)
    rows = []
    for j in range(10):
        sel = (q >= j/10)&(q < (j+1)/10 if j<9 else q<=1)
        total = p[sel].sum()
        if total < 1e-8:
            continue
        prob = p[sel]/total
        ranks = [prob @ (sums[f"Al_fraction_sublattice_rank_{s}"][sel]/counts[sel]) for s in range(1,5)]
        rows.append(dict(q_range=[j/10,(j+1)/10],probability=total,
                         mean_x=prob @ k[sel]/n, mean_v=prob @ volume[sel],
                         sorted_sublattice_Al_fractions=ranks,
                         second_minus_third_Al_fraction=ranks[1]-ranks[2]))
    return rows


def terminal_domains(manifests):
    result = []
    for manifest in manifests:
        for row in csv.DictReader(manifest.open()):
            path = manifest.parent / (row["run"]+".state")
            if not path.is_file():
                continue
            nx,ny,nz = map(int,row["shape"].split("x"))
            index = np.array(list(np.ndindex(nx,ny,nz,2)))
            i,j,z,b = index.T
            r = np.array([(i+b/3)/nx,(j+b/3)/ny,(z+b/2)/nz]).T
            sub = 2*((i+b)%2)+(j+b)%2
            occ = np.loadtxt(path,skiprows=1,usecols=0)
            if len(occ)!=len(i):
                raise ValueError("Terminal state size differs from geometry "+str(path))
            # A directed antiphase modulation of any D019 variant gives power
            # in at least one of these sidebands. Removing the common mode
            # suppresses a plain concentration wave independent of the variant.
            waves = np.exp(2j*np.pi * np.array([[1,0,0],[0,1,0],[1,1,0],[0,0,1]]) @ r.T)
            rng = np.random.default_rng(767217+int(row["seed"]))
            replicas = 256
            amps = np.empty((replicas+1,4,4),dtype=complex)
            for s in range(4):
                sel = sub==s
                local = occ[sel] - occ[sel].mean()
                shuffled = np.vstack([local, *[rng.permutation(local) for _ in range(replicas)]])
                amps[:,:,s] = shuffled @ waves[:,sel].T
            amps -= amps.mean(axis=2)[:,:,None]
            power = np.abs(amps)**2
            maxpower = power.sum(axis=2).max(axis=1)/len(occ)
            obs = maxpower[0]
            ranked = np.array([occ[sub==g].mean() for g in range(4)])
            result.append(dict(run=row["run"], N=len(occ), cap=float(row["cap"]),
                               x=occ.mean(), q=4*(ranked.max()-occ.mean())/3,
                               sideband_power=obs, null_percentile95=float(np.quantile(maxpower[1:],.95)),
                               permutation_p=(1+np.count_nonzero(maxpower[1:]>=obs))/(replicas+1),
                               dominant_variant=int(np.argmax(ranked))))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifests",type=Path,nargs="+")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--bootstrap",type=int,default=0)
    a = ap.parse_args()
    data = load_runs(a.manifests,None,0)
    output = {"susceptibility_note":"Peaks refer to the reweighted composition variance, without a Q cut. "
              "Its two standard deviations approach the binodal width only in the asymptotic two-phase regime.",
              "spatial_note":"A terminal snapshot per window. Null permutations preserve all four sublattice "
              "occupancies. Significant power is compatible with spatial chemical correlations, including domains.",
              "groups":[], "terminal_states":terminal_domains(a.manifests)}
    rng = np.random.default_rng(721654)
    size_inputs = {}
    for (n,shape,cap,axial),runs in data.items():
        k,q,specs,blocks,_,sums = make_data(n,runs)
        hist,vsum = aggregate(blocks,len(specs))
        ref,base,f,_ = fit_mbar(k,q,specs,hist)
        cnt = hist.sum(axis=0)
        v = vsum/np.maximum(cnt,1)
        profiles = [concentration_profile(n,ref,k,base)]
        size_inputs.setdefault((cap,axial),[]).append((n,profiles))
        result = dict(N=n,shape=shape,cap=cap,axial_free=axial,
                      susceptibility_peaks=susceptibility(n,k,q,base,ref,cnt),
                      cut_sensitivity=cut_sensitivity(n,k,q,base,ref,cnt,v))
        for peak in result["susceptibility_peaks"]:
            peak["Q_binned_phase_observables"] = order_bins(n,k,q,base,ref,peak["mu_eV"],cnt,v,sums)
        boot = []
        for _ in range(a.bootstrap):
            h,_ = aggregate(blocks,len(specs),rng)
            r,logp,_,_ = fit_mbar(k,q,specs,h,f)
            boot.append(susceptibility(n,k,q,logp,r,h.sum(axis=0)))
            profiles.append(concentration_profile(n,r,k,logp))
        result["bootstrap_susceptibility_peaks"] = boot
        output["groups"].append(result)
        print(n,cap,[(p["mu_eV"],p["mean_x"],p["two_sigma_x"],p["fourth_cumulant_apparent_gap"])
                     for p in result["susceptibility_peaks"]],flush=True)
    output["size_crossing_note"] = ("Compare mean concentration at the same field for successive sizes. "
        "The strongest positive-slope crossing is a field estimate. Optional endpoint estimates use a "
        "two-phase limiting mixture and linear 1/N corrections to variance and third central moment.")
    output["size_crossings"] = []
    for (cap,axial),profiles in size_inputs.items():
        profiles.sort(key=lambda x:x[0])
        for left,right in zip(profiles,profiles[1:]):
            samples = [size_crossing(l,r) for l,r in zip(left[1],right[1])]
            row = dict(N=[left[0],right[0]],cap=cap,axial_free=axial,candidates=samples[0])
            resolved = [roots[0] for roots in samples[1:] if roots]
            row["bootstrap_resolved"] = len(resolved)
            if resolved:
                for name in ("mu_eV","x_cross","apparent_moment_extrapolated_ends"):
                    row[name+"_bootstrap_95"] = np.quantile([r[name] for r in resolved],[.025,.975],axis=0).tolist()
            output["size_crossings"].append(row)
    a.output.write_text(json.dumps(output,indent=2)+"\n")


if __name__ == "__main__":
    main()
