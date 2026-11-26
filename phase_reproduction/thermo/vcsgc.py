"""Reconstruct an isotherm from saved VCSGC composition windows (S4–S5)."""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.sparse.csgraph import connected_components

KB = 8.617333262145e-5


def lower_hull(x, f, min_span):
    hull = []
    for k in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2:]
            if (f[b]-f[a])*(x[k]-x[b]) < (f[k]-f[b])*(x[b]-x[a]):
                break
            hull.pop()
        hull.append(k)
    return [dict(alpha_at_pct=float(100*x[a]), alpha2_at_pct=float(100*x[b]),
                 window_edge=bool(a == 0 or b == len(x)-1))
            for a,b in zip(hull,hull[1:]) if b>a+1 and x[b]-x[a]>min_span]


def integrate(means, mu, min_span):
    order = np.argsort(means)
    x, y = means[order], mu[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("Coincident mean concentrations give an ambiguous integration")
    grid = np.linspace(x[0],x[-1], max(101,int(np.ceil((x[-1]-x[0])/.00005))+1))
    spline = PchipInterpolator(x,y).antiderivative()
    f = spline(grid)-spline(x[0])
    return grid, f, lower_hull(grid, f, min_span)


def wham(n_al, samples, settings, n, t, min_span):
    hist = np.array([np.bincount(s, minlength=n+1)[n_al] for s in samples])
    active = hist > 0
    parts = connected_components(active.astype(int) @ active.astype(int).T,
                                 directed=False, return_labels=False)
    result = {"overlap_components": int(parts)}
    if parts != 1:
        return result, []
    sizes, h = hist.sum(axis=1), hist.sum(axis=0)
    c = n_al/n
    c0, k, mu0 = np.asarray(settings).T
    kbt = KB*t
    gauge = float(np.mean(mu0))
    b = n*(k[:,None]*(c-c0[:,None])**2 - (mu0[:,None]-gauge)*c/kbt)
    def likelihood(z):
        log_terms = np.log(sizes[:,None]) - b - np.r_[0,z][:,None]
        den = logsumexp(log_terms, axis=0)
        value = (h @ den + sizes[1:] @ z)/h.sum()
        grad = (sizes - np.exp(log_terms-den) @ h)/h.sum()
        return value, grad[1:]
    opt = minimize(likelihood, np.zeros(len(samples)-1), jac=True, method="L-BFGS-B",
                   options={"ftol":1.e-15, "gtol":1.e-10, "maxiter":20000, "maxcor":30})
    err = np.max(abs(likelihood(opt.x)[1]))
    result["offset_gradient_max_abs"] = float(err)
    if not opt.success or err > 1.e-6:
        result["status"] = "normalization_failed"
        return result, []
    den = logsumexp(np.log(sizes[:,None])-b-np.r_[0,opt.x][:,None],axis=0)
    f = -kbt/n*(np.log(h)-den)+gauge*c
    result.update(status="connected",contacts=lower_hull(c,f,min_span))
    return result, zip(c, f-f[0])


def read_inputs(paths):
    data = defaultdict(lambda: defaultdict(list))
    for path in paths:
        open_text = gzip.open if path.suffix==".gz" else open
        with open_text(path, "rt", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                n, t = int(row["N"]), float(row["T_K"])
                c0, k, mu0 = (float(row[s]) for s in ("c0","kappa","delta_mu0_eV"))
                count, sweep = int(row["n_Al"]), int(row["sweep"])
                if not (np.isfinite((c0,k,mu0,t)).all() and 0<=c0<=1 and k>0 and n>0 and t>0 and 0<=count<=n):
                    raise ValueError("Invalid VCSGC sample")
                order=float(row["m"]) if row.get("m") else np.nan
                data[(row["model"],t,n)][(c0,k,mu0,row["replica"])].append((sweep,count,order))
    return data


def correlation_time(y):
    """Integrated autocorrelation in saved frames, initial-positive-pair convention."""
    y=np.array(y,dtype=float)
    if np.any(~np.isfinite(y)) or len(y)<4 or np.ptp(y)==0:
        return None
    y-=y.mean()
    if np.dot(y,y)==0:
        return None
    n=len(y)
    ft=np.fft.rfft(y,n=2*n)
    corr=np.fft.irfft(ft*ft.conj(),n=2*n)[:n]
    corr/=corr[0]
    pairs=corr[:len(corr)//2*2].reshape(-1,2).sum(axis=1)
    stop=np.where(pairs<=0)[0]
    pairs=pairs[:stop[0]] if len(stop) else pairs
    return float(max(.5,pairs.sum()-.5))


def analyze(windows, t, n, min_span, bootstrap, blocks, rng):
    by_window = defaultdict(list)
    order_blocks=defaultdict(list)
    diagnostics=[]
    for (c0,k,mu0,rep), rows in sorted(windows.items()):
        rows.sort()
        if any(b[0]<=a[0] for a,b in zip(rows,rows[1:])):
            raise ValueError("Duplicate production sweeps in a window/replica")
        sweep,counts,order=zip(*rows)
        by_window[(c0,k,mu0)].append(np.asarray(counts))
        if len(order)>=blocks and np.isfinite(order).all():
            order_blocks[(c0,k,mu0)].extend(v.mean() for v in np.array_split(np.asarray(order),blocks))
        diagnostics.append((c0,k,mu0,rep,len(sweep),len(sweep)//blocks,
                            correlation_time(counts),correlation_time(order)))
    if len(by_window)<4:
        raise ValueError("Need at least four sampled windows for a free-energy reconstruction")
    settings = list(by_window)
    c0,k,mu0 = np.asarray(settings).T
    samples = [np.concatenate(r) for r in by_window.values()]
    means = np.array([s.mean()/n for s in samples])
    def chemical(c):
        return mu0-2*KB*t*k*(c-c0)
    mu = chemical(means)
    grid, f, contacts = integrate(means,mu,min_span)
    raw_internal=[p for p in lower_hull(grid,f,0) if not p["window_edge"]]
    maximum_width=max((p["alpha2_at_pct"]-p["alpha_at_pct"] for p in raw_internal),default=0.)
    boot = {}
    interior = [p for p in contacts if not p["window_edge"]]
    def measure_order(source_contacts):
        if len(order_blocks)!=len(settings):
            return
        mean_m=np.array([np.mean(order_blocks[w]) for w in settings])
        se_m=np.array([np.std(order_blocks[w],ddof=1)/np.sqrt(len(order_blocks[w])) for w in settings])
        order=np.argsort(means)
        for p in source_contacts:
            ab=np.array([p["alpha_at_pct"],p["alpha2_at_pct"]])/100
            m=np.interp(ab,means[order],mean_m[order])
            se=np.interp(ab,means[order],se_m[order])
            difference=float(m[1]-m[0])
            direction="uncertain"
            if abs(difference)>2*se.sum():
                direction="rising" if difference>0 else "falling"
            p.update(order_left=float(m[0]),order_right=float(m[1]),order_direction=direction,
                     order_se_left=float(se[0]),order_se_right=float(se[1]))
    measure_order(contacts)
    # Gap selection for the Ti-rich alpha -> alpha2 branch uses chemical order
    # increasing with Al. Other convex-hull facets remain recorded.
    increasing=[p for p in interior if p.get("order_direction")=="rising"]
    if bootstrap:
        pieces = []
        for replicas in by_window.values():
            runs = []
            for arr in replicas:
                if len(arr)<blocks*2:
                    raise ValueError(f"Need at least {2*blocks} samples/replica for {blocks} blocks")
                runs.append(np.array([(a.sum(),len(a)) for a in np.array_split(arr,blocks)]))
            pieces.append(runs)
        accepted = [[] for _ in interior]
        no_contact=one_contact=multiple=invalid=0
        for _ in range(bootstrap):
            c = []
            for replicas in pieces:
                sums = sum(block[rng.integers(0,blocks,blocks)].sum(axis=0) for block in replicas)
                c.append(sums[0]/sums[1]/n)
            c=np.array(c)
            try:
                candidates=integrate(c,chemical(c),min_span)[2]
            except ValueError:
                invalid+=1
                continue
            internal=[p for p in candidates if not p["window_edge"]]
            if len(internal)==0:
                no_contact+=1
            elif len(internal)==1:
                one_contact+=1
            else:
                multiple+=1
            for ref,matches in zip(interior,accepted):
                mid=.5*(ref["alpha_at_pct"]+ref["alpha2_at_pct"])
                match=[p for p in internal if p["alpha_at_pct"]<mid<p["alpha2_at_pct"]]
                if len(match)==1:
                    matches.append((match[0]["alpha_at_pct"],match[0]["alpha2_at_pct"]))
        boot=dict(draws=bootstrap,resolved=None,blocks_per_replica=blocks,
                  baseline_interior_candidates=len(interior),draws_without_interior_contact=no_contact,
                  draws_with_one_interior_contact=one_contact,draws_with_multiple_interior_contacts=multiple,
                  draws_invalid=invalid)
        for p,matches in zip(interior,accepted):
            stats=dict(resolved=len(matches))
            if len(matches)>=2:
                a,b=np.transpose(matches)
                stats.update(alpha_95=np.percentile(a,[2.5,97.5]).tolist(),
                             alpha2_95=np.percentile(b,[2.5,97.5]).tolist(),
                             width_95=np.percentile(b-a,[2.5,97.5]).tolist())
            p["bootstrap_midpoint_match"]=stats
        if len(increasing)==1:
            boot.update(increasing[0]["bootstrap_midpoint_match"])
        elif len(interior)==1 and "order_direction" not in interior[0]:
            boot.update(interior[0]["bootstrap_midpoint_match"])
    n_al=np.unique(np.concatenate(samples))
    hw, whamcurve=wham(n_al,samples,settings,n,t,min_span)
    measure_order(hw.get("contacts",[]))
    return (dict(windows=len(settings),TI_contacts=contacts,min_width_at_pct=100*min_span,
                 largest_unthresholded_interior_width_at_pct=maximum_width,bootstrap_conditional=boot,WHAM=hw),
            zip(c0,k,mu0,means,mu,[len(s) for s in samples]), zip(grid,f), whamcurve, diagnostics)


def run(args):
    groups=read_inputs(args.samples)
    if not groups or args.blocks<2 or args.bootstrap<0 or args.min_span<=0:
        raise ValueError("Invalid data or settings")
    args.out.mkdir(parents=True,exist_ok=True)
    summary=[]
    with (args.out/"window_correlations.csv").open("w",newline="") as diagfile, (
            args.out/"window_means.csv").open("w",newline="") as meanfile, (
            args.out/"mc_free_energy.csv").open("w",newline="") as curvefile:
        means_writer,curve_writer=csv.writer(meanfile),csv.writer(curvefile)
        diag_writer=csv.writer(diagfile)
        diag_writer.writerow(("model","T_K","N","c0","kappa","delta_mu0_eV","replica","samples",
                              "block_frames","tau_int_count_frames","tau_int_order_frames"))
        means_writer.writerow(("model","T_K","N","c0","kappa","delta_mu0_eV","mean_c","mean_mu_eV","samples"))
        curve_writer.writerow(("model","T_K","N","method","c_Al","relative_f_eV_atom"))
        for key,windows in sorted(groups.items()):
            model,t,n=key
            res,means,curve,wcurve,diag=analyze(windows,t,n,args.min_span,args.bootstrap,args.blocks,np.random.default_rng(args.seed))
            summary.append(dict(model=model,T_K=t,N=n,**res))
            means_writer.writerows((*key,*row) for row in means)
            diag_writer.writerows((*key,*row) for row in diag)
            for method,values in (("TI",curve),("WHAM",wcurve)):
                curve_writer.writerows((*key,method,*row) for row in values)
    (args.out/"mc_contacts.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("samples",type=Path,nargs="+")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--min-span",type=float,default=.005)
    p.add_argument("--bootstrap",type=int,default=0)
    p.add_argument("--blocks",type=int,default=20)
    p.add_argument("--seed",type=int,default=17137)
    run(p.parse_args())
