"""Temperature-resolved direct MTP analysis. Each manifest belongs to one NPT temperature."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "solvus_1000"))
import analysis as mbar


def load(manifests, discard):
    temperatures = defaultdict(list)
    settings = {}
    for path in manifests:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        t, = {float(r["T_K"]) for r in rows}
        if not np.isfinite(t) or t <= 0:
            raise ValueError(f"Invalid sampling temperature for {path}")
        temperatures[t].append(path)
        for r in rows:
            if r["run"] in settings:
                raise ValueError("Duplicate trajectory " + r["run"])
            settings[r["run"]] = r
    for t, paths in temperatures.items():
        for (n, shape, cap, axial), runs in mbar.load_runs(paths, None, 0, t).items():
            for r in runs:
                row = settings[r["run"]]
                burn, length, stride = (int(row[key]) for key in ("burn","sweeps","stride"))
                expected = np.arange((burn//stride+1)*stride, burn+length+1, stride)
                if not np.array_equal(expected, r["sweep"]):
                    raise ValueError("Incomplete, duplicate or truncated production steps for " + r["run"])
                if discard:
                    first = int(len(expected)*discard)
                    for key in r:
                        if isinstance(r[key], np.ndarray):
                            r[key] = r[key][first:]
            yield t, n, shape, cap, axial, runs


def scan(n, runs, kt, bounds, bandwidth):
    """Isotherm without splitting P(Q). Raw/Kish effective counts concern sampled states only."""
    k,q,specs,blocks,diag,moments = mbar.make_data(n,runs)
    hist,vsum = mbar.aggregate(blocks,len(specs))
    h = hist.sum(axis=0)
    ref,base,_,overlap = mbar.fit_mbar(k,q,specs,hist,kt=kt)
    g = np.array([d["statistical_inefficiency"] for d in diag])
    counts = np.array([np.asarray(block[1].sum(axis=0)).ravel() for block in blocks])
    inflated_counts = g @ counts
    means = np.array([moments[f"Al_fraction_sublattice_rank_{j+1}"]/h for j in range(4)])
    bc = mbar.connectivity(hist,k,q,n)
    tree = minimum_spanning_tree(-bc).data
    fields = np.unique(np.r_[np.linspace(*bounds,81),specs[:,0]])
    records = []
    for mu in fields:
        logits = base+(mu-ref)*k/kt
        w = np.exp(logits-logsumexp(logits))
        mean_k,mean_q = w@k,w@q
        grid,density,peakidx = mbar.order_profile(q,w,bandwidth)
        peaks = []
        for i in peakidx:
            if density[i] > .02*np.max(density):
                peaks.append([float(grid[i]),float(density[i])])
        pair = None
        for i,index in enumerate(peakidx):
            for j in peakidx[i+1:]:
                if grid[j]-grid[index] < 2*bandwidth:
                    continue
                valley = index+np.argmin(density[index:j+1])
                if valley in (index,j) or density[valley] <= 0:
                    continue
                high = q>grid[valley]
                p_hi = float(w[high].sum())
                if not .05 < p_hi < .95:
                    continue
                height = float(np.log(min(density[index],density[j])/density[valley]))
                if pair is None or height > pair["barrier_from_smaller_peak_kBT"]:
                    pair = dict(peaks_Q=grid[[index,j]].tolist(),valley_Q=float(grid[valley]),
                                barrier_from_smaller_peak_kBT=height,ordered_side_weight=p_hi)
        per_frame = w/h
        var_x=float((w@(k-mean_k)**2)/n**2)
        records.append(dict(mu_eV=float(mu),x_Al=float(mean_k/n),mean_Q=float(mean_q),
                            dx_dmu_eV_inv=n*var_x/kt,composition_sd=var_x**.5,
                            maximum_conditional_mean_gap_at_odds_four=2.5*var_x**.5,
                            Nvar_Q=float(n*(w@(q-mean_q)**2)),
                            Al_fraction_ranked_sublattices=(means@w).tolist(),
                            mean_V_A3_atom=float(w@(vsum/h)),mean_c_a=float(w@(moments["mean_c_a"]/h)),
                            effective_frames=float(1/(h@(per_frame**2))),
                            effective_frames_with_time_correlation=float(1/(inflated_counts@(per_frame**2))),
                            local_modes_Q_and_density=peaks,two_well_profile=pair))
    eigenvalues = sorted(np.real(np.linalg.eigvals(overlap)),reverse=True)
    return dict(grid=records, weakest_tree_Bhattacharyya=float(np.min(-tree)) if len(tree)==len(specs)-1 and len(tree) else 0.,
                mbar_second_eigenvalue=float(eigenvalues[1]) if len(eigenvalues)>1 else None,
                direct_windows=diag)


def analyze(manifests, output, boot=0, discard=0., field_bounds=None, bandwidth=.04, seed=26713, quick=False,descending=False,
            restrict_k0_cut=None):
    ensembles = []
    for t,n,shape,cap,axial,runs in load(manifests,discard):
        kt = mbar.KB*t
        if restrict_k0_cut is not None:
            for r in runs:
                basin=0
                if r["spec"][1]==0:
                    trace=4*r["code"]/(3*n)
                    if np.all(trace<=restrict_k0_cut):
                        basin=-1
                    elif np.all(trace>restrict_k0_cut):
                        basin=1
                    else:
                        raise ValueError("Unbiased trajectory crossed the selected basin cutoff: "+r["run"])
                r["spec"]=(*r["spec"],basin,restrict_k0_cut)
        fields = sorted({r["spec"][0] for r in runs})
        bounds = field_bounds or [fields[0] - .015, fields[-1] + .015]
        raw_scan = scan(n,runs,kt,bounds,bandwidth)
        direction=-1 if descending else 1
        result = dict(temperature_K=t,N=n,shape=shape,cap=cap,axial_free=axial,pressure_GPa=0.,direction=direction,
                      conditional_gap_note="At a given sampled field, any partition of P(x) into groups of weights "
                      "1/5 and 4/5 satisfies |mean_x1-mean_x2| <= 2.5 sd(x), by the law of total variance. This "
                      "is a bound for that finite-cell distribution, not an uncertainty on a bulk phase boundary.",
                      n_runs=len(runs), searched_mu_range_eV=bounds, scan=raw_scan)
        if restrict_k0_cut is not None:
            result["restricted_K0_basin_cut_Q"]=restrict_k0_cut
            result["restricted_K0_note"]="Each unbiased run stayed in one assigned basin. K=0 partition functions " \
                "are separate for Q <= cutoff and Q > cutoff; the number of starts sets no phase weights."
        if not quick:
            fit=mbar.analysis(n,runs,boot,seed,bandwidth,kt=kt,field_bounds=bounds,direction=direction)
            result["coexistence"] = fit
            for name,part in [("early",True),("late",False)]:
                short=mbar.half_runs(runs,part)
                if short:
                    result.setdefault("time_halves",{})[name]=mbar.central_subset(n,short,bandwidth,kt=kt,field_bounds=bounds,direction=direction)
        ensembles.append(result)
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix(output.suffix+".tmp")
    tmp.write_text(json.dumps(ensembles,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(output)
    for item in ensembles:
        fit = item.get("coexistence",{}).get("equal_single_variant")
        r=item["scan"]["grid"]
        print(f"T={item['temperature_K']} N={item['N']} W={item['cap']}: x={100*r[0]['x_Al']:.2f}..{100*r[-1]['x_Al']:.2f}%, "
              f"min tree BC={item['scan']['weakest_tree_Bhattacharyya']:.3f}; "
              f"candidate {None if fit is None else [fit[k] for k in ['mu_eV','x_alpha','x_alpha2']]}")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests",nargs="+",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=0)
    p.add_argument("--discard",type=float,default=0.)
    p.add_argument("--field-bounds",type=float,nargs=2)
    p.add_argument("--bandwidth",type=float,default=.04)
    p.add_argument("--seed",type=int,default=26713)
    p.add_argument("--quick",action="store_true",help="omit fitting phase ratios and return field scan only")
    p.add_argument("--descending",action="store_true",help="test loss of order as the Al fraction rises")
    p.add_argument("--restrict-k0-cut",type=float,help="treat non-switching K=0 runs as distinct hard-basin ensembles")
    a=p.parse_args()
    if a.bootstrap<0 or not 0<=a.discard<1 or not 0<a.bandwidth<.25 or (a.field_bounds is not None and a.field_bounds[0]>=a.field_bounds[1]):
        p.error("invalid analysis parameters")
    analyze(a.manifests,a.out,a.bootstrap,a.discard,a.field_bounds,a.bandwidth,a.seed,a.quick,a.descending,a.restrict_k0_cut)
