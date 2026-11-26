"""Reweight single-temperature umbrella trajectories locally in T and chemical field.

E is the total unbiased configurational energy in eV. All volume/shape measure
factors cancel at p=0 with identical cell coordinates, cage and shape measure.
Independent source temperatures are kept separate to test their predictions.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "solvus_temperatures/analysis"))
from isotherm import load, mbar


def source_temperature(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return float(next(csv.DictReader(f)).get("T_K") or 1000.)


def load_source(manifests):
    # The historical 1000 K manifests predate the temperature column.
    if all(source_temperature(p) == 1000 for p in manifests):
        groups = mbar.load_runs(manifests, None, 0, 1000.)
        (n, shape, cap, axial), runs = next(iter(groups.items()))
        if len(groups) != 1:
            raise ValueError("Only one cell and mechanical ensemble can be reweighted at a time")
        return 1000., n, shape, cap, axial, runs
    src, = list(load(manifests, 0))
    return src


def reference(n, runs, t0):
    k, q, specs, blocks, diag, _ = mbar.make_data(n, runs)
    count, vsum = mbar.aggregate(blocks, len(specs))
    h = count.sum(axis=0)
    ref, logmass, offsets, overlap = mbar.fit_mbar(k, q, specs, count, kt=mbar.KB*t0)
    code = np.rint(3*n*q/4).astype(int)
    occupied = k*(3*n+1)+code
    energies, inflation, indices, segments = [], [], [], []
    offset=0
    for run, diagnostic in zip(runs, diag):
        sweeps, energy = np.loadtxt(run["path"], delimiter=",", skiprows=1, usecols=(0,8), unpack=True)
        present = np.searchsorted(sweeps, run["sweep"])
        if not np.array_equal(sweeps[present], run["sweep"]):
            raise ValueError("Energy and composition are out of alignment in "+run["run"])
        # Large constant offsets are harmless but centering improves precision.
        emu = energy[present] - ref*run["k"]
        energies.append(emu)
        g = max(mbar.tau_int(emu), diagnostic["statistical_inefficiency"])
        inflation.append(np.full(len(emu),g))
        indices.append(np.searchsorted(occupied,run["k"]*(3*n+1)+run["code"]))
        bias_id=next(i for i,s in enumerate(specs) if tuple(s)==tuple(run["spec"]))
        segments.append((offset,offset+len(emu),bias_id,g))
        offset+=len(emu)
    idx = np.concatenate(indices)
    if not np.array_equal(np.bincount(idx),h):
        raise ValueError("Compressed and frame-level states disagree")
    intrinsic = np.concatenate(energies)
    return dict(k=k,q=q,ref=ref,h=h,volume=vsum/h,idx=idx,
                energy=intrinsic-np.mean(intrinsic),g=np.concatenate(inflation),
                log_frame=(logmass-np.log(h))[idx],specs=specs,offsets=offsets,hist=count,segments=segments,
                lambda2=float(np.sort(np.real(np.linalg.eigvals(overlap)))[-2]) if len(specs)>1 else 0.)


def tilted_weights(model, t0, t):
    idx, k = model["idx"], model["k"]
    # At the common reference field the exact temperature tilt is
    # exp[-(beta-beta0)(E-mu_ref*n_Al)]. No phonon/mass model enters.
    a = model["log_frame"] - (1/t-1/t0)/mbar.KB * model["energy"]
    base = np.full(len(k), -np.inf)
    np.logaddexp.at(base, idx, a)
    log_w_conditional = a-base[idx]
    inv_neff = np.bincount(idx,weights=np.exp(2*log_w_conditional)*model["g"],minlength=len(k))
    return base, inv_neff


def solve(model, n, t0, t, fields, bandwidth):
    k,q,ref = model["k"],model["q"],model["ref"]
    base,inverse = tilted_weights(model,t0,t)
    kt=mbar.KB*t
    def stats(mu, cut):
        logs=base+(mu-ref)*k/kt
        w=np.exp(logs-logsumexp(logs))
        out=[]
        for sel in (q<=cut,q>cut):
            p=w[sel]/w[sel].sum()
            out.append(dict(mean_x_atpct=float(100*p@k[sel]/n),mean_Q=float(p@q[sel]),
                            effective_frames_time_corrected=float(1/((p*p)@inverse[sel]))))
        return out
    roots={}
    for branch,direction in (("Ti_rich",1),("Al_rich",-1)):
        sep=mbar.height_separator(k,q,base,ref,bandwidth,kt=kt,field_bounds=fields,direction=direction)
        result=None
        if sep:
            fit=mbar.individual_fit(n,k,q,base,ref,model["h"],model["volume"],bandwidth,4,
                                    fixed_cut=sep["order_cut"],kt=kt,field_bounds=fields,direction=direction)
            result=dict(equal_height_field=sep["mu_eV"],beta_barrier=sep["barrier_kBT"],
                        Q_modes=sep["basin_mode_q"],Q_cut=sep["order_cut"],
                        height_phase_stats=stats(sep["mu_eV"],sep["order_cut"]),single_variant=None)
            if fit:
                result["single_variant"]=dict(mu_eV=fit["mu_eV"],width_pp=100*fit["width"],
                                               phases=stats(fit["mu_eV"],sep["order_cut"]))
        roots[branch]=result
    return dict(T_K=t, roots=roots)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests",type=Path,nargs="+")
    p.add_argument("--temps",type=float,nargs="+",required=True)
    p.add_argument("--field-range",type=float,nargs=2,required=True)
    p.add_argument("--bandwidth",type=float,default=.04)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    t0,n,shape,cap,axial,runs=load_source(a.manifests)
    model=reference(n,runs,t0)
    data=dict(source_T_K=t0,N=n,shape=shape,W_max=cap,axial_free=axial,pressure_GPa=0.,
              source_fields=sorted({r["spec"][0] for r in runs}), lambda2=model["lambda2"],
              field_bounds=a.field_range,Q_bandwidth=a.bandwidth,
              manifests=[str(p.resolve().relative_to(ROOT.parent).as_posix()) for p in a.manifests],
              notes="Local temperature reweighting with unbiased total E. Secondary check only. Disconnected or low-ESS phases and fields far outside the source windows can give spurious roots.",
              temperatures=[solve(model,n,t0,t,a.field_range,a.bandwidth) for t in a.temps])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
