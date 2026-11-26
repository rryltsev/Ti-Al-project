"""Independent unrestrained-order controls at MBAR fields, conditional on each phase."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from analysis import KT, tau_int

NAMES = ["x", "Q", "volume_A3_atom", "c_a", "wall", "U_eV_atom",
         "Al_richest_sublattice", "Al_2nd_sublattice", "Al_3rd_sublattice", "Al_4th_sublattice"]


def read_chains(manifest):
    chains = []
    for row in csv.DictReader(manifest.open()):
        assert float(row["K_eV"]) == 0 and int(row["axial_free"]) == 1
        allrows = np.genfromtxt(manifest.parent/(row["run"]+".csv"),names=True,delimiter=",")
        r = allrows[allrows["sweep"]>int(row["burn"])]
        n = int(row["N"])
        ordered = np.sort(np.stack([r["n"+str(i)] for i in range(4)],axis=1),axis=1)[:,::-1]
        assert np.all(ordered.sum(axis=1)==r["n_Al"])
        obs = np.column_stack([r["n_Al"]/n,r["Q"],r["volume_A3_atom"],r["c_a"],r["wall"],
                               r["E_eV"]/n,4*ordered/n])
        chains.append(dict(name=row["run"], seed=int(row["seed"]),start=row["start"],N=n,cap=float(row["cap"]),
                           mu=float(row["mu_eV"]),k=r["n_Al"],sweep=r["sweep"],q=r["Q"],obs=obs,
                           whole_q=allrows["Q"],whole_sweep=allrows["sweep"]))
    if len({(r["N"],r["cap"],r["mu"]) for r in chains}) != 1:
        raise ValueError("Expected one size, cage and sampled field per control batch")
    return chains


def summarize(chains,mu,cut):
    # Each phase gets its own normalization. Initial replica counts do not
    # define a relative phase free energy and never enter the coexistence field.
    result = []
    for high in (False,True):
        logs,values,gs = [],[],[]
        for r in chains:
            take = (r["q"]>cut)==high
            logs.append(((mu-r["mu"])*r["k"]/KT)[take])
            values.append(r["obs"][take])
            gs.append(np.full(np.sum(take),r["g"]))
        ell,observables,g = np.concatenate(logs),np.vstack(values),np.concatenate(gs)
        if not len(ell):
            result.append(None)
            continue
        w = np.exp(ell-logsumexp(ell))
        means = w @ observables
        result.append(dict(means=dict(zip(NAMES,means)),frames=len(ell),
                           semi_grand_U_minus_muN_eV_atom=means[5]-mu*means[0],
                           Kish_frames=1/np.sum(w**2),approx_correlated_Kish=1/np.sum(g*w**2)))
    return result


def transitions(r,lo,hi,cut):
    phase = None
    entries = []
    for q,s in zip(r["q"],r["sweep"]):
        hit = "alpha" if q<=lo else "alpha2" if q>=hi else None
        if hit and hit!=phase:
            entries.append([int(s),hit])
            phase = hit
    return dict(core_visits=entries,core_to_core_transitions=max(0,len(entries)-1),
                q_cut_crossings=int(np.count_nonzero(np.diff(r["q"]>cut))),
                samples=[int(np.sum(r["q"]<=cut)),int(np.sum(r["q"]>cut))])


def split_halves(chains,half):
    out = []
    for r in chains:
        a = len(r["q"])//2
        only = slice(0,a) if half==0 else slice(a,None)
        out.append({**r,**{k:r[k][only] for k in ("q","obs","k","sweep")}})
    return out


def x_sample(r,mu,cut,draws):
    # Work only with composition and weights inside the bootstrap loops.
    ks,qs,ns,source = r
    w = np.exp((mu-source)*(ks-ks.mean())/KT)*draws
    sums = []
    for mask in (qs<=cut,qs>cut):
        ww = w[mask]
        sums.append(ww @ ks[mask]/(ns*ww.sum()) if ww.sum() else np.nan)
    return np.asarray(sums)


def sample_intervals(chains,mu,cut,mbar,draw_count,seed):
    rng = np.random.default_rng(seed)
    groups = [[i for i,r in enumerate(chains) if r["start"]==s] for s in ("random","ordered")]
    sizes = np.array([len(r["k"]) for r in chains])
    lengths = np.array([min(n,max(1,int(np.ceil(2*r["g"])))) for r,n in zip(chains,sizes)])
    args = (np.concatenate([r["k"] for r in chains]),np.concatenate([r["q"] for r in chains]),
            chains[0]["N"],chains[0]["mu"])
    store = {design:{"conditional":[],"with_field":[],"minus_umbrella":[]} for design in ("replica","block")}
    for _ in range(draw_count):
        joint = mbar[rng.integers(len(mbar))]
        bootn = np.zeros(len(chains),dtype=int)
        for group in groups:
            bootn += np.bincount(rng.choice(group,len(group)),minlength=len(chains))
        replica = np.repeat(bootn,sizes)
        allblocks = []
        for n,block in zip(sizes,lengths):
            # Circular moving-block bootstrap within each actual replica.
            starts = rng.integers(n,size=(n+block-1)//block)
            idx = (starts[:,None]+np.arange(block))%n
            allblocks.append(np.bincount(idx.ravel()[:n],minlength=n))
        for label,counts in (("replica",replica),("block",np.concatenate(allblocks))):
            fixed = x_sample(args,mu,cut,counts)
            target = x_sample(args,joint[0],cut,counts)
            if np.isfinite(fixed).all() and np.isfinite(target).all():
                store[label]["conditional"].append(fixed)
                store[label]["with_field"].append(target)
                store[label]["minus_umbrella"].append(target-joint[1:3])
    result = {}
    for design,records in store.items():
        result[design] = {key:np.quantile(v,[.025,.975],axis=0).tolist() if v else None for key,v in records.items()}
        result[design]["resolved"] = len(records["conditional"])
    return result


def group_fit(manifest,results,bootstrap):
    chains = read_chains(manifest)
    n,cap = chains[0]["N"],chains[0]["cap"]
    group, = [g for g in results.values() if g["N"]==n and g["cap"]==cap and g["axial_free"]==1]
    fit = group["equal_single_variant"]
    mu,cut = fit["mu_eV"],fit["order_cut"]
    modes = np.array(fit["basin_mode_q"])
    lo,hi = .5*(modes+cut)
    records = []
    for r in chains:
        with_indicator = np.column_stack([r["obs"][:,:4],r["q"]>cut])
        gs = [tau_int(col) for col in with_indicator.T]
        r["g"] = max(gs)
        complete = transitions(dict(q=r["whole_q"],sweep=r["whole_sweep"]),lo,hi,cut)
        records.append(dict(name=r["name"],seed=r["seed"],start=r["start"],inefficiencies_x_q_v_ca_indicator=gs,
                            **transitions(r,lo,hi,cut),including_burn_in=complete,
                            phase_results=summarize([r],mu,cut)))
    by_phase = summarize(chains,mu,cut)
    interval = group.get("bootstrap",{}).get("joint_mu_xalpha_xalpha2",[])
    if not interval:
        interval = [[mu,fit["x_alpha"],fit["x_alpha2"]]]
    estimates = [s["means"]["x"] if s else None for s in by_phase]
    report = dict(N=n,cap=cap,series=manifest.parent.name,
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  sampled_mu_eV=chains[0]["mu"],target_mu_eV=mu,MBAR_order_cut=cut,
                  q_core_limits=[lo,hi],MBAR_phase_x=[fit["x_alpha"],fit["x_alpha2"]],
                  MBAR_phase_volumes_A3_atom=fit["phase_volumes_A3_atom"],controls_phase_x=estimates,
                  phase_results=by_phase,replicas=records,
                  production_halves=[summarize(split_halves(chains,h),mu,cut) for h in (0,1)],
                  divider_check=[dict(cut=c,phase_results=summarize(chains,mu,c)) for c in (cut-.05,cut+.05)],
                  bootstrap=sample_intervals(chains,mu,cut,np.asarray(interval),bootstrap,394678+n+round(100*cap)))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("analysis",type=Path)
    p.add_argument("manifests",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=1000)
    a = p.parse_args()
    data = json.loads(a.analysis.read_text())
    report = dict(method="Separate normalization conditional on phase. Bootstrap columns are alpha, alpha2; "
                  "rows are 2.5% and 97.5% quantiles. Replica draws are stratified by starting phase. "
                  "Block draws keep replicas fixed. Field uncertainty uses joint umbrella bootstrap values.",
                  umbrella_analysis=a.analysis.name,umbrella_sha256=hashlib.sha256(a.analysis.read_bytes()).hexdigest(),
                  groups=[group_fit(m,data,a.bootstrap) for m in a.manifests])
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    for g in report["groups"]:
        print(g["N"],g["cap"],g["target_mu_eV"],g["controls_phase_x"],g["MBAR_phase_x"])


if __name__=="__main__":
    main()
