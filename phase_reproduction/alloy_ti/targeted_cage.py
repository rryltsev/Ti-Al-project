"""High-statistics two-proposal audit of four dominant Ti-Al decorations in the W<0.40 domain."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np
from scipy.linalg import null_space
from scipy.special import logsumexp

CAP = .40
SEED = 8693777
BLOCKS = 4
IDS = ("k02_0-13", "k03_0-2-13", "k03_0-1-14", "k04_0-2-13-15")
METHODS = ("gaussian", "student")
KT = 8.617333262145e-5*1000.
MODEL_HASH = "704f7fd900c7446286066b741084c30ebc7fcf26e5b9b8fd2f29a18ba4ffd661"


def gaussian(n,dim,rng):
    from pilot import SCALES
    z=rng.normal(size=(n,dim))*SCALES[np.arange(n)%len(SCALES),None]
    r2=np.sum(z*z,axis=1)
    density=-dim*np.log(SCALES)-r2[:,None]/(2*SCALES**2)
    return z,r2,np.log(len(SCALES))-.5*r2-logsumexp(density,axis=1)


def generate(task):
    from pilot import Frame, evaluate, force_constants, site_planes
    from hcp_orbits import hcp_geometry
    from tail_check import proposal
    row, method, group, n, mlp, model, directory=task
    cell,sites=hcp_geometry((2,2,2))
    ids=row["representative_Al_indices"]
    types=np.zeros(16,dtype=int)
    types[ids]=1
    def calc(frames):
        return evaluate(frames,mlp,model,3)
    frame=calc([Frame(cell,np.array(row["relaxed_positions_A"]),types)])[0]
    if (abs(frame.energy-row["energy_eV_cell"])>2e-6 or
            abs((frame.positions-sites).mean(axis=0)).max()>1e-8):
        raise ValueError("Relaxed energy or displacement gauge mismatch")
    basis=null_space(np.tile(np.eye(3),(1,16)))
    vals,vec=np.linalg.eigh(basis.T@force_constants(calc,frame,.0025)@basis)
    if vals.min()<=0 or abs(.5*KT*np.log(vals/(2*np.pi*KT)).sum()-row["harmonic_config_eV_cell"])>2e-5:
        raise ValueError("Inconsistent Cartesian harmonic partition")
    transform=(basis@vec*np.sqrt(KT/vals)).T
    planes=site_planes(cell,sites)
    seed=[SEED,16,sum(1<<i for i in ids),METHODS.index(method),group]
    sampler=gaussian if method=="gaussian" else proposal
    z,r2,rho=sampler(n,len(vals),np.random.default_rng(seed))
    positions=frame.positions+(z@transform).reshape(n,16,3)
    wall=np.einsum("bni,nki->bnk",positions-sites,planes).max(axis=(1,2))
    valid=np.flatnonzero(wall<CAP)
    delta=np.zeros(n)
    for i in range(0,len(valid),256):
        idx=valid[i:i+256]
        energies=[r.energy for r in calc([Frame(cell,positions[j],types) for j in idx])]
        delta[idx]=np.asarray(energies)-frame.energy-.5*KT*r2[idx]
    lh=np.where(wall<CAP,rho,-np.inf)
    la=lh-delta/KT
    lognorm=logsumexp(la)
    filename=f"{method}_{row['id']}_g{group}.npz"
    np.savez_compressed(directory/filename,delta_eV_cell=delta,log_harmonic_to_proposal=rho,
                        max_wall_fraction=wall,rng_seed=seed,cap=CAP)
    return dict(id=row["id"],method=method,group=group,raw_file=filename,
                draws=n,inside=int(len(valid)),log_Z0_over_unrestricted_HA=float(logsumexp(lh)-np.log(n)),
                log_Z1_over_unrestricted_HA=float(lognorm-np.log(n)),
                endpoint_ess=float(np.exp(2*lognorm-logsumexp(2*la))),
                largest_weight=float(np.exp(la.max()-lognorm)))


def run(args):
    if not 1<=args.workers<=12 or args.draws<2 or args.draws%2:
        raise ValueError("Use 1–12 workers and a positive even number of draws per group")
    dest=args.output_dir.resolve()
    dest.mkdir(parents=True,exist_ok=True)
    model=args.model.resolve(strict=True)
    ref=json.loads(args.static.read_text())
    if (sha256(model.read_bytes()).hexdigest()!=MODEL_HASH or ref["model_sha256"]!=MODEL_HASH
            or ref["temperature_K"]!=1000 or ref["N"]!=16 or ref["shape"]!=[2,2,2]):
        raise ValueError("Unexpected input model, temperature or cell")
    rows={r["id"]:r for r in ref["results"]}
    tasks=[(rows[name],m,g,args.draws,args.mlp,str(model),dest)
           for g in range(BLOCKS) for name in IDS for m in METHODS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        summaries=list(pool.map(generate,tasks))
    output=dict(model_sha256=MODEL_HASH,static_sha256=sha256(args.static.read_bytes()).hexdigest(),
                cap=CAP,N=16,temperature_K=1000,ids=IDS,methods=METHODS,groups=BLOCKS,
                draws_per_group=args.draws,seed_root=SEED,blocks=summaries)
    (dest/"targeted_blocks.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")


def saved_student(archive_path):
    data={}
    with tarfile.open(archive_path) as tar:
        for entry in tar:
            if entry.name.startswith("cage_k"):
                with np.load(BytesIO(tar.extractfile(entry).read())) as f:
                    if float(f["cap"])!=CAP:
                        raise ValueError("Prior Student run inconsistent cap")
                    data[Path(entry.name).stem.removeprefix("cage_")]=f["data"].transpose(1,0,2)
    return data


def load_new(path,blocks):
    from cage_sensitivity import npz_data
    answer={}
    with tarfile.open(path) as tar:
        for method in METHODS:
            for name in IDS:
                draws=[]
                for g in range(BLOCKS):
                    meta=blocks[(method,name,g)]
                    raw=npz_data(BytesIO(tar.extractfile(meta["raw_file"]).read()))
                    d,r,w=raw
                    lh=np.where(w<CAP,r,-np.inf)
                    if (len(d)!=meta["draws"] or
                            abs(logsumexp(lh)-np.log(len(d))-meta["log_Z0_over_unrestricted_HA"])>1e-10 or
                            abs(logsumexp(lh-d/KT)-np.log(len(d))-meta["log_Z1_over_unrestricted_HA"])>1e-10):
                        raise ValueError("Archived draws do not reproduce block partition")
                    draws.append(raw)
                answer[(method,name)]=np.array(draws).transpose(1,0,2)
    return answer


def influence(measured,rows):
    by_name={}
    for k,coeff in ((2,1),(3,-2),(4,1)):
        group=[r for r in rows if r["N_Al"]==k]
        logd=np.log([r["degeneracy"] for r in group])
        def lp(method):
            return logd-np.array([r["harmonic_config_total_eV_cell"]+
                                  measured[method][r["id"]][1][0] for r in group])/KT
        g,t=lp("gaussian"),lp("student")
        total=logsumexp(g)
        logr=logsumexp(t)-total
        divdiff=1. if abs(logr)<1e-14 else logr/np.expm1(logr)
        share=-coeff*1000*KT*divdiff*(np.exp(t-total)-np.exp(g-total))
        for r,val in zip(group,share):
            by_name[r["id"]]=float(val)
    return dict(sorted(by_name.items(),key=lambda v:-abs(v[1])))


def analyze(args):
    from cage_sensitivity import archive_data, measure, n16_observables
    orig=args.runs/"alloy-ti-n16ti-01"/"outputs"
    refined=args.runs/"alloy-ti-n16ref-01"/"outputs"
    caged=args.runs/"alloy-ti-caged-01"/"outputs"
    meta=json.loads(args.blocks.read_text())
    run=json.loads((refined/"n16_refined.json").read_text())
    if (meta["model_sha256"]!=run["model_sha256"] or meta["static_sha256"]!=run["static_sha256"]
            or meta["cap"]!=CAP or meta["groups"]!=BLOCKS or set(meta["ids"])!=set(IDS)):
        raise ValueError("Incompatible targeted and original inputs")
    blocks={(r["method"],r["id"],r["group"]):r for r in meta["blocks"]}
    if len(meta["blocks"])!=32 or len(blocks)!=32:
        raise ValueError("Expected exactly 32 independent blocks")
    new=load_new(args.archive,blocks)
    rows=run["results"]
    first=archive_data(orig/"n16_draws.tar.gz","raw_")
    first.update(archive_data(refined/"n16_refined_draws.tar.gz","raw_"))
    data={"gaussian":first,"student":saved_student(caged/"caged_draws.tar.gz")}
    names={r["id"] for r in rows}
    if any(set(v)!=names for v in data.values()):
        raise ValueError("Original orbits incomplete")
    result={}
    for cap in (.35,.40):
        original={m:{name:measure(v,cap) for name,v in d.items()} for m,d in data.items()}
        measured={m:dict(orbitvals) for m,orbitvals in original.items()}
        for (m,name),draws in new.items():
            measured[m][name]=measure(draws,cap)
        summary={m:n16_observables(measured[m],rows) for m in METHODS}
        prior={m:n16_observables(original[m],rows) for m in METHODS}
        discrepancy={}
        for quantity in ("central_second_difference_target_eV","central_second_difference_anharmonic_shift_eV",
                         "middle_excess_target_meV_atom","middle_excess_anharmonic_shift_meV_atom"):
            g,t=(summary[m][quantity] for m in METHODS)
            discrepancy[quantity]=dict(student_minus_gaussian=t["value"]-g["value"],
                                       independent_SE=float(np.hypot(t["se"],g["se"])))
        result[f"{cap:.2f}"]=dict(estimates=summary,independent_discrepancy=discrepancy,
                                  prior_target_second_difference_eV={m:prior[m]["central_second_difference_target_eV"] for m in METHODS},
                                  student_minus_gaussian_orbit_contributions_meV_cell=influence(measured,rows),
                                  targeted_orbit_diagnostics={m:{j:measured[m][j][2] for j in IDS} for m in METHODS})
    output=dict(cap=CAP,inputs={p.name:sha256(p.read_bytes()).hexdigest() for p in
                (args.blocks,args.archive,orig/"n16_draws.tar.gz",refined/"n16_refined_draws.tar.gz",
                 refined/"n16_refined.json",caged/"caged_draws.tar.gz")},
                replaced_ids=IDS,retained_ids=sorted(names-set(IDS)),draws_per_targeted_orbit_per_method=BLOCKS*meta["draws_per_group"],
                scope="Each estimate combines its own independent replacement samples for four orbits with its matching previous estimate for the other 39. The cage is imposed in both harmonic and direct configurational partitions.",
                caps=result)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    sub=cli.add_subparsers(dest="command",required=True)
    r=sub.add_parser("run")
    r.add_argument("--model",type=Path,required=True)
    r.add_argument("--mlp",required=True)
    r.add_argument("--static",type=Path,default=Path("n16_static.json"))
    r.add_argument("--workers",type=int,default=12)
    r.add_argument("--draws",type=int,default=8192)
    r.add_argument("--output-dir",type=Path,default=Path("."))
    a=sub.add_parser("analyze")
    a.add_argument("--runs",type=Path,default=Path(__file__).resolve().parents[1]/"cluster/runs")
    a.add_argument("--blocks",type=Path,required=True)
    a.add_argument("--archive",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    (run if args.command=="run" else analyze)(args)
