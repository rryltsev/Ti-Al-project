"""Compare long random/ordered S3 checks and replace the tested window in the TI isotherm."""
import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
from vcsgc import KB, correlation_time, integrate


def load(path):
    groups=defaultdict(list)
    with gzip.open(path,"rt",newline="") as f:
        for row in csv.DictReader(f):
            key=(row["model"],int(float(row["T_K"])),int(row["N"]),float(row["c0"]),
                 float(row["kappa"]),float(row["delta_mu0_eV"]),int(row["replica"]))
            groups[key].append((int(row["sweep"]),int(row["n_Al"])/key[2],float(row["m"]),float(row["volume_A3_atom"])))
    arrays={}
    for key,rows in groups.items():
        data=np.asarray(rows)
        data=data[np.argsort(data[:,0])]
        if not np.all(np.diff(data[:,0])==10):
            raise ValueError(f"Noncontiguous production sampling in {key}")
        arrays[key]=data[:,1:]
    return arrays


def stats(x, parts=20):
    chunks=np.array_split(x,parts)
    avg=np.stack([c.mean(axis=0) for c in chunks])
    return x.mean(axis=0),avg.std(axis=0,ddof=1)/np.sqrt(parts)


def standardized(a,b,sa,sb):
    den=np.hypot(sa,sb)
    return np.divide(np.abs(a-b),den,out=np.zeros_like(den),where=den>1e-12)


def contact_set(settings, means, temperature):
    c,order=means[:,:2].T
    center,k,mu0=settings.T
    _,_,all_contacts=integrate(c,mu0-2*KB*temperature*k*(c-center),.005)
    sort=np.argsort(c)
    result=[]
    for p in all_contacts:
        if not p["window_edge"]:
            ends=np.array([p["alpha_at_pct"],p["alpha2_at_pct"]])
            m=np.interp(ends/100,c[sort],order[sort])
            result.append((ends,m[1]-m[0]))
    return result


def run(args):
    seeds={}
    with args.windows.open() as f:
        for row in csv.reader(f,delimiter="\t"):
            if row[0].startswith("#") or row[0]=="model":
                continue
            if row[10] not in ("random","ordered") or tuple(map(int,row[8:10]))!=(50000,400000):
                raise ValueError("Unexpected long-run settings")
            seeds[int(row[7])]=row[10]
    long={}
    for path in args.long:
        part=load(path)
        if long.keys() & part.keys():
            raise ValueError("Duplicate long trajectory across files")
        long.update(part)
    observed={k[-1] for k in long}
    if (not observed<=seeds.keys() or (not args.partial and observed!=seeds.keys())):
        raise ValueError("Missing or duplicate long trajectories")
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/"long_mixing.csv").open("w",newline="") as fm, (
            args.out/"long_contact_sensitivity.csv").open("w",newline="") as fc:
        wm,wc=csv.writer(fm),csv.writer(fc)
        wm.writerow(("model","T_K","c0","observable","base_mean","base_SE","base_tau","random_mean","random_SE",
                     "random_tau","ordered_mean","ordered_SE","ordered_tau","rand_vs_order_z","base_vs_rand_z",
                     "base_vs_order_z","rand_halves_z","order_halves_z","base_block_frames","long_block_frames"))
        wc.writerow(("model","T_K","c0_replaced","trajectory_choice","global_internal_gaps",
                     "rising_gaps","alpha_at_pct","alpha2_at_pct","width_at_pct","delta_width_from_baseline",
                     "other_gaps_at_pct"))
        for key in sorted({k[:-1] for k in long}):
            model,t,n,c0,kappa,mu0=key
            by_start={seeds[k[-1]]:x for k,x in long.items() if k[:-1]==key}
            if set(by_start)!={"random","ordered"} or any(len(x)!=40000 for x in by_start.values()):
                raise ValueError(f"Expected two complete 40000-frame checks for {key}")
            orig=load(args.base/f"{model}_{t}"/"samples.csv.gz")
            candidates=[x for k,x in orig.items() if k[:-1]==key]
            if len(candidates)!=1 or len(candidates[0])!=4000:
                raise ValueError("Could not match a unique baseline window")
            base=candidates[0]
            arrays=[base,by_start["random"],by_start["ordered"]]
            summaries=[stats(x) for x in arrays]
            taus=[[correlation_time(y) for y in x.T] for x in arrays]
            contrasts=[standardized(summaries[a][0],summaries[b][0],summaries[a][1],summaries[b][1])
                       for a,b in ((1,2),(0,1),(0,2))]
            halves=[]
            for x in arrays[1:]:
                first,second=np.array_split(x,2)
                av1,se1=stats(first,10); av2,se2=stats(second,10)
                halves.append(standardized(av1,av2,se1,se2))
            for j,label in enumerate(("c_Al","order_m","volume_A3_atom")):
                details=[z for h,(v,se) in enumerate(summaries) for z in (v[j],se[j],taus[h][j])]
                wm.writerow((model,t,c0,label,*details,*(z[j] for z in contrasts),*(z[j] for z in halves),200,2000))
            ordered_keys=sorted(orig)
            settings=np.array([[k[3],k[4],k[5]] for k in ordered_keys])
            means=np.stack([orig[k].mean(axis=0) for k in ordered_keys])
            idx=next(i for i,k in enumerate(ordered_keys) if k[:-1]==key)
            contrasts={"baseline":summaries[0][0],"random":summaries[1][0],"ordered":summaries[2][0],
                       "pooled_long":.5*(summaries[1][0]+summaries[2][0])}
            w0=None
            for name,value in contrasts.items():
                means[idx]=value
                gaps=contact_set(settings,means,t)
                rising=[p for p,d in gaps if d>0]
                width=rising[0][1]-rising[0][0] if len(rising)==1 else None
                if name=="baseline":
                    w0=width
                wc.writerow((model,t,c0,name,len(gaps),len(rising),*(rising[0] if len(rising)==1 else ["",""]),
                             width,width-w0 if w0 is not None and width is not None else "",
                             "; ".join(f"{p[0]:.4f} to {p[1]:.4f}" for p,d in gaps if d<=0)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",type=Path,required=True,help="S3 parent of per-isotherm directories")
    p.add_argument("--long",type=Path,required=True,nargs="+")
    p.add_argument("--windows",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--partial",action="store_true",help="input contains complete paired checks for a subset of isotherms")
    run(p.parse_args())
