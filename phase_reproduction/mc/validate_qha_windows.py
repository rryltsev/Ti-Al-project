"""Check sampled QHA volumes and MC incremental energy/field consistency."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean

import numpy as np


HERE = Path(__file__).resolve().parent
LOG = re.compile(r"(\d+): samples (\d+) accepted (\d+) accepted_volume (\d+) drift_eV (\S+) max_field_error_eV (\S+)")


def analyze(run, fit_volumes):
    windows = [line.split() for line in (run/"windows.tsv").read_text().splitlines() if line.strip()]
    logs = {}
    for line in (run/"window_logs.txt").read_text().splitlines():
        m = LOG.fullmatch(line.strip())
        if not m:
            raise ValueError(f"Unexpected log record {line!r}")
        index, n, moves, vmoves = map(int,m.groups()[:4])
        if index in logs:
            raise ValueError("Duplicate window log")
        logs[index] = (n, moves, vmoves, float(m[5]), float(m[6]))
    if set(logs) != set(range(1,len(windows)+1)):
        raise ValueError("Incomplete MC window logs")
    drift, field = [], []
    qs = {}
    hashes = json.loads((run/"inputs.json").read_text())["sha256"]
    coefficient_limits = {}
    for index,w in enumerate(windows,1):
        sites = 2*math.prod(map(int,w[2:5]))
        n, _, vmoves, e, h = logs[index]
        if n != int(w[9])//10 or abs(e)/sites > 1e-7 or abs(h) > 1e-7 or not math.isfinite(e+h):
            raise ValueError(f"MC invariant failure at {run.name}/{index}")
        drift.append(abs(e)/sites)
        field.append(abs(h))
        if w[0] != "atat_QHA":
            continue
        vol_path = HERE/"inputs"/run.name/w[11]
        if w[11] not in coefficient_limits:
            staged=vol_path.read_bytes().replace(b"\r\n",b"\n")
            if hashlib.sha256(staged).hexdigest() != hashes[w[11]]:
                raise ValueError("Coefficients differ from deployed copy")
            v = np.loadtxt(vol_path,ndmin=2)[:,0]
            if not np.allclose(v,fit_volumes,rtol=0,atol=1e-11):
                raise ValueError("MC volume grid differs from fitted volume grid")
            coefficient_limits[w[11]]=(float(v[0]),float(v[-1]))
        lo,hi=coefficient_limits[w[11]]
        step = float(w[12])
        seed = int(w[7])
        if seed in qs:
            raise ValueError("Duplicate run seed")
        qs[seed] = dict(run=run.name, window=index, T_K=int(w[1]), N=sites, c0=float(w[5]),
                        n=0, sum_v=0., sum_c=0., min_v=math.inf,max_v=-math.inf,n_low=0,n_high=0,
                        v_lo=lo,v_hi=hi,proposal=step,expected_n=n,start=int(w[8]),
                        volume_acceptance_all=vmoves/(int(w[8])+int(w[9])),
                        abs_drift_eV_atom=abs(e)/sites,max_field_error_eV=abs(h))
    log_metrics = dict(n_windows=len(logs), max_abs_energy_drift_eV_atom=max(drift),
                       max_field_error_eV=max(field), compiled_limit_eV_atom_or_field=1e-7)
    if not qs:
        return log_metrics,[]
    models_by_seed = {int(w[7]): w[0] for w in windows}
    with gzip.open(run/"samples.csv.gz","rt",newline="") as f:
        for row in csv.DictReader(f):
            seed = int(row["replica"])
            if models_by_seed.get(seed) != row["model"]:
                raise ValueError("Unknown sample seed or model")
            if row["model"] != "atat_QHA":
                continue
            d = qs[seed]
            x = float(row["volume_A3_atom"])
            if (row["model"]!="atat_QHA" or d["T_K"]!=int(row["T_K"]) or d["N"]!=int(row["N"])
                or d["c0"]!=float(row["c0"]) or not d["v_lo"]-1e-9 <= x <= d["v_hi"]+1e-9):
                raise ValueError("Invalid window or volume sample")
            d["n"]+=1
            if int(row["sweep"])!=d["start"]+10*d["n"]:
                raise ValueError("Window is missing samples or has wrong stride")
            d["sum_v"]+=x
            d["sum_c"]+=int(row["n_Al"])/d["N"]
            d["min_v"]=min(d["min_v"],x)
            d["max_v"]=max(d["max_v"],x)
            d["n_low"]+=x <= d["v_lo"]+d["proposal"]
            d["n_high"]+=x >= d["v_hi"]-d["proposal"]
    for d in qs.values():
        if d["n"]!=d.pop("expected_n"):
            raise ValueError("Wrong sample count")
        d.pop("start")
        d["mean_v"]=d.pop("sum_v")/d["n"]
        d["mean_cAl"]=d.pop("sum_c")/d["n"]
        d["fraction_low"]=d["n_low"]/d["n"]
        d["fraction_high"]=d["n_high"]/d["n"]
    return log_metrics,list(qs.values())


def summarize(rows):
    def side(key):
        counts=[r[key] for r in rows]
        high=max(rows,key=lambda r:r[key]/r["n"])
        return dict(samples=sum(counts),windows_with_hits=sum(x>0 for x in counts),
                    largest_fraction=high[key]/high["n"],worst_c0=high["c0"] if any(counts) else None,
                    worst_mean_cAl=high["mean_cAl"] if any(counts) else None)
    return dict(windows=len(rows),saved=sum(r["n"] for r in rows),min_sample=min(r["min_v"] for r in rows),
                max_sample=max(r["max_v"] for r in rows),lower=side("n_low"),upper=side("n_high"),
                mean_volume_acceptance=fmean(r["volume_acceptance_all"] for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs",nargs="*",help="run directory names, defaults to all six ATAT production runs")
    p.add_argument("--output",type=Path,default=HERE/"diagnostics",help="write reports to this directory")
    args=p.parse_args()
    names=args.runs or [f"atat_{model}{suffix}" for model in ("static","ha","qha") for suffix in ("","_section")]
    dest=args.output
    dest.mkdir(parents=True,exist_ok=True)
    vs=np.load(HERE.parent/"atomistic/atat/qha_fit/grid.npz")["volume_A3_atom"]
    summary=dict(volume_bounds_A3_atom=[float(vs[0]),float(vs[-1])],logs={},volume={})
    all_rows=[]
    for name in names:
        log,rows=analyze(HERE/"runs"/name,vs)
        summary["logs"][name]=log
        all_rows.extend(rows)
        if rows:
            summary["volume"][name]={str(t):summarize([r for r in rows if r["T_K"]==t])
                                      for t in sorted({r["T_K"] for r in rows})}
    if all_rows:
        with (dest/"qha_volume_windows.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
    (dest/"qha_volume_summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    main()
