"""Estimate one wall's vertex or compare independent wall estimates at N=2304."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "solvus_closure"
sys.path.insert(0, str(PREVIOUS))
import pooled


def source_paths(wall, temperatures, fields, available, previous):
    choices = sorted((HERE/"atomistic/series").glob("*/windows.csv"))
    if previous and wall == .40:
        choices.extend(PREVIOUS/"atomistic/series"/s/"windows.csv" for s in ("zc4523c","zc4523s"))
    kept,missing = [],[]
    for path in choices:
        with path.open(encoding="utf-8-sig",newline="") as f:
            rows = list(csv.DictReader(f))
        if any(float(r["K_eV"]) <= 0 for r in rows):
            if any(float(r["K_eV"]) > 0 for r in rows):
                raise ValueError(f"Mixed biased and unbiased series: {path}")
            continue
        if not all(float(r["cap"]) == wall and int(r["N"]) == 2304 for r in rows):
            continue
        if temperatures and not all(float(r["T_K"]) in temperatures for r in rows):
            continue
        if fields and not all(fields[0] <= float(r["mu_eV"]) <= fields[1] for r in rows):
            continue
        status_path = path.with_name("window_status.json")
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf8"))["windows"]
            if set(status) != {r["run"] for r in rows} or not all(
                status[r["run"]]["status"]=="ok" and status[r["run"]]["exit_code"]==0
                and path.with_name(r["run"]+".csv").is_file() for r in rows
            ):
                raise ValueError(f"Incomplete or failed window in {path}")
            kept.append(path)
        else:
            missing.append(path.parent.name)
    if missing and not available:
        raise ValueError("Awaiting submitted windows: "+", ".join(missing))
    if not kept:
        raise ValueError("No verified positive-umbrella trajectories for this selection")
    return kept,missing


def fit(a):
    kept,missing = source_paths(a.wall,a.temperatures,a.fields,a.available,a.previous)
    seed = 4173129 if a.wall == .35 else 4253129
    temp_file=a.out.with_suffix(".partial")
    args = SimpleNamespace(manifests=kept,stride=a.stride,cut=a.cut,bounds=a.bounds,
                           seed=seed,draws=a.draws,out=temp_file)
    pooled.main(args)
    result = json.loads(temp_file.read_text(encoding="utf8"))
    result["pending_at_fit"] = missing
    result["historical_windows_allowed"] = a.previous
    result["new_ensembles_only"] = not any(PREVIOUS in p.parents for p in kept)
    result["temperature_interpolation"] = min(result["source_T_K"]) <= result["central"]["T_K"] <= max(result["source_T_K"])
    temp_file.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    temp_file.replace(a.out)
    print(a.wall,result["source_T_K"],result["central"]["T_K"],result["central"]["mu_eV"],flush=True)


def difference(a):
    tight,loose = [json.loads(p.read_text(encoding="utf8")) for p in [a.tight,a.loose]]
    if (tight["W"],loose["W"]) != (.35,.4) or tight["N"]!=loose["N"] or tight["Q_divider"]!=loose["Q_divider"]:
        raise ValueError("Expected two matched wall fits")
    if tight["bootstrap_columns"]!=loose["bootstrap_columns"]:
        raise ValueError("Incompatible bootstrap outputs")
    if not tight["replicates"] or not loose["replicates"]:
        raise ValueError("Both temperatures need independent block resampling")
    values = {}
    for i,key in enumerate(tight["bootstrap_columns"]):
        v1 = np.asarray(tight["replicates"])[:,i]
        v2 = np.asarray(loose["replicates"])[:,i]
        v = (v1[:,None]-v2).ravel()
        values[key] = dict(central=tight["central"][key]-loose["central"][key],
                           independent_block_025_50_975=np.quantile(v,[.025,.5,.975]).tolist())
    sources = [os.path.relpath(p,HERE) for p in (a.tight,a.loose)]
    result=dict(N=2304,pressure_GPa=0,free_axial=True,definition="value(W=0.35) minus value(W=0.40)",
                sources=sources,cut_Q=tight["Q_divider"],by_observable=values,
                temperatures_bracketed=[z["temperature_interpolation"] for z in (tight,loose)],
                source_T_K=[z["source_T_K"] for z in (tight,loose)],
                note="Each wall was simulated independently. Percentiles are taken over the product "
                     "of the two block-bootstrap distributions. They quantify finite-trajectory "
                     "uncertainty at the given cell, basin divider and ensemble.")
    a.out.parent.mkdir(exist_ok=True,parents=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(values["T_K"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    p=sub.add_parser("fit")
    p.add_argument("--wall",type=float,required=True,choices=[.35,.4])
    p.add_argument("--temperatures",type=float,nargs="+")
    p.add_argument("--fields",type=float,nargs=2)
    p.add_argument("--available",action="store_true",help="Allow pilot fits on only finished groups.")
    p.add_argument("--previous",action="store_true",help="Include the original 1245 K broad-wall data.")
    p.add_argument("--bounds",type=float,nargs=4,default=[1200.,1320.,3.43,3.68])
    p.add_argument("--cut",type=float,default=.37)
    p.add_argument("--stride",type=int,default=12)
    p.add_argument("--draws",type=int,default=64)
    p.add_argument("--out",type=Path,required=True)
    q=sub.add_parser("difference")
    q.add_argument("tight",type=Path)
    q.add_argument("loose",type=Path)
    q.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    with threadpool_limits(limits=1):
        if a.action=="fit":
            fit(a)
        else:
            difference(a)
