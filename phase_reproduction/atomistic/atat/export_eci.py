"""Export one fitted Ti-Al ECI slice for native ATAT corrdump predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hcp_orbits import MULTIPLICITIES, ORBIT_NAMES, VREF


def exact_index(grid, value):
    indices = np.flatnonzero(np.isclose(grid,value,rtol=1e-9,atol=1e-9))
    if len(indices)!=1:
        raise ValueError(f"Value {value} missing from grid")
    return indices[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("design",type=Path)
    p.add_argument("fit",type=Path,help="static_grid.npz or grid.npz in S1 order")
    p.add_argument("output",type=Path)
    p.add_argument("--ratio",default=1.,type=float)
    p.add_argument("--temperature",type=float,help="include Fvib at this temperature")
    p.add_argument("--corrdump",help="optional native CPU binary for a full prediction round-trip")
    args=p.parse_args()
    data=np.load(args.fit)
    if tuple(data["orbit_names"])!=ORBIT_NAMES:
        raise ValueError("Input orbit order mismatch")
    v=exact_index(data["volume_A3_atom"],VREF*args.ratio)
    eci=data["J_static"][v].copy()
    if args.temperature is not None:
        t=exact_index(data["temperature_K"],args.temperature)
        eci+=data["J_vib"][v,t]
    native=json.loads((args.design/"design.json").read_text())["native_names"]
    reorder=[ORBIT_NAMES.index(s) for s in native]
    with (args.design/"design.csv").open(newline="") as f:
        records=list(csv.DictReader(f))
    x=np.array([[float(row[key]) for key in ORBIT_NAMES] for row in records])
    raw_corr=np.vstack([np.loadtxt(args.design/"selected"/row["id"]/"corr.out") for row in records])
    expected=x@eci
    # corrdump -eci multiplies its input by orbit multiplicity per primitive
    # cell (two HCP sites). Divide physical J by two for outputs in eV/atom.
    eci_for_native=eci[reorder]/2
    pred=raw_corr@(2*MULTIPLICITIES[reorder]*eci_for_native)
    if np.max(abs(pred-expected))>1e-9:
        raise ValueError("Native correlation normalization mismatch")
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ["lat.in","clusters.out"]:
        shutil.copyfile(args.design/name,args.output/name)
    shutil.copyfile(args.design/"native/selected.out",args.output/"selected.out")
    with (args.output/"eci.out").open("w",newline="\n") as stream:
        np.savetxt(stream,eci_for_native,fmt="%.17g")
    with (args.output/"eci.per_cluster_eV").open("w",newline="\n") as stream:
        np.savetxt(stream,eci[reorder],fmt="%.17g")
    with (args.output/"interactions.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["orbit_S1","multiplicity_per_atom","J_eV_per_cluster","corrdump_eci_eV"])
        w.writerows(zip(ORBIT_NAMES,MULTIPLICITIES,eci,eci/2))
    native_error=None
    if args.corrdump:
        command=[args.corrdump,"-c","-l=lat.in","-s=selected.out","-eci=eci.out","-sig=14"]
        result=subprocess.run(command,cwd=args.output,check=True,capture_output=True,text=True)
        (args.output/"predicted.out").write_bytes(result.stdout.encode("ascii"))
        direct=np.loadtxt(result.stdout.splitlines(),ndmin=1)
        if direct.shape!=expected.shape:
            raise ValueError("Native prediction count mismatch")
        native_error=float(np.max(abs(direct-expected)))
        if native_error>1e-9:
            raise ValueError(f"Native ECI round-trip failed with {native_error} eV/atom")
    with (args.output/"predictions.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["id","split","predicted_eV_atom"])
        w.writerows((row["id"],row["split"],value) for row,value in zip(records,expected))
    (args.output/"export.json").write_text(json.dumps(dict(
        fitted_grid_sha256=hashlib.sha256(args.fit.read_bytes()).hexdigest(),ratio=args.ratio,
        temperature_K=args.temperature,native_names=native,
        convention="eci.out = J(per cluster)/2 in native order; corrdump -eci=eci.out predicts eV per atom",
        max_algebraic_roundtrip_error_eV_atom=float(np.max(abs(pred-expected))),
        max_native_binary_roundtrip_error_eV_atom=native_error),indent=2))


if __name__=="__main__":
    main()
