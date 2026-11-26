"""Compute and validate a CPU-only independent-ECI mean-field temperature sweep."""
import csv
import gzip
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy
from mfa import run_grid


def one_temperature(arg):
    grid,models,t=arg
    target=Path("isotherms")/str(t)
    run_grid(grid,target,[t],models)
    return target


def main():
    grid,=Path(".").glob("*.npz")
    with np.load(grid) as fit:
        if "J_vib" not in fit:
            models=["static"]
        elif len(fit["volume_A3_atom"])==1:
            models=["static","HA"]  # shared mask from phonon stability at vref
        else:
            models=["QHA"]
        temps=list(map(int,fit["temperature_K"])) if "J_vib" in fit else list(range(600,1801,25))
    workers=int(os.environ["SLURM_CPUS_PER_TASK"])
    ordered=[1000,*[t for t in temps if t!=1000]]
    if set(ordered)!=set(temps):
        raise ValueError("1000 K missing from fitted temperatures")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for dest in pool.map(one_temperature,[(grid,models,t) for t in ordered]):
            print(models,dest.name,"K",flush=True)
    files=["grid_mfa_contacts.csv","grid_mfa_free_energy.csv"]
    for file in files:
        compressed=file=="grid_mfa_free_energy.csv"
        out=open(file,"w",newline="") if not compressed else gzip.open(file+".gz","wt",newline="")
        with out as stream:
            writer=None
            for t in sorted(temps):
                with (Path("isotherms")/str(t)/file).open() as f:
                    reader=csv.DictReader(f)
                    if writer is None:
                        writer=csv.DictWriter(stream,fieldnames=reader.fieldnames)
                        writer.writeheader()
                    vals=list(reader)
                if (not vals or any(float(r["T_K"])!=t for r in vals) or {r["model"] for r in vals}!=set(models)
                        or (compressed and (len(vals)!=701*len(models) or any(not np.isfinite(float(r["f_eV_atom"])) for r in vals)))):
                    raise ValueError(f"Invalid output {t} {file}")
                if compressed:
                    for m in models:
                        ordered_c=[float(r["c_Al"]) for r in vals if r["model"]==m]
                        if not np.allclose(ordered_c,np.linspace(0,.35,701),rtol=0,atol=1e-12):
                            raise ValueError(f"Composition grid mismatch at {t}")
                writer.writerows(vals)
    info={"SLURM_JOB_ID":os.environ["SLURM_JOB_ID"],"input_sha256":hashlib.sha256(grid.read_bytes()).hexdigest(),
          "temperatures":sorted(temps),"models":models,"scipy":scipy.__version__,"numpy":np.__version__,"workers":workers}
    Path("validation.json").write_text(json.dumps(info,indent=2)+"\n")


if __name__=="__main__":
    main()
