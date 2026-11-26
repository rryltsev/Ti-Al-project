"""Sensitivity of the congruent-temperature estimate to finite-cell extrapolation."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"analysis/thermal"


def main():
    sources=["pooled_n768","pooled_n1400_all","direct_1245_n2304"]
    data=[json.loads((DATA/(name+".json")).read_text(encoding="utf8")) for name in sources]
    n=np.array([z["N"] for z in data])
    t=np.array([z["central"]["T_K"] for z in data])
    rng=np.random.default_rng(4023129)
    draws=[]
    for z in data:
        store=z["replicates"]
        ts=np.array([v["T_K"] if isinstance(v,dict) else v[z["bootstrap_columns"].index("T_K")] for v in store])
        if len(ts)<16 or not z["central"]["profile"]:
            raise ValueError("Insufficient sampled roots or missing intervening order valley")
        draws.append(rng.choice(ts,10000,replace=True))
    boot=np.stack(draws)
    fits=[]
    for exponent,which in [(1.,np.arange(3)),(1.,np.arange(1,3)),(2/3,np.arange(3))]:
        v=n[which].astype(float)**(-exponent)
        design=np.column_stack([np.ones(len(v)),v/v.mean()])
        weights=np.linalg.pinv(design)[0]
        b=weights@boot[which]
        fits.append(dict(inverse_size_exponent=exponent,cells=n[which].tolist(),
                         intercept_K=float(weights@t[which]),
                         conditional_bootstrap_025_50_975_K=np.quantile(b,[.025,.5,.975]).tolist()))
    cells=[]
    for z in data:
        c=z["central"]; B=c["profile"]["lower_barrier"]
        cells.append(dict(N=z["N"],vertex_K=c["T_K"],x_at_pct=c["x_atpct"],jump_Q=c["jump_Q"],
                          heat_meV_atom=c["latent_meV_atom"],barrier_kBT=B,barrier_per_N23=B/z["N"]**(2/3)))
    result=dict(cells=cells,estimates=fits,source_files=[f"analysis/thermal/{v}.json" for v in sources],
                note="Unweighted fits on three sizes or the two largest. The bulk intercept depends on the model "
                     "for finite-size corrections. The resampling intervals omit inter-source equilibration "
                     "scatter, the order divider, cell shape and the displacement wall. The inverse-N^(2/3) "
                     "fit tests sensitivity only. Barrier values refer to the smaller peak at the 4:1 vertex.")
    (ROOT/"analysis/vertex_size_limit.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
