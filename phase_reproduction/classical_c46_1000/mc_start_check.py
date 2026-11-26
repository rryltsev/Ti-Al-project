"""Compare independently generated ordered/classical and random/quantum chains."""
import json
from pathlib import Path

import numpy as np


HERE=Path(__file__).resolve().parent


def main():
    files=[HERE/f"paired5120_{k}_start" for k in ("quantum","classical")]
    q,c=[json.loads(f.with_suffix(".json").read_text()) for f in files]
    assert q["bootstrap_seed"]!=c["bootstrap_seed"]
    assert q["N"]==c["N"]==5120 and q["window_centers"]==c["window_centers"]
    results={}
    with np.load(files[0].with_suffix(".npz")) as a, np.load(files[1].with_suffix(".npz")) as b:
        for i,k in enumerate(["quantum","classical","classical_minus_quantum"]):
            if k=="classical_minus_quantum":
                central=np.array(q[k])-np.array(c[k])
            else:
                central=np.array(q["estimates"][i])-np.array(c["estimates"][i])
            # Disjoint seeds, initializations and source ensembles. Use the empirical
            # product distribution of the two independent block bootstrap samples.
            spread=(a[k][:,None,:]-b[k][None,:,:]).reshape(-1,3)
            se=np.sqrt(a[k].var(axis=0,ddof=1)+b[k].var(axis=0,ddof=1))
            results[k]=dict(random_quantum_source_minus_ordered_classical_source=central.tolist(),
                            interval_95=np.percentile(spread,[2.5,97.5],axis=0).T.tolist(),
                            combined_se=se.tolist(),observed_in_combined_se=(central/se).tolist(),
                            random_source_se=np.std(a[k],axis=0,ddof=1).tolist(),
                            ordered_source_se=np.std(b[k],axis=0,ddof=1).tolist())
    output=dict(columns=q["columns"],quantum_random_draws=q["bootstrap_resolved"],
                classical_ordered_draws=c["bootstrap_resolved"],results=results,
                convention="Difference between independent estimates of the same target. The random-start chains sampled "
                "the quantum Hamiltonian; the ordered-start chains sampled the classical one. Values from the alternate "
                "ensemble are obtained by one-sided exact Boltzmann weights, independently inside each window. Percentiles "
                "use the empirical product of block bootstrap distributions. Their conditional uncertainty may underestimate "
                "mixing or initialization sensitivity. These two runs confound initialization and sampled Hamiltonian.")
    (HERE/"mc_start_sensitivity.json").write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(results["classical"]))
    print(json.dumps(results["classical_minus_quantum"]))


if __name__=="__main__":
    main()
