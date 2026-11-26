"""Independent repeated 1245 K W=0.40 run at common field on 2304 sites."""
import numpy as np
from isotherms import HERE,load_json,atomic_json


def compare():
    prior=HERE/"validation_1245_W40.json"
    fresh=HERE/"final_T1245_W0p40.json"
    if not(prior.is_file() and fresh.is_file()):return
    old,new=load_json(prior),load_json(fresh)
    result=[]
    for c in ("0.37","0.47"):
        rows=[];draws=[]
        for r in (old,new):
            i=r["direct_fields_eV"].index(3.545)
            field=r["fixed_cut"][c]["sampled_field_scores"][i]
            b=r["bootstrap"]["cuts"][c]
            v=np.asarray([rec["actual_field_scores"][i] for rec in b["replicates"]])
            rows.append(dict(S=field["score"],S_95=np.quantile(v,[.025,.975]).tolist(),delta_h_meV=field["delta_h_meV"],
                B_kBT=(field["profiles"]["0.04"] or {}).get("barrier"),n_runs=r["direct_runs"]))
            draws.append(v)
        delta=draws[1][:,None]-draws[0][None,:]
        result.append(dict(cut_Q=float(c),prior=rows[0],fresh=rows[1],fresh_minus_prior_S=rows[1]["S"]-rows[0]["S"],
            fresh_minus_prior_S_interval_95=np.quantile(delta,[.025,.975]).tolist()))
    atomic_json(HERE/"historical_replicate_1245.json",dict(T_K=1245,W=.4,N=2304,mu_eV=3.545,records=result,
              note="Two studies use separate trajectories and umbrella seeds, evaluated at exactly the same field."))


if __name__=="__main__":compare()
