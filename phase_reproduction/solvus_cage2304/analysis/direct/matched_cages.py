"""Compare independently sampled cages at matched T and directly simulated chemical fields."""
import csv
from collections import defaultdict
import numpy as np

from isotherms import HERE,load_json,atomic_json


def ci(values):
    return np.quantile(values,[.025,.975]).tolist() if len(values)>1 else None


def compare():
    models=defaultdict(dict)
    for path in HERE.glob("final_T[0-9]*.json"):
        r=load_json(path);models[r["T_K"]][r["cap"]]=r
    comparisons=[]
    for t,pair in sorted(models.items()):
        if set(pair)!={.35,.40}:continue
        a,b=pair[.35],pair[.40]
        for cut in ("0.37","0.47"):
            sa=a["fixed_cut"][cut];sb=b["fixed_cut"][cut]
            ra=a["bootstrap"]["cuts"][cut]["replicates"];rb=b["bootstrap"]["cuts"][cut]["replicates"]
            # Cartesian product respects independent trajectories and needs no random pairing.
            rep_S=np.array([v["apex"][4] for v in ra])[:,None]-np.array([v["apex"][4] for v in rb])[None,:]
            row=dict(T_K=t,cut_Q=float(cut),delta_Smax=sa["maximum"]["score"]-sb["maximum"]["score"],
                delta_Smax_interval_95=ci(rep_S.ravel()),both_maxima_interior=not sa["maximum_on_edge"] and not sb["maximum_on_edge"],
                matched_fields=[],branches={})
            fields_a={v["mu_eV"]:(i,v) for i,v in enumerate(sa["sampled_field_scores"])}
            fields_b={v["mu_eV"]:(i,v) for i,v in enumerate(sb["sampled_field_scores"])}
            for mu in sorted(fields_a.keys()&fields_b.keys()):
                ia,va=fields_a[mu];ib,vb=fields_b[mu]
                delta=np.array([x["actual_field_scores"][ia] for x in ra])[:,None]-np.array(
                    [x["actual_field_scores"][ib] for x in rb])[None,:]
                fac=-1000*.00008617333262145*t/a["N"]
                row["matched_fields"].append(dict(mu_eV=mu,delta_S=va["score"]-vb["score"],delta_S_interval_95=ci(delta.flatten()),
                     delta_ordered_minus_disordered_meV_atom=fac*(va["score"]-vb["score"]),
                     delta_free_energy_interval_95_meV_atom=ci(fac*delta.flatten())))
            central_a={x["branch"]:x for x in sa["crossings"]};central_b={x["branch"]:x for x in sb["crossings"]}
            for side in ("Ti","Al"):
                va=central_a.get(side);vb=central_b.get(side)
                av=np.array([x["roots"][side] for x in ra if side in x["roots"]]);bv=np.array([x["roots"][side] for x in rb if side in x["roots"]])
                entry=dict(p_both_cages_have_branch=len(av)*len(bv)/(len(ra)*len(rb)))
                for k,label,scale in ((0,"mu_eV",1),(1,"x_low",100),(2,"x_high",100),(3,"gap",100)):
                    entry["central_delta_"+label]=scale*(va[label]-vb[label]) if va and vb else None
                    entry["conditional_delta_"+label+"_95"]=ci((scale*(av[:,None,k]-bv[None,:,k])).flatten()) if av.size and bv.size else None
                row["branches"][side]=entry
            comparisons.append(row)
    atomic_json(HERE/"matched_cage_effects.json",dict(convention="All differences W=0.35 minus W=0.40. "
                    "Compositions and signed-width shifts are in percentage points; mu shifts in eV.",
                    records=comparisons))


if __name__=="__main__":
    compare()
