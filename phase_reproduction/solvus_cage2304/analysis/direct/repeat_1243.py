"""Separate the original seven-field 1243 K W=0.40 grid and independent longer repeat."""
import argparse
import time
import numpy as np
from isotherms import HERE,STUDY,load_json,atomic_json,discover,manual


def update():
    base=STUDY/"atomistic"/"series"
    groups,pending=discover(base)
    ready={p.parent.name:p for p in groups.get((1243.,.4),[])}
    models={}
    for label,series,seed in (("initial",["cg4043"+tag for tag in "abcdefg"],751234),
                              ("long_repeat",["cg40r43"+tag for tag in "abc"],850842)):
        if not all(s in ready for s in series):continue
        out=HERE/(label+"_T1243_W0p40.json")
        if not out.exists():
            manual([ready[s] for s in series],out,64,True,seed)
        models[label]=load_json(out)
    if len(models)!=2:return False
    records=[]
    for cut in ("0.37","0.47"):
        estimates=[];distributions=[];maxima=[]
        for label,d in models.items():
            idx=d["direct_fields_eV"].index(3.545)
            z=d["fixed_cut"][cut];b=d["bootstrap"]["cuts"][cut]
            at=z["sampled_field_scores"][idx]
            s=np.array([r["actual_field_scores"][idx] for r in b["replicates"]])
            sm=np.array([r["apex"][4] for r in b["replicates"]])
            estimates.append(dict(source=label,S_at_3p545=at["score"],S_at_3p545_interval_95=np.quantile(s,[.025,.975]).tolist(),
                 Smax=z["maximum"]["score"],Smax_interval_95=b["apex"]["interval_95"][4],Smax_mu=z["maximum"]["mu_eV"],
                 Smax_on_direct_field_edge=z["maximum_on_edge"],actual_field_span=d["direct_fields_eV"],
                 crossings=z["crossings"],both_roots_fraction=b["fractional_both_roots_inside"],
                 Smax_halves=[d["halves"][k][cut]["maximum"]["score"] for k in ("first","last")]))
            distributions.append(s);maxima.append(sm)
        delta=distributions[1][:,None]-distributions[0][None,:]
        dm=maxima[1][:,None]-maxima[0][None,:]
        records.append(dict(cut=float(cut),estimates=estimates,delta_long_minus_initial_S_at_3p545=estimates[1]["S_at_3p545"]-estimates[0]["S_at_3p545"],
             delta_S_at_3p545_interval_95=np.quantile(delta,[.025,.975]).tolist(),
             delta_long_minus_initial_Smax=estimates[1]["Smax"]-estimates[0]["Smax"],
             delta_Smax_interval_95=np.quantile(dm,[.025,.975]).tolist()))
    atomic_json(HERE/"repeat_consistency_1243.json",dict(T_K=1243,W=.4,N=2304,records=records,
                note="The long repeat scans three fields. A missing root outside that field span has no width "
                     "assigned in the independent comparison. All results at 3.545 eV use exactly a sampled field."))
    return True


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch",action="store_true")
    args=p.parse_args()
    while not update() and args.watch:time.sleep(30)
