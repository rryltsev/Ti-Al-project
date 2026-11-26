"""Check each of the pooled MBAR roots against independently reconstructed adjacent fields."""
from collections import defaultdict

from isotherms import HERE,load_json,atomic_json


def assess():
    by=defaultdict(lambda:defaultdict(list))
    for f in (HERE/"fields").glob("*.json"):
        a=load_json(f)
        for field in a["fields"]:
            for cut,z in field["cuts"].items():
                by[a["T_K"],a["W"],cut][field["mu_eV"]].append(dict(series=f.stem,**z))
    checks=[]
    for f in sorted(HERE.glob("final_T[0-9]*.json")):
        model=load_json(f)
        for cut in ("0.37","0.47"):
            index=by[model["T_K"],model["cap"],cut]
            central=model["fixed_cut"][cut]
            grid=[]
            for item in central["sampled_field_scores"]:
                mu=item["mu_eV"]
                sets=index.get(mu,[])
                grid.append(dict(mu_eV=mu,pooled_S=item["score"],single_field_estimates=[
                    dict(series=a["series"],S=a["S"],interval_95_S=a["interval_95_S"]) for a in sets]))
            roots=[]
            for r in central["crossings"]:
                ml,mr=r["direct_field_bracket_eV"];b=r["branch"]
                x,y=index.get(ml,[]),index.get(mr,[])
                raw=dict(branch=b,mu_root=r["mu_eV"],enclosing_actual_fields=[ml,mr],
                         left_single_field_estimates=x,right_single_field_estimates=y)
                if x and y:
                    sign=1 if b=="Ti" else -1
                    raw["independent_point_estimates_bracket_root"]=bool(max(sign*g["S"] for g in x)<0<min(sign*g["S"] for g in y))
                    raw["at_least_one_independent_pair_brackets_root"]=bool(min(sign*g["S"] for g in x)<0<max(sign*g["S"] for g in y))
                    raw["independent_95_intervals_exclude_zero_at_both_ends"]=bool(
                        max(sign*i for g in x for i in g["interval_95_S"])<0<min(sign*i for g in y for i in g["interval_95_S"]))
                roots.append(raw)
            checks.append(dict(T_K=model["T_K"],W=model["cap"],cut_Q=float(cut),field_comparisons=grid,
                               roots=roots,complete_single_field_comparisons=all(x["single_field_estimates"] for x in grid)))
    atomic_json(HERE/"independent_brackets.json",dict(records=checks,
        note="Each chemical field is reconstructed using independent umbrella chains sampled only at that field. "
             "Adjacent-field endpoint checks complement the full multi-field MBAR; a root can remain inside the "
             "sampled field range with one or both single-field scores too noisy to confirm its bracket."))


if __name__=="__main__":
    assess()
