"""Interpolate adjacent directly sampled isotherms at the extremum of the order odds."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def points(paths,cut):
    out={}
    for p in paths:
        for d in json.loads(p.read_text(encoding="utf8")):
            a=next((r for r in d["apex"] if r and abs(r["cut_Q"]-cut)<1e-7),None)
            if a is None:
                continue
            t,n=d["T_K"],d["N"]
            pair=a.get("basin_profiles_by_bandwidth_Q",{}).get(str(d["bandwidth_Q"]),a["pair_at_max"])
            lo,hi=a["conditional_low_high"]
            boot=d.get("bootstrap",{}).get("apex_fixed_cuts",{}).get(str(cut))
            deriv=n*a["delta_U_minus_mu_x_meV_atom"]*.001/(8.617333262145e-5*t*t) if "delta_U_minus_mu_x_meV_atom" in a else None
            sampled=d.get("direct_fields_eV")
            point=dict(T_K=t,N=n,mu_eV=a["mu"],x_pct=100*a["x"],S=a["log_weight_per_variant"],
                       field_edge=a["at_field_edge"],inside_simulated_fields=bool(sampled and sampled[0]<=a["mu"]<=sampled[-1]),
                       modes_Q=pair["modes_Q"] if pair else None,barrier_from_lower_peak_kBT=pair["barrier"] if pair else None,
                       valley_Q=pair["cut"] if pair else None,
                       divider_separates_modes=bool(pair and pair["modes_Q"][0]<cut<pair["modes_Q"][1]),
                       delta_conditional_x_pp=100*(hi["x"]-lo["x"]),delta_conditional_Q=hi["Q"]-lo["Q"],
                       conditional_Q_low_high=[lo["Q"],hi["Q"]],
                       conditional_time_ESS=[lo.get("approx_time_ESS"),hi.get("approx_time_ESS")],
                       curvature_eV_inv2=a.get("d2_log_odds_dmu2_eV_inv2"),
                       semigrand_energy_gap_meV_atom=a.get("delta_U_minus_mu_x_meV_atom"),
                       estimated_local_dS_dT=deriv,local_linear_zero_K=t-a["log_weight_per_variant"]/deriv if deriv else None,
                       S_interval_95=boot["interval_95"][2] if boot and boot["interval_95"] else None,
                       src=p.name)
            if d.get("halves"):
                slices={name:next((v for v in subset["apex"] if v and abs(v["cut_Q"]-cut)<1e-7),None)
                        for name,subset in d["halves"].items()}
                point["S_time_halves"]={name:(v["log_weight_per_variant"] if v else None) for name,v in slices.items()}
                point["delta_U_minus_mu_x_time_halves_meV_atom"]={name:(v.get("delta_U_minus_mu_x_meV_atom") if v else None)
                                                                  for name,v in slices.items()}
            if boot and boot.get("interval_95") and len(boot["columns"])>=6:
                reps=np.asarray(boot["replicates"])
                neg=reps[:,5]<0
                local=t-reps[neg,2]/reps[neg,5]
                point["semigrand_energy_gap_interval_95_meV_atom"]=boot["interval_95"][4]
                point["local_linear_zero_interval_95_conditional_negative_slope"]=np.quantile(local,[.025,.975]).tolist() if len(local)>1 else None
                point["negative_energy_slope_fraction"]=float(neg.mean())
            peakboot=d.get("bootstrap",{}).get("apex_profiles",{}).get(str(cut))
            if peakboot:
                point["bootstrap_resolved_mode_pairs"]=peakboot["pairs_found"]
                qs=peakboot["interval_95_conditional_two_modes"]
                point["barrier_from_lower_peak_interval_95_conditional_two_modes"]=qs[3] if qs else None
                point["valley_Q_interval_95_conditional_two_modes"]=qs[2] if qs else None
            out.setdefault(n,[]).append((point,boot["replicates"] if boot else []))
    return {n:sorted(arr,key=lambda r:r[0]["T_K"]) for n,arr in out.items()}


def roots(z):
    out=[]
    for (a,aa),(b,bb) in zip(z,z[1:]):
        if a["field_edge"] or b["field_edge"] or a["S"]*b["S"]>0 or a["S"]==b["S"]:
            continue
        wt=a["S"]/(a["S"]-b["S"])
        rec=dict(bracket_T_K=[a["T_K"],b["T_K"]],
                 interpolated_T_K=a["T_K"]+wt*(b["T_K"]-a["T_K"]),
                 interpolated_mu_eV=a["mu_eV"]+wt*(b["mu_eV"]-a["mu_eV"]),
                 interpolated_x_pct=a["x_pct"]+wt*(b["x_pct"]-a["x_pct"]),
                 both_endpoints_separate_modes=bool(a["divider_separates_modes"] and b["divider_separates_modes"]),
                 both_fields_within_sampled_range=bool(a["inside_simulated_fields"] and b["inside_simulated_fields"]))
        if aa and bb:
            aS=np.array(aa)[:,None,2]; bS=np.array(bb)[None,:,2]
            den=aS-bS
            good=den>0
            with np.errstate(divide="ignore",invalid="ignore"):
                r=a["T_K"]+aS/den*(b["T_K"]-a["T_K"])
            rec.update(negative_slope_fraction=float(np.mean(good)),
                       fraction_of_all_bootstrap_pairs_bracketing_root=float(np.mean((aS>=0)&(bS<=0))),
                       interval_95_conditional_negative_slope=np.quantile(r[good],[.025,.975]).tolist() if good.any() else None)
        out.append(rec)
    return out


if __name__=="__main__":
    ps=argparse.ArgumentParser(description=__doc__)
    ps.add_argument("paths",nargs="*",type=Path)
    ps.add_argument("--out",type=Path,default=HERE/"apex_temperature.json")
    args=ps.parse_args()
    paths=args.paths or sorted(HERE.glob("T[0-9]*_N[0-9]*.json"))
    records=[]
    for c in (.42,.47,.5):
        for n,z in points(paths,c).items():
            records.append(dict(N=n,cut_Q=c,observed=[v[0] for v in z],crossings=roots(z)))
    out=dict(definition="S=max_mu log[P(Q>cut)/(4P(Q<=cut))]. The zero is a finite-cell equal-variant-weight "
             "condition only where the fixed separating cut lies in a physical valley between distinct modes. "
             "The local slope assumes conditional semi-grand energy at the interior maximum; linear zeros are "
             "diagnostic. Temperature interpolation resamples independent isotherms and does not quantify bulk or "
             "divider uncertainty.",records=records)
    args.out.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf8")
