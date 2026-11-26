"""Publish the checked vertex, wall shift and 1243 K contacts from the primary fits."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
AN=HERE/"analysis"
DIRECT=AN/"direct"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def write_report():
    paths=[DIRECT/"final_summary.csv",DIRECT/"matched_cage_effects.json",
           DIRECT/"direct_temperature_zeros.json",AN/"wall_shift.json",AN/"wall_shift_local.json",
           HERE/"atomistic/series_summary.csv",HERE/"atomistic/provenance.json",
           HERE/"atomistic/slurm_accounting.csv"]
    with (HERE/"atomistic/series_summary.csv").open(encoding="utf8",newline="") as f:
        jobs=list(csv.DictReader(f))
    if not jobs or not all(j["slurm_state"]=="COMPLETED" and j["slurm_exit"]=="0:0" for j in jobs):
        raise ValueError("UMT series not all verified")
    wall_specs={}
    rows=[]
    for w,tag in [(.35,"35"),(.40,"40")]:
        merged=AN/f"W{tag}_all.json"
        single=DIRECT/f"final_T1243_W0p{tag}.json"
        pooled=AN/f"contacts_W{tag}_1243.json"
        paths.extend((merged,single,pooled))
        z=read(merged)
        d=read(single)
        contacts=read(pooled)
        if z["pending_at_fit"] or contacts["pending_at_fit"] or not z["temperature_interpolation"]:
            raise ValueError("Incomplete primary data or temperature outside the sampled interval")
        cut=str(z["Q_divider"])
        direct=d["fixed_cut"][cut]
        boot=d["bootstrap"]["cuts"][cut]
        intervals=boot["apex"]["interval_95"]
        cross={}
        for item in direct["crossings"]:
            cross[item["branch"]]=item
        wall_specs[tag]=dict(T_vertex_K=z["central"]["T_K"],T_conditional_025_50_975=z["interval_95"][0],
            vertex=z["central"],observable_order=z["bootstrap_columns"],q_cut=z["Q_divider"],
            intervals=z["interval_95"],divider_comparison=[dict(cut=x["cut"],T_K=x["estimate"]["T_K"])
                for x in z["divider_sensitivity"] if x["estimate"]],
            direct_fields_1243_eV=d["direct_fields_eV"],
            direct_Smax_1243=direct["maximum"]["score"],direct_Smax_interval95=intervals[4],
            direct_both_contacts_fraction=boot["fractional_both_roots_inside"],
            pooled_1243_Smax=contacts["center"]["Smax"],
            pooled_1243_Smax_025_50_975=contacts["Smax_025_50_975"],
            pooled_both_contacts_count=contacts["both_contacts_replicates"],pooled_draws=contacts["draws"])
        for branch in ("Ti","Al"):
            root=cross.get(branch)
            ci=boot["branches"].get(branch) or {}
            ci=list(ci.get("interval_95") or [])
            row=dict(T_K=1243,N=2304,W=w,branch=branch,method="direct_1243",
                mu_eV=root["mu_eV"] if root else None,
                field_low_eV=root["direct_field_bracket_eV"][0] if root else None,
                field_high_eV=root["direct_field_bracket_eV"][1] if root else None,
                x_alpha_atpct=100*root["x_low"] if root else None,
                x_alpha2_atpct=100*root["x_high"] if root else None,
                signed_gap_pp=100*root["gap"] if root else None,
                signed_gap_low95_pp=100*ci[3][0] if ci else None,
                signed_gap_high95_pp=100*ci[3][1] if ci else None,
                branch_bootstrap_count=boot["branches"][branch]["count"] if boot["branches"].get(branch) else 0,
                bootstrap_draws=d["bootstrap"]["count"],both_fraction=boot["fractional_both_roots_inside"])
            rows.append(row)
    with (HERE/"contacts_1243.csv").open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    effects=read(DIRECT/"matched_cage_effects.json")["records"]
    entry,=[r for r in effects if r["cut_Q"]==.37 and r["T_K"]==1243.]
    shared,=[r for r in entry["matched_fields"] if abs(r["mu_eV"]-3.545)<1e-9]
    paths.append(HERE/"contacts_1243.csv")
    direct_analysis=read(DIRECT/"direct_temperature_zeros.json")
    direct_roots={}
    for r in direct_analysis["records"]:
        if r["cut_Q"]==.37:
            w=str(round(100*r["W"]))
            apex,=r["weighted_monotone_crossings"]
            direct_roots[w]=dict(T_K=apex,T_interval95=r["conditional_interpolated_T_interval_95"],
                                 x_atpct=r["interpolated_x_pct"],x_interval95=r["conditional_x_pct_interval_95"],
                                 source_T_K=r["usable_T"])
    comparison,=[r for r in direct_analysis["W35_minus_W40"] if r["cut_Q"]==.37]
    direct_shift=dict(T_K=direct_roots["35"]["T_K"]-direct_roots["40"]["T_K"],
                      interval95=comparison["conditional_W35_minus_W40_interval_95"],
                      fraction_both_temperatures_bracketed=comparison["fraction_both_T_bracketed"])
    for tag in ("35","40"):
        check=AN/f"W{tag}_all_stride4.json"
        if check.is_file():
            q=read(check)
            paths.append(check)
            wall_specs[tag]["all_sources_stride4"]={"T_K":q["central"]["T_K"],"x_atpct":q["central"]["x_atpct"]}
    for p in (HERE/"report").glob("TiAl_solvus_cage2304.*"):
        if p.suffix in [".tex",".pdf"]:
            paths.append(p)
    z=dict(account="u3129",N=2304,p_GPa=0.,axial_free=True,nuclear_statistics="classical",
           potential_sha256=read(HERE/"atomistic/provenance.json")["source_sha256"]["fitted.mtp"],
           jobs=len(jobs),trajectories=sum(int(j["windows"]) for j in jobs),
           production_frames=sum(int(j["production_records"]) for j in jobs),
           allocated_CPU_h=sum(float(j["allocated_cpu_hours"]) for j in jobs),
           primary_direct_T=direct_roots,primary_direct_T35_minus_T40=direct_shift,
           pooled_wall_results=wall_specs,pooled_T35_minus_T40=read(AN/"wall_shift.json"),
           common_field_1243_cage_effect=shared,
           files=[dict(path=p.relative_to(HERE).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    (HERE/"results.json").write_text(json.dumps(z,indent=2,allow_nan=False)+"\n",encoding="utf8")


if __name__=="__main__":
    write_report()
