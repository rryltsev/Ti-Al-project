"""Collect reviewed direct binodals and independent same-potential closure estimates."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
AN=HERE/"analysis"
TH=AN/"thermal"
KB=8.617333262145e-5


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(name,rows):
    with (HERE/name).open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    with (AN/"closure_summary.csv").open(encoding="utf8",newline="") as f:
        measurements=list(csv.DictReader(f))
    size_rows=[]
    for row in measurements:
        n,t=int(row["N"]),float(row["T_K"])
        barrier=float(row["equal_height_barrier_kBT"]) if row["equal_height_barrier_kBT"] else None
        size_rows.append(dict(T_K=t,N=n,N_two_thirds=n**(2/3),branch=row["branch"],status=row["status"],
                 alpha_at_pct=row["x_alpha_pct"],alpha2_at_pct=row["x_alpha2_pct"],
                 abs_width_pp=abs(float(row["signed_gap_pp"])) if row["signed_gap_pp"] else None,
                 equal_height_barrier_kBT=barrier,
                 barrier_eV_cell=barrier*KB*t if barrier is not None else None,
                 beta_barrier_per_N_two_thirds=barrier/n**(2/3) if barrier is not None else None,
                 bootstrap_barrier_025=row["barrier_low95_kBT"],bootstrap_barrier_975=row["barrier_high95_kBT"],
                 source="analysis/"+row["source"]))
    save_csv("size_dependence.csv",size_rows)
    vertex_rows=[]
    for path in sorted([*TH.glob("direct_*.json"), *TH.glob("boot_1273*.json")]):
        z=read(path)
        c=z["central"]; profile=c["profile"] or {}; q=z["conditional_95"]
        ts=[s["root"]["T_K"] for s in z.get("divider_sensitivity",[]) if s["root"] and s["root"]["profile"]]
        lo,_,hi=q.get("T_K",[None]*3)
        vertex_rows.append(dict(estimate="single_T",source_T_K=z["source_T_K"],N=z["N"],time_part=z["time_part"],
                    divider=z["Q_divider"],T_vertex_K=c["T_K"],conditional_025_K=lo,conditional_975_K=hi,
                    T_min_divider_scan_K=min(ts) if ts else None,T_max_divider_scan_K=max(ts) if ts else None,
                    Al_at_pct=c["x_atpct"],mu_cfg_eV=c["mu_eV"],delta_Q=c["jump_Q"],
                    latent_meV_atom=c["latent_meV_atom"],barrier_at_odds_four_kBT=profile.get("lower_barrier"),
                    valley_Q=profile.get("Q_valley"),ESS_disordered=c["phases"][0]["effective_frames"],
                    ESS_ordered=c["phases"][1]["effective_frames"],bootstrap_converged=z["converged"],
                    bootstrap_draws=z["draws"],source=path.relative_to(HERE).as_posix()))
    for path in sorted(TH.glob("pooled_n*.json")):
        z=read(path)
        c=z["central"]; profile=c["profile"] or {}
        interval=(z.get("interval_95") or [[None,None,None]])[0]
        ts=[a["estimate"]["T_K"] for a in z["divider_sensitivity"] if a["estimate"] and a["estimate"]["profile"]]
        vertex_rows.append(dict(estimate="multitemperature_MBAR",source_T_K="/".join(map(str,z["source_T_K"])),
                  N=z["N"],time_part="all",divider=z["Q_divider"],T_vertex_K=c["T_K"],
                  conditional_025_K=interval[0],conditional_975_K=interval[2],
                  T_min_divider_scan_K=min(ts) if ts else None,T_max_divider_scan_K=max(ts) if ts else None,
                  Al_at_pct=c["x_atpct"],mu_cfg_eV=c["mu_eV"],delta_Q=c["jump_Q"],
                  latent_meV_atom=c["latent_meV_atom"],barrier_at_odds_four_kBT=profile.get("lower_barrier"),
                  valley_Q=profile.get("Q_valley"),ESS_disordered=c["phases"][0]["effective_frames"],
                  ESS_ordered=c["phases"][1]["effective_frames"],bootstrap_converged=z["resolved"],
                  bootstrap_draws=z["draws"],source=path.relative_to(HERE).as_posix()))
    if vertex_rows:
        vertex_rows.sort(key=lambda r:(r["N"],r["estimate"],str(r["source_T_K"]),r["time_part"]))
        save_csv("congruent_temperatures.csv",vertex_rows)
    files=[AN/"closure_summary.csv", HERE/"size_dependence.csv", HERE/"congruent_temperatures.csv",
           AN/"reviewed_status.json",TH/"validation.json",HERE/"atomistic/series_summary.csv",
           HERE/"atomistic/provenance.json",HERE/"atomistic/slurm_accounting.csv",
           AN/"vertex_size_limit.json",AN/"critical_order_profiles.csv",AN/"n2304_widths.json",
           TH/"direct_1245_n2304.json",TH/"direct_1245_n2304_early.json",TH/"direct_1245_n2304_late.json",
           TH/"pooled_n768.json",TH/"pooled_n1400_all.json",TH/"stride4_n768.json",TH/"stride4_n1400.json",
           HERE/"report/TiAl_solvus_closure.tex",HERE/"report/TiAl_solvus_closure.pdf"]
    record=dict(potential="../inputs/fitted.mtp",potential_sha256=sha(HERE.parent/"inputs/fitted.mtp"),
                pressure_GPa=0,axial_free=True,displacement_wall=.40,account="u3129",
                direct_table="size_dependence.csv",congruent_table="congruent_temperatures.csv",
                note="Widths for confirmed equilibrium must be read together with status. "
                     "Barriers at equal peak heights differ from the barriers at corrected 4:1 basin weights. "
                     "The N^(2/3) normalization is a size diagnostic, not a measured interfacial tension.",
                files=[dict(file=p.relative_to(HERE).as_posix(),sha256=sha(p)) for p in files if p.is_file()])
    if (AN/"vertex_size_limit.json").is_file() and (HERE/"atomistic/provenance.json").is_file():
        audit=read(HERE/"atomistic/provenance.json")["verified_series"]
        if len(audit)==36 and all(row["slurm_state"]=="COMPLETED" and row["slurm_exit"]=="0:0" for row in audit):
            z=read(TH/"direct_1245_n2304.json")
            record["closure"]=dict(
                working_bulk_T_K=1244,working_bulk_range_K=[1240,1247],Al_at_pct_rounded=27.2,
                scope="Classical site-confined HCP Ti–Al, W<0.40, hydrostatic p=0, free V and c/a.",
                uncertainty="Working range reflecting between-isotherm and finite-size model sensitivity. "
                            "It is not a frequentist confidence interval on the infinite-size limit.",
                largest_cell=dict(N=z["N"],source_T_K=z["source_T_K"],central=z["central"],
                                  conditional_intervals_025_50_975=z["conditional_95"],
                                  early_late_T_K=[read(TH/f"direct_1245_n2304_{p}.json")["central"]["T_K"]
                                                  for p in ("early","late")]),
                finite_size_comparison=read(AN/"vertex_size_limit.json"),
                UMT_completed_jobs=len(audit),
            )
    (HERE/"results.json").write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
