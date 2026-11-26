"""Collect the checked temperature-specific fits into a compact data record for the report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source(path):
    return {"path":path.resolve().relative_to(HERE).as_posix(),
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def compact(path):
    data, = read(path)
    analysis = data.get("coexistence", {})
    fit = analysis.get("equal_single_variant") or {}
    target = None
    if fit:
        intervals = analysis.get("bootstrap",{}).get("intervals",{})
        target = {"Al_at_pct":[100*fit[x] for x in ("x_alpha","x_alpha2")],
                  "Al_at_pct_conditional_95":[
                      [100 * intervals[x]["quantile_025_50_975"][i] for i in (0,2)]
                      if x in intervals else None for x in ("x_alpha","x_alpha2")],
                  "Q_boundary":fit["order_cut"],"Q_modes":fit["basin_mode_q"],"mu_cfg_eV":fit["mu_eV"],
                  "equal_height_barrier_kBT":(analysis.get("equal_height_separator") or {}).get("barrier_kBT"),
                  "volume_A3_atom":fit["phase_volumes_A3_atom"],
                  "c_a":fit.get("phase_observables",{}).get("mean_c_a"),
                  "phase_Kish_approx":fit.get("phase_Kish_approx_with_run_inefficiency"),
                  "bootstrap_resolved":analysis.get("bootstrap",{}).get("resolved"),
                  "bootstrap_count":analysis.get("bootstrap",{}).get("count")}
    scan = data["scan"]
    supported = [p for p in scan["grid"] if p["effective_frames_with_time_correlation"]>=10]
    return {"source":source(path),"T_K":data["temperature_K"],"N":data["N"],"cap":data["cap"],
            "axial_free":data["axial_free"],"n_runs":data["n_runs"],"phase_fit":target,
            "weakest_overlap_Bhattacharyya":scan["weakest_tree_Bhattacharyya"],
            "supported_x_range_at_pct": [100*supported[0]["x_Al"],100*supported[-1]["x_Al"]] if supported else None}


def response(path):
    r, = read(path)
    band = r["statistics"]["experimental_alpha_alpha2"]
    point = band["max_chemical_response"]
    return {"source":source(path),"N":r["N"],"band_at_pct":[100*v for v in band["x_range"]],
            "max_dx_dmu_eV_inv":point["dx_dmu_eV_inv"],
            "max_dx_dmu_block_95":r["bootstrap_max_chi_x_and_NvarQ_95"]["experimental_alpha_alpha2"][0],
            "at_mean_Al_at_pct":100*point["x_Al"],"observed_composition_modes_at_pct":point["composition_modes_at_percent"],
            "scale_for_experimental_gap_equal_total_weights":r["diagnostic_dx_dmu_for_3p5pp_equal_total"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-low",type=Path,required=True)
    p.add_argument("--low-checks",type=Path,nargs="+",required=True)
    p.add_argument("--high-profiles",type=Path,nargs="+",required=True)
    p.add_argument("--high-checks",type=Path,nargs="*",default=[])
    p.add_argument("--high-note",required=True,help="Conclusion after finite-size and independent-start checks")
    p.add_argument("--primary-high",type=Path,help="Only when two equilibrium compositions are resolved")
    a=p.parse_args()
    low=compact(a.primary_low)
    assert low["T_K"]==600 and low["cap"]==.4 and low["axial_free"] and low["phase_fit"]
    old=read(HERE.parent/"solvus_1000/analysis1400_free.json")
    base,=[g for g in old.values() if g["N"]==1400 and g["cap"]==.4 and g["axial_free"]]
    original=base["equal_single_variant"]
    high=compact(a.primary_high)["phase_fit"] if a.primary_high else None
    if a.primary_high and (not high or not compact(a.primary_high)["axial_free"]):
        raise ValueError("Invalid primary high coexistence fit")
    with (HERE/"calphad/liang_contacts.csv").open(encoding="utf-8") as f:
        calphad = [{"T_K":float(r["T_K"]),"Al_at_pct":[float(r["hcp_Al_at_pct"]),float(r["D019_Al_at_pct"])]}
                  for r in csv.DictReader(f) if r["model"]=="binary10" and r["branch"]=="Ti-rich"]
    c46 = []
    for t in ("600","1273.15"):
        fname = HERE/"c46"/f"T{t}"/"c5120/result.json"
        saved = read(fname)
        c46.append({"source":source(fname),"T_K":saved["T_K"],"N":saved["N"],
                    "classical_contact":saved["class_direct_tangent"],"contact_columns":saved["contact_columns"],
                    "classical_95":saved.get("classical_95"),"range_x_Al":saved["range_x_Al"],
                    "bootstrap_resolved":saved["class_bootstrap_resolved"],"bootstrap_count":saved["bootstrap_draws"]})
    report = {"pressure_GPa":0,"MTP_sha256":hashlib.sha256((HERE.parent/"inputs/fitted.mtp").read_bytes()).hexdigest(),
              "MTP_solvus":[{"temperature_K":600,"Al_at_pct":low["phase_fit"]["Al_at_pct"],"primary":low},
                            {"temperature_K":1000,"Al_at_pct":[100*original[key] for key in ("x_alpha","x_alpha2")],
                             "source":"../solvus_1000/analysis1400_free.json"},
                            {"temperature_K":1273.15,"Al_at_pct":high["Al_at_pct"] if high else None,
                             "interpretation":a.high_note}],
              "low_checks":[compact(f) for f in a.low_checks],
              "high_checks":[compact(f) for f in a.high_profiles+a.high_checks],
              "high_profile_files":[source(f)["path"] for f in a.high_profiles],
              "C46_classical":c46,"CALPHAD_Liang":calphad,
              "large_cell_high_response":[response(HERE/"analysis"/f"1273_n{n}_response.json") for n in [768,1400]],
              "independent_600_K_controls":source(HERE/"analysis/600_independent_endpoints.json"),
              "full_run_inventory":source(HERE/"atomistic/series_summary.csv"),
              "experimental_table":"calphad/kainuma_table1.csv"}
    (HERE/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
