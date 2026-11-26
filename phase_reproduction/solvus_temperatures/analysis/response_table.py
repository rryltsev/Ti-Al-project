"""Compact finite-size chemical/order diagnostics from the temperature-resolved MBAR files."""
import argparse
import csv
import json
from pathlib import Path


def make_rows(paths):
    cells=[d for p in paths for d in json.loads(p.read_text(encoding="utf8"))]
    for d in sorted(cells,key=lambda cell:cell["N"]):
        for reg,stat in d["statistics"].items():
            chi=stat["max_chemical_response"]
            order=stat["max_order_response"]
            intervals=d["bootstrap_max_chi_x_and_NvarQ_95"].get(reg,[[None,None],[None,None]])
            modal=stat.get("composition_mode_scan",{})
            pair=modal.get("strongest_pair") or {}
            yield dict(T_K=d["temperature_K"],N=d["N"],W=d["cap"],region=reg,
                       x_min_at_percent=100*stat["x_range"][0],x_max_at_percent=100*stat["x_range"][1],
                       max_dx_dmu=chi["dx_dmu_eV_inv"],max_dx_dmu_lower95=intervals[0][0],
                       max_dx_dmu_upper95=intervals[0][1],mu_at_max_dx_dmu=chi["mu_eV"],
                       x_at_max_dx_dmu_percent=100*chi["x_Al"],
                       diagnostic_equal_total_dx_dmu_for_3p5pp=d["diagnostic_dx_dmu_for_3p5pp_equal_total"],
                       max_NvarQ=order["Nvar_Q"],max_NvarQ_lower95=intervals[1][0],max_NvarQ_upper95=intervals[1][1],
                       mu_at_max_NvarQ=order["mu_eV"],x_at_max_NvarQ_percent=100*order["x_Al"],
                       dx_dmu_at_max_NvarQ=order["dx_dmu_eV_inv"],mean_Q_at_max_NvarQ=order["mean_Q"],
                       x_low_Q_quartile_at_max_NvarQ_percent=order["low_Q_quartile"]["mean_x_at_percent"],
                       x_high_Q_quartile_at_max_NvarQ_percent=order["high_Q_quartile"]["mean_x_at_percent"],
                       max_NvarQ_at_search_edge=order["at_field_interval_edge"],
                       strongest_composition_barrier_kBT=pair.get("barrier_from_smaller_peak_kBT"),
                       strongest_composition_mode_gap_pp=pair.get("delta_modes_pp"),
                       composition_mode_x_at_max_dx_dmu_percent=json.dumps(chi["composition_modes_at_percent"]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs",nargs="+",type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    rows=list(make_rows(args.inputs))
    with args.out.open("w",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
