"""Summarize independent 5120-site near-endpoint checks against v16 SI S3."""
import argparse
import csv
import json
from pathlib import Path


TARGETS={("atat_static",1600):(22.89,24.16), ("atat_static",1625):(24.12,24.96),
         ("atat_static",1650):None, ("atat_HA",1200):(22.60,24.03), ("atat_HA",1225):None,
         ("atat_QHA",1200):(23.46,24.91), ("atat_QHA",1225):None}


def summarize(files, output):
    rows=[]
    seen=set()
    for file in files:
        with file.with_name("window_correlations.csv").open(encoding="utf8") as f:
            diagnostics=list(csv.DictReader(f))
        for e in json.loads(file.read_text(encoding="utf8")):
            m,t=e["model"],e["T_K"]
            if e["N"]!=5120 or (m,t) not in TARGETS:
                continue
            if (m,t) in seen:
                raise ValueError(f"Duplicate upper-scan record {m} {t}")
            seen.add((m,t))
            all_intervals=[p for p in e["TI_contacts"] if not p["window_edge"] and
                        p["alpha2_at_pct"]-p["alpha_at_pct"]>.5]
            candidates=[p for p in all_intervals if p.get("order_direction")=="rising"]
            bounds=([candidates[0]["alpha_at_pct"],candidates[0]["alpha2_at_pct"]]
                    if len(candidates)==1 else None)
            ref=TARGETS[m,t]
            b=e["bootstrap_conditional"]
            d=[r for r in diagnostics if r["model"]==m and float(r["T_K"])==t and
               r["N"]=="5120" and r["tau_int_order_frames"]]
            longest=max(d,key=lambda r:float(r["tau_int_order_frames"])) if d else None
            status="interior_tangent" if bounds else "multiple" if candidates else "unresolved"
            rows.append((m,int(t),e["windows"],status,*(bounds or ("","")),
                         bounds[1]-bounds[0] if bounds else "",*b.get("width_95",("","")),
                         e["largest_unthresholded_interior_width_at_pct"],
                         "interior_tangent" if ref else "unresolved",*(ref or ("","")),
                         b["draws"],b["draws_without_interior_contact"],b["draws_with_one_interior_contact"],
                         b["draws_with_multiple_interior_contacts"],b["draws_invalid"],
                         b.get("resolved"),e["WHAM"]["overlap_components"],
                         longest["tau_int_order_frames"] if longest else "",
                         longest["c0"] if longest else "",longest["block_frames"] if longest else "",
                         len(all_intervals),"; ".join(f'{p["alpha_at_pct"]:.4f} to {p["alpha2_at_pct"]:.4f} ({p["order_direction"]})'
                                                    for p in all_intervals)))
    with output.open("w",newline="") as f:
        out=csv.writer(f)
        out.writerow(("model","T_K","windows","status","alpha_at_pct","alpha2_at_pct","width_at_pct",
                      "width_conditional_lo","width_conditional_hi","widest_unthresholded_gap_at_pct",
                      "S3_reference_status","S3_alpha_at_pct","S3_alpha2_at_pct","bootstrap_draws",
                      "draws_zero","draws_one","draws_multiple","draws_invalid","matched_baseline_draws",
                      "wham_components","max_tau_order_samples","c0_of_max_tau","block_samples",
                      "all_internal_contact_count","all_internal_gaps_at_pct_with_order"))
        out.writerows(sorted(rows,key=lambda r:(list(TARGETS).index((r[0],r[1])))))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("files",type=Path,nargs="+")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    summarize(a.files,a.out)
