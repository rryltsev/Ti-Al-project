"""Write branch records for the temperature-closure report, retaining unresolved cases."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent


def rows(inputs,review):
    for f in inputs:
        for z in json.loads(f.read_text(encoding="utf8")):
            for tag,b in z["branches"].items():
                p=b.get("equal_variant")
                h=b.get("equal_height")
                sb=z.get("bootstrap",{}).get("branches",{}).get(tag,{})
                ints=sb.get("interval_95")
                ess=p.get("phase_time_ess_approx",[]) if p else []
                strongest=z.get("strongest_Q_pair")
                status=review.get(f.stem,{}).get(tag,"candidate" if p else "unresolved")
                if status not in ("candidate","unresolved","resolved","inadequate_coverage"):
                    raise ValueError("Unknown report status "+str(status))
                yield dict(T_K=z["T_K"],N=z["N"],W=z["cap"],axial_free=z["axial_free"],
                    branch=tag,status=status,mu_eV=p["mu_eV"] if p else None,
                    x_alpha_pct=100*p["x_alpha"] if p else None,x_alpha2_pct=100*p["x_alpha2"] if p else None,
                    signed_gap_pp=100*(p["x_alpha2"]-p["x_alpha"]) if p else None,
                    equal_height_barrier_kBT=h["barrier_kBT"] if h else None,
                    mu_sigma_eV=np.std(np.array(sb["replicates"])[:,0],ddof=1) if ints else None,
                    mu_low95=ints[0][0] if ints else None,mu_high95=ints[0][1] if ints else None,
                    gap_low95_pp=100*ints[4][0] if ints else None,gap_high95_pp=100*ints[4][1] if ints else None,
                    barrier_low95_kBT=ints[3][0] if ints else None,barrier_high95_kBT=ints[3][1] if ints else None,
                    BC_tree_min=z["BC_tree_min"],phase_ess_low=ess[0] if ess else None,
                    phase_ess_high=ess[1] if ess else None,mu_min=z["mu_range_eV"][0],mu_max=z["mu_range_eV"][1],
                    x_min_pct=100*z["x_range"][0],x_max_pct=100*z["x_range"][1],max_chi_x=z["max_chemical_response"]["chi_x"],
                    max_NvarQ=z["max_order_response"]["NvarQ"],peakQ_x_pct=100*z["max_order_response"]["x"],
                    strongest_barrier_kBT=strongest["pair"]["barrier"] if strongest else None,source=f.name)


def fixed_rows(inputs):
    for f in inputs:
        for z in json.loads(f.read_text(encoding="utf8")):
            for apex in z["apex"]:
                if not apex or apex["cut_Q"] not in (.42,.47,.50):
                    continue
                for item in apex.get("fixed_cut_crossings",[]):
                    tag=item["branch"]
                    boot=z.get("bootstrap",{}).get("fixed_cut_crossings",{}).get(str(apex["cut_Q"]),{}).get(tag,{})
                    ints=boot.get("interval_95")
                    pair=item["pair"]
                    yield dict(T_K=z["T_K"],N=z["N"],cut_Q=apex["cut_Q"],branch=tag,
                        mu_eV=item["mu_eV"],x_lowQ_pct=100*item["x_lowQ"],x_highQ_pct=100*item["x_highQ"],
                        signed_gap_pp=100*item["signed_gap"],cut_between_modes=item["cut_between_modes"],
                        modes_Q=json.dumps(pair["modes_Q"]) if pair else None,
                        smaller_barrier_kBT=pair["barrier"] if pair else None,profile_valley_Q=pair["cut"] if pair else None,
                        time_ESS=item.get("time_ESS"),bootstrap_root_count=boot.get("n_roots"),
                        bootstrap_count=z.get("bootstrap",{}).get("count"),gap_low95_pp=100*ints[3][0] if ints else None,
                        gap_high95_pp=100*ints[3][1] if ints else None,source=f.name)


def write(path,rows):
    if not rows:
        return
    tmp=path.with_suffix(".csv.tmp")
    with tmp.open("w",newline="",encoding="utf8") as fp:
        wr=csv.DictWriter(fp,fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    tmp.replace(path)


if __name__=="__main__":
    files=[Path(i) for i in sys.argv[1:]] if len(sys.argv)>1 else sorted(HERE.glob("T[0-9]*_N[0-9]*.json"))
    review=HERE/"reviewed_status.json"
    rr=list(rows(files,json.loads(review.read_text(encoding="utf8")) if review.is_file() else {}))
    if not rr:
        raise SystemExit("No checked direct isotherms are available")
    rr.sort(key=lambda r:(r["T_K"],r["N"],r["W"],r["source"],r["branch"]))
    write(HERE/"closure_summary.csv",rr)
    write(HERE/"fixed_cut_contacts.csv",list(fixed_rows(files)))
