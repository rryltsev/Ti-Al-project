"""Read the existing 5120-site constrained MC trajectories at 1000 K."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def analyze():
    output = []
    for src,result in [("section5120","mc_section5120"),("atat_static_section","atat_static_section"),
                       ("atat_ha_section","atat_ha_section"),("atat_qha_section","atat_qha_section")]:
        groups = defaultdict(lambda:np.zeros(4))
        with gzip.open(BASE/"mc/runs"/src/"samples.csv.gz","rt",newline="") as f:
            for row in csv.DictReader(f):
                n = int(row["N"])
                assert n==5120 and float(row["T_K"])==1000
                c,m = int(row["n_Al"])/n,float(row["m"])
                # vcs.cpp uses m=4(q_max-c)/3. Check integer site populations.
                q = c+.75*m
                max_al = q*n/4
                if abs(max_al-round(max_al))>1e-7:
                    raise ValueError("Order and site count disagree")
                groups[row["model"],float(row["c0"])] += [c,m,q,1]
        refs = json.loads((BASE/"thermo/results"/result/"mc_contacts.json").read_text())
        for entry in refs:
            model = entry["model"]
            data = [v[:3]/v[3] for (name,center),v in groups.items() if name==model]
            c,m,q = np.array(sorted(data,key=lambda x:x[0])).T
            tangents = [s for s in entry["TI_contacts"] if not s["window_edge"]]
            if len(tangents)!=1 or len(c)!=35:
                raise ValueError("Expected a single contact and 35 windows")
            for condition,value in [("alpha2_contact",tangents[0]["alpha2_at_pct"]/100),
                                    ("common_23.5_at_pct",.235),("stoichiometry_25_at_pct",.25)]:
                assert c[0]<value<c[-1]
                order = np.interp(value,c,m)
                preferred = np.interp(value,c,q)
                other = (4*value-preferred)/3
                output.append(dict(model=model,sampled_state=condition,c_Al_at_pct=100*value,
                    order_parameter=order,Al_preferred_at_pct=100*preferred,
                    Al_other_at_pct=100*other))
    path = HERE/"robustness"/"mc_sublattice_occupancy.csv"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf8") as f:
        w = csv.DictWriter(f,fieldnames=output[0].keys())
        w.writeheader()
        w.writerows(output)
    for row in output:
        if row["sampled_state"]=="alpha2_contact":
            print(row)


if __name__=="__main__":
    analyze()
