"""Extract distinct HCP_A3/ALTI3_D019 tie lines in the Ti-Al section."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pycalphad
from pycalphad import Database, equilibrium, variables as v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tdb", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--temperatures", nargs="+", type=float, help="temperatures in kelvin")
    p.add_argument("--pressure", type=float, default=101325, help="pressure in pascals")
    args = p.parse_args()
    db = Database(str(args.tdb))
    hcp = ["HCP_A3", "ALTI3_D019"]
    # The SI reports ten phases. These are the Ti-Al phases selected here
    # from the Liang database; BCC_ORD includes its disordered reference.
    binary = hcp + ["BCC_ORD","FCC_A1","LIQUID","AL11TI5_D023","AL2TI","AL3TI","AL5TI3","ALTI"]
    if not set(binary).issubset(db.phases):
        raise ValueError("unexpected phase names in TDB")
    args.output.mkdir(exist_ok=True,parents=True)
    temperatures = args.temperatures or [*range(600,1401,25),1425,1430,1435,1437,1440,1450,1500,1600]
    comps = ["AL","TI","VA"]
    compositions = np.unique(np.r_[np.arange(.005,.401,.005),np.arange(.28,.321,.0005)])
    with (args.output/"liang_contacts.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model","T_K","hcp_Al_at_pct","D019_Al_at_pct","branch","bulk_states_supporting_tie"])
        for scope, phases in [("binary10",binary),("hcp",hcp)]:
            for t in temperatures:
                eq = equilibrium(db,comps,phases,{v.T:t,v.P:args.pressure,v.N:1,v.X("AL"):compositions})
                names = eq.Phase.values.reshape(len(compositions),-1)
                weights = eq.NP.values.reshape(names.shape)
                xs = eq.X.sel(component="AL").values.reshape(names.shape)
                ties = {}
                for names_i, np_i, x_i in zip(names, weights, xs):
                    active = (np_i>1e-7) & (names_i!="")
                    if sorted(names_i[active])!=sorted(hcp):
                        continue
                    pair = [float(x_i[names_i==ph][0])*100 for ph in hcp]
                    branch = "Ti-rich" if pair[0]<pair[1] else "Al-rich"
                    ties.setdefault(branch,[]).append(pair)
                for branch, x in sorted(ties.items()):
                    x = np.asarray(x)
                    if np.ptp(x,axis=0).max()>.05:
                        raise ValueError(f"inconsistent contacts within {scope} {t}K {branch}")
                    writer.writerow([scope,t,*np.median(x,axis=0),branch,len(x)])
                f.flush()
    metadata = {"pycalphad":pycalphad.__version__,"T_K":temperatures,"P_Pa":args.pressure,"bulk_compositions":compositions.tolist(),
                "phases":binary,"source_doi":"10.1007/s11669-024-01124-9",
                "tdb_sha256":hashlib.sha256(args.tdb.read_bytes()).hexdigest(),
                "selection":"ten binary phases inferred from v16 SI and Liang phase definitions; HCP-restricted calculation also reported"}
    (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__=="__main__":
    main()
