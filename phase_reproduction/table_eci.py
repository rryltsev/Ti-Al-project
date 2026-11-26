"""Write the fitted static and vibrational Ti–Al cluster coefficients as a long CSV."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"atomistic"))
from hcp_orbits import ORBIT_NAMES, VREF, MULTIPLICITIES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("grid",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    with np.load(args.grid,allow_pickle=False) as g:
        names=g["orbit_names"].astype(str)
        vs,ts=g["volume_A3_atom"],g["temperature_K"]
        js,jv=g["J_static"],g["J_vib"]
        if tuple(names)!=ORBIT_NAMES or js.shape!=(len(vs),len(names)) or jv.shape!=(len(vs),len(ts),len(names)):
            raise ValueError("unexpected fit grid")
        if not np.isfinite(js).all() or not np.isfinite(jv).all():
            raise ValueError("non-finite ECI")
        with args.output.open("w",encoding="utf-8",newline="") as f:
            w=csv.writer(f)
            w.writerow(["V_over_Vref","V_A3_per_atom","T_K","orbit","multiplicity_per_atom",
                        "Jstatic_eV_per_cluster","Jvib_eV_per_cluster","Jeff_eV_per_cluster"])
            for i,v in enumerate(vs):
                for k,t in enumerate(ts):
                    w.writerows((v/VREF,v,t,name,MULTIPLICITIES[j],js[i,j],jv[i,k,j],js[i,j]+jv[i,k,j])
                                for j,name in enumerate(names))
    print(f"{len(vs)} volumes, {len(ts)} temperatures, {len(names)} orbits")


if __name__=="__main__":
    main()
