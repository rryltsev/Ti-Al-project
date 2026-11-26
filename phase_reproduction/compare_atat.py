"""Compare a freshly fitted ATAT Hamiltonian with S1 and a separately labelled set."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"atomistic"))
from hcp_orbits import ORBIT_NAMES, MULTIPLICITIES, STATIC_S1, VREF


def data(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def features(row):
    return np.asarray([float(row[name]) for name in ORBIT_NAMES])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fit",type=Path)
    p.add_argument("design",type=Path)
    p.add_argument("output",type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with np.load(args.fit, allow_pickle=False) as fit:
        if tuple(fit["orbit_names"].astype(str))!=ORBIT_NAMES:
            raise ValueError("Unexpected cluster order")
        selected = np.flatnonzero(np.isclose(fit["volume_A3_atom"],VREF,rtol=0,atol=1e-6))
        if len(selected)!=1:
            raise ValueError("reference volume missing")
        J = fit["J_static"][selected[0]]
    with (args.output/"eci_vs_S1.csv").open("w",newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["orbit","multiplicity_per_site","ATAT_fit_eV","paper_S1_eV","fit_minus_S1_eV"])
        w.writerows(zip(ORBIT_NAMES,MULTIPLICITIES,J,STATIC_S1,J-STATIC_S1))
    reference = np.asarray([features(row) for row in data(args.design/"design.csv")])
    rows = data(ROOT/"atomistic/design/design.csv")
    labels = ROOT/"cluster/runs/design-static-01/outputs/results.tar.gz"
    errors = []
    domain_errors = []
    with tarfile.open(labels,"r:gz") as tar, (args.output/"transfer.csv").open("w",newline="") as out:
        w = csv.writer(out)
        w.writerow(["independent_id","n_Al","n_sites","identical_correlation_vector_in_ATAT_design",
                    "MTP_eV_atom","ATAT_CE_eV_atom","S1_eV_atom","ATAT_minus_MTP_meV_atom"])
        for row in rows:
            ident = row["id"]
            f = features(row)
            duplicate = bool(np.any(np.max(np.abs(reference-f),axis=1)<1e-9))
            label = json.load(tar.extractfile(f"results/{ident}_v1.00/label.json"))
            if row["cell_sha256"]!=label["source_cell_sha256"]:
                raise ValueError("label metadata mismatch")
            observed = label["static_eV_atom"]
            residual = 1000*(f@J-observed)
            w.writerow([ident,row["N_Al"],row["N"],duplicate,observed,f@J,f@STATIC_S1,residual])
            if not duplicate:
                values = [residual,1000*(f@STATIC_S1-observed)]
                errors.append(values)
                if int(row["N_Al"])/int(row["N"])<=.35:
                    domain_errors.append(values)
    e = np.asarray(errors)
    summary = {"nonoverlapping_rows":len(errors),"overlapping_rows_excluded":len(rows)-len(errors),
        "new_ATAT_fit_transfer_RMSE_meV_atom":float(np.sqrt(np.mean(e[:,0]**2))),
        "S1_transfer_RMSE_meV_atom":float(np.sqrt(np.mean(e[:,1]**2))),
        "max_absolute_ATAT_error_meV_atom":float(np.max(np.abs(e[:,0])))}
    if domain_errors:
        in_range = np.asarray(domain_errors)
        summary["Ti_rich_nonoverlapping_rows"] = len(in_range)
        summary["Ti_rich_fit_transfer_RMSE_meV_atom"] = float(np.sqrt(np.mean(in_range[:,0]**2)))
        summary["Ti_rich_S1_transfer_RMSE_meV_atom"] = float(np.sqrt(np.mean(in_range[:,1]**2)))
    (args.output/"transfer_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
