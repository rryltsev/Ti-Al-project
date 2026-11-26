"""Publish direct MTP energies per chemical cell, plus phonon free energies where available."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/"atomistic"))
from fit_eci import Labels, read_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("design",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("archives",nargs="+",type=Path)
    p.add_argument("--ratios",nargs="+",type=float)
    p.add_argument("--thermal",action="store_true")
    args=p.parse_args()
    design={r["id"]:r for r in read_rows(args.design/"design.csv")}
    source=Labels(args.archives[0],args.archives[1:])
    rows=[]
    for job in read_rows(args.design/"jobs.csv"):
        ratio=float(job["ratio"])
        if args.ratios and ratio not in args.ratios:
            continue
        meta=design[job["id"]]
        info=json.loads(source.read(job["label_subdir"],"label.json"))
        if meta["cell_sha256"]!=info["source_cell_sha256"] or abs(ratio-info["volume_ratio"])>1e-8:
            raise ValueError(f"mismatch for {job['id']}")
        row={"id":job["id"],"split":meta["split"],"N":int(meta["N"]),"N_Al":int(meta["N_Al"]),
             "V_over_Vref":ratio,"V_A3_per_atom":info["volume_A3_atom"],
             "E_MTP_eV_atom":info["static_eV_atom"],"E_MTP_eV_cell":info["n_atoms"]*info["static_eV_atom"],
             "fmax_eV_A":info["fmax_eV_A"],"stable_mesh":info["stable_mesh"],"cell_sha256":meta["cell_sha256"]}
        if args.thermal and info["stable_mesh"]:
            for obs in csv.DictReader(io.StringIO(source.read(job["label_subdir"],"vibrations.csv"))):
                key=f"Fvib_{int(float(obs['temperature_K']))}K_eV_atom"
                row[key]=float(obs["Fvib_eV_atom"])
        rows.append(row)
    if not rows:
        raise ValueError("no labels found")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with args.out.open("w",newline="",encoding="utf-8") as stream:
        w=csv.DictWriter(stream,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} independently evaluated structure-volume labels")


if __name__=="__main__":
    main()
