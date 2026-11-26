"""Prepare the author's 13 QHA windows with orbit records for paired estimators."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.interpolate import CubicSpline


HERE = Path(__file__).resolve().parent
KBT = 8.617333262145e-5 * 1000


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--size", type=int, choices=[768, 5120], required=True)
    p.add_argument("--eq", type=int, required=True)
    p.add_argument("--prod", type=int, required=True)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--start", choices=["random", "ordered"], required=True)
    p.add_argument("--source", choices=["quantum","classical"], default="quantum")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--midpoints", action="store_true",
                   help="sample the 12 midpoints between the original 13 centers")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/"samples").mkdir()
    author = json.loads((HERE.parent/"solvus_1000/c46_reference.json").read_text())
    points = author["windows_c_Al_and_mu_occ_eV"]
    author_centers = np.arange(13, 26) / 100
    mu_estimates = [mu_i + 2*KBT*100*(x-c0)
                    for (x,mu_i),c0 in zip(points,author_centers,strict=True)]
    if np.ptp(mu_estimates)>2e-10:
        raise ValueError(f"Cannot recover author's window fields {mu_estimates}")
    mu = np.mean(mu_estimates)
    settings = np.arange(135, 246, 10) / 1000 if a.midpoints else author_centers
    for model in ("quantum", "classical"):
        grid = np.load(HERE/f"{model}_grid.npz",allow_pickle=False)
        v = grid["volume_A3_atom"]
        j = grid["J_static"] + grid["J_vib"][:,0,:]
        second = CubicSpline(v,j,axis=0)(v,2)
        np.savetxt(a.out/f"{model}.txt",np.column_stack([v,j,second]),
                   header="volume J[20] d2J_dv2[20]",fmt="%.17g")
    for src in [HERE.parent/"mc/vcs.cpp", HERE.parent/"mc/static/motifs.txt"]:
        shutil.copyfile(src, a.out/src.name)
    dims = (8,8,6) if a.size==768 else (16,16,10)
    cfg = dict(size=a.size,eq=a.eq,prod=a.prod,stride=a.stride,start=a.start,
               source=a.source,seed=a.seed,shape=dims,mu=float(mu),kappa=100,
               T_K=1000,volume_step_A3_atom=.035,coeff_interpolation="not-a-knot",
               windows=settings.tolist(),domain="flat dv at zero pressure")
    (a.out/"manifest.json").write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")
    with (a.out/"windows.tsv").open("w",newline="") as stream:
        wr=csv.writer(stream,delimiter="\t",lineterminator="\n")
        for i,c0 in enumerate(settings):
            wr.writerow([a.source,1000,*dims,c0,format(mu,".16g"),a.seed+i,a.eq,a.prod,a.start,a.source+".txt",.035,"orbit-sums"])


if __name__=="__main__":
    main()
