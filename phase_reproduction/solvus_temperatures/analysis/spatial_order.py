"""Check chemical domains in terminal states, separately from finite-cell phase weights."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import uniform_filter

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"solvus_1000"))
from phase_checks import terminal_domains


def block_order(t, mask, size):
    # Even basal extents give equal site counts for all four sublattices in each block.
    by_color=np.stack([uniform_filter((t*(mask==j)).sum(-1).astype(float),size=size,mode="wrap")*2 for j in range(4)])
    return float(np.mean(4*(by_color.max(axis=0)-by_color.mean(axis=0))/3))


def main(args):
    states={s["run"]:s for s in terminal_domains(args.manifests)}
    output=[]
    for m in args.manifests:
        for row in csv.DictReader(m.open(encoding="utf-8-sig")):
            path=m.parent/(row["run"]+".state")
            if not path.is_file():
                raise FileNotFoundError(path)
            shape=tuple(int(z) for z in row["shape"].split("x"))
            types=np.loadtxt(path,skiprows=1,usecols=0,dtype=int).reshape(*shape,2)
            i,j,z,b=np.indices(types.shape)
            sub=2*((i+b)%2)+(j+b)%2
            groups=[sub==j for j in range(4)]
            rng=np.random.default_rng(int(row["seed"])+349)
            sizes=[s for s in ((2,2,2),(4,4,3),(6,6,4)) if all(a<=b for a,b in zip(s,shape)) and s!=shape]
            obs={str(s):block_order(types,sub,s) for s in sizes}
            null={str(s):[] for s in sizes}
            for _ in range(96):
                shuffled=types.copy()
                for group in groups:
                    shuffled[group]=rng.permutation(types[group])
                for s in sizes:
                    null[str(s)].append(block_order(shuffled,sub,s))
            data={s:dict(mean_folded_Q=obs[s],conditional_null_mean=float(np.mean(ns)),
                         conditional_null_95=np.quantile(ns,[.025,.975]).tolist()) for s,ns in null.items()}
            output.append(dict(**states[row["run"]],start=row["start"],mu_eV=float(row["mu_eV"]),
                               K_eV=float(row["K_eV"]),q0=float(row["q0"]),temperature_K=float(row["T_K"]),
                               block_order=data))
    result=dict(note="One terminal snapshot per window. Fourier sidebands and sliding-block chemical order are "
                     "compared with permutations preserving the four global sublattice occupancies. Spatial correlations "
                     "can include short-range order and antiphase domains; these measures do not independently define "
                     "coexisting bulk phases.",windows=output)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests",type=Path,nargs="+")
    p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
