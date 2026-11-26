"""Compare composition and order across independently reweighted finite cells."""
import argparse
import json
from pathlib import Path

import numpy as np


def matched_x(scans, xs):
    out = []
    rng = np.random.default_rng(3129600)
    for x in xs:
        row = dict(x_Al=float(x), cells=[])
        for s in scans:
            grid = s["scan"]["grid"]
            concentration = [r["x_Al"] for r in grid]
            if not concentration[0] <= x <= concentration[-1]:
                continue
            def interpolate(key):
                return float(np.interp(x, concentration, [r[key] for r in grid]))
            n=s["N"]
            k=round(n*x)
            groups=rng.multivariate_hypergeometric([n//4]*4,k,size=4096)
            random_order=4*(4*groups.max(axis=1)-k)/(3*n)
            row["cells"].append(dict(N=n,mu_eV=interpolate("mu_eV"),mean_Q=interpolate("mean_Q"),
                random_decoration_mean_Q_at_nearest_fixed_k=float(random_order.mean()),
                random_decoration_standard_error=float(np.std(random_order,ddof=1)/np.sqrt(len(groups))),
                Nvar_Q=interpolate("Nvar_Q"), dx_dmu_eV_inv=interpolate("dx_dmu_eV_inv"),
                effective_frames_with_time_correlation=interpolate("effective_frames_with_time_correlation")))
        if len(row["cells"])>=2:
            out.append(row)
    return out


def main(args):
    scans=[s for path in args.inputs for s in json.loads(path.read_text(encoding="utf-8"))]
    groups={}
    for s in scans:
        spec=(s["temperature_K"],s["cap"],s["axial_free"])
        group=groups.setdefault(spec,{})
        if s["N"] in group:
            raise ValueError("Select exactly one merged isotherm per size and condition")
        group[s["N"]]=s
    result=[]
    for (t,cap,axial),g in groups.items():
        data=[g[k] for k in sorted(g)]
        curves=matched_x(data,np.arange(.02,.501,.01))
        result.append(dict(temperature_K=t,W_max=cap,axial_free=axial,matched_composition=curves,
                           note="Linear interpolation at a common ensemble mean Al fraction. Random-mixing reference "
                           "is the independent expectation of the folded order at the nearest fixed number of Al "
                           "atoms (finite hypergeometric draws); local chemical correlations can increase the observed "
                           "order even without long-range ordering. Weak double peaks in small Q histograms alone "
                           "do not locate a bulk first-order transition."))
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",nargs="+",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
