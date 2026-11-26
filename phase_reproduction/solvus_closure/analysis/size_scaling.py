"""Compare order odds across independently sampled sizes at common temperatures."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from apex_temperature import points


KB = 8.617333262145e-5
HERE = Path(__file__).resolve().parent


def intervals(values):
    return np.quantile(values, [.025, .975]).tolist() if len(values) else None


def pairs_at_common_temperatures(records, rng, repeats):
    by_size = {n: {point["T_K"]: (point, np.asarray(boot)) for point, boot in obs}
               for n, obs in records.items() if n >= 768}
    output = []
    for small, big in combinations(sorted(by_size), 2):
        differences = []
        draws_at = {}
        for t in sorted(by_size[small].keys() & by_size[big].keys()):
            pa, aa = by_size[small][t]
            pb, bb = by_size[big][t]
            if pa["field_edge"] or pb["field_edge"]:
                continue
            diff = pb["S"] - pa["S"]
            rate = pb["estimated_local_dS_dT"] - pa["estimated_local_dS_dT"]
            factor = 1000*KB*t/(big-small)
            entry = dict(T_K=t,delta_S=diff,ordered_advantage_meV_atom=diff*factor,
                         both_dividers_separate_modes=pa["divider_separates_modes"] and pb["divider_separates_modes"],
                         both_mu_within_sampled_range=pa["inside_simulated_fields"] and pb["inside_simulated_fields"],
                         local_d_delta_S_dT=rate,
                         locally_predicted_zero_K=t-diff/rate if rate < 0 else None)
            if aa.size and bb.size:
                ia = rng.integers(len(aa),size=repeats)
                ib = rng.integers(len(bb),size=repeats)
                ds = bb[ib,2]-aa[ia,2]
                draws_at[t] = ds
                entry.update(delta_S_interval_95=intervals(ds),positive_diff_bootstrap_fraction=float(np.mean(ds>0)),
                             ordered_advantage_interval_95_meV_atom=intervals(ds*factor))
                if aa.shape[1]>=6 and bb.shape[1]>=6:
                    dr = bb[ib,5]-aa[ia,5]
                    good=dr<0
                    entry.update(locally_negative_slope_fraction=float(good.mean()),
                                 conditional_local_zero_interval_95_K=intervals(t-ds[good]/dr[good]))
            differences.append(entry)
        zeros = []
        for a,b in zip(differences,differences[1:]):
            da,db=a["delta_S"],b["delta_S"]
            if not da>=0>=db or da==db:
                continue
            ta,tb=a["T_K"],b["T_K"]
            record=dict(bracket_K=[ta,tb],interpolated_K=ta+(tb-ta)*da/(da-db))
            if ta in draws_at and tb in draws_at:
                ds1,ds2=draws_at[ta],draws_at[tb]
                good=ds1>ds2
                est=ta+(tb-ta)*ds1[good]/(ds1[good]-ds2[good])
                record.update(fraction_bracketing=float(np.mean((ds1>=0)&(ds2<=0))),
                              fraction_negative_slope=float(good.mean()),
                              conditional_zero_interval_95_K=intervals(est))
            zeros.append(record)
        output.append(dict(sizes=[small,big],common_temperatures=differences,direct_sign_crossings=zeros))
    return output


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=HERE/"size_scaling.json")
    args=parser.parse_args()
    paths=sorted(HERE.glob("T[0-9]*_N[0-9]*.json"))
    rng=np.random.default_rng(124511)
    result=dict(note="The slope of S_N with N estimates beta times the bulk ordered free-energy advantage if "
                     "both restricted basins are pure phases and their finite-volume offsets are N-independent. "
                     "These assumptions and the infinite-size limit are not established by the fit. Only "
                     "interior field maxima are used. Bootstrap quantifies independent isotherm statistics.",
                by_cut={str(c):pairs_at_common_temperatures(points(paths,c),rng,40000) for c in (.42,.47,.5)})
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
