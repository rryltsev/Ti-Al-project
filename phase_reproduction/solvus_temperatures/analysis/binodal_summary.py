"""Collect branch estimates and independent, paired-observable uncertainty on differences."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np


def compact(data):
    fit = data["coexistence"]
    central = fit["equal_single_variant"]
    if central is None:
        raise ValueError("No resolved pair in this low-temperature input")
    sims = np.asarray(fit["bootstrap"]["joint_mu_xalpha_xalpha2"])
    vals = np.array([central[k] for k in ("mu_eV","x_alpha","x_alpha2")])
    vals[1:] *= 100
    sims[:,1:] *= 100
    def reduced(entry):
        if not entry:
            return None
        return [entry["mu_eV"],100*entry["x_alpha"],100*entry["x_alpha2"]]
    info = dict(N=data["N"],cap=data["cap"],axial_free=data["axial_free"],shape=data["shape"],
                resolved_bootstrap=fit["bootstrap"]["resolved"],requested_bootstrap=fit["bootstrap"]["count"],
                mu_eV_xalpha_xalpha2_at_percent=vals.tolist(),
                interval_95=np.quantile(sims,[.025,.975],axis=0).T.tolist(),
                equal_height_barrier_kBT=fit["equal_height_separator"]["barrier_kBT"],
                folded_order_modes=central["basin_mode_q"],
                phase_effective_frames=central["phase_Kish_approx_with_run_inefficiency"],
                weakest_joint_histogram_tree_BC=data["scan"]["weakest_tree_Bhattacharyya"],
                mean_axial_ratio=central["phase_observables"]["mean_c_a"],
                volume_A3_atom=central["phase_volumes_A3_atom"],
                mean_max_wall_coordinate=central["phase_observables"]["mean_wall"],
                bandwidth_sensitivity={k:reduced(v) for k,v in fit["bandwidth_sensitivity"].items()},
                halves={k:reduced(v) for k,v in data["time_halves"].items()})
    return info, vals, sims


def main(args):
    source=[]
    for path in args.inputs:
        for d in json.loads(path.read_text(encoding="utf-8")):
            source.append((path.name,*compact(d)))
    results=[]
    for (fa,aa,va,ba),(fb,ab,vb,bb) in combinations(source,2):
        if sum(aa[k]!=ab[k] for k in ("N","cap","axial_free")) != 1:
            continue
        # Independent trajectories across the two specified series. Within each
        # replicate the two endpoints and their chemical field remain paired.
        pairwise = (bb[None,:,:]-ba[:,None,:]).reshape(-1,3)
        results.append(dict(change_from=fa,change_to=fb,delta_mu_eV_xalpha_xalpha2_pp=(vb-va).tolist(),
                            interval_95=np.quantile(pairwise,[.025,.975],axis=0).T.tolist()))
    out=dict(cells=[dict(source=f,**i) for f,i,_,_ in source],independent_differences=results,
             note="95% percentile intervals are conditional on the sampled trajectories, within-trajectory "
             "block resampling and the resolved branch rule. Differences preserve joint covariance of field "
             "and branch estimates within a series. No assumed finite-size extrapolation is used.")
    args.out.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",nargs="+",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
