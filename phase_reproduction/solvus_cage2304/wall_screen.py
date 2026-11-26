"""Check support for the tighter wall in the old, broader-cage Ti–Al ensembles."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "solvus_closure"
sys.path.insert(0, str(PREVIOUS))
from thermal_check import load_source, reference, mbar
from vertex_bootstrap import solve


def screen(names):
    manifests = [PREVIOUS/"atomistic/series"/name/"windows.csv" for name in names]
    t0, n, shape, wall, axial, runs = load_source(manifests)
    model = reference(n, runs, t0)
    assert wall == .4 and axial == 1
    k = model["k"][model["idx"]]
    q = model["q"][model["idx"]]
    w = np.concatenate([r["wall"] for r in runs])
    tight = w < .35
    wide = solve(model,model["log_frame"],n,t0,.37,[1244,3.545],[1200,1290,3.43,3.68],.04)
    if not wide or not wide["profile"]:
        raise ValueError("Broad-wall vertex unconfirmed")
    t = wide["T_K"]
    mu = wide["mu_eV"]
    logw = (model["log_frame"] + (1/t0-1/t)/mbar.KB*model["energy"]
            + (mu-model["ref"])*k/(mbar.KB*t))
    fractions = []
    for mask in (q <= .37, q > .37):
        retained = mask & tight
        retained_logw = logw[retained]
        if np.any(retained):
            total = logsumexp(retained_logw)
            prob = np.exp(retained_logw-total)
            fraction = float(np.exp(total-logsumexp(logw[mask])))
            neff = float(1/((prob**2)@model["g"][retained]))
        else:
            fraction,neff = 0.,0.
        fractions.append(dict(frames=int(retained.sum()),fraction=fraction,effective_frames=neff))
    local_shift = None
    if min(p["fraction"] for p in fractions)>0:
        local_shift = (mbar.KB*t*t * np.log(fractions[1]["fraction"]/fractions[0]["fraction"])
                       / (n * wide["latent_meV_atom"] * .001))
    return dict(N=n,source_temperature_K=t0,shape=shape,manifests=names,production=len(w),
                T_at_evaluation=t,field_eV=mu,broad_ensemble_phase_cut=.37,
                retained_by_phase=fractions,local_shift_K=local_shift,
                note="Conditional on the original W=0.40 shape guard. A first-order thermodynamic "
                     "linearization, only for choosing direct simulation temperatures.")


if __name__ == "__main__":
    cases = [["zc237m"],["zc458a"],["zc4514l","zc4514c","zc4514r"],["zc4523c","zc4523s"]]
    dest = HERE/"analysis/wall_screen.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    data = [screen(names) for names in cases]
    dest.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf8")
    for row in data:
        print(row["N"],row["retained_by_phase"],row["local_shift_K"])
