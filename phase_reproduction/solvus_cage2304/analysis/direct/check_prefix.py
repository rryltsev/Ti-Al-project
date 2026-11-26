"""Read local immutable production prefixes for selecting a temperature and field span."""
import argparse
from pathlib import Path
import json
import time
import numpy as np

from isotherms import HERE,STUDY,manual,mbar,atomic_json,load_json


def summarize(path,full):
    info=load_json(path.parent/"prefix.json")
    cuts={}
    for cut in (".37",".47"):
        key=str(float(cut));f=full["fixed_cut"][key]["maximum"]
        reps=full.get("bootstrap",{}).get("cuts",{}).get(key,{}).get("replicates",[])
        slopes=np.asarray([r["apex"][5] for r in reps])*full["N"]*.001/(mbar.KB*full["T_K"]**2)
        sig=slopes<0
        delta=full["T_K"]-np.asarray([r["apex"][4] for r in reps])[sig]/slopes[sig]
        cuts[key]=dict(mu=f["mu_eV"],S=f["score"],delta_h=f["delta_h_meV"],slope_K_inv=f["dS_dT_fixed_mu"],
            locally_estimated_T_K=f["T_local_linear_zero"],negative_bootstrap_slope=np.mean(sig),
            conditional_local_T_interval_95=np.quantile(delta,[.025,.975]).tolist() if np.sum(sig)>1 else None,
            phase_ESS=[k["time_ESS"] for k in f["conditional"]],
            S_halves=[v[key]["maximum"]["score"] for v in full["halves"].values()],
            Q_pair=f["profiles"]["0.04"])
    return dict(manifest=str(path),prefix_through_sweep=info["through_sweep"],cuts=cuts)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch",action="store_true")
    p.add_argument("--name",default="cg3543d")
    args=p.parse_args()
    base=STUDY/"atomistic"/"prefixes"
    while True:
        for prefix in sorted(base.rglob("prefix.json")):
            manifest=prefix.parent/"windows.csv"
            if not manifest.is_file() or args.name not in manifest.read_text(encoding="utf-8-sig"):
                continue
            info=load_json(prefix)
            tag=f"prefix_{args.name}_{info['through_sweep']}"
            short=HERE/(tag+"_estimate.json")
            if short.exists():continue
            full=manual([manifest],HERE/(tag+".json"),32,True,999341)
            compact=summarize(manifest,full)
            atomic_json(short,compact)
            print(json.dumps(compact),flush=True)
        if not args.watch or (STUDY/"atomistic"/"series"/args.name/"window_status.json").exists():break
        time.sleep(8)
