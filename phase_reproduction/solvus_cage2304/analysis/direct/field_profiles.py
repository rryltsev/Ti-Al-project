"""Exact MBAR field scan at 1243 K, with direct chemical fields and their block intervals."""
import argparse
import csv
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
from isotherms import HERE,load,mbar,load_json
from profiles import write_csv


def rows(path):
    d=load_json(path);z=d["fixed_cut"]["0.37"];fields=np.asarray(d["direct_fields_eV"])
    t,n,_,w,_,rr=next(load(list(map(Path,d["manifests"])),0))
    runs=[r for r in rr if r["spec"][1]>0]
    k,q,spec,blocks,_,_=mbar.make_data(n,runs)
    hist,_=mbar.aggregate(blocks,len(spec))
    kt=t*mbar.KB;ref,base,_,_=mbar.fit_mbar(k,q,spec,hist,kt=kt)
    ks,ik=np.unique(k,return_inverse=True)
    spectra=[]
    for mask in (q<=.37,q>.37):
        sp=np.full(len(ks),-np.inf);np.logaddexp.at(sp,ik[mask],base[mask]);spectra.append(sp)
    root={r["mu_eV"]:r["branch"] for r in z["crossings"]}
    ci=d.get("bootstrap",{}).get("cuts",{}).get("0.37",{}).get("actual_field_scores",{}).get("interval_95")
    mu_opt=z["maximum"]["mu_eV"]
    for mu in np.unique(np.r_[np.linspace(fields.min(),fields.max(),351),fields,list(root),mu_opt]):
        tilt=(mu-ref)*ks/kt;S=logsumexp(spectra[1]+tilt)-logsumexp(spectra[0]+tilt)-np.log(4.)
        i,=np.nonzero(fields==mu)
        yield dict(T_K=t,W=w,mu_eV=mu,score=S,is_simulated_field=int(bool(len(i))),
                   lower95=ci[i[0]][0] if len(i) and ci else None,upper95=ci[i[0]][1] if len(i) and ci else None,
                   is_maximum=int(mu==mu_opt),root_branch=root.get(mu),source=path.name,n_actual_fields=len(fields))


def update():
    parts=[]
    for cap in ("0p35","0p40"):
        names=[HERE/(s+"_T1243_W"+cap+".json") for s in ("final","initial","scout")]
        src=next((p for p in names if p.is_file()),None)
        if not src:continue
        out=HERE/("field_curve_T1243_W"+cap+".csv")
        if not out.exists() or out.stat().st_mtime<src.stat().st_mtime:
            write_csv(out,list(rows(src)))
        with out.open(encoding="utf8",newline="") as f:parts.extend(csv.DictReader(f))
    if parts:write_csv(HERE/"field_profiles_1243.csv",parts)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--watch",action="store_true");args=p.parse_args()
    while True:
        update()
        if not args.watch or all((HERE/f"final_T1243_W{w}.json").is_file() for w in ("0p35","0p40")):break
        time.sleep(25)
