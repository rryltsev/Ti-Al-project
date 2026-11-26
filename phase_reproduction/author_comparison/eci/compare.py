"""Apply the existing TiAl fit implementation to the original C46 observations and cross-score both data sets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT/"atomistic"))
from hcp_orbits import ORBIT_NAMES, MULTIPLICITIES, VREF
from fit_eci import fit_volume


def rows(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(p, records):
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0])
        w.writeheader()
        w.writerows(records)


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k:f[k] for k in f}


def rmse(values):
    return float(np.sqrt(np.mean(np.square(values))))


def run(src, out):
    out.mkdir(parents=True,exist_ok=True)
    static_file = src/"models/static/volume_temperature_coefficients.npz"
    thermal_file = src/"models/HA_QHA/volume_temperature_coefficients.npz"
    s,h = load_npz(static_file),load_npz(thermal_file)
    for key in ("X","training","stable","temperatures_K","volumes_A3_atom"):
        if not np.array_equal(s[key],h[key]):
            raise ValueError(f"Contradictory author arrays for {key}")
    g = json.loads((src/"models/geometry.json").read_text(encoding="utf-8"))
    orbit_map = {"empty":"constant", "t222_capped":"t222_1","t222_uncapped":"t222_2"}
    names = tuple(orbit_map.get(o["name"],o["name"]) for o in g["orbits"])
    if names != ORBIT_NAMES or not np.array_equal(MULTIPLICITIES,[o["multiplicity_per_atom"] for o in g["orbits"]]):
        raise ValueError("Check orbit names, order and weights")
    idx = rows(src/"structures/index.csv")
    if not np.array_equal([int(r["index"]) for r in idx],np.arange(len(idx))):
        raise ValueError("Configuration index order differs from design")
    train,stable,x=s["training"],s["stable"],s["X"]
    if not np.array_equal(train,[r["role"]=="fit" for r in idx]):
        raise ValueError("Inconsistent split")
    volumes,temps=s["volumes_A3_atom"],s["temperatures_K"]
    vref, = np.flatnonzero(np.isclose(volumes,VREF,rtol=0,atol=1e-8))
    if np.max(abs(s["E_static_eV_atom"][vref]-np.array([float(r["static_target_eV_atom"]) for r in idx])))>1e-11:
        raise ValueError("Index and source energies differ at reference volume")
    if not (np.isfinite(h["Y_eV_atom"][stable]).all() and np.isfinite(s["E_static_eV_atom"]).all()):
        raise ValueError("Missing labels among included configurations")
    author_stat,author_vib,replay=[],[],[]
    # Include both E and Fvib as in our label reader, then reconstruct F_eff.
    for iv,v in enumerate(volumes):
        use = stable[iv]
        e = s["E_static_eV_atom"][iv,use,None]
        y = h["Y_eV_atom"][iv,use]
        j,err,metrics = fit_volume(x[use],np.column_stack([e,y-e]),train[use])
        js,jv = j[:,0],j[:,1:].T
        jtot = jv+js
        stat_delta = float(np.max(abs(js-s["eci"][iv])))
        eff_delta = float(np.max(abs(jtot-h["eci"][iv])))
        vib_delta = float(np.max(abs(jv-(h["eci"][iv]-s["eci"][iv]))))
        for n in (stat_delta,eff_delta,vib_delta):
            if n > 1e-9:
                raise ValueError(f"Numerical replay mismatch {n:g}")
        iloo=metrics["loo_rmse"]
        replay.append(dict(volume_ratio=v/VREF,n_fit=metrics["train"],n_control=metrics["control"],
            rank=int(np.linalg.matrix_rank(x[use&train])),condition=metrics["condition_number"],
            excluded="|".join(r["configuration"] for r,ok in zip(idx,use) if not ok),
            max_static_J_error_eV=stat_delta,max_vib_J_error_eV=vib_delta,
            max_full_J_error_eV=eff_delta,
            static_train_rmse_meV=1000*rmse(err[train[use],0]),static_loo_rmse_meV=1000*iloo[0],
            static_control_rmse_meV=1000*metrics["control_rmse"][0]))
        author_stat.append(js)
        author_vib.append(jv)
    a_stat,a_vib=np.array(author_stat),np.array(author_vib)
    np.savez(out/"author_refit_grid.npz",orbit_names=ORBIT_NAMES,volume_A3_atom=volumes,
             temperature_K=temps,J_static=a_stat,J_vib=a_vib)
    save_csv(out/"author_replay_by_volume.csv",replay)

    # Verify the CSVs are another view of the published NPZ, including both affine terms.
    csv_grid = rows(src/"ECI_TiAl.csv")
    if len(csv_grid)!=len(volumes)*len(temps)*len(names):
        raise ValueError("Unexpected number of tabulated coefficients")
    tv={int(t):i for i,t in enumerate(temps)}
    vv={round(v,9):i for i,v in enumerate(volumes)}
    seen=set()
    csv_max=0.
    for r in csv_grid:
        vi,ti,k=vv[round(float(r["V_A3_per_atom"]),9)],tv[int(r["T_K"])],int(r["index"])
        if (vi,ti,k) in seen or g["orbits"][k]["name"]!=r["cluster"]:
            raise ValueError("Missing or duplicate keyed data in ECI_TiAl")
        seen.add((vi,ti,k))
        val=s["eci"][vi,ti,k]
        target=h["eci"][vi,ti,k]
        csv_max=max(csv_max,abs(float(r["J_static_eV"])-val),abs(float(r["J_effective_eV"])-target),
                    abs(float(r["delta_J_vib_eV"])-(target-val)))
    if csv_max>1e-10:
        raise ValueError("CSV and NPZ disagree")

    own_grid=load_npz(PROJECT/"atomistic/atat/qha_fit/grid.npz")
    own_t=own_grid["temperature_K"]
    if tuple(own_grid["orbit_names"])!=ORBIT_NAMES or not np.allclose(own_grid["volume_A3_atom"],volumes,atol=1e-8,rtol=0):
        raise ValueError("Change of basis or volume grid")
    shared=np.array([tv[t] for t in own_t])
    own_cells=rows(PROJECT/"atomistic/atat/output/design.csv")
    own_x=np.array([[float(r[k]) for k in ORBIT_NAMES] for r in own_cells])
    own_train=np.array([r["split"]=="train" for r in own_cells])
    own_labels=rows(PROJECT/"atat_comparison/MTP_labels_414.csv")
    lookup={(r["id"],round(float(r["V_over_Vref"]),6)):r for r in own_labels}
    oxs,oxv,ostab=[],[],[]
    for v in volumes:
        label=[lookup[r["id"],round(v/VREF,6)] for r in own_cells]
        oxs.append([float(r["E_MTP_eV_atom"]) for r in label])
        stab=np.array([r["stable_mesh"].lower()=="true" for r in label])
        ostab.append(stab)
        oxv.append([[float(r[f"Fvib_{t:g}K_eV_atom"]) if ok else np.nan for t in own_t]
                       for r,ok in zip(label,stab)])
    own_s,own_v,own_mask=np.array(oxs),np.array(oxv),np.array(ostab)
    same=np.max(abs(x[:,None]-own_x[None,:]),axis=2)<1e-8
    coefs={"author":(a_stat,a_vib[:,shared]),"ours":(own_grid["J_static"],own_grid["J_vib"])}
    observations={"author":(x,s["E_static_eV_atom"],h["Y_eV_atom"][:,:,shared]-s["E_static_eV_atom"][:,:,None],train,stable,[r["configuration"] for r in idx]),
        "ours":(own_x,own_s,own_v,own_train,own_mask,[r["id"] for r in own_cells])}
    summary,individual=[],[]
    for source,(js,jv) in coefs.items():
        for dataset,(xx,es,fv,is_train,mask,labels) in observations.items():
            for iv,v in enumerate(volumes):
                train_same=(same[:,own_train&own_mask[iv]].any(axis=1) if dataset=="author" else
                            same[train&stable[iv],:].any(axis=0))
                subsets={"train":is_train&mask[iv],"control":~is_train&mask[iv]}
                if dataset=="author":
                    last_pair=np.array([int(r["atoms"])==64 and int(r["Al_atoms"])==22 for r in idx])
                else:
                    last_pair=np.array([int(r["N"])==64 and int(r["N_Al"])==22 for r in own_cells])
                subsets["control_22Al_of_64"]=subsets["control"]&last_pair
                subsets["other_controls"]=subsets["control"]&~last_pair
                if source!=dataset:
                    subsets["control_unique_X"]=subsets["control"]&~train_same
                sres=xx@js[iv]-es[iv]
                vres=xx@jv[iv].T-fv[iv]
                for ti,t in enumerate(own_t):
                    total=sres+vres[:,ti]
                    for split,ok in subsets.items():
                        if not ok.any():continue
                        summary.append(dict(ECI=source,data=dataset,split=split,n=int(ok.sum()),
                            V_over_Vref=v/VREF,T_K=int(t),static_RMSE_meV_atom=1000*rmse(sres[ok]),
                            vib_RMSE_meV_atom=1000*rmse(vres[ok,ti]),total_RMSE_meV_atom=1000*rmse(total[ok])))
                    if iv==vref and t==1000:
                        for k in np.flatnonzero(mask[iv]):
                            individual.append(dict(ECI=source,data=dataset,id=labels[k],split="train" if is_train[k] else "control",
                                X_in_other_training=bool(train_same[k]),static_MTP_eV_atom=es[iv,k],Fvib_MTP_eV_atom=fv[iv,k,ti],
                                static_error_meV_atom=1000*sres[k],vib_error_meV_atom=1000*vres[k,ti],
                                total_error_meV_atom=1000*total[k]))
    save_csv(out/"cross_validation.csv",summary)
    save_csv(out/"vref_1000K_predictions.csv",individual)
    differences=[]
    for iv,v in enumerate(volumes):
        for k,label in enumerate(ORBIT_NAMES):
            for ti,t in enumerate(own_t):
                ja,jav=a_stat[iv,k],a_vib[iv,shared[ti],k]
                jo,jov=own_grid["J_static"][iv,k],own_grid["J_vib"][iv,ti,k]
                differences.append(dict(V_over_Vref=v/VREF,T_K=int(t),orbit=label,author_static_eV=ja,our_static_eV=jo,
                    author_vib_eV=jav,our_vib_eV=jov,author_total_eV=ja+jav,our_total_eV=jo+jov,
                    static_ours_minus_author_meV=1000*(jo-ja),vib_ours_minus_author_meV=1000*(jov-jav),
                    total_ours_minus_author_meV=1000*(jo+jov-ja-jav)))
    save_csv(out/"matched_ECI.csv",differences)
    report=dict(volume_ratios=(volumes/VREF).tolist(),author_temperature_K=temps.tolist(),our_temperature_K=own_t.tolist(),
                author_stable=int(stable.sum()),our_stable=int(own_mask.sum()),shared_X_rows=int(same.sum()),
                csv_NPZ_error_eV=csv_max,
                max_replay_static_eV=max(r["max_static_J_error_eV"] for r in replay),
                max_replay_vib_eV=max(r["max_vib_J_error_eV"] for r in replay),
                max_replay_total_eV=max(r["max_full_J_error_eV"] for r in replay),
                source_sha256={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (static_file,thermal_file)})
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:val for k,val in report.items() if k not in ("volume_ratios","author_temperature_K","our_temperature_K","source_sha256")},indent=2))
    print(replay[vref])
    for r in summary:
        if r["T_K"]==1000 and abs(r["V_over_Vref"]-1)<1e-8 and r["split"]!="train": print(r)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("author_directory",type=Path)
    parser.add_argument("--out",type=Path,default=HERE)
    args=parser.parse_args()
    run(args.author_directory,args.out)
