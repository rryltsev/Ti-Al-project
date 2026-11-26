"""C46 classical harmonic QHA at the requested T, using the original author's fit mask."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BASE = PROJECT / "classical_c46_1000"
KB = 8.617333262145e-5


def tables(src, t):
    with (src / "structures/index.csv").open(encoding="utf-8-sig", newline="") as f:
        configs = [r["configuration"] for r in csv.DictReader(f)]
    with (BASE / "structure_labels.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = {(float(r["volume_A3_atom"]),r["configuration"]):r for r in csv.DictReader(f)}
    with np.load(src / "models/HA_QHA/volume_temperature_coefficients.npz") as d:
        mask, fit, x = d["stable"],d["training"],d["X"]
        temps = d["temperatures_K"]
        v = d["volumes_A3_atom"]
        j_q = CubicSpline(temps,d["eci"],axis=1)(t)
        hermite = CubicHermiteSpline(temps,d["eci"],d["d_eci_dT_K"],axis=1)(t)
        coarse = CubicSpline(temps[::2],d["eci"][:,::2],axis=1)(t)
    with np.load(BASE/"classical_grid.npz") as d:
        j_static, old_class = d["J_static"],d["J_vib"][:,0,:]
        names=d["orbit_names"]
    assert len(rows)==int(mask.sum())==409
    j = []
    checks = []
    for i, vol in enumerate(v):
        records = [rows[vol, configs[a]] for a in np.flatnonzero(mask[i]&fit)]
        old = np.array([float(r["Fcl_eV_atom"]) for r in records])
        modes = np.array([float(r["effective_mode_weight"]) for r in records])
        design = x[mask[i]&fit]
        assert np.linalg.matrix_rank(design)==20
        old_diff = np.max(np.abs(np.linalg.lstsq(design,old,rcond=None)[0] - old_class[i]))
        assert old_diff < 2e-13
        labels = t/1000 * old - modes*KB*t*np.log(t/1000)
        j.append(np.linalg.lstsq(design,labels,rcond=None)[0])
        control = [rows[vol, configs[a]] for a in np.flatnonzero(mask[i]&~fit)]
        y_control = [t/1000*float(r["Fcl_eV_atom"]) - float(r["effective_mode_weight"])*KB*t*np.log(t/1000)
                     for r in control]
        checks.append(dict(V_A3_atom=float(vol), n_training=len(records), n_control=len(control),
                           J_vib_classical_check_1000_K_max_abs_eV=float(old_diff),
                           training_RMSE_vib_meV_atom=1000*np.sqrt(np.mean((labels-design@j[-1])**2)),
                           control_RMSE_vib_meV_atom=1000*np.sqrt(np.mean((y_control-x[mask[i]&~fit]@j[-1])**2))))
    grids = {"quantum":j_q, "classical":j_static+np.array(j)}
    if t in temps:
        with np.load(src/"models/HA_QHA/volume_temperature_coefficients.npz") as d:
            np.testing.assert_allclose(j_q,d["eci"][:,list(temps).index(t),:],rtol=0,atol=2e-13)
    for name,coeff in grids.items():
        folder=HERE/f"T{t:g}"
        folder.mkdir(exist_ok=True)
        np.savez(folder/f"{name}.npz", J_static=j_static, J_vib=(coeff-j_static)[:,None,:],
                 volume_A3_atom=v,temperature_K=[t],orbit_names=names)
        np.savetxt(folder/f"{name}.txt",np.column_stack([v,coeff,CubicSpline(v,coeff)(v,2)]),
                   header="volume J[20] d2J_dv2[20]",fmt="%.17g")
    source_files = [BASE/"structure_labels.csv", BASE/"classical_grid.npz", src/"structures/index.csv",
                    src/"models/HA_QHA/volume_temperature_coefficients.npz"]
    metadata = dict(T_K=t, n_stable=409, n_orbits=20, volumes=checks,
                    sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                    target_quantum_temperature_is_original=bool(t in temps),
                    quantum_temperature_Hermite_check_max_abs_label_eV_atom=float(np.max(abs(x@(hermite-j_q).T))),
                    quantum_temperature_50_vs_25K_max_abs_label_eV_atom=float(np.max(abs(x@(coarse-j_q).T))),
                    temperature_bracket_quantum_K=[float(temps[temps<=t][-1]),float(temps[temps>=t][0])])
    (folder/"fit.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return folder


def main(a):
    folder=tables(a.package,a.temperature)
    archive=None
    if a.temperature==600:
        with (a.package/"phase_check/MC_window_means.csv").open() as f:
            archive=np.array([(float(r["mean_c_Al"]),float(r["mean_delta_mu_eV"]))
                              for r in csv.DictReader(f) if r["model"]=="QHA" and float(r["T_K"])==600 and int(r["N"])==5120])
        c0=np.arange(15,275,5)/1000
        muvals=archive[:,1]+2*KB*600*100*(archive[:,0]-c0)
        assert np.ptp(muvals)<1e-9
        mu=muvals.mean()
        np.savetxt(folder/"author_window_means.csv",archive,delimiter=",",header="x_Al,mu_eV",comments="")
        if a.coarse:
            c0=c0[::2]
    else:
        mu=3.60
        c0=np.arange(130,381,10 if a.coarse else 5)/1000
    dest=folder/a.run
    dest.mkdir(exist_ok=False)
    (dest/"samples").mkdir()
    for f in ("classical.txt","quantum.txt"):
        shutil.copyfile(folder/f,dest/f)
    shutil.copyfile(PROJECT/"mc/static/motifs.txt",dest/"motifs.txt")
    nx,ny,nz=(8,8,6) if a.size==768 else (16,16,10)
    cfg=dict(size=a.size,T_K=a.temperature,shape=[nx,ny,nz],source=a.model,start=a.start,
             volume_step_A3_atom=.035,kappa=100,mu=float(mu),windows=c0.tolist(),
             eq=a.eq,prod=a.prod,stride=a.stride,seed=a.seed,
             coeff_interpolation="not-a-knot",grid=str(folder.name))
    (dest/"manifest.json").write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")
    with (dest/"windows.tsv").open("w",newline="") as f:
        w=csv.writer(f,delimiter="\t",lineterminator="\n")
        for i,c in enumerate(c0):
            w.writerow([a.model,a.temperature,nx,ny,nz,c,format(mu,".16g"),a.seed+i,a.eq,a.prod,
                        a.start,a.model+".txt",.035,"orbit-sums"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("package",type=Path)
    p.add_argument("temperature",type=float)
    p.add_argument("run")
    p.add_argument("--size",type=int,choices=(768,5120),required=True)
    p.add_argument("--model",choices=("classical","quantum"),default="classical")
    p.add_argument("--start",choices=("random","ordered"),required=True)
    p.add_argument("--eq",type=int,required=True)
    p.add_argument("--prod",type=int,required=True)
    p.add_argument("--stride",type=int,default=10)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--coarse",action="store_true")
    main(p.parse_args())
