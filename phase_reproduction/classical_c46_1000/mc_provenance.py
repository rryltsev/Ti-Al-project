"""Validate new trajectories against analytic MC tests, original C46 spline, and saved logs."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from scipy.interpolate import CubicSpline,PPoly

HERE=Path(__file__).resolve().parent


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def original_spline(table,t):
    with table.open() as f:
        assert f.readline().strip()=="QHA_CE_V1"
        nv,nt,nc=map(int,f.readline().split())
        assert abs(float(f.readline())-8.617333262145e-5)<1e-18
        v=np.fromstring(f.readline(),sep=" ")
        temperatures=np.fromstring(f.readline(),sep=" ")
        rows=np.loadtxt(f).reshape(nt,nv-1,nc,8)
    it,=np.flatnonzero(temperatures==t)
    return PPoly(rows[it,:,:,:4][:,:,::-1].transpose(2,0,1),v,extrapolate=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs",nargs="+",type=Path)
    p.add_argument("--exe",required=True,type=Path)
    p.add_argument("--author-package",required=True,type=Path)
    p.add_argument("--out",default=HERE/"mc_validation.json",type=Path)
    a=p.parse_args()
    reference=a.author_package/"models/HA_QHA/qha_ce.dat"
    field=original_spline(reference,1000)
    data=np.load(HERE/"quantum_grid.npz")
    q=CubicSpline(data["volume_A3_atom"],data["J_static"]+data["J_vib"][:,0,:],axis=0)
    mapfile=HERE.parent/"author_comparison/cells/orbit_mapping.csv"
    with mapfile.open(newline="") as f:
        mapping=list(csv.DictReader(f))
    idx=[int(s["author_column_index"]) for s in sorted(mapping,key=lambda s:int(s["working_S1_index"]))]
    probes=np.linspace(field.x[0],field.x[-1],129)
    qe=float(np.abs(field(probes)[:,idx]-q(probes)).max())
    if qe>1e-10: raise ValueError(f"Quantum grid and author cubic polynomials disagree by {qe}")
    cdata=np.load(HERE/"classical_grid.npz")
    c=CubicSpline(cdata["volume_A3_atom"],cdata["J_static"]+cdata["J_vib"][:,0,:],axis=0)
    ideal=HERE.parent/"mc/check_sampler.py"
    refdata=subprocess.run([sys.executable,str(ideal),str(a.exe.resolve())],check=True,capture_output=True,text=True).stdout
    m=re.search(r"nAl exact ([\d.]+), sampled ([\d.]+)\s+v exact ([\d.]+), sampled ([\d.]+)",refdata)
    expected_n,observed_n,expected_v,observed_v=map(float,m.groups())
    out=dict(engine_source_sha256=sha(HERE.parent/"mc/vcs.cpp"),binary_sha256=sha(a.exe),
             compiler="Microsoft Visual C++ 2017 x64, /O2 /EHsc /std:c++14",binary_name=a.exe.name,
             preparation_sha256=sha(HERE/"prepare_mc.py"),local_driver_sha256=sha(HERE/"run_local.py"),
             paired_analysis_sha256=sha(HERE/"paired_mc.py"),author_original_qha_ce_dat_sha256=sha(reference),
             quantum_grid_sha256=sha(HERE/"quantum_grid.npz"),classical_grid_sha256=sha(HERE/"classical_grid.npz"),
             quantum_spline_comparison=dict(temperatures_K=[1000],volumes=129,orbit_count=20,
                                            max_abs_difference_eV_cluster=qe),
             exact_96_site_test=dict(driver_sha256=sha(ideal),n_Al_exact=expected_n,n_Al_measured=observed_n,
                                     v_exact_A3_atom=expected_v,v_measured_A3_atom=observed_v,
                                     raw_output=refdata.strip()),runs={})
    for run in a.runs:
        cfg=json.loads((run/"manifest.json").read_text())
        logs=list(sorted((run/"samples").glob("*.log"))); files=list(sorted((run/"samples").glob("*.csv.gz")))
        if len(logs)!=len(cfg["windows"]) or len(files)!=len(logs): raise ValueError(f"Incomplete {run}")
        drift=[];fields=[];frames=[];max_de=[];ess=[];maxw=[];orders=[];volumes=[];h={}
        for i in range(1,len(files)+1):
            log=run/"samples"/f"{i}.log"; fp=run/"samples"/f"{i}.csv.gz"
            s=re.fullmatch(r"samples (\d+) accepted \d+ accepted_volume \d+ drift_eV (\S+) max_field_error_eV (\S+)",log.read_text().strip())
            frames.append(int(s[1])); drift.append(abs(float(s[2]))); fields.append(abs(float(s[3])))
            with gzip.open(fp,"rt",newline="") as f:
                rr=csv.DictReader(f); orbit=rr.fieldnames[12:]; rows=list(rr)
            assert len(rows)==frames[-1]==cfg["prod"]//cfg["stride"]
            arr=np.array([[float(r[n]) for n in ["n_Al","m","volume_A3_atom","Heff_eV_atom",*orbit]] for r in rows])
            k,order,v,saved_E=arr[:,:4].T
            n=cfg["size"]
            totals=np.column_stack((np.full(len(v),n/2),2*k-n,arr[:,4:]))
            jq,jc=field(v)[:,idx],c(v)
            e=np.einsum("ij,ij->i",totals,jq if cfg["source"]=="quantum" else jc)/n
            max_de.append(float(np.max(np.abs(e-saved_E))))
            d=np.einsum("ij,ij->i",totals,jc-jq)/.08617333262145
            lw=d if cfg["source"]=="classical" else -d
            w=np.exp(lw-lw.max());w/=w.sum()
            maxw.append(float(w.max()));ess.append(float(1/(len(w)*sum(w*w))))
            orders.append(float(order.mean())); volumes.extend([float(v.min()),float(v.max())])
            h[str(fp.relative_to(run)).replace("\\","/")]=sha(fp)
        if max(max_de)>2e-9: raise ValueError(f"{run}: stored vs recalculated energy")
        out["runs"][run.name]=dict(source=cfg["source"],initialization=cfg["start"],sites=n,
              seed_first=cfg["seed"],seed_last=cfg["seed"]+len(files)-1,
              equilibration_sweeps=cfg["eq"],production_sweeps=cfg["prod"],stride=cfg["stride"],
              windows=len(files),retained_frames=sum(frames),per_window=frames,largest_final_drift_eV_cell=max(drift),
              largest_local_field_error_eV=max(fields),largest_frame_energy_difference_eV_atom=max(max_de),
              minimum_Kish_fraction_target_per_window=min(ess),largest_normalized_weight=max(maxw),
              volume_range_A3_atom=[min(volumes),max(volumes)],mean_order_first_last=[orders[0],orders[-1]],
              input_and_manifest_sha256={name:sha(run/name) for name in
              ("vcs.cpp","quantum.txt","classical.txt","motifs.txt","manifest.json","windows.tsv")},
              trajectory_sha256=h)
    a.out.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("verified",len(out["runs"]),"runs,",sum(r["retained_frames"] for r in out["runs"].values()),"frames; spline error",qe)


if __name__=="__main__":
    main()
