"""Verify staged hashes, completed trajectories and thermodynamic conventions.

Run batch.py accounting after retrieving the last group and before the final audit.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main(include_partial):
    source={"fitted.mtp":PROJECT/"inputs/fitted.mtp","sgmc.cpp":HERE/"sgmc.cpp","run_windows.py":HERE/"run_windows.py",
            "engine.hpp":PROJECT/"solvus_1000/engine.hpp","job.sh":PROJECT/"solvus_1000/job.sh"}
    hashes={name:digest(path) for name,path in source.items()}
    accounting={}
    if (HERE/"slurm_accounting.csv").exists() or not include_partial:
        with (HERE/"slurm_accounting.csv").open() as stream:
            accounting={r["series"]:r for r in csv.DictReader(stream)}
    result=[]
    for job in sorted((HERE/"runs").iterdir()):
        series=HERE/"series"/job.name
        statusfile=series/"window_status.json"
        if not statusfile.exists() and include_partial:
            continue
        intent=json.loads((job/"intent.json").read_text())
        sub=json.loads((job/"submission.json").read_text())
        status=json.loads(statusfile.read_text())["windows"]
        assert intent["user"]=="u3129"
        assert all(intent["sha256"][key]==expected for key,expected in hashes.items())
        assert digest(series/"windows.csv")==intent["sha256"]["windows.csv"]
        rawsha=digest(job/"outputs/results.tar.gz")
        assert rawsha==(job/"outputs/raw.sha256").read_text().split()[0]
        with (series/"windows.csv").open() as f:
            windows=list(csv.DictReader(f))
        assert {s["run"] for s in windows}==set(status)
        t,n,cap,ax=(windows[0][key] for key in ["T_K","N","cap","axial_free"])
        mus=sorted({float(r["mu_eV"]) for r in windows})
        springs=sorted({float(r["K_eV"]) for r in windows})
        restraint="unbiased" if max(springs)==0 else "harmonic_Q_umbrella"
        prod=burn=0
        total_time=0.
        volume_min,volume_max,axial_min,axial_max,guard_min=math.inf,-math.inf,math.inf,-math.inf,math.inf
        for r in windows:
            assert [r[k] for k in ["T_K","N","cap","axial_free"]]==[t,n,cap,ax]
            run=r["run"]
            frames=0
            with (series/(run+".csv")).open() as f:
                for v in csv.DictReader(f):
                    frames+=1
                    assert float(v["T_K"])==float(t)
                    assert float(v["wall"])<float(cap)
                    assert sum(int(v["n"+str(i)]) for i in range(4))==int(v["n_Al"])
                    assert abs(4*(4*max(int(v["n"+str(i)]) for i in range(4))-int(v["n_Al"]))/(3*int(n))-float(v["Q"]))<1e-9
                    assert int(v["sweep"])==frames*int(r["stride"])
                    assert float(ax) or abs(float(v["c_a"])-4.6046/2.9446)<1.e-10
                    if int(v["sweep"])>int(r["burn"]):
                        volume,axial=float(v["volume_A3_atom"]),float(v["c_a"])
                        basal=(4*volume/(math.sqrt(3)*axial))**(1/3)
                        nearest=basal*min(1.,math.sqrt(1./3+axial**2/4))
                        clearance=(1-float(cap))*nearest-1.68052
                        assert 15 <= volume <= 21 and 1.45 <= axial <= 1.72 and clearance >= -1e-9
                        volume_min=min(volume,volume_min)
                        volume_max=max(volume,volume_max)
                        axial_min=min(axial,axial_min)
                        axial_max=max(axial,axial_max)
                        guard_min=min(clearance,guard_min)
            assert status[run]["status"]=="ok" and status[run]["exit_code"]==0
            assert frames==status[run]["frames"]==(int(r["burn"])+int(r["sweeps"]))//int(r["stride"])
            warm=int(r["burn"])//int(r["stride"])
            burn+=warm
            prod+=frames-warm
            total_time+=status[run]["elapsed_seconds"]
        slurm=accounting.get(job.name,{})
        if accounting:
            assert slurm["State"]=="COMPLETED" and slurm["ExitCode"]=="0:0" and slurm["JobID"]==sub["job_id"]
        result.append(dict(series=job.name,job_id=sub["job_id"],slurm_state=slurm.get("State"),
                    slurm_exit=slurm.get("ExitCode"),allocated_cpu_hours=int(slurm.get("CPUTimeRAW",0))/3600,
                    T_K=float(t),N=int(n),W_max=float(cap),
                    ensemble="site_confined_semigrand_NPT",pressure_GPa=0,restraint=restraint,
                    mu_cfg_min_eV=min(mus),mu_cfg_max_eV=max(mus),n_mu=len(mus),spring_constants_eV=springs,
                    starts="/".join(sorted({r["start"] for r in windows})),
                    axial_free=bool(int(ax)),windows=len(windows),thermalization_records=burn,
                    production_records=prod,window_wall_hours=total_time/3600,manifest_sha256=intent["sha256"]["windows.csv"],
                    V_A3_atom_min=volume_min,V_A3_atom_max=volume_max,axial_ratio_min=axial_min,axial_ratio_max=axial_max,
                    geometric_lower_distance_guard_clearance_A_min=guard_min,
                    results_sha256=rawsha,submitted_utc=sub["utc"],status="all_windows_ok_and_archive_sha256_verified"))
    data=dict(account="u3129",host="umt.imm.uran.ru",pressure_GPa=0,
              fields="configurational mu_Al - mu_Ti in eV per replacement; K in eV per cell",
              ensemble="flat dV and d(log(c/a)) when axial freedom is enabled, V^N Jacobian",
              source_sha256=hashes,verified_series=result,windows=sum(r["windows"] for r in result),
              production_records=sum(r["production_records"] for r in result))
    (HERE/"provenance.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    with (HERE/"series_summary.csv").open("w",encoding="utf-8",newline="") as stream:
        fields=("series","job_id","slurm_state","slurm_exit","ensemble","pressure_GPa","T_K","N","W_max","axial_free","restraint","n_mu",
                "mu_cfg_min_eV","mu_cfg_max_eV","starts","windows","thermalization_records",
                "production_records","window_wall_hours","allocated_cpu_hours","status","results_sha256")
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result)
    print(len(result),"completed series,",data["windows"],"chains,",data["production_records"],"production frames")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include-finished-only",action="store_true")
    args=p.parse_args()
    main(args.include_finished_only)
