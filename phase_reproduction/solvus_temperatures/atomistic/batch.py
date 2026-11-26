"""Prepare, submit and recover temperature-explicit MTP umbrella trajectories."""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = PROJECT / "solvus_1000"
sys.path.insert(0, str(PROJECT / "cluster"))
import umt_job as umt


def checkname(name):
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,32}", name):
        raise ValueError(name)
    return name


def make(a):
    name = checkname(a.name)
    target = HERE / "series" / name
    target.mkdir(parents=True, exist_ok=False)
    n = 2*a.nx*a.nx*a.nz
    strength = (5+.064*n)*a.temperature/1000 if a.strength is None else a.strength
    centers = [0.] if a.windows == 1 else [
        a.qmin + (a.qmax-a.qmin)*j/(a.windows-1) for j in range(a.windows)]
    with (target / "windows.csv").open("x", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream,lineterminator="\n")
        writer.writerow("run N shape cap mu_eV K_eV q0 seed start burn sweeps stride axial_free T_K".split())
        for h,mu in enumerate(a.fields):
            for j,center in enumerate(centers):
                run = f"{name}-h{h:02}q{j:02}"
                seed = int.from_bytes(hashlib.sha256(run.encode()).digest()[:4], "big")
                start = a.start[(h*len(centers)+j)%len(a.start)]
                writer.writerow([run,n,f"{a.nx}x{a.nx}x{a.nz}",a.cap,mu,round(strength,8),
                                 round(center,8),seed,start,a.burn,a.sweeps,a.stride,a.axial,a.temperature])


def submit(name, cpus, minutes):
    checkname(name)
    target = HERE / "series" / name
    files = [OLD/"job.sh", HERE/"run_windows.py", HERE/"sgmc.cpp", OLD/"engine.hpp",
             PROJECT/"inputs/fitted.mtp", target/"windows.csv"]
    with (target/"windows.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    files.extend(target/f for f in sorted({r["start"] for r in rows}-{"ordered","random","window"}))
    args=SimpleNamespace(run=name, cpus=min(cpus,len(rows)), minutes=minutes, memory=4, files=files)
    umt.submit(args, HERE/"runs"/name, umt.REMOTE+"/"+name)


def fetch(name):
    checkname(name)
    record=HERE/"runs"/name
    dest=HERE/"series"/name
    umt.fetch_raw(record,umt.REMOTE+"/"+name)
    with tarfile.open(record/"outputs/results.tar.gz") as arc:
        state=json.load(arc.extractfile("window_status.json"))
        if any(v["status"]!="ok" for v in state["windows"].values()):
            raise ValueError(state)
        if arc.extractfile("windows.csv").read()!=(dest/"windows.csv").read_bytes():
            raise ValueError("Manifest mismatch: "+name)
        for m in arc:
            if not m.isfile() or Path(m.name).name!=m.name:
                raise ValueError(m.name)
            if m.name=="window_status.json" or m.name.startswith(name+"-"):
                b=arc.extractfile(m).read()
                out=dest/m.name
                if out.exists() and out.read_bytes()!=b:
                    raise FileExistsError(out)
                if not out.exists():
                    out.write_bytes(b)
    print("verified",name,len(state["windows"]),flush=True)


def collect(names):
    jobs={}
    for name in names:
        j=json.loads((HERE/"runs"/checkname(name)/"submission.json").read_text())["job_id"]
        if not (HERE/"series"/name/"window_status.json").exists():
            jobs[j]=name
    failures={}
    while jobs:
        try:
            states=umt.ssh("sacct -nXP -j "+",".join(jobs)+" --format=JobID,State").decode()
        except subprocess.CalledProcessError as ex:
            if ex.returncode!=255: raise
            time.sleep(45)
            continue
        for line in states.splitlines():
            job, state=line.split("|")[:2]
            if job not in jobs or state.split()[0] in {"PENDING","RUNNING","COMPLETING","CONFIGURING","REQUEUED"}:
                continue
            name=jobs[job]
            if state!="COMPLETED":
                failures[name]=state
                print("failed",name,state,flush=True)
            else:
                try:
                    fetch(name)
                except subprocess.CalledProcessError as ex:
                    if ex.returncode!=255: raise
                    continue
            del jobs[job]
        if jobs: time.sleep(55)
    if failures: raise RuntimeError(failures)


def accounting():
    job_ids={json.loads(p.read_text())["job_id"]:p.parent.name for p in (HERE/"runs").glob("*/submission.json")}
    if not job_ids or not all(j.isdigit() for j in job_ids):
        raise ValueError("Expected project Slurm job IDs")
    fields=["JobID","State","ExitCode","Partition","NodeList","AllocCPUS","ElapsedRaw","CPUTimeRAW"]
    returned=umt.ssh("sacct -nXP -j "+",".join(job_ids)+" --format="+",".join(fields)).decode()
    records=[]
    for line in returned.splitlines():
        job=dict(zip(fields,line.rstrip("|").split("|"),strict=True))
        if job["State"]!="COMPLETED" or job["ExitCode"]!="0:0":
            raise RuntimeError(job)
        records.append(dict(series=job_ids[job["JobID"]],**job))
    if {j["JobID"] for j in records}!=set(job_ids) or len(records)!=len(job_ids):
        raise ValueError("Missing or duplicate Slurm jobs")
    with (HERE/"slurm_accounting.csv").open("w",encoding="ascii",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["series"]+fields)
        writer.writeheader()
        writer.writerows(sorted(records,key=lambda r:r["series"]))
    print(len(records),"Slurm jobs completed with zero exit codes")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest="action",required=True)
    c=sp.add_parser("make")
    c.add_argument("name")
    c.add_argument("temperature",type=float)
    c.add_argument("nx",type=int)
    c.add_argument("nz",type=int)
    c.add_argument("cap",type=float)
    c.add_argument("burn",type=int)
    c.add_argument("sweeps",type=int)
    c.add_argument("--fields",nargs="+",type=float,required=True)
    c.add_argument("--windows",type=int,default=10)
    c.add_argument("--qmin",type=float,default=.025)
    c.add_argument("--qmax",type=float,default=.985)
    c.add_argument("--strength",type=float)
    c.add_argument("--start",nargs="+",default=["window"])
    c.add_argument("--stride",type=int,default=1)
    c.add_argument("--axial",type=int,choices=[0,1],default=1)
    s=sp.add_parser("submit")
    s.add_argument("names",nargs="+")
    s.add_argument("--cpus",type=int,default=16)
    s.add_argument("--minutes",type=int,default=60)
    f=sp.add_parser("fetch")
    f.add_argument("names",nargs="+")
    s=sp.add_parser("status")
    s.add_argument("names",nargs="+")
    s=sp.add_parser("progress")
    s.add_argument("name")
    s=sp.add_parser("collect")
    s.add_argument("names",nargs="+")
    sp.add_parser("accounting")
    a=p.parse_args()
    if a.action=="make":
        make(a)
    elif a.action=="submit":
        for name in a.names:
            submit(name,a.cpus,a.minutes)
    elif a.action=="fetch":
        for name in a.names:
            fetch(name)
    elif a.action=="collect":
        collect(a.names)
    elif a.action=="accounting":
        accounting()
    elif a.action=="progress":
        name=checkname(a.name)
        print(umt.ssh(f"cd {umt.REMOTE}/{name} && wc -l {name}-*.csv && cat window_status.json").decode())
    else:
        ids=[json.loads((HERE/"runs"/checkname(name)/"submission.json").read_text())["job_id"] for name in a.names]
        print(umt.ssh("sacct -nXP -j "+",".join(ids)+" --format=JobID,JobName%32,State,Elapsed,NodeList").decode())
