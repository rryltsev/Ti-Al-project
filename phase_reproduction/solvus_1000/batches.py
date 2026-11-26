"""Submit, inspect and retrieve the solvus umbrella series on UMT u3129."""
import argparse
import csv
import hashlib
import io
import json
import re
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
CLUSTER = HERE.parent / "cluster"
sys.path.insert(0, str(CLUSTER))
import umt_job


def series_path(name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,44}", name):
        raise ValueError(name)
    return HERE / "series" / name


def submit(names, minutes, dependency):
    for name in names:
        series = series_path(name)
        manifest = series/"windows.csv"
        with manifest.open() as stream:
            rows = list(csv.DictReader(stream))
        cpus = min(16, len(rows))
        files = [HERE/f for f in ("job.sh", "run_windows.py", "sgmc.cpp", "engine.hpp")]
        files += [HERE.parent/"inputs"/"fitted.mtp", manifest]
        restarts = {row["start"] for row in rows} - {"random", "ordered", "window"}
        files.extend(series/path for path in sorted(restarts))
        cmd = [sys.executable, "-B", str(CLUSTER/"umt_job.py"), "submit", "--cpus", str(cpus),
               "--minutes", str(minutes), "--memory", "4"]
        if dependency:
            cmd.extend(["--dependency",dependency])
        subprocess.run([*cmd,name,*map(str,files)],check=True)


def inspect(names):
    ids = []
    directories = []
    for name in names:
        series_path(name)
        record = CLUSTER/"runs"/name/"submission.json"
        if record.is_file():
            ids.append(json.loads(record.read_text())["job_id"])
            directories.append(umt_job.REMOTE+"/"+name)
    if not ids:
        return
    lines = [f"sacct -nXP -j {','.join(ids)} --format=JobID,JobName%32,State,Elapsed,NodeList"]
    for d in directories:
        # Only completed checkpoint intervals are visible before frames are flushed.
        lines.append(f"if test -d {shlex.quote(d)}; then cd {shlex.quote(d)} && "
                     "f=$(find . -name '*-q00.csv' -print -quit); "
                     'if test -n "$f"; then echo "$f $(tail -1 "$f" | cut -d, -f1)"; fi; fi')
    print(umt_job.ssh("; ".join(lines)).decode(), end="")


def retrieve(names):
    for name in names:
        series = series_path(name)
        record = CLUSTER/"runs"/name
        if not (record/"outputs"/"results.tar.gz").exists():
            for attempt in range(4):
                try:
                    umt_job.fetch_raw(record,umt_job.REMOTE+"/"+name)
                    break
                except subprocess.CalledProcessError as error:
                    if error.returncode != 255 or attempt == 3:
                        raise
                    time.sleep(3*(attempt+1))
        with tarfile.open(record/"outputs"/"results.tar.gz","r:gz") as archive:
            for item in archive:
                if not item.isfile():
                    continue
                # Only known basenames from this job can enter the series directory.
                fname = Path(item.name).name
                if not (fname.startswith(name+"-q") or fname in {"windows.csv","window_status.json","manifest.csv"}):
                    continue
                data = archive.extractfile(item).read()
                target = series/fname
                if target.is_file() and target.read_bytes() != data:
                    raise FileExistsError(target)
                target.write_bytes(data)


def snapshot(names, destination):
    """Capture flushed, complete rows for diagnostics while a job is running."""
    code = """import io,os,sys,tarfile
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for job in sys.argv[1:]:
        for name in os.listdir(job):
            if not name.endswith('.csv'): continue
            with open(os.path.join(job,name),'rb') as src:
                data = src.read()
            data=data[:data.rfind(b'\\n')+1]
            entry=tarfile.TarInfo(job+'/'+name)
            entry.size=len(data)
            archive.addfile(entry,io.BytesIO(data))
"""
    for name in names:
        series_path(name)
    payload=umt_job.ssh("cd "+shlex.quote(umt_job.REMOTE)+" && python3 -c "+
                        shlex.quote(code)+" "+shlex.join(names))
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for item in archive:
            job,filename=item.name.split("/")
            if job not in names or not re.fullmatch(r"[a-zA-Z0-9_.-]+\.csv",filename):
                raise ValueError(item.name)
            target=destination/job/filename
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.extractfile(item).read())


def collect(names):
    """Retrieve every completed batch and surface scheduler/worker failures."""
    jobs = {}
    for name in names:
        series_path(name)
        job = json.loads((CLUSTER/"runs"/name/"submission.json").read_text())["job_id"]
        jobs[job] = name
    while jobs:
        ids = ",".join(jobs)
        try:
            raw = umt_job.ssh(f"sacct -nXP -j {ids} --format=JobID,State")
        except subprocess.CalledProcessError as error:
            if error.returncode != 255:
                raise
            time.sleep(20)
            continue
        for line in raw.decode().splitlines():
            job,state = line.strip().split("|")[:2]
            if job not in jobs:
                continue
            if state in {"PENDING","RUNNING","COMPLETING","CONFIGURING","REQUEUED"}:
                continue
            name = jobs[job]
            if state != "COMPLETED":
                raise RuntimeError(f"{name} job {job} ended in {state}")
            retrieve([name])
            workers=json.loads((series_path(name)/"window_status.json").read_text())["windows"]
            if not workers or any(w["status"]!="ok" for w in workers.values()):
                raise RuntimeError("Incomplete series "+name)
            del jobs[job]
            print(f"{name}: verified {len(workers)} windows", flush=True)
        if jobs:
            time.sleep(40)


def audit(names, output):
    batches=[]
    expected_model=hashlib.sha256((HERE.parent/"inputs"/"fitted.mtp").read_bytes()).hexdigest()
    for name in names:
        path=series_path(name)
        record=CLUSTER/"runs"/name
        intent=json.loads((record/"intent.json").read_text())
        job=json.loads((record/"submission.json").read_text())["job_id"]
        counts=json.loads((path/"window_status.json").read_text())["windows"]
        with (path/"windows.csv").open() as f:
            rows=list(csv.DictReader(f))
        if {r["run"] for r in rows}!=set(counts):
            raise ValueError("Worker manifest mismatch "+name)
        if intent["user"]!="u3129" or intent["sha256"]["fitted.mtp"]!=expected_model:
            raise ValueError("Account or potential mismatch "+name)
        digests=intent["sha256"]
        for filename in ["engine.hpp","sgmc.cpp","run_windows.py","job.sh","windows.csv"]:
            source=(path if filename=="windows.csv" else HERE)/filename
            if hashlib.sha256(source.read_bytes()).hexdigest()!=digests[filename]:
                raise ValueError("Source differs from staged "+name+"/"+filename)
        for row in rows:
            case=counts[row["run"]]
            count=(int(row["burn"])+int(row["sweeps"]))//int(row["stride"])
            if case["status"]!="ok" or case["exit_code"]!=0 or case["frames"]!=count:
                raise ValueError("Incomplete window "+row["run"])
        archive=record/"outputs"/"results.tar.gz"
        actual=hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual!=(archive.parent/"raw.sha256").read_text().split()[0]:
            raise ValueError("Archive hash mismatch "+name)
        batches.append(dict(series=name,job_id=job,windows=len(rows),N=int(rows[0]["N"]),
            cap=float(rows[0]["cap"]),axial_free=int(rows[0]["axial_free"]),
            umbrella=any(float(r["K_eV"]) for r in rows),
            production_sweeps=sum(int(r["sweeps"]) for r in rows),results_sha256=actual))
    data=dict(account="u3129",model_sha256=expected_model,temperature_K=1000,pressure_GPa=0,
              batches=batches,completed_chains=sum(b["windows"] for b in batches),
              production_sweeps=sum(b["production_sweeps"] for b in batches))
    output.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["submit","status","fetch","snapshot","collect","audit"])
    p.add_argument("series",nargs="+")
    p.add_argument("--minutes",type=int,default=180)
    p.add_argument("--dependency")
    p.add_argument("--snapshot-root",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.action=="submit":
        submit(a.series,a.minutes,a.dependency)
    elif a.action=="status":
        inspect(a.series)
    elif a.action=="snapshot":
        if a.snapshot_root is None:
            p.error("--snapshot-root is required")
        snapshot(a.series,a.snapshot_root)
    elif a.action=="collect":
        collect(a.series)
    elif a.action=="audit":
        if a.output is None:
            p.error("--output is required for audit")
        audit(a.series,a.output)
    else:
        retrieve(a.series)


if __name__=="__main__":
    main()
