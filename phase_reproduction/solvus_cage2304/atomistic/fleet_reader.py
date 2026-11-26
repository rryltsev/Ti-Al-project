"""UMT side of a framed, read-only harvest of completed Slurm results."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

series={}
for s in sys.argv[1:]:
    name,job=s.split("=")
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,32}",name) or not job.isdigit():
        raise ValueError(s)
    series[job]=name
states=subprocess.check_output(["sacct","-nXP","-j",",".join(series),"--format=JobID,State,ExitCode"]).decode()
done=[]
for line in states.splitlines():
    job,state,exitcode=line.rstrip("|").split("|")
    if state=="COMPLETED" and exitcode=="0:0":
        done.append(series[job])
    elif state.split()[0] not in {"PENDING","RUNNING","COMPLETING","CONFIGURING","REQUEUED"}:
        raise RuntimeError((job,state,exitcode))
out=sys.stdout.buffer
def header(obj):
    out.write((json.dumps(obj)+"\n").encode())
    out.flush()
header({"protocol":"umt-raw-1","available":len(done)})
for name in done[:6]:
    p=Path(name)/"results.tar.gz"
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    header({"series":name,"size":p.stat().st_size,"sha256":h.hexdigest()})
    with p.open("rb") as f:
        shutil.copyfileobj(f,out,1024*1024)
    out.flush()
header({"end":True})
