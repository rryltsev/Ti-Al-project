"""Return a consistent immutable CSV prefix from active trajectories on UMT."""
import csv
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

target=int(sys.argv[1])
with open("windows.csv",newline="") as stream:
    rows=list(csv.DictReader(stream))
files={}
hashes={}
for row in rows:
    if not int(row["burn"])<target<=int(row["burn"])+int(row["sweeps"]) or int(row["stride"])!=1:
        raise ValueError("The diagnostic prefix must include a positive number of production steps")
    path=Path(row["run"]+".csv")
    if not path.is_file():
        print("WAIT missing",path)
        sys.exit(0)
    with path.open("rb") as f:
        included=[]
        for i in range(target+1):
            line=f.readline()
            if not line.endswith(b"\n"):
                print("WAIT",path,"flushed through",max(0,i-1),"of",target)
                sys.exit(0)
            included.append(line)
    if int(included[-1].split(b",",1)[0])!=target:
        raise RuntimeError("Unexpected recorded timestep")
    files[path.name]=b"".join(included)
    hashes[path.name]=hashlib.sha256(files[path.name]).hexdigest()
    row["sweeps"]=target-int(row["burn"])
manifest=io.StringIO()
writer=csv.DictWriter(manifest,fieldnames=list(rows[0]),lineterminator="\n")
writer.writeheader();writer.writerows(rows)
files["prefix.json"]=(json.dumps(dict(through_sweep=target,source="in-progress direct Slurm trajectories",sha256=hashes),indent=2)+"\n").encode()
files["windows.csv"]=manifest.getvalue().encode()
with tarfile.open(mode="w|gz",fileobj=sys.stdout.buffer) as ar:
    for name,data in files.items():
        ti=tarfile.TarInfo(name)
        ti.size=len(data)
        ar.addfile(ti,io.BytesIO(data))
