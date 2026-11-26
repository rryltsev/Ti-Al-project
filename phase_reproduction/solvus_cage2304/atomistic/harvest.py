"""Collect multiple Slurm archives per SSH connection and verify each one independently."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

from cage_runs import HERE, batch, workflow, check_prefixes, retry_connection


def read_header(stream):
    line=stream.readline()
    if not line.endswith(b"\n"):
        raise ConnectionError("Incomplete UMT response")
    return json.loads(line)


def receive(names):
    args=[]
    for name in names:
        path=HERE/"runs"/batch.checkname(name)/"submission.json"
        if path.is_file():
            job=json.loads(path.read_text())["job_id"]
            if not job.isdigit():
                raise ValueError(job)
            args.append(f"{name}={job}")
    if not args:
        return 0
    cmd="cd "+batch.umt.REMOTE+" && python3 - "+" ".join(args)
    with subprocess.Popen(batch.umt.ssh_args()+[cmd],stdin=subprocess.PIPE,stdout=subprocess.PIPE) as proc:
        try:
            proc.stdin.write((HERE/"fleet_reader.py").read_bytes())
            proc.stdin.close()
            hello=read_header(proc.stdout)
            if hello.get("protocol")!="umt-raw-1":
                raise RuntimeError(hello)
            while True:
                hdr=read_header(proc.stdout)
                if hdr.get("end"): break
                name=hdr["series"]
                if name not in names or not 0<hdr["size"]<200*1024*1024:
                    raise RuntimeError(hdr)
                data=proc.stdout.read(hdr["size"])
                if len(data)!=hdr["size"]:
                    raise ConnectionError("Interrupted archive transfer: "+name)
                if hashlib.sha256(data).hexdigest()!=hdr["sha256"]:
                    raise ValueError("Archive checksum mismatch: "+name)
                dest=HERE/"runs"/name/"outputs"
                dest.mkdir(exist_ok=True)
                archive=dest/"results.tar.gz"
                if archive.exists() and workflow.provenance.digest(archive)!=hdr["sha256"]:
                    raise FileExistsError(archive)
                if not archive.exists():
                    tmp=dest/"results.fleet.part"
                    tmp.write_bytes(data)
                    tmp.replace(archive)
                (dest/"raw.sha256").write_text(hdr["sha256"]+"  results.tar.gz\n",encoding="ascii")
                if not workflow.restore_local(name):
                    raise RuntimeError(name)
                print("verified",name,hdr["size"],flush=True)
        except (ConnectionError,BrokenPipeError):
            proc.stdout.close()
            proc.wait()
            if proc.returncode not in (0,255,-1,4294967295):
                raise RuntimeError("Remote Slurm or integrity check failed (exit %d)"%proc.returncode)
            raise ConnectionError("UMT transport interrupted")
        if proc.wait()!=0:
            raise ConnectionError("Transport failed after result stream")
    return hello["available"]


def main():
    while True:
        groups=[p.parent.name for p in sorted((HERE/"series").glob("*/windows.csv"))]
        missing=[n for n in groups if not (HERE/"series"/n/"window_status.json").is_file()
                 or not (HERE/"runs"/n/"outputs/raw.sha256").is_file()]
        if not missing:
            retry_connection(workflow.resume)
            check_prefixes()
            return
        try:
            ready=receive(missing)
            if ready<6: time.sleep(50)
        except ConnectionError as e:
            print(e,flush=True)
            time.sleep(15)


if __name__=="__main__":
    main()
