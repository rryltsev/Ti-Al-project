"""Prepare and verify direct N2304 isotherms at both displacement limits."""
import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/"solvus_closure"/"atomistic"))
import closure_runs as workflow

workflow.HERE=HERE
workflow.batch.HERE=HERE
workflow.provenance.HERE=HERE
workflow.batch.umt.REMOTE="/home6/u3129/tial-cage2304-20261126"
batch=workflow.batch
KEYS="run N shape cap mu_eV K_eV q0 seed start burn sweeps stride axial_free T_K".split()


def make(name,t,cap,windows):
    directory=HERE/"series"/batch.checkname(name)
    directory.mkdir(parents=True,exist_ok=False)
    with (directory/"windows.csv").open("x",encoding="ascii",newline="") as f:
        writer=csv.writer(f,lineterminator="\n")
        writer.writerow(KEYS)
        for seq,(mu,q,spring,start,burn,prod) in enumerate(windows):
            run=f"{name}-q{seq:02d}"
            seed=int.from_bytes(hashlib.sha256(run.encode()).digest()[:4],"big")
            writer.writerow([run,2304,"12x12x8",cap,mu,round(spring,8),round(q,8),seed,start,burn,prod,1,1,t])


def grid(main):
    return [3.510,3.522,3.533,3.545,3.557,3.568,3.580] if main else [3.510,3.5275,3.545,3.5625,3.580]


def umbrella(name,t,cap,mu,dense,central):
    nq=14 if dense or central else 12
    burn,prod=(625,1100) if dense or central else (475,900)
    spring=(5+.064*2304)*t/1000
    centers=[.025+.84*j/(nq-1) for j in range(nq)]
    windows=[(mu,q,spring,"window",burn,prod) for q in centers]
    if dense or central:
        windows.extend([(mu,centers[2],spring,"ordered",burn,prod),
                        (mu,centers[-4],spring,"random",burn,prod)])
    make(name,t,cap,windows)


def prepare():
    workflow.prepare_inputs()
    for t in [1243,1242,1244,1245]:
        for cap in [.35,.40]:
            tag=f"cg{int(round(cap*100))}{int(t)%100:02d}"
            fields=grid(t==1243)
            for j,mu in enumerate(fields):
                umbrella(tag+chr(ord("a")+j),t,cap,mu,t==1243,mu==3.545)
            unbiased=[(mu,0.,0.,start,625,1150) for mu in fields[::1 if t==1243 else 2]
                      for start in ("ordered","random")]
            make(tag+"u",t,cap,unbiased)


def first_extension():
    for j,mu in enumerate((3.458,3.482,3.600,3.624)):
        umbrella("cg35x43"+chr(ord("a")+j),1243,.35,mu,True,False)
    for t in (1254,1262):
        temperature_triad(t,.35)


def temperature_triad(t,cap):
    for j,mu in enumerate((3.520,3.545,3.570)):
        umbrella(f"cg{int(round(cap*100))}{t%100:02d}"+chr(ord("a")+j),t,cap,mu,False,mu==3.545)


def repeat_1243():
    t=1243
    spring=(5+.064*2304)*t/1000
    centers=[.02+.84*j/13 for j in range(14)]
    for j,mu in enumerate((3.530,3.545,3.560)):
        primary=[(mu,q,spring,("random","window","ordered","window")[i%4],700,1800)
                 for i,q in enumerate(centers)]
        primary.extend([(mu,centers[2],spring,"random",700,1800),
                        (mu,centers[-4],spring,"random",700,1800)])
        make("cg40r43"+chr(ord("a")+j),t,.40,primary)


def priority():
    groups=[]
    for man in sorted((HERE/"series").glob("*/windows.csv")):
        with man.open() as f: rows=list(csv.DictReader(f))
        mu=float(rows[0]["mu_eV"])
        t=float(rows[0]["T_K"])
        cap=float(rows[0]["cap"])
        name=man.parent.name
        # Establish absolute order weights everywhere first, followed by flanks close to 1243 K.
        if len({float(r["mu_eV"]) for r in rows})==1 and abs(mu-3.545)<1e-7:
            rank=0
        elif t==1243 and min(abs(mu-3.533),abs(mu-3.557))<1e-7:
            rank=1
        elif t==1243:
            rank=2
        elif rows[0]["K_eV"]=="0.0":
            rank=5
        else:
            rank=3
        groups.append((rank,abs(t-1243),mu,cap,name))
    return [r[-1] for r in sorted(groups)]


def doctor():
    workflow.prepare_inputs()
    print(batch.umt.ssh("set -eu; test \"$(id -un)\" = u3129; id -un; hostname; command -v sbatch; "
        "command -v g++; test -x /home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python; "
        "sinfo -h -p apollo,nompi,debug -o '%P|%a|%D|%c'").decode())


def recover(name):
    record=HERE/"runs"/batch.checkname(name)
    intent=json.loads((record/"intent.json").read_text())
    remote=intent["remote"]
    if remote!=batch.umt.REMOTE+"/"+name or intent["user"]!="u3129":
        raise ValueError("Unexpected recovery directory")
    staged=batch.umt.ssh(f"if test -s {remote}/inputs.sha256 && (cd {remote} && sha256sum --status -c inputs.sha256); "
                         "then echo staged; else echo incomplete; fi").decode().strip()
    if staged!="staged":
        paths=[batch.OLD/"job.sh",HERE/"run_windows.py",HERE/"sgmc.cpp",batch.OLD/"engine.hpp",
               PROJECT/"inputs/fitted.mtp",HERE/"series"/name/"windows.csv"]
        files={p.name:p.read_bytes() for p in paths}
        hashes={n:hashlib.sha256(data).hexdigest() for n,data in files.items()}
        if hashes!=intent["sha256"]:
            raise ValueError("Local files changed after submission intent")
        files["inputs.sha256"]="".join(f"{h}  {n}\n" for n,h in hashes.items()).encode()
        archive=io.BytesIO()
        with tarfile.open(fileobj=archive,mode="w:gz") as tar:
            for n,data in files.items():
                ti=tarfile.TarInfo(n);ti.size=len(data)
                tar.addfile(ti,io.BytesIO(data))
        batch.umt.ssh(f"set -eu; test \"$(id -un)\" = u3129; umask 077; mkdir -p {remote}; cd {remote}; "
                      "test ! -e job.id; tar -xzf -; sha256sum --status -c inputs.sha256",archive.getvalue())
    workflow.recover_submission(name)


def submit(names,minutes):
    for name in names:
        while not (HERE/"runs"/name/"submission.json").is_file():
            try:
                if (HERE/"runs"/name/"intent.json").exists():
                    recover(name)
                else:
                    batch.submit(name,16,minutes)
            except subprocess.CalledProcessError as error:
                if error.returncode not in (255,-1,4294967295):
                    raise
                time.sleep(12)


def retry_connection(operation):
    while True:
        try:
            return operation()
        except subprocess.CalledProcessError as error:
            if error.returncode not in (255,-1,4294967295):
                raise
            time.sleep(15)


def watch():
    """Collect completed immutable archives while later submissions are still queued."""
    while True:
        folders=sorted((HERE/"series").glob("*/windows.csv"))
        names=[p.parent.name for p in folders]
        outstanding=[name for name in names if not (HERE/"series"/name/"window_status.json").is_file()]
        if not outstanding:
            retry_connection(workflow.resume)
            check_prefixes()
            return
        ids={json.loads(p.read_text())["job_id"]:name for name in outstanding
             if (p:=HERE/"runs"/name/"submission.json").is_file()}
        try:
            if ids:
                states=batch.umt.ssh("sacct -nXP -j "+",".join(ids)+" --format=JobID,State").decode()
                for line in states.splitlines():
                    job,state=line.rstrip("|").split("|")[:2]
                    if state=="COMPLETED":
                        batch.fetch(ids[job])
                    elif state.split()[0] not in {"PENDING","RUNNING","COMPLETING","CONFIGURING","REQUEUED"}:
                        raise RuntimeError((ids[job],job,state))
        except subprocess.CalledProcessError as error:
            if error.returncode not in (255,-1,4294967295):
                raise
        time.sleep(55)


def preview(name,sweeps):
    """Diagnostic prefix of every chain; does not alter the Slurm job or its archive."""
    name=batch.checkname(name)
    directory=HERE/"prefixes"/name/str(sweeps)
    if (directory/"windows.csv").is_file():
        print(directory)
        return True
    reader=(HERE/"prefix_reader.py").read_bytes()
    data=batch.umt.ssh(f"cd {batch.umt.REMOTE}/{name} && python3 - {sweeps}",reader)
    if data.startswith(b"WAIT "):
        print(data.decode().strip())
        return False
    directory.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).name!=member.name:
                raise ValueError(member.name)
            (directory/member.name).write_bytes(archive.extractfile(member).read())
    print(directory,flush=True)
    return True


def early(names,sweeps):
    pending=names[:]
    while pending:
        for name in pending[:]:
            try:
                if preview(name,sweeps):
                    pending.remove(name)
            except subprocess.CalledProcessError as error:
                if error.returncode not in (255,-1,4294967295):
                    raise
        if pending:
            time.sleep(55)


def check_prefixes():
    verified=[]
    for path in sorted((HERE/"prefixes").glob("*/*/prefix.json")):
        name=path.parent.parent.name
        info=json.loads(path.read_text())
        for run,expected in info["sha256"].items():
            if hashlib.sha256((path.parent/run).read_bytes()).hexdigest()!=expected:
                raise ValueError("Diagnostic prefix checksum mismatch: "+run)
            with (HERE/"series"/name/run).open("rb") as f:
                actual=b"".join(f.readline() for i in range(info["through_sweep"]+1))
            if hashlib.sha256(actual).hexdigest()!=expected:
                raise ValueError("Final record differs from in-progress prefix: "+run)
        verified.append(dict(series=name,through_sweep=info["through_sweep"],traces=len(info["sha256"])))
    if verified:
        (HERE/"prefix_audit.json").write_text(json.dumps(verified,indent=2)+"\n",encoding="utf8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest="action",required=True)
    for op in ("prepare","doctor","plan","checkpoint","resume","accounting","watch","first-extension","repeat-1243"):
        sp.add_parser(op)
    audit_parser=sp.add_parser("audit")
    audit_parser.add_argument("--partial",action="store_true")
    submit_parser=sp.add_parser("submit")
    submit_parser.add_argument("names",nargs="*")
    submit_parser.add_argument("--limit",type=int)
    submit_parser.add_argument("--minutes",type=int,default=155)
    rec=sp.add_parser("recover");rec.add_argument("name")
    stat=sp.add_parser("status");stat.add_argument("names",nargs="*")
    prev=sp.add_parser("preview");prev.add_argument("name");prev.add_argument("sweeps",type=int)
    first=sp.add_parser("early");first.add_argument("sweeps",type=int);first.add_argument("names",nargs="+")
    prog=sp.add_parser("progress");prog.add_argument("name")
    extra=sp.add_parser("extend")
    extra.add_argument("prefix");extra.add_argument("temperature",type=float)
    extra.add_argument("cap",type=float,choices=[.35,.40]);extra.add_argument("fields",type=float,nargs="+")
    extra.add_argument("--unbiased",action="store_true")
    extra.add_argument("--short",action="store_true",help="Use 12 Q centers and 475+900 sweeps.")
    upper=sp.add_parser("tight-upper")
    upper.add_argument("temperature",type=int)
    triad=sp.add_parser("triad")
    triad.add_argument("temperature",type=int);triad.add_argument("cap",type=float,choices=[.35,.4])
    a=p.parse_args()
    if a.action=="prepare": prepare()
    elif a.action=="first-extension": first_extension()
    elif a.action=="repeat-1243": repeat_1243()
    elif a.action=="tight-upper": temperature_triad(a.temperature,.35)
    elif a.action=="triad": temperature_triad(a.temperature,a.cap)
    elif a.action=="doctor": doctor()
    elif a.action=="plan": print(" ".join(priority()))
    elif a.action=="submit":
        names=a.names or priority()
        pending=[n for n in names if not (HERE/"runs"/n/"submission.json").exists()]
        submit(pending[:a.limit],a.minutes)
    elif a.action=="recover": recover(a.name)
    elif a.action=="checkpoint": retry_connection(workflow.checkpoint)
    elif a.action=="resume":
        retry_connection(workflow.resume)
        check_prefixes()
    elif a.action=="accounting": batch.accounting()
    elif a.action=="audit":
        workflow.provenance.main(a.partial)
        if not a.partial:
            check_prefixes()
    elif a.action=="watch": watch()
    elif a.action=="preview": preview(a.name,a.sweeps)
    elif a.action=="early": early(a.names,a.sweeps)
    elif a.action=="progress":
        name=batch.checkname(a.name)
        cmd=f"cd {batch.umt.REMOTE}/{name} && wc -l {name}-*.csv"
        print(retry_connection(lambda:batch.umt.ssh(cmd)).decode())
    elif a.action=="extend":
        if a.unbiased:
            make(a.prefix+"u",a.temperature,a.cap,[(mu,0.,0.,start,625,1150) for mu in a.fields
                                                   for start in ("ordered","random")])
        else:
            for j,mu in enumerate(a.fields):
                umbrella(a.prefix+chr(ord("a")+j),a.temperature,a.cap,mu,not a.short,False)
    elif a.action=="status":
        names=a.names or [p.parent.name for p in (HERE/"runs").glob("*/submission.json")]
        ids=[json.loads((HERE/"runs"/batch.checkname(n)/"submission.json").read_text())["job_id"] for n in names]
        cmd="sacct -nXP -j "+",".join(ids)+" --format=JobID,JobName%32,State,Elapsed,NodeList"
        print(retry_connection(lambda:batch.umt.ssh(cmd)).decode())


if __name__=="__main__":
    main()
