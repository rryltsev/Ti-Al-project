"""Submit the temperature-closure series with the validated Ti–Al kernel."""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REFERENCE = PROJECT / "solvus_temperatures" / "atomistic"
sys.path.insert(0, str(REFERENCE))
import batch
import provenance

batch.HERE = HERE
batch.umt.REMOTE = "/home6/u3129/tial-closure-20261126"
provenance.HERE = HERE
provenance.PROJECT = PROJECT
EXPECTED_MTP = "704f7fd900c7446286066b741084c30ebc7fcf26e5b9b8fd2f29a18ba4ffd661"


def prepare_inputs():
    model = PROJECT / "inputs/fitted.mtp"
    if hashlib.sha256(model.read_bytes()).hexdigest() != EXPECTED_MTP:
        raise ValueError("Unexpected MTP identity")
    for name in ["sgmc.cpp", "run_windows.py"]:
        data = (REFERENCE / name).read_bytes()
        copy = HERE / name
        if copy.exists():
            if copy.read_bytes() != data:
                raise ValueError("MTP simulation source differs from the validated implementation: " + name)
        else:
            copy.write_bytes(data)


def prepare(name, temperature, nx, nz, fields, windows, burn, sweeps, start, cap=.4, strength=None,
            qmin=.025,qmax=.955):
    prepare_inputs()
    batch.make(SimpleNamespace(name=name, temperature=temperature, nx=nx, nz=nz, cap=cap, burn=burn,
        sweeps=sweeps, fields=fields, windows=windows, qmin=qmin, qmax=qmax, strength=strength,
        start=start, stride=1, axial=1))


def screen():
    for t, shift in [(1100, 0.), (1150, .027), (1200, .058), (1250, .088)]:
        for letter, nx, nz, fields, windows, burn, sweeps in [
            ("s", 4, 3, [3.29, 3.33, 3.37, 3.41, 3.45, 3.50, 3.56], 7, 250, 550),
            ("m", 6, 4, [3.31, 3.34, 3.37, 3.40, 3.43, 3.46, 3.49], 8, 350, 650),
        ]:
            prepare(f"zc{t}{letter}", t, nx, nz, [round(x + shift, 3) for x in fields],
                    windows, burn, sweeps, ["window"])


def closure_candidates():
    for t, label, left, center, right in [(1225, "22", 3.462, 3.54, 3.612),
                                          (1250, "50", 3.517, 3.546, 3.582)]:
        for side, mus in [("l", [left-.009,left+.009]), ("r", [center,right])]:
            prepare(f"zc{label}8{side}", t, 8, 6, [round(x,3) for x in mus], 12, 400, 800, ["window"])
        for side, mu in [("l", left), ("c", center)]:
            prepare(f"zc{label}14{side}", t, 10, 7, [mu], 14, 475, 850, ["window"])
        paired=[round(x,3) for x in [left-.022,left-.006,left+.012,center,right-.01,right+.01,right+.035]
                for k in range(2)]
        prepare(f"zc{label}8u", t, 8, 6, paired, 1, 500, 850, ["random","ordered"], strength=0)
    prepare("zc237m",1237.5,6,4,[3.425,3.455,3.485,3.515,3.545,3.575,3.605,3.635],
            8,350,700,["window"])


def apex():
    for t, label, fields in [(1240,"40",[3.510,3.540,3.575]),
                             (1245,"45",[3.520,3.545,3.570])]:
        prepare(f"zc{label}8a",t,8,6,fields,12,475,850,["window"])
    for side,mu in [("l",3.520),("c",3.545),("r",3.570)]:
        prepare(f"zc4514{side}",1245,10,7,[mu],14,500,900,["window"])
    prepare("zc458u",1245,8,6,[mu for mu in (3.505,3.525,3.545,3.565,3.585) for _ in range(2)],
            1,550,950,["random","ordered"],strength=0)


def lower_ties():
    for temperature, fields in [(1100,[3.345,3.355]),(1150,[3.385,3.400]),(1200,[3.428,3.448])]:
        prepare(f"zc{temperature}8",temperature,8,6,fields,13,475,850,["window"])


def largest_cell():
    prepare("zc4523c",1245,12,8,[3.545],20,725,1500,["window"],qmin=.018,qmax=.925)
    prepare("zc4523s",1245,12,8,[3.535,3.555],4,600,1050,["random","ordered"],qmin=.08,qmax=.80)


def final_bracket(t):
    label={1240:"40",1247.5:"47"}[t]
    prepare(f"zc{label}14c",t,10,7,[3.545],16,600,1125,["window"],qmin=.02,qmax=.93)
    prepare(f"zc{label}14s",t,10,7,[3.533,3.557],4,550,950,["ordered","random"],qmin=.075,qmax=.81)


def recover_submission(name):
    """Reconcile a broken SSH receipt with job.id, squeue and accounting."""
    run=HERE/"runs"/batch.checkname(name)
    intent=json.loads((run/"intent.json").read_text())
    if (run/"submission.json").exists():
        return
    remote=intent["remote"]
    if remote != batch.umt.REMOTE+"/"+name or intent["user"]!="u3129":
        raise ValueError("Unexpected remote directory or account")
    receipt=batch.umt.ssh(f"set -eu; cd {remote}; sha256sum -c inputs.sha256 >/dev/null; "
                          "if test -s job.id; then cat job.id; fi").decode().strip().split(";")[0]
    scheduler=batch.umt.ssh("set -eu; squeue -h --user=u3129 --name="+intent["job_name"]+" -o %i; "
        "sacct -nXP --starttime="+intent["utc"][:10]+" --user=u3129 --name="+intent["job_name"]+
        " --format=JobID").decode()
    known={s.strip().rstrip("|") for s in scheduler.splitlines() if s.strip()}
    if receipt:
        known.add(receipt)
    if not known:
        cmd=shlex.join(intent["sbatch"])
        receipt=batch.umt.ssh(f"set -eu; cd {remote}; if test -s job.id; then cat job.id; "
                              f"else {cmd} > job.id; cat job.id; fi").decode().strip().split(";")[0]
        known.add(receipt)
    if len(known)!=1 or not next(iter(known)).isdigit():
        raise ValueError("Expected one confirmed Slurm allocation: "+str(known))
    job_id=known.pop()
    batch.umt.save(run/"submission.json",dict(job_id=job_id,utc=batch.umt.stamp(),reconciled=True))
    print(name,job_id,flush=True)


def run_names():
    runs=sorted((HERE/"runs").glob("*/intent.json"))
    for p in runs:
        if not (p.parent/"submission.json").exists():
            raise RuntimeError("Reconcile this submission first: "+p.parent.name)
    return [p.parent.name for p in runs]


def checkpoint():
    names=run_names()
    ids={json.loads((HERE/"runs"/n/"submission.json").read_text())["job_id"]:n for n in names}
    state=batch.umt.ssh("sacct -nXP -j "+",".join(ids)+" --format=JobID,State,ExitCode,Elapsed,NodeList").decode()
    rows=[]
    for line in state.splitlines():
        j,s,code,elapsed,node=line.strip().rstrip("|").split("|")
        name=ids[j]
        rows.append(dict(series=name,slurm_id=j,state=s,exit_code=code,elapsed=elapsed,node=node,
                         local_archive=(HERE/"runs"/name/"outputs/raw.sha256").exists()))
    if len(rows)!=len(names) or {r["slurm_id"] for r in rows}!=set(ids):
        raise RuntimeError("Incomplete Slurm snapshot")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path=HERE/"checkpoints"/(stamp+".json")
    path.parent.mkdir(exist_ok=True)
    data=dict(utc=stamp,account="u3129",remote=batch.umt.REMOTE,runs=rows)
    raw=(json.dumps(data,indent=2)+"\n").encode()
    path.write_bytes(raw)
    remote_path=batch.umt.REMOTE+"/checkpoints/"+path.name
    checksum=batch.umt.ssh(f"set -eu; mkdir -p {batch.umt.REMOTE}/checkpoints; set -C; "
                           f"cat > {remote_path}; sha256sum {remote_path}",raw).decode().split()[0]
    if checksum!=hashlib.sha256(raw).hexdigest():
        raise RuntimeError("Remote checkpoint checksum mismatch")
    print(str(path),remote_path,flush=True)
    print({s:sum(r["state"]==s for r in rows) for s in sorted({r["state"] for r in rows})},flush=True)


def restore_local(name):
    output=HERE/"runs"/name/"outputs"
    archive=output/"results.tar.gz"
    if not archive.exists() or not (output/"raw.sha256").exists():
        return False
    if provenance.digest(archive)!=(output/"raw.sha256").read_text().split()[0]:
        raise ValueError("Local archive checksum mismatch: "+name)
    directory=HERE/"series"/name
    with tarfile.open(archive) as ar:
        if ar.extractfile("windows.csv").read()!=(directory/"windows.csv").read_bytes():
            raise ValueError("Archive manifest mismatch: "+name)
        for m in ar:
            if not m.isfile() or Path(m.name).name!=m.name:
                raise ValueError(m.name)
            if m.name=="window_status.json" or m.name.startswith(name+"-"):
                dest=directory/m.name
                payload=ar.extractfile(m).read()
                if not dest.exists() or dest.read_bytes()!=payload:
                    part=dest.with_suffix(dest.suffix+".resume.part")
                    part.write_bytes(payload)
                    part.replace(dest)
    return True


def resume():
    names=run_names()
    missing=[n for n in names if not restore_local(n)]
    while missing:
        try:
            batch.collect(missing)
            break
        except subprocess.CalledProcessError as error:
            if error.returncode not in (255,-1,4294967295):
                raise
            time.sleep(10)
            missing=[n for n in missing if not restore_local(n)]
    for name in missing:
        if not restore_local(name):
            batch.fetch(name)
            if not restore_local(name):
                raise RuntimeError("Missing verified archive: "+name)
    batch.accounting()
    provenance.main(False)

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cmd = cli.add_subparsers(dest="op", required=True)
    cmd.add_parser("doctor")
    cmd.add_parser("screen")
    cmd.add_parser("candidates")
    cmd.add_parser("apex")
    cmd.add_parser("lower")
    cmd.add_parser("largest")
    bp=cmd.add_parser("bracket")
    bp.add_argument("temperature",type=float,choices=[1240.,1247.5],nargs="?",default=1247.5)
    rec=cmd.add_parser("recover")
    rec.add_argument("name")
    progress=cmd.add_parser("progress")
    progress.add_argument("name")
    make = cmd.add_parser("make")
    make.add_argument("name")
    make.add_argument("temperature", type=float)
    make.add_argument("nx", type=int)
    make.add_argument("nz", type=int)
    make.add_argument("burn", type=int)
    make.add_argument("sweeps", type=int)
    make.add_argument("--fields", type=float, nargs="+", required=True)
    make.add_argument("--windows", type=int, required=True)
    make.add_argument("--start", nargs="+", default=["window"])
    make.add_argument("--cap", type=float, default=.4)
    make.add_argument("--strength", type=float)
    for action in ["submit", "collect", "status", "queue"]:
        parser=cmd.add_parser(action)
        parser.add_argument("names", nargs="+")
        if action=="submit":
            parser.add_argument("--cpus",type=int,default=16)
            parser.add_argument("--minutes",type=int,default=75)
    cmd.add_parser("accounting")
    cmd.add_parser("checkpoint")
    cmd.add_parser("resume")
    audit=cmd.add_parser("audit")
    audit.add_argument("--partial",action="store_true",help="Audit only retrieved archives while the rest are running.")
    args=cli.parse_args()
    if args.op=="doctor":
        prepare_inputs()
        out=batch.umt.ssh("set -eu; id -un; hostname; command -v sbatch; command -v g++; "
                          "test -x /home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python; "
                          "sinfo -h -p apollo,nompi,debug -o '%P %a %D'").decode()
        if out.splitlines()[0]!="u3129":
            raise RuntimeError(out)
        print(out.strip())
    elif args.op=="screen":
        screen()
    elif args.op=="candidates":
        closure_candidates()
    elif args.op=="apex":
        apex()
    elif args.op=="lower":
        lower_ties()
    elif args.op=="largest":
        largest_cell()
    elif args.op=="bracket":
        final_bracket(args.temperature)
    elif args.op=="recover":
        recover_submission(args.name)
    elif args.op=="progress":
        name=batch.checkname(args.name)
        print(batch.umt.ssh(f"cd {batch.umt.REMOTE}/{name} && wc -l {name}-*.csv").decode())
    elif args.op=="make":
        prepare(args.name,args.temperature,args.nx,args.nz,args.fields,args.windows,args.burn,args.sweeps,
                args.start,args.cap,args.strength)
    elif args.op=="submit":
        for name in args.names:
            batch.submit(name,args.cpus,args.minutes)
    elif args.op=="collect":
        batch.collect(args.names)
    elif args.op in {"status","queue"}:
        jobs=[json.loads((HERE/"runs"/batch.checkname(name)/"submission.json").read_text())["job_id"]
              for name in args.names]
        if args.op=="status":
            cmd="sacct -nXP -j "+",".join(jobs)+" --format=JobID,JobName%32,State,Elapsed,NodeList"
        else:
            cmd="squeue --start -h -j "+",".join(jobs)+" -o '%i|%T|%S|%C|%R'; sinfo -h -p apollo,nompi -o '%P|%c|%a|%D'"
        print(batch.umt.ssh(cmd).decode())
    elif args.op=="accounting":
        batch.accounting()
    elif args.op=="checkpoint":
        checkpoint()
    elif args.op=="resume":
        resume()
    elif args.op=="audit":
        provenance.main(args.partial)


if __name__=="__main__":
    main()
