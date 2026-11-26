"""Resume analysis of complete, locally downloaded UMT groups after shutdown."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from reproduce import CASES, DATA, HERE, command


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def ready(directory):
    contents=read_json(directory/"window_status.json")
    manifest=directory/"windows.csv"
    if not contents or not manifest.is_file():
        return False
    with manifest.open(encoding="utf-8-sig",newline="") as fp:
        names={r["run"] for r in csv.DictReader(fp)}
    windows=contents.get("windows",{})
    return bool(names and names==windows.keys() and all(
        windows[key].get("status")=="ok" and windows[key].get("exit_code")==0 and
        all((directory/(key+ext)).is_file() for ext in (".csv",".state")) for key in names))


def run(args):
    subprocess.run(list(map(str,args)),check=True,env={**os.environ,"OPENBLAS_NUM_THREADS":"1"})


def updated(dest,inputs):
    return dest.is_file() and all(i.stat().st_mtime<=dest.stat().st_mtime for i in inputs)


def run_cached(name,series,nrep):
    out=HERE/(name+".json")
    records=read_json(out)
    if records:
        rec=records[0]
        if sorted(rec.get("series",[]))==sorted(series) and rec.get("bootstrap",{}).get("count",0)>=nrep and rec.get("halves"):
            return out
        if name=="T1225_N768" and sorted(rec.get("series",[]))==["zc228l","zc228r"]:
            import shutil
            shutil.copy2(out,HERE/"control_T1225_N768_lr.json")
    args=command(name,nrep,True)
    temp=out.with_suffix(".inprogress")
    args[args.index("--out")+1]=str(temp)
    run(args)
    temp.replace(out)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-all",action="store_true",help="exit 2 if remote series remain unavailable")
    parser.add_argument("--status",action="store_true",help="only display availability and existing full analyses")
    opts=parser.parse_args()
    pending={}
    for name,(series,_) in CASES.items():
        if name=="control_T1225_N768_lr":
            continue
        missing=[s for s in series if not ready(DATA/s)]
        if missing:
            pending[name]=missing
            print("PENDING",name,*missing,flush=True)
            continue
        n=int(name.split("_N")[1].split("_")[0])
        paths=[DATA/s/"windows.csv" for s in series]
        if opts.status:
            records=read_json(HERE/(name+".json"))
            done=bool(records and sorted(records[0].get("series",[]))==sorted(series) and records[0].get("halves"))
            print("ANALYZED" if done else "READY",name,flush=True)
            continue
        if name.startswith("control_"):
            dest=HERE/("K0_"+name.split("control_")[1].split("_K0")[0]+".json")
            if not updated(dest,paths):
                run([sys.executable,HERE/"hysteresis.py",*paths,"--out",dest])
            continue
        print("ANALYZE",name,flush=True)
        profile=run_cached(name,series,32 if n==96 else 48 if n==288 else 64)
        if n>=768 and name.startswith(("T1240_","T1245_","T1247p5_","T1250_")):
            dest=HERE/("subsets_"+name+".json")
            if not updated(dest,[profile]):
                run([sys.executable,HERE/"check_subsets.py",profile,"--out",dest])
        if n>=768 and name.startswith("T1245_"):
            dest=HERE/("terminal_"+name+".json")
            if not updated(dest,paths):
                run([sys.executable,HERE/"terminal_order.py",*paths,"--out",dest])
    if not opts.status:
        for script in ("summarize.py","apex_temperature.py","size_scaling.py"):
            run([sys.executable,HERE/script])
        if list(HERE.glob("T1245_N*.json")):
            run([sys.executable,HERE/"critical_profiles.py"])
    print(f"Primary and diagnostic groups pending: {len(pending)}",flush=True)
    sys.exit(2 if opts.require_all and pending else 0)
