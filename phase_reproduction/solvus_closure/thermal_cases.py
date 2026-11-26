"""Reproduce the congruent-temperature estimates using the archived production windows."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


HERE=Path(__file__).resolve().parent
OUT=HERE/"analysis/thermal"
DIRECT=HERE/"atomistic/series"
OLD=HERE.parent/"solvus_temperatures/atomistic/series"
CASES={
    "direct_1237p5_n288":["zc237m"],
    "direct_1250_n288":["zc1250m"],
    "direct_1225_n768":["zc228l","zc228r"],
    "direct_1240_n768":["zc408a"],
    "direct_1245_n768":["zc458a"],
    "direct_1250_n768":["zc508l","zc508r"],
    "direct_1240_n1400":["zc4014c","zc4014s"],
    "direct_1245_n1400":["zc4514l","zc4514c","zc4514r"],
    "direct_1247p5_n1400":["zc4714c","zc4714s"],
    "direct_1250_n1400":["zc5014l","zc5014c"],
    "direct_1245_n2304":["zc4523c","zc4523s"],
    "boot_1273_n768":["at1273-n768-"+tag for tag in ("u","v","o","p","peak","sh","ex")],
    "boot_1273_n1400":["at1273-n1400-"+tag for tag in ("pure","q","r","sh","rich")],
}
POOLED={
    "pooled_n288":["zc237m","zc1250m"],
    "pooled_n768":["zc228l","zc228r","zc408a","zc458a","zc508l","zc508r"],
    "pooled_n1400":["zc4514l","zc4514c","zc4514r","zc4714c","zc4714s","zc5014l","zc5014c"],
    "pooled_n1400_all":["zc4014c","zc4014s","zc4514l","zc4514c","zc4514r","zc4714c","zc4714s","zc5014l","zc5014c"],
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("names",nargs="*")
    p.add_argument("--ready",action="store_true",help="Run missing cases with all their local trajectories")
    p.add_argument("--show",action="store_true")
    p.add_argument("--draws",type=int,default=64)
    p.add_argument("--stride",type=int,default=8,help="Thinning interval for unbinned multi-T fits")
    p.add_argument("--part",choices=["all","early","late"],default="all")
    p.add_argument("--workers",type=int,default=1)
    a=p.parse_args()
    cmds=[]
    for name in a.names or [*CASES,*POOLED]:
        base=OLD if name.startswith("boot_1273") else DIRECT
        folders=[base/label for label in (POOLED[name] if name in POOLED else CASES[name])]
        tag=name if a.part=="all" else name+"_"+a.part
        dest=OUT/(tag+".json")
        if a.ready and (dest.exists() or not all((x/"window_status.json").exists() for x in folders)):
            continue
        draws=48 if name.startswith("boot_1273") and a.draws==64 else a.draws
        if name in POOLED:
            if a.part!="all":
                raise ValueError("Use the independent isotherms to check time halves")
            cmd=[sys.executable,HERE/"pooled.py",*[d/"windows.csv" for d in folders],
                 "--cut",.37,"--stride",a.stride,"--draws",draws,"--out",dest]
        else:
            cmd=[sys.executable,HERE/"vertex_bootstrap.py",*[d/"windows.csv" for d in folders],
                 "--cut",.37,"--part",a.part,"--draws",draws,"--out",dest]
        cmd=list(map(str,cmd))
        cmds.append(cmd)
    if a.show:
        print("\n".join(subprocess.list2cmdline(cmd) for cmd in cmds))
        return
    def execute(cmd):
        subprocess.run(cmd,check=True,env={**os.environ,"OPENBLAS_NUM_THREADS":"1"})
        print(Path(cmd[-1]).name,flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        list(executor.map(execute,cmds))


if __name__=="__main__":
    main()
