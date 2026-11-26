"""Recompute direct, same-temperature closure profiles from the downloaded manifests."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/"atomistic"/"series"
CASES={}
for temp in (1100,1150,1200,1250):
    for count,suffix in ((96,"s"),(288,"m")):
        CASES[f"T{temp}_N{count}"]=([f"zc{temp}{suffix}"],None)
CASES["T1237p5_N288"]=(["zc237m"],None)
for t,span in ((1100,(3.330,3.370)),(1150,(3.366,3.414)),(1200,(3.409,3.466))):
    CASES[f"T{t}_N768"]=([f"zc{t}8"],span)
CASES["T1240_N768"]=(["zc408a"],(3.502,3.592))
CASES["T1240_N1400"]=(["zc4014c","zc4014s"],(3.515,3.575))
CASES["T1245_N768"]=(["zc458a"],(3.507,3.591))
CASES["T1245_N1400"]=(["zc4514l","zc4514c","zc4514r"],(3.510,3.582))
CASES["T1245_N2304"]=(["zc4523c","zc4523s"],(3.523,3.567))
CASES["T1200_N1400"]=(["zc120014"],(3.425,3.457))
CASES["T1247p5_N1400"]=(["zc4714c","zc4714s"],(3.521,3.570))
CASES["control_T1245_N768_K0"]=(["zc458u"],None)
for temp,tag,left,rich in ((1225,"22",3.462,3.612),(1250,"50",3.517,3.582)):
    CASES[f"T{temp}_N768"]=([f"zc{tag}8l",f"zc{tag}8r"],(left-.018,rich+.018))
    CASES[f"T{temp}_N1400"]=([f"zc{tag}14l",f"zc{tag}14c"],(left-.012,3.55 if temp==1225 else 3.56))
    # The unconstrained K=0 starts are a separate diagnostic unless they interconvert.
    CASES[f"control_T{temp}_N768_K0"]=([f"zc{tag}8u"],None)
CASES["T1225_N768"]=(["zc228l","zc228r","zc228h"],(3.444,3.663))
CASES["control_T1225_N768_lr"]=(["zc228l","zc228r"],(3.444,3.630))


def command(name,bootstrap,halves):
    dirs,field=CASES[name]
    args=[sys.executable,HERE/"closure.py",*[DATA/x/"windows.csv" for x in dirs],"--out",HERE/(name+".json"),
          "--bootstrap",str(bootstrap),"--seed","56823"]
    if halves:
        args.append("--halves")
    if field:
        args.extend(["--field",*map(str,field)])
    return list(map(str,args))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names",nargs="*",choices=list(CASES))
    parser.add_argument("--bootstrap",type=int,default=0)
    parser.add_argument("--halves",action="store_true")
    parser.add_argument("--show",action="store_true")
    args=parser.parse_args()
    if not args.names: print(*CASES,sep="\n")
    for name in args.names:
        cmd=command(name,args.bootstrap,args.halves)
        if args.show: print(subprocess.list2cmdline(cmd))
        else: subprocess.run(cmd,check=True,env={**os.environ,"OPENBLAS_NUM_THREADS":"1"})
