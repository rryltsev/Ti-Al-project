"""Run a prepared set of independent spin-MC windows on the local CPU."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("exe",type=Path)
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    exe=a.exe.resolve()
    cfg=json.loads((a.run/"manifest.json").read_text())
    rows=list(csv.reader((a.run/"windows.tsv").read_text().splitlines(),delimiter="\t"))
    def one(item):
        i,row=item
        model,t,nx,ny,nz,c0,mu,seed,eq,prod,start,coeff,step,orbit=row
        final=a.run/"samples"/f"{i}.csv.gz"
        if final.exists():
            return
        tmp=final.with_suffix("")
        tm=time.monotonic()
        args=[str(exe),"motifs.txt",coeff,nx,ny,nz,t,c0,"100",mu,eq,prod,str(cfg["stride"]),seed,start,model,step,
              str(tmp.resolve()),orbit]
        calc=subprocess.run(args,cwd=a.run,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        with tmp.open("rb") as inp, gzip.open(str(final)+".tmp","wb") as out:
            shutil.copyfileobj(inp,out)
        Path(str(final)+".tmp").replace(final)
        tmp.unlink()
        (a.run/"samples"/f"{i}.log").write_text(calc.stdout,encoding="ascii")
        print(i,c0,round(time.monotonic()-tm,2),calc.stdout.strip(),flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(one,enumerate(rows,1)))


if __name__=="__main__":
    main()
