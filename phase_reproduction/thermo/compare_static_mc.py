"""Compare the static temperature sweep with the markers in v16 Figure 5b."""
import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot(article, files, output):
    with ZipFile(article) as z:
        image=np.asarray(Image.open(io.BytesIO(z.read("word/media/image5.png"))).convert("RGB"))
    if image.shape[:2]!=(3150,3840):
        raise ValueError("Unexpected Figure 5 raster")
    x_fit=np.polyfit([2154,2386,2619,2852,3084,3317,3550,3782],np.arange(0,36,5),1)
    y_fit=np.polyfit([600,900,1200,1500,1800],[2127,1682,1236,790,345],1)
    # Filled blue N=768 markers have unobscured centres at these isotherms.
    # The 600/1000 K markers overlap other marks; 1600 K shows N=5120.
    targets={}
    for t in (700,800,900,1100,1200,1300,1400,1500):
        row=image[round(np.polyval(y_fit,t)),2154:3785,:]
        xs=np.flatnonzero((row[:,0]<60)&(row[:,1]>70)&(row[:,2]>170))+2154
        groups=np.split(xs,np.flatnonzero(np.diff(xs)>1)+1)
        if len(groups)!=2:
            raise ValueError(f"Ambiguous MC raster at {t}")
        targets[t]=np.polyval(x_fit,[g.mean() for g in groups])
    results={}
    for file in files:
        for e in json.loads(file.read_text(encoding="utf8")):
            if e["model"]=="static_S1":
                key=(e["N"],int(e["T_K"]))
                if e["windows"]<=results.get(key,{}).get("windows",0):
                    continue
                results[key]=e
    fig,ax=plt.subplots(figsize=(5.9,4.8))
    with (output/"mc_static_comparison.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("N","T_K","windows","status","alpha_at_pct","alpha2_at_pct",
                         "source","alpha_ref_at_pct","alpha2_ref_at_pct","delta_alpha_at_pct","delta_alpha2_at_pct"))
        for n in sorted({n for n,t in results}):
            points=[]
            for _,t in sorted(key for key in results if key[0]==n):
                e=results[n,t]
                intervals=[p for p in e["TI_contacts"] if not p["window_edge"]]
                src,ref="",None
                if n==768 and t in targets:
                    src,ref="Fig. 5b raster",targets[t]
                elif n==5120 and t in (1000,1600,1625):
                    src,ref="Table 1 or SI S3",{1000:(11.45,20.47),1600:(22.89,24.16),1625:(24.12,24.96)}[t]
                status="interior" if len(intervals)==1 else "multiple" if intervals else "unresolved"
                if len(intervals)==1:
                    a,b=(intervals[0][k] for k in ("alpha_at_pct","alpha2_at_pct"))
                    writer.writerow((n,t,e["windows"],status,a,b,src,*(list(ref)+[a-ref[0],b-ref[1]] if ref is not None else [""]*4)))
                    points.append((t,a,b))
                else:
                    writer.writerow((n,t,e["windows"],status,"","",src,*(ref if ref is not None else ["",""]),"",""))
            if points:
                temp,a,b=np.array(points).T
                style="o-" if len(points)>1 else "s"
                color="#176a89" if n==768 else "#c75b37"
                ax.plot(a,temp,style,color=color,ms=4,label=f"Расчёт, N = {n}")
                ax.plot(b,temp,style,color=color,ms=4)
    for i in range(2):
        ax.plot([pt[i] for pt in targets.values()],list(targets),"o",mec="#3b4347",mfc="none",ms=8,
                label="Рис. 5б, N = 768" if i==0 else None)
    ax.set(xlabel="Al, ат.%",ylabel="Температура, K",title="Статическая бинодаль Монте-Карло")
    ax.legend(frameon=False)
    ax.spines[["right","top"]].set_visible(False)
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(output/"mc_static_comparison.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("article",type=Path)
    cli.add_argument("files",type=Path,nargs="+")
    cli.add_argument("--out",type=Path,required=True)
    args=cli.parse_args()
    plot(args.article,args.files,args.out)
