"""Compare fresh ATAT→MTP→ECI results with numeric targets of manuscript v16."""
import argparse
import csv
import json
import io
from zipfile import ZipFile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from read_v16 import word_tables


def run(article, mfa_files, mc_files, output):
    output.mkdir(parents=True,exist_ok=True)
    paper={(("static" if "вибрационного" in m else m),method):np.array([float(a),float(b)])
           for m,method,a,b in word_tables(article)[0][1:]}
    observations=[]
    uncertainty={}
    for mfa in mfa_files:
        for row in csv.DictReader(mfa.open(encoding="utf8")):
            model,t,status=row["model"],float(row["T_K"]),row["status"]
            if row.get("volume_endpoint","").lower()=="true":
                status="volume_edge"
            pair=np.array([float(row[k]) for k in ("alpha_at_pct","alpha2_at_pct")]) if status=="interior_tangent" else None
            observations.append((model,"MFA",t,None,None,status,pair))
    for file in mc_files:
        for row in json.loads(file.read_text(encoding="utf8")):
            if not row["model"].startswith("atat_"):
                raise ValueError("Expected newly fitted ATAT model, got "+row["model"])
            model=row["model"].removeprefix("atat_")
            all_intervals=[p for p in row["TI_contacts"] if (not p["window_edge"]
                     and p["alpha2_at_pct"]-p["alpha_at_pct"]>.5)]
            matched=([p for p in all_intervals if p.get("order_direction")=="rising"]
                     if any("order_direction" in p for p in all_intervals) else all_intervals)
            pair=np.array([matched[0][k] for k in ("alpha_at_pct","alpha2_at_pct")]) if len(matched)==1 else None
            status="interior_tangent" if pair is not None else "multiple" if matched else "unresolved"
            observations.append((model,"MC",row["T_K"],row["N"],row["windows"],status,pair))
            uncertainty[model,row["T_K"],row["N"],row["windows"]]=row["bootstrap_conditional"]
    with (output/"atat_comparison.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(("model","method","T_K","N","windows","status","alpha_at_pct","alpha2_at_pct",
                    "reference","paper_status","paper_alpha_at_pct","paper_alpha2_at_pct","delta_alpha_at_pct","delta_alpha2_at_pct",
                    "alpha_conditional_lo","alpha_conditional_hi","alpha2_conditional_lo","alpha2_conditional_hi",
                    "bootstrap_resolved","bootstrap_draws"))
        for model,method,t,n,nw,status,pair in observations:
            r=paper[model,method] if t==1000 else None
            source="v16 Table 1" if r is not None else ""
            if method=="MC" and t==600 and model in ("HA","QHA"):
                r=np.array((6.71,22.61) if model=="HA" else (4.59,24.09))
                source="v16 SI S2"
            if method=="MC" and n==5120 and (model,t) in {("HA",1200),("QHA",1200),("static",1600),("static",1625)}:
                r=np.array({("HA",1200):(22.60,24.03),("QHA",1200):(23.46,24.91),
                            ("static",1600):(22.89,24.16),("static",1625):(24.12,24.96)}[model,t])
                source="v16 SI S3"
            expected_status="interior_tangent" if r is not None else ""
            if method=="MC" and n==5120 and (model,t) in {("static",1650),("HA",1225),("QHA",1225)}:
                expected_status="unresolved"
                source="v16 SI S3"
            stat=uncertainty.get((model,t,n,nw),{})
            w.writerow((model,method,t,n,nw,status,*(pair if pair is not None else ["",""]),
                        source,expected_status,*(r if r is not None else ["",""]),*(pair-r if pair is not None and r is not None else ["",""]),
                        *stat.get("alpha_95",["",""]),*stat.get("alpha2_95",["",""]),stat.get("resolved",""),stat.get("draws","")))
    fig,ax=plt.subplots(figsize=(7.4,4.4))
    groups=[key for key in paper if any((m,met)==key and t==1000 for m,met,t,*_ in observations)]
    if not groups:
        plt.close(fig)
        return
    for i,key in enumerate(groups):
        ax.plot(paper[key],[i,i],"o",mec="#313e47",mfc="none",ms=8,label="Статья, табл. 1" if i==0 else None)
        candidates=[e for e in observations if e[:2]==key and e[2]==1000]
        row=max(candidates,key=lambda e:(e[3] or 0,e[4] or 0))
        if row[-1] is not None:
            ax.plot(row[-1],[i-.08]*2,"s-",color="#23726c",ms=5,label="Независимая CE" if i==0 else None)
        else:
            ax.text(1,i-.13,"касательная не разрешена",fontsize=8)
    ylabels=[("Статика" if m=="static" else m)+", "+met for m,met in groups]
    ax.set(yticks=range(len(groups)),yticklabels=ylabels,ylim=(len(groups)-.55,-.6),
           xlabel="Al, ат.%",xlim=(0,30),title="Новая подгонка по MTP для структур ATAT, 1000 K")
    ax.grid(axis="x",alpha=.17)
    ax.spines[["right","top"]].set_visible(False)
    ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.17),ncol=2)
    fig.subplots_adjust(left=.22,right=.97,bottom=.25,top=.88)
    fig.savefig(output/"atat_1000K.png",dpi=180)
    plt.close(fig)
    if len(set(row[2] for row in observations))<3:
        return
    with ZipFile(article) as z:
        ref_image=Image.open(io.BytesIO(z.read("word/media/image5.png"))).convert("RGB")
    if ref_image.size!=(3840,3150):
        raise ValueError("Unexpected v16 Figure 5 source size")
    fig,panels=plt.subplots(1,2,figsize=(10,5.3),sharey=True)
    colors={"static":"#166f98","HA":"#b95340","QHA":"#aa7420"}
    for axis, method, x0 in zip(panels,["MFA","MC"],[346,2154]):
        # Use the paper's native axes and exact colors to extract thin reference
        # traces. White gaps in its dashed curves and hollow markers are preserved.
        rgb=np.asarray(ref_image.crop((x0,345,x0+1628,2128))).astype(int)
        blue=(rgb[:,:,0]<70)&(rgb[:,:,1]>70)&(rgb[:,:,2]>170)
        red=(rgb[:,:,0]>170)&(rgb[:,:,1]<60)&(rgb[:,:,2]<100)
        orange=(rgb[:,:,0]>200)&(rgb[:,:,1]>75)&(rgb[:,:,1]<170)&(rgb[:,:,2]<70)
        img=np.zeros((*rgb.shape[:2],4),dtype=np.uint8)
        img[:,:,:3]=np.array([100,103,104])
        img[:,:,3]=(blue|red|orange)*93
        axis.imshow(img,extent=[0,35,600,1800],origin="upper",aspect="auto",interpolation="bilinear")
        for model in colors:
            available=[e for e in observations if e[:2]==(model,method)]
            for size in dict.fromkeys(e[3] for e in available):
                by_t={}
                for _,_,t,n,nw,status,pair in available:
                    if n==size and (t not in by_t or (nw or 0)>by_t[t][0]):
                        by_t[t]=(nw or 0,pair)
                temps=sorted(by_t)
                vals=np.array([by_t[t][1] if by_t[t][1] is not None else [np.nan,np.nan] for t in temps])
                # Small-cell series contains every main isotherm. A NaN breaks
                # the connecting segment. Large-cell refinements are standalone
                # observations rather than a curve through unmeasured temperatures.
                style="-" if method=="MFA" else "o-" if size==768 else "o"
                for side in range(2):
                    axis.plot(vals[:,side],temps,style,color=colors[model],
                              mfc="white" if size==5120 else colors[model], ms=5 if size==5120 else 3,
                              lw=1.6,label=("Статика" if model=="static" else model) if method=="MFA" and side==0 else None)
        axis.set(title=method,xlabel="Al, ат.%",xlim=(0,35),ylim=(590,1810))
        axis.spines[["top","right"]].set_visible(False)
    panels[0].set_ylabel("Температура, K")
    fig.suptitle("Бинодали по новой кластерной модели ATAT–MTP",y=.97)
    fig.text(.5,.045,"Серым показан рис. 5 статьи. Закрашенные точки MC соответствуют 768 узлам, пустые — 5120.",
             ha="center",fontsize=9)
    panels[0].legend(frameon=False,loc="lower left",ncol=3,bbox_to_anchor=(0,-.29))
    fig.subplots_adjust(left=.085,right=.98,bottom=.23,top=.84,wspace=.15)
    fig.savefig(output/"atat_phase_comparison.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("article",type=Path)
    p.add_argument("mfa",type=Path,nargs="+")
    p.add_argument("--mc",type=Path,nargs="+",default=[])
    p.add_argument("--out",type=Path,required=True)
    arg=p.parse_args()
    run(arg.article,arg.mfa,arg.mc,arg.out)
