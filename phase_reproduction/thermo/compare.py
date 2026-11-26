"""Compare independently calculated binodals with v16 Fig. 5a and Tables 1/S6."""
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
from read_v16 import word_tables


def fig5_static_mfa(path):
    with ZipFile(path) as z:
        image=np.asarray(Image.open(io.BytesIO(z.read("word/media/image5.png"))).convert("RGB"))
    if image.shape[:2]!=(3150,3840):
        raise ValueError("Unexpected Fig. 5a raster size")
    # Axis tick centres measured from the native left panel (x in at.%, y in K).
    x_fit=np.polyfit([346,578,811,1044,1276,1509,1742,1974], np.arange(0,36,5),1)
    y_fit=np.polyfit([600,900,1200,1500,1800],[2127,1682,1236,790,345],1)
    values={}
    for t in range(600,1801,100):
        row=image[round(np.polyval(y_fit,t)),346:1975,:]
        xs=np.flatnonzero((row[:,0]<60)&(row[:,1]>70)&(row[:,2]>170))+346
        groups=np.split(xs,np.flatnonzero(np.diff(xs)>1)+1)
        if len(groups)!=2 or any(len(g)<2 for g in groups):
            raise ValueError(f"Cannot resolve both blue branches at {t} K")
        values[t]=[float(np.polyval(x_fit,g.mean())) for g in groups]
    return values


def compare(pd,results,mc_files,independent):
    article,=pd.glob("*v16_статья.docx")
    table=word_tables(article)[0]
    ref={(("static" if "вибрационного" in model else model),method):(float(a),float(b))
         for model,method,a,b in table[1:]}
    drawn=fig5_static_mfa(article)
    calculated=list(csv.DictReader((results/"mfa_contacts.csv").open(encoding="utf8")))
    selected={}
    fit_results={}
    if independent:
        for r in csv.DictReader(independent.open(encoding="utf8")):
            if float(r["T_K"])==1000 and r["status"]=="interior_tangent":
                name=r["model"]
                fit_results[(name,"MFA")]=tuple(float(r[k]) for k in ("alpha_at_pct","alpha2_at_pct"))
    fig, (axis, sect)=plt.subplots(1,2,figsize=(9.8,4.9),gridspec_kw={"width_ratios":[1,1.15]})
    with (results/"paper_comparison.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("model","method","T_K","reference","alpha_calc_at_pct","alpha2_calc_at_pct",
                         "alpha_ref_at_pct","alpha2_ref_at_pct","delta_alpha_at_pct","delta_alpha2_at_pct"))
        def save(name,method,t,xr,comp,source):
            writer.writerow((name,method,t,source,*comp,*xr,*(np.subtract(comp,xr))))
        for r in calculated:
            if r["status"]!="interior_tangent":
                continue
            t=int(r["T_K"]); name="static" if r["model"]=="static_S1" else "HA"
            comp=tuple(float(r[k]) for k in ("alpha_at_pct","alpha2_at_pct"))
            if name=="static":
                save(name,"MFA",t,drawn[t],comp,"Fig. 5a raster")
            if t==1000:
                selected[(name,"MFA")]=comp
                save(name,"MFA",t,ref[name,"MFA"],comp,"Table 1")
        precision={}
        for (name,method),comp in fit_results.items():
            save(name+" independent fit",method,1000,ref[name,method],comp,"Table 1")
        for path in mc_files:
            for entry in json.loads(path.read_text(encoding="utf8")):
                if entry["T_K"]!=1000:
                    continue
                name=entry["model"].lower()
                name="static" if "stat" in name else "HA" if "ha" in name and "qha" not in name else None
                if not name:
                    continue
                contacts=[p for p in entry["TI_contacts"] if not p["window_edge"]]
                if len(contacts)==1:
                    comp=tuple(contacts[0][k] for k in ("alpha_at_pct","alpha2_at_pct"))
                    rank=(entry["N"],entry["windows"])
                    if rank>precision.get(name,(0,0)):
                        selected[name,"MC"]=comp
                        precision[name]=rank
                    save(name,f"MC N={entry['N']} W={entry['windows']}",1000,ref[name,"MC"],comp,"Table 1")
    rows=[r for r in calculated if r["model"]=="static_S1" and r["status"]=="interior_tangent"]
    times=[int(r["T_K"]) for r in rows]
    left=[float(r["alpha_at_pct"]) for r in rows]
    right=[float(r["alpha2_at_pct"]) for r in rows]
    axis.fill_betweenx(times,left,right,color="#d8ebf5")
    axis.plot(left,times,"-",color="#146b9b",label="Расчёт по табл. S1")
    axis.plot(right,times,"-",color="#146b9b")
    for b in range(2):
        axis.plot([drawn[t][b] for t in times],times,"o",mec="#2b3339",mfc="none",ms=4,
                  label="Статья, рис. 5а" if b==0 else None)
    axis.set(xlabel="Al, ат.%",ylabel="Температура, K", title="Статическая бинодаль MFA",
             xlim=(0,27), ylim=(550,1850))
    axis.text(1.1,1310,"α",fontsize=14)
    axis.text(23,1310,"α₂",fontsize=14)
    axis.legend(frameon=False,loc="lower left",bbox_to_anchor=(0,-.28))
    ordering=[("static","MFA"),("HA","MFA"),("QHA","MFA"),("static","MC"),("HA","MC"),("QHA","MC")]
    for i,key in enumerate(ordering):
        a,b=ref[key]
        sect.plot((a,b),(i,i),color="#abb5bd",lw=1.2)
        sect.plot((a,b),(i,i),"o",mfc="white",mec="#323c42",ms=6,label="Табл. 1" if i==0 else None)
        if key in selected:
            sect.plot(selected[key],(i-.12,i-.12),"x",color="#be4e2a",label="Пересчёт" if i==0 else None)
        if key in fit_results:
            sect.plot(fit_results[key],(i+.13,i+.13),"^",mfc="none",mec="#258066",ms=5,
                      label="Новая подгонка" if key==next(iter(fit_results)) else None)
    sect.set(xlabel="Al, ат.%",title="Составы при 1000 K",xlim=(4,24),ylim=(5.5,-.7),
             yticks=np.arange(6),yticklabels=[f"{'Статика' if k=='static' else k}, {m}" for k,m in ordering])
    sect.legend(frameon=False,loc="lower left",bbox_to_anchor=(-.12,-.27),ncol=2,fontsize=9)
    for ax in (axis,sect):
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="x",alpha=.16)
    fig.subplots_adjust(left=.078,right=.98,bottom=.24,top=.9,wspace=.47)
    fig.savefig(results/"phase_comparison.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("pd",type=Path)
    p.add_argument("results",type=Path)
    p.add_argument("--mc",nargs="+",type=Path,default=[])
    p.add_argument("--independent",type=Path)
    args=p.parse_args()
    compare(args.pd,args.results,args.mc,args.independent)
