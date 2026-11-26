"""Show identical-input checks and the source of the Ti-rich alpha2 shift at 1000 K."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent


def read(name):
    with (HERE/name).open(newline="",encoding="utf8") as f:
        return list(csv.DictReader(f))


def run():
    models=["static","HA","QHA"]
    mfa={(r["source"],r["model"]):float(r["alpha2_at_pct"]) for r in
         read("thermo/mfa_1000K.csv") if r["volume_spline"]!="natural"}
    mc={(r["source"],r["model"]):float(r["author_alpha2"]) for r in
        read("mc/mc_same_windows.csv") if r["T_K"]=="1000" and r["N"]=="5120"}
    fixed={(r["model"],r["orbit"]):float(r["path_alpha2_at_pct"]) for r in
           read("thermo/exact_orbit_attribution.csv")}
    qha={r["orbit"]:float(r["full_at_pct"]) for r in read("qha/qha_attribution.csv")
         if r["spline"]=="not-a-knot"}
    fig,(a,b)=plt.subplots(1,2,figsize=(11.2,6.2),gridspec_kw={"width_ratios":[1,1.12]})
    c1,c2="#274e6e","#bf572c"
    labels=[]
    for i,(model,method) in enumerate((m,method) for m in models for method in ["MFA","MC"]):
        table=mfa if method=="MFA" else mc
        xa,xb=table["author",model],table["ours",model]
        a.plot([xa,xb],[i,i],color="#b4b9bd",lw=2)
        a.scatter(xa,i,s=52,marker="o",color=c1,zorder=3)
        a.scatter(xb,i,s=50,marker="s",color=c2,zorder=3)
        labels.append(("Статика" if model=="static" else model)+" · "+method)
    a.set(yticks=range(6),yticklabels=labels,ylim=(5.55,-.55),xlim=(19.0,23.05),
          xlabel="Состав α₂, ат.% Al")
    a.set_title("Правая граница сосуществования",loc="left",fontsize=12,pad=12)
    a.legend(handles=[Line2D([],[],marker="o",ls="",color=c1,label="Данные C46"),
                      Line2D([],[],marker="s",ls="",color=c2,label="Наша выборка")],
             loc="upper center",bbox_to_anchor=(.5,-.15),frameon=False,ncol=2,columnspacing=1.1)
    rows=["t334","t123","t136","p3","point"]
    for i,(model,color) in enumerate(zip(models,["#274e6e","#288273","#bf572c"])):
        y=np.arange(len(rows))+(i-1)*.22
        vals=[qha[nm] if model=="QHA" else fixed.get((model,nm),0.) for nm in rows]
        b.barh(y,vals,height=.19,color=color,label="Статика" if i==0 else model)
    b.set(yticks=range(len(rows)),yticklabels=rows[:-1]+["Одноузельный"],ylim=(4.5,-.5),
          xlabel="Вклад в сдвиг α₂, процентные пункты",xlim=(-.78,1.5))
    b.set_title("Влияние отдельных ECI в среднем поле",loc="left",fontsize=12,pad=12)
    b.axvline(0,color="#555",lw=.65)
    b.legend(loc="upper center",bbox_to_anchor=(.5,-.15),frameon=False,ncol=3,columnspacing=.9)
    for ax in (a,b):
        ax.grid(axis="x",alpha=.19)
        ax.set_axisbelow(True)
        ax.spines[["left","top","right"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
    fig.suptitle("Что меняет выбор 46 структур Ti–Al при 1000 K",x=.12,y=.97,ha="left",
                 fontsize=16,weight="bold")
    fig.text(.12,.015,"MFA использует один способ интерполяции объёма. Обе серии MC обработаны авторским методом, N = 5120.",
             fontsize=9,color="#444")
    fig.subplots_adjust(left=.12,right=.98,top=.82,bottom=.25,wspace=.50)
    fig.savefig(HERE/"comparison_1000K.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    run()
