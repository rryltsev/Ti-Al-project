"""Show MFA attribution of the 1000 K alpha2 shift and training-cell sensitivity."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent


def rows(path):
    with path.open(newline="",encoding="utf8") as f:
        return list(csv.DictReader(f))


def main():
    attr = rows(HERE/"mfa/results/orbit_attribution.csv")
    loo = rows(HERE/"robustness/leave_one_out_contributions.csv")
    modes = ("static","HA")
    au = {(r["mode"],r["orbit"]):float(r["path_total_at_pct"])
          for r in attr if r["branch"]=="alpha2"}
    names = sorted({name for _,name in au}, key=lambda nm:-max(abs(au[m,nm]) for m in modes))
    col = {"static":"#205e84","HA":"#ba582c"}
    fig,axes = plt.subplots(1,2,figsize=(10.6,6.6),gridspec_kw={"width_ratios":[1.25,1]})
    left,right = axes
    for m,offset in zip(modes,[-.17,.17]):
        y = np.arange(len(names))+offset
        left.barh(y,[au[m,n] for n in names],height=.27,color=col[m],zorder=2)
    left.set(yticks=np.arange(len(names)),yticklabels=names,ylim=(len(names)-.5,-.65),
             xlim=(-.77,1.53),xlabel="Вклад в сдвиг границы α₂, ат.% Al")
    left.set_title("Замена ECI статьи на наши",fontsize=12,loc="left",pad=9)
    shortlist = ["t334","t123","t136","p3","t114"]
    for m,offset in zip(modes,[-.11,.11]):
        for i,n in enumerate(shortlist):
            values = [float(r["contribution_at_pct"]) for r in loo
                      if r["model"]==m and r["orbit"]==n and r["omitted"]!="none"]
            y = i+offset
            right.plot([min(values),max(values)],[y,y],color=col[m],lw=2)
            right.scatter([au[m,n]],[y],color=col[m],s=29,zorder=3)
    right.set(yticks=range(len(shortlist)),yticklabels=shortlist,ylim=(4.6,-.6),
              xlabel="Вклад в сдвиг границы α₂, ат.% Al",xlim=(-1.85,2.58))
    right.set_title("Исключение одной структуры из 37",fontsize=12,loc="left",pad=9)
    for a in axes:
        a.axvline(0,color="#444",lw=.75)
        a.grid(axis="x",color="#ccc",alpha=.45,lw=.6)
        a.spines[["right","top","left"]].set_visible(False)
        a.tick_params(axis="y",length=0,labelsize=9)
    legend = [Line2D([],[],color=col[m],lw=5,label=label) for m,label in zip(modes,["Статика","HA"])]
    fig.legend(handles=legend,loc="upper left",bbox_to_anchor=(.067,.911),ncol=2,frameon=False)
    fig.suptitle("Влияние кластерных взаимодействий на границу α₂ при 1000 K",
                 x=.075,ha="left",y=.968,fontsize=16,weight="semibold")
    fig.text(.075,.04,"Среднее поле, опорный объём. Справа точки показывают подгонку по всем 37 структурам,\n"
             "отрезки — диапазон вкладов при поочерёдном исключении каждой структуры.",size=9,color="#444")
    fig.subplots_adjust(left=.085,right=.975,bottom=.16,top=.8,wspace=.36)
    fig.savefig(HERE/"alpha2_eci_attribution.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    main()
