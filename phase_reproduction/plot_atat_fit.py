"""Plot fitted pair/triplet ECIs and per-cell residuals against the paper's static S1."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("eci_csv",type=Path)
    p.add_argument("design",type=Path)
    p.add_argument("residuals",type=Path)
    p.add_argument("output",type=Path)
    args = p.parse_args()
    c = read(args.eci_csv)[2:]
    fig = plt.figure(figsize=(11.4,6.6),layout="constrained")
    gs = fig.add_gridspec(2,2,height_ratios=[1,1.18])
    for ax,rows,title in [(fig.add_subplot(gs[0,0]),c[:7],"Парные взаимодействия"),
                          (fig.add_subplot(gs[0,1]),c[7:],"Тройные взаимодействия")]:
        x = np.arange(len(rows))
        for off,key,color,marker,label in [(-.1,"ATAT_fit_eV","#146a77","o","Наши ECI"),
                                            (.1,"paper_S1_eV","#b44929","x","Таблица S1")]:
            ax.scatter(x+off,[1000*float(r[key]) for r in rows],color=color,marker=marker,
                       s=45,label=label,zorder=3)
        ax.set_xticks(x,[r["orbit"].replace("t222_","222·").removeprefix("t") for r in rows],
                      rotation=35 if len(rows)>8 else 0)
        ax.axhline(0,color=".60",lw=.8)
        ax.set_ylabel("J, мэВ на кластер")
        ax.set_title(title)
    fig.axes[0].legend(frameon=False,fontsize=9)
    ax = fig.add_subplot(gs[1,:])
    data = {r["id"]:r for r in read(args.design)}
    entries = [r for r in read(args.residuals) if abs(float(r["volume_ratio"])-1)<1e-8]
    for group,color,marker,label in [("train","#146a77","o","Обучение"),
                                     ("control","#b44929","D","Контроль")]:
        selected = [r for r in entries if data[r["id"]]["split"]==group]
        ax.scatter([100*int(data[r["id"]]["N_Al"])/int(data[r["id"]]["N"]) for r in selected],
                   [1000*float(r["static"]) for r in selected],
                   color=color,s=40,marker=marker,label=label,alpha=.8)
    ax.axhline(0,color=".55",lw=.8)
    ax.set(ylabel="CE − MTP, мэВ/атом",xlabel="Содержание Al, ат.%",
           title="Отклонение от прямой MTP-энергии после внутренней релаксации")
    ax.legend(frameon=False,ncol=2)
    for a in fig.axes:
        a.spines[["top","right"]].set_visible(False)
        a.grid(axis="y",color=".92",zorder=0)
    fig.savefig(args.output,dpi=180)
    plt.close(fig)


if __name__=="__main__":
    main()
