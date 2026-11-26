"""Plot contributions to each 1000 K contact and the five distinguishable MFA classes."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def read(path):
    with path.open(encoding="utf8") as file:
        return list(csv.DictReader(file))


def main(out):
    rows = read(out/"orbit_attribution.csv")
    classes = read(out/"mfa_orbit_classes.csv")
    cols = [("static","alpha"), ("static","alpha2"), ("HA","alpha"), ("HA","alpha2")]
    orbits = [r["orbit"] for r in rows if (r["mode"],r["branch"])==cols[0]]
    keys = [r["orbits"] for r in classes if (r["mode"],r["branch"])==cols[0]]
    fig, (left,right) = plt.subplots(1,2,figsize=(12.3,8.7),width_ratios=(1,1.25))
    norm = TwoSlopeNorm(0,vmin=-1.5,vmax=1.5)

    def matrix(ax, data, key, labels, title):
        lookup = {(r[key],r["mode"],r["branch"]):float(r["path_total_at_pct"]) for r in data}
        vals = np.array([[lookup[(s,*col)] for col in cols] for s in labels])
        ax.imshow(vals,cmap="RdBu_r",norm=norm,aspect="auto")
        for (i,k),val in np.ndenumerate(vals):
            ax.text(k,i,f"{val:+.2f}",ha="center",va="center",fontsize=9,
                    color="white" if abs(val)>.85 else "#171717")
        ax.set_yticks(range(len(labels)),[s.replace("t222_1","t222₁").replace("t222_2","t222₂") for s in labels])
        ax.set_xticks(range(4),["static\nα","static\nα₂","HA\nα","HA\nα₂"])
        ax.xaxis.tick_top()
        ax.tick_params(length=0,pad=5)
        ax.axvline(1.5,color="white",lw=3)
        ax.set_title(title,loc="left",fontweight="bold",fontsize=13,pad=15)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return vals.sum(axis=0)

    sums = matrix(left,rows,"orbit",orbits,"По орбитам кластерного разложения")
    left.axhline(6.5,color="white",lw=2.5)
    matrix(right,classes,"orbits",keys,"По различимым в MFA классам")
    right.set_yticklabels(["p1 p2 p5\np6 p7","p3 p4","t112 t117 t125\nt222₁ t222₂ t226","t114 t123\nt136 t247","t334"])
    fig.subplots_adjust(left=.076,right=.98,bottom=.07,top=.82,wspace=.48)
    right.set_position([.65,.43,.32,.39])
    fig.suptitle("Сдвиг границ α и α₂ при 1000 K",x=.076,ha="left",y=.98,fontsize=21,fontweight="bold")
    fig.text(.076,.923,"Собственные ECI минус реконструкция статьи, процентные пункты Al",fontsize=12)
    fig.text(.65,.35,"Сумма по всем орбитам",fontsize=13,fontweight="bold")
    fig.text(.65,.30,f"static    α  {sums[0]:+.3f}     α₂  {sums[1]:+.3f}",fontsize=13)
    fig.text(.65,.26,f"HA         α  {sums[2]:+.3f}     α₂  {sums[3]:+.3f}",fontsize=13)
    fig.text(.65,.16,"В одном классе корреляции после деления\nна кратность совпадают при любых q и r\nдля заполнений подрешёток q, r, r, r.",fontsize=10,linespacing=1.4)
    fig.savefig(out/"attribution_1000K.png",dpi=175,bbox_inches="tight",pad_inches=.24)
    plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parent/"results")
    main(p.parse_args().out)
