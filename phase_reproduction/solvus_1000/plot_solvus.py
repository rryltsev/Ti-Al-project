"""Plot finite-cell compositions, conditional intervals and optional order profiles.

Example: python plot_solvus.py "results/analysis*.json" --profiles --output solvus.png
The input is the group dictionary from analysis.py. An unresolved group can also
be identified explicitly with --unresolved GLOB after reviewing its diagnostics.
"""
import argparse
import glob
import json
import math
from fnmatch import fnmatchcase
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import ScaledTranslation


HERE = Path(__file__).resolve().parent
INK = "#263744"
GRID = "#dce2e5"
GRAY = "#596b77"
COLORS = ("#00796f", "#bb501c", "#486aa3", "#865b8b")
MARKERS = ("o", "s", "^", "D")


def load_groups(patterns, include, unresolved):
    groups = {}
    for pattern in patterns:
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            raise ValueError("No input matches " + pattern)
        for path in files:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            for key, row in data.items():
                if not isinstance(row, dict) or not {"N","cap","axial_free","status"} <= row.keys():
                    continue
                if include and not any(fnmatchcase(key,p) for p in include):
                    continue
                if key in groups:
                    raise ValueError("Group appears more than once: " + key)
                fit = row.get("equal_single_variant")
                if row["status"] not in ("candidate","resolved","accepted") or any(
                        fnmatchcase(key,p) for p in unresolved):
                    fit = None
                if fit:
                    a,b = [float(fit[k]) for k in ("x_alpha","x_alpha2")]
                    if not 0 <= a < b <= 1:
                        raise ValueError("Invalid contacts in " + key)
                groups[key] = (row, fit)
    if not groups:
        raise ValueError("No selected groups with the analysis.py schema")
    signatures = [(r["N"],r["cap"],r["axial_free"]) for r,_ in groups.values()]
    if len(set(signatures)) != len(signatures):
        raise ValueError("Select a single cell shape per N, cap and c/a setting using --include")
    return list(groups.values())


def number(x, digits):
    return f"{x:.{digits}f}".replace(".",",")


def clean_axis(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["bottom","left"]].set_color("#8c9ba3")
    ax.tick_params(direction="out",length=3,color="#8c9ba3")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: f"{x:g}".replace(".",",")))
    ax.grid(axis="y", color=GRID, linewidth=.65)
    ax.set_axisbelow(True)


def compositions(ax, phase, rows, colors, ref, sizes):
    xmax = max(1000/n for n in sizes)
    ax.set_xlim(-.07*xmax,1.10*xmax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5,steps=[1,2,2.5,5,10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: number(x,2).rstrip("0").rstrip(",")))
    ax.set_xlabel(r"$10^3/N$")
    top = ax.secondary_xaxis("top")
    top.set_xticks([1000/n for n in sizes], [str(n) for n in sizes])
    top.set_xlabel("Число узлов $N$", labelpad=4, color=GRAY)
    top.spines["top"].set_visible(False)
    top.tick_params(length=0,colors=GRAY,pad=1,labelsize=9)
    if len(sizes)>2:
        for tick in top.get_xticklabels():
            tick.set_rotation(48)
            tick.set_horizontalalignment("left")
    clean_axis(ax)
    central = ref["Al_at_percent"]
    low,high = ref["Al_at_percent_conditional_95"]
    ax.axhspan(low,high,color="#d5dce0",zorder=0)
    ax.axhline(central,color=GRAY,lw=1.3,ls=(0,(4,3)),zorder=1)
    y_values = [low,high]
    caps = sorted({r["cap"] for r,_ in rows})
    for row,fit in rows:
        cap = row["cap"]
        color,marker = colors[cap]
        offset = 6*(caps.index(cap)-(len(caps)-1)/2)
        shift = ScaledTranslation(offset/72,0,ax.figure.dpi_scale_trans)
        x = 1000/row["N"]
        if not fit:
            ax.plot(x,.045,marker="x",ms=7,mew=1.7,color=color,ls="",
                    transform=ax.get_xaxis_transform()+shift,clip_on=False,zorder=3)
            continue
        val = 100*fit["x_"+phase]
        y_values.append(val)
        trans = ax.transData+shift
        interval = row.get("bootstrap",{}).get("intervals",{}).get("x_"+phase,{})
        if "quantile_025_50_975" in interval:
            lo,_,hi = [100*v for v in interval["quantile_025_50_975"]]
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise ValueError("Invalid bootstrap interval")
            ax.plot([x,x],[lo,hi],color=color,lw=1.5,transform=trans)
            ax.plot([x,x],[lo,hi],color=color,marker="_",ms=7,ls="",transform=trans)
            y_values.extend([lo,hi])
        ax.plot([x],[val],color=color,marker=marker,mfc="white",mew=1.8,ms=6,
                ls="",transform=trans,zorder=4)
    lo,hi = min(y_values), max(y_values)
    span = max(hi-lo,1.2)
    ax.set_ylim((lo+hi)/2-.69*span,(lo+hi)/2+.69*span)
    ax.set_ylabel("Al, ат. %")


def order_panel(ax, rows, colors):
    nmax = max(r["N"] for r,_ in rows)
    ax.set(xlabel="Параметр порядка $Q$", ylabel="Плотность $P(Q)$",xlim=(0,1))
    clean_axis(ax)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: f"{x:g}".replace(".",",")))
    lines = []
    for row,fit in sorted(rows,key=lambda item:item[0]["cap"]):
        if row["N"] != nmax:
            continue
        if fit and fit.get("order_profile"):
            profile = fit["order_profile"]
            xs,ys = profile["Q"],profile["density"]
            label = f"W < {number(row['cap'],2)}, " + r"$\mu^*$"
            style = "-"
        elif row.get("reference_profile"):
            profile = row["reference_profile"]
            xs,ys = profile["q"],profile["density"]
            label = f"W < {number(row['cap'],2)}, " + r"$\mu$=" + number(row["mu_reference_eV"],3) + " эВ"
            style = "--"
        else:
            continue
        line, = ax.plot(xs,ys,color=colors[row["cap"]][0],lw=1.7,ls=style,label=label)
        lines.append(line)
    if lines:
        ax.legend(handles=lines,frameon=False,fontsize=8,loc="upper left")
        ax.set_ylim(bottom=0)
    else:
        ax.text(.5,.5,"Профиль не сохранён",ha="center",va="center",color=GRAY,transform=ax.transAxes)
    return nmax


def plot(rows, reference, output, profiles):
    caps = sorted({r["cap"] for r,_ in rows})
    if len(caps)>len(COLORS):
        raise ValueError("At most four cage limits can be compared in one panel")
    colors = dict(zip(caps, zip(COLORS,MARKERS)))
    axial = sorted({int(r["axial_free"]) for r,_ in rows},reverse=True)
    ncols = 3 if profiles else 2
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.labelcolor":INK,
                         "text.color":INK,"xtick.color":INK,"ytick.color":INK,
                         "pdf.fonttype":42,"svg.fonttype":"none","axes.unicode_minus":True})
    fig, axes = plt.subplots(len(axial),ncols,squeeze=False,figsize=(4*ncols,2.95*len(axial)+1.6),
                             layout="constrained")
    tags = ["α","α₂"]
    for index,flag in enumerate(axial):
        selected = [(r,f) for r,f in rows if int(r["axial_free"])==flag]
        sizes = sorted({r["N"] for r,_ in selected},reverse=True)
        mode = "свободное $c/a$" if flag else "фиксированное $c/a$"
        for col,phase in enumerate(("alpha","alpha2")):
            ax = axes[index,col]
            compositions(ax,phase,selected,colors,reference["phase"][phase],sizes)
            ax.set_title(f"{tags[col]}  ·  {mode}",loc="left",fontsize=12,y=1.30,pad=0,fontweight="bold")
        if profiles:
            ax = axes[index,2]
            nmax = order_panel(ax,selected,colors)
            ax.set_title(f"Распределение порядка  ·  $N={nmax}$",loc="left",fontsize=12,y=1.30,pad=0,fontweight="bold")
    handles = [Line2D([],[],ls="",marker=m,color=c,mfc="white",mew=1.8,label=f"W < {number(cap,2)}")
               for cap,(c,m) in colors.items()]
    handles.append(Line2D([],[],ls="--",color=GRAY,mfc="none",label="C46, 5120 узлов (95%)"))
    if any(not f for _,f in rows):
        handles.append(Line2D([],[],ls="",marker="x",color=GRAY,label="Две фазы не разрешены"))
    fig.suptitle("Границы α и α₂ в Ti–Al при 1000 K и нулевом давлении",fontsize=15,fontweight="bold")
    intervals = any(f and r.get("bootstrap",{}).get("intervals") for r,f in rows)
    legend_title = "Составы фаз в конечных ячейках"
    if intervals:
        legend_title += "  ·  отрезки — 95%-е интервалы блочного бутстрепа"
    fig.legend(handles=handles,loc="outside lower center",frameon=False,ncol=min(4,len(handles)),
               title=legend_title,title_fontsize=9)
    fig.savefig(output,dpi=220,bbox_inches="tight",pad_inches=.14)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",nargs="+",help="analysis.py JSON files or quoted glob patterns")
    parser.add_argument("--include",action="append",default=[],metavar="GROUP_GLOB")
    parser.add_argument("--unresolved",action="append",default=[],metavar="GROUP_GLOB")
    parser.add_argument("--reference",type=Path,default=HERE/"c46_reference.json")
    parser.add_argument("--profiles",action="store_true")
    parser.add_argument("--output",type=Path,required=True,help="PNG, SVG or PDF path")
    args=parser.parse_args()
    plot(load_groups(args.inputs,args.include,args.unresolved),
         json.loads(args.reference.read_text(encoding="utf-8")),args.output,args.profiles)


if __name__=="__main__":
    main()
