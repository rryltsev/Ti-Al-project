"""Summarize grid-level agreement and render three phase strips."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def table(name):
    with (HERE/name).open(encoding="utf8") as f:
        return list(csv.DictReader(f))


def main():
    rows=table("mfa_comparison_grid.csv")
    summary={}
    fig,axes=plt.subplots(1,3,figsize=(10.2,4.7),sharey=True)
    for ax,model in zip(axes,("static","HA","QHA")):
        bc="not-a-knot" if model=="QHA" else "fixed"
        sample=[r for r in rows if r["model"]==model and r["volume_spline"]==bc]
        author=[r for r in sample if r["source"]=="author"]
        present=[r for r in author if r["author_grid_listed"]=="True"]
        selected=[r for r in author if r["status"].startswith("interior")]
        if {r["T_K"] for r in present}!={r["T_K"] for r in selected}:
            raise ValueError("Disagreement in reported grid coverage")
        checks={k:max(abs(float(r["grid_minus_author_"+k+"_pp"])) for r in present)
                for k in ("alpha","alpha2")}
        summary[model]=dict(temperatures_considered=len(author),resolved_temperatures=len(selected),
                            max_abs_deviation_grid_pp=checks)
        for source,label,color,style in [("author","ECI авторов","#126966","-"),
                                         ("ours","Независимые ECI","#c35636","--")]:
            r=sorted((x for x in sample if x["source"]==source),key=lambda x:int(x["T_K"]))
            T=[int(x["T_K"]) for x in r]
            for index,k in enumerate(("alpha","alpha2")):
                c=[float(x["grid_"+k+"_at_pct"]) if x["status"].startswith("interior") else np.nan for x in r]
                ax.plot(c,T,color=color,ls=style,lw=1.8,label=label if index==0 else None)
        ax.set(xlim=(0,35),ylim=(585,1815),xlabel="Al, ат.%",title="Статика" if model=="static" else model)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.18)
    axes[0].set_ylabel("Температура, K")
    fig.suptitle("Бинодали α / α₂ в одной реализации среднего поля",fontsize=15,y=.98)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",bbox_to_anchor=(.51,.005),ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.09,1,.94))
    fig.savefig(HERE/"mfa_exact_vs_independent.png",dpi=180)
    plt.close(fig)
    for source in ("author","ours"):
        qp={bc:{int(r["T_K"]):r for r in rows if r["source"]==source and r["model"]=="QHA"
                and r["volume_spline"]==bc} for bc in ("natural","not-a-knot")}
        n,k=qp["natural"],qp["not-a-knot"]
        both=[t for t in sorted(n) if n[t]["status"].startswith("interior") and k[t]["status"].startswith("interior")]
        diff=np.array([[float(n[t]["grid_"+side+"_at_pct"])-float(k[t]["grid_"+side+"_at_pct"])
                        for side in ("alpha","alpha2")] for t in both])
        summary[source+"_QHA_spline"]=dict(shared_resolved_temperatures=len(both),
            temperatures_with_grid_difference=[t for t,delta in zip(both,diff) if np.max(abs(delta))>1e-8],
            max_abs_difference_pp=abs(diff).max(axis=0).tolist(),
            status_disagreements=[t for t in n if n[t]["status"]!=k[t]["status"]])
    exact=table("mfa_1000K.csv")
    ref=[r for r in exact if r["source"]=="author" and r["volume_spline"]!="natural"]
    summary["max_refined_difference_pp"]=max(abs(float(r["refined_minus_author_"+side+"_pp"]))
                                              for r in ref for side in ("alpha","alpha2"))
    (HERE/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
