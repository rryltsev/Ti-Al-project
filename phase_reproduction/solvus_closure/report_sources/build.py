"""Generate a standalone Russian XeLaTeX report from reviewed closure isotherms."""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
OLD = STUDY.parent / "solvus_1000"
STATUS = {"candidate":"К", "resolved":"П", "unresolved":"Н", "inadequate_coverage":"В"}
COLORS = {96:"muted",288:"ochre",768:"sky",1400:"teal",2304:"plum"}


def legacy():
    rows = []
    for filename in ("pilot96", "pilot288", "analysis768_free", "analysis1400_free"):
        data = json.loads((OLD/(filename+".json")).read_text(encoding="utf8"))
        for key, r in data.items():
            if "_cap0.4_axial1" not in key:
                continue
            q, h = r["equal_single_variant"], r["equal_height_separator"]
            ci=r.get("bootstrap",{}).get("intervals",{}).get("width",{}).get("quantile_025_50_975")
            rows.append(dict(T_K=1000.,N=r["N"],branch="Ti_rich",status="resolved",
                             x_alpha_pct=100*q["x_alpha"],x_alpha2_pct=100*q["x_alpha2"],
                             gap_low95_pp=100*ci[0] if ci else None,gap_high95_pp=100*ci[-1] if ci else None,
                             signed_gap_pp=100*q["width"],equal_height_barrier_kBT=h["barrier_kBT"]))
    assert len(rows)==4
    prior = STUDY.parent/"solvus_temperatures/analysis"
    for name in ("1273_n96", "1273_n288", "1273_n768_merged", "1273_n1400_merged"):
        r,=json.loads((prior/(name+".json")).read_text(encoding="utf8"))
        assert r["temperature_K"]==1273.15 and r["cap"]==.4 and r["axial_free"]
        result=r["coexistence"]
        assert result["equal_single_variant"] is None
        h=result["equal_height_separator"] or {}
        rows.append(dict(T_K=1273.15,N=r["N"],branch="Ti_rich",status="unresolved",
                         equal_height_barrier_kBT=h.get("barrier_kBT")))
    return rows


def val(row,key):
    return None if row.get(key) in ("",None) else float(row[key])


def fmt(x,d=2):
    return "---" if x is None else f"{x:.{d}f}".replace(".",",")


def table(rows):
    result=[r"\begin{center}\small\renewcommand{\arraystretch}{1.08}\setlength{\tabcolsep}{3.3pt}\begin{tabular}{@{}rrrrrrcc@{}}\toprule",
            r"\(T\), К & \(N\) & \(x_\alpha\) & \(x_{\alpha_2}\) & \(\Delta x\), 95\% & \(\beta\Delta F\) & Ti & Al\\\midrule"]
    for t in sorted(set(x["T_K"] for x in rows)):
        group=[x for x in rows if x["T_K"]==t]
        n=max(r["N"] for r in group)
        group=[x for x in group if x["N"]==n]
        l=next(x for x in group if x["branch"]=="Ti_rich")
        r=next((x for x in group if x["branch"]=="Al_rich"),{})
        usable=l["status"] in ("candidate","resolved")
        def measured(key):
            return val(l,key) if usable else None
        lo,hi=measured("gap_low95_pp"),measured("gap_high95_pp")
        ci="---" if lo is None or hi is None else "["+fmt(lo)+"; "+fmt(hi)+"]"
        result.append(" & ".join((fmt(t,0 if t.is_integer() else 2),str(n),
                     fmt(measured("x_alpha_pct")),fmt(measured("x_alpha2_pct")),ci,
                     fmt(measured("equal_height_barrier_kBT")),STATUS[l["status"]],
                     STATUS.get(r.get("status"),"---")))+r"\\")
    result.extend([r"\bottomrule\end{tabular}\end{center}",
        r"{\small Составы в ат.\% Al на Ti-богатой ветви приведены для наибольшей ячейки изотермы. Для \(\Delta x=x_{\alpha_2}-x_\alpha\) показан условный 95\%-й интервал в п.п. Обозначения ветвей следующие. П --- подтверждённое сосуществование, К --- слабый кандидат, Н --- пара не разрешена, В --- недостаточное покрытие полей. Прочерк означает отсутствие оценки.}\par"])
    return "\n".join(result)


def right_contacts(rows):
    accepted=[r for r in rows if r["branch"]=="Al_rich" and r["status"] in ("resolved","candidate")
              and val(r,"x_alpha_pct") is not None and r["N"]>=768]
    if not accepted:
        return ""
    items=[]
    for t in sorted(set(r["T_K"] for r in accepted)):
        group=[r for r in accepted if r["T_K"]==t]
        row=max(group,key=lambda r:(r["status"]=="resolved",r["N"]))
        label="подтверждённая пара" if row["status"]=="resolved" else "кандидат"
        items.append(fmt(t,0 if t.is_integer() else 2)+" К, \\(N="+str(row["N"])+r"\) --- "+
                     fmt(val(row,"x_alpha_pct"))+"/"+fmt(val(row,"x_alpha2_pct"))+" ("+label+")")
    return (r"Оценка Al-богатой ветви (\(x_\alpha/x_{\alpha_2}\), ат.\% Al) получена при "+"; ".join(items)+".")


def thermal_table():
    lines=[r"\begin{center}\small\setlength{\tabcolsep}{3.6pt}\begin{tabular}{@{}rrrrrrr@{}}\toprule",
           r"\(T_0\), К & \(N\) & \(T_*\), К & 95\% для \(T_*\), К & \(x_*\) & \(\ell_{\rm dis}\) & \(B_{\rm low}\)\\\midrule"]
    for name in (HERE/"thermal_files.txt").read_text(encoding="utf8").splitlines():
        if not name.strip():
            continue
        r=json.loads((STUDY/"analysis/thermal"/name.strip()).read_text(encoding="utf8"))
        assert r["Q_divider"]==.37
        if isinstance(r["source_T_K"],list):
            assert r["W"]==.40 and r["free_axial"]
            lo,_,hi=r["interval_95"][0]
            source=r"\(\sum T\)"
        else:
            assert r["cap"]==.40 and r["axial_free"]
            lo,_,hi=r["conditional_95"]["T_K"]
            t=float(r["source_T_K"])
            source=fmt(t,0 if t.is_integer() else 2)
        q=r["central"]
        lines.append(" & ".join((source,str(r["N"]),fmt(q["T_K"],1),
                     "["+fmt(lo,1)+"; "+fmt(hi,1)+"]",fmt(q["x_atpct"]),fmt(q["latent_meV_atom"],1),
                     fmt((q["profile"] or {}).get("lower_barrier"))))+r"\\")
    lines.extend([r"\bottomrule\end{tabular}\end{center}",
            r"{\small \(T_0\) --- температура прямых траекторий, \(\sum T\) обозначает совместный MBAR по изотермам, "
            r"\(x_*\) в ат.\% Al, \(\ell_{\rm dis}\) в мэВ/атом. "
            r"\(B_{\rm low}\) --- глубина минимума в единицах \(k_{\rm B}T\) от меньшего \(Q\)-пика при \(R=1\). "
            r"На рис.~\ref{fig:eqbarrier} поле отдельно подстроено до равных высот.}\par"])
    return "\n".join(lines)


def axis(lines, options):
    return "\n".join([r"\begin{center}\begin{tikzpicture}\begin{axis}[width=.91\linewidth,height=51mm,"
             r"axis x line*=bottom,axis y line*=left,ymajorgrids,y grid style={hair!70},"
             r"tick label style={font=\sffamily\footnotesize},label style={font=\sffamily\small},"
             r"legend columns=4,legend style={font=\sffamily\scriptsize,draw=none,column sep=3pt,"
             r"at={(.5,1.01)},anchor=south},enlarge y limits={upper,value=.13},"+
             options+"]",*lines,r"\end{axis}\end{tikzpicture}\end{center}"])


def order_profile():
    path=STUDY/"analysis/critical_order_profiles.csv"
    if not path.is_file():
        return ""
    with path.open(encoding="utf8",newline="") as f:
        data=list(csv.DictReader(f))
    lines=[]
    for n in sorted({int(r["N"]) for r in data}):
        points=[]
        for r in data:
            if n!=int(r["N"]) or float(r["T_K"])!=1245.:
                continue
            assert abs(float(r["cut_Q"])-.47)<1e-8
            y=val(r,"delta_F_kBT")
            ess=val(r,"local_time_ESS_approx")
            points.append((float(r["Q"]),y if y is not None and ess is not None and ess>=5 else None))
        if any(y is not None for _,y in points):
            color=COLORS[n]
            lines.append(r"\addplot["+color+r",line width=1pt,no marks,unbounded coords=jump] coordinates {"+
                " ".join(f"({x:.4f},{y:.5f})" if y is not None else f"({x:.4f},nan)" for x,y in sorted(points))+"};")
            lines.append(r"\addlegendentry{\(N="+str(n)+r"\)}")
    if not lines:
        return ""
    chart=axis(lines,r"height=40mm,xmin=0,xmax=1,ymin=0,ymax=5.5,xtick={0,.2,.4,.6,.8,1},"
                     r"xlabel={Параметр порядка \(Q\)},ylabel={\(-\ln[P(Q)/P_{\max}]\)}")
    return (chart+"\n"+r"\captionof{figure}{Профили при 1245~К в поле максимума отношения весов при \(q_c=0{,}47\). "
            r"Нуль выбран для наиболее вероятного \(Q\) каждого размера. "
            r"Локальное эффективное число кадров после поправки на корреляции не ниже пяти.}\label{fig:order}")


def traces(rows,category,xkey,ykey):
    series=[]
    for r in rows:
        if ykey!="signed_gap_pp" and r["branch"]!="Ti_rich":
            continue
        y=val(r,ykey)
        status=r["status"]
        if status=="unresolved" and ykey=="equal_height_barrier_kBT" and y is not None:
            status="only_peak"
        if status not in ("resolved","candidate","only_peak"):
            continue
        if y is not None:
            lok,hik=(("gap_low95_pp","gap_high95_pp") if ykey=="signed_gap_pp"
                     else ("barrier_low95_kBT","barrier_high95_kBT"))
            series.append((r[category],r[xkey],y,status,val(r,lok),val(r,hik)))
    lines=[]
    palette=["teal","sky","ochre","plum","rust","black","green!45!black","muted","blue!55!black"]
    markers=[("o","*"),("triangle","triangle*"),("square","square*"),("diamond","diamond*"),("pentagon","pentagon*")]
    sizes=sorted(set(r[xkey] for r in rows))
    for i,key in enumerate(sorted(set(d[0] for d in series))):
        color=COLORS.get(key,palette[i%len(palette)]) if category=="N" else palette[i%len(palette)]
        op,fill=markers[i%len(markers)]
        if category=="T_K":
            pts=sorted((x,y) for k,x,y,s,lo,hi in series if k==key and s!="only_peak")
            for (x1,y1),(x2,y2) in zip(pts,pts[1:]):
                if sizes.index(x2)==sizes.index(x1)+1:
                    lines.append(r"\addplot[no marks,forget plot,line width=.5pt,"+color+
                                 f"!65] coordinates {{({x1:.6f},{y1:.6f}) ({x2:.6f},{y2:.6f})}};")
        for k,x,y,s,lo,hi in series:
            if k==key and lo is not None and hi is not None:
                lines.append(r"\addplot[no marks,forget plot,line width=.45pt,"+color+
                             f"] coordinates {{({x:.6f},{lo:.6f}) ({x:.6f},{hi:.6f})}};")
        # Marker outline denotes the scientific status; hue identifies N or T.
        for flag,marker in (("resolved",fill),("candidate",op),("only_peak","x")):
            pts=sorted((x,y) for k,x,y,s,lo,hi in series if k==key and s==flag)
            if pts:
                lines.append(r"\addplot[only marks,forget plot,mark="+marker+",mark size=2.5pt,"
                             +color+",line width=.8pt] coordinates {"+
                             " ".join(f"({x:.6f},{y:.6f})" for x,y in pts)+"};")
        lines.append(r"\addlegendimage{only marks,mark="+fill+r",mark size=1.7pt,"+color+"}")
        unit=r"\(N="+str(int(key))+r"\)" if category=="N" else fmt(key,0 if float(key).is_integer() else 2)+" К"
        lines.append(r"\addlegendentry{"+unit+"}")
    return lines


def plots(rows):
    n_values=sorted(set(r["N"] for r in rows))
    for r in rows:
        r["area"]=r["N"]**(2/3)
    negative=any(r["status"] in ("resolved","candidate") and (val(r,"signed_gap_pp") or 0)<0 for r in rows)
    lowerbound="ymin=-1.5,ytick={-1,0,2,4,6,8}," if negative else "ymin=0,"
    lines=traces(rows,"N","T_K","signed_gap_pp")
    if negative:
        lines.insert(0,r"\addplot[hair,dashed,no marks,forget plot] coordinates {(983,0) (1286,0)};")
    upper=axis(lines,lowerbound+
              r"xmin=983,xmax=1286,xtick={1000,1050,1100,1150,1200,1250,1273.15},"
              r"xticklabels={1000,1050,1100,1150,1200,1250,1273},xlabel={Температура, К},"
              r"ylabel={\(\Delta x\), п.п. Al}")
    lower=axis(traces(rows,"T_K","area","equal_height_barrier_kBT"),
              "ymin=0,xtick={"+",".join(f"{n**(2/3):.6f}" for n in n_values)+"},"
              "xticklabels={"+",".join(str(n) for n in n_values)+"},"
              r"xlabel={Число атомов \(N\), шкала \(N^{2/3}\)},"
              r"ylabel={\(\beta\Delta F\)}")
    return "\n".join((upper,r"\captionof{figure}{Разность составов \(x_{\alpha_2}-x_\alpha\). Ti-богатой ветви соответствует положительный знак, Al-богатой --- отрицательный. Цвет обозначает размер ячейки. Закрашенный символ отмечает разрешённое сосуществование, полый --- слабый кандидат. Отрезки показывают условные блочные 95\%-е интервалы там, где они оценены. Пропуски оставлены пустыми.}",
                    r"\vspace{1mm}",lower,r"\captionof{figure}{Барьер на Ti-богатой стороне между равновысокими максимумами плотности по \(Q\). Цвет обозначает температуру. Линии соединяют соседние размеры, для которых оценены составы. Крестик обозначает пару максимумов \(Q\) без подтверждения отдельных составов при заданном отношении весов. Отсутствие оценки оставляет соответствующую позицию пустой.}\label{fig:eqbarrier}"))


def main(a):
    rr=[]
    summary=STUDY/"analysis/closure_summary.csv"
    if summary.exists():
        with summary.open(encoding="utf8",newline="") as f:
            for r in csv.DictReader(f):
                if float(r["W"])==.4 and r["axial_free"] in ("1","True","1.0"):
                    r.update(T_K=float(r["T_K"]),N=int(r["N"]))
                    rr.append(r)
    if not a.preview and (not summary.is_file() or not (STUDY/"analysis/reviewed_status.json").is_file()):
        raise RuntimeError("A final report requires reviewed status and the closure summary.")
    seen=set()
    for r in rr:
        k=(r["T_K"],r["N"],r["branch"])
        if k in seen:
            raise ValueError(f"Unresolved duplicate experimental series {k}")
        seen.add(k)
    rr.extend(r for r in legacy() if (r["T_K"],r["N"],r["branch"]) not in seen)
    body=(HERE/"template.tex").read_text(encoding="utf8").replace("@SUMMARYTABLE@",table(rr)).replace("@PLOTS@",plots(rr))
    body=body.replace("@RIGHTBRANCH@",right_contacts(rr))
    body=body.replace("@PROFILE@",order_profile())
    for key in ("finding","interpretation","thermal","limits"):
        body=body.replace("@"+key.upper()+"@",(HERE/(key+".tex")).read_text(encoding="utf8").strip())
    body=body.replace("@THERMALTABLE@",thermal_table())
    a.dest.mkdir(parents=True,exist_ok=True)
    (a.dest/"TiAl_solvus_closure.tex").write_text(body,encoding="utf8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview",action="store_true")
    parser.add_argument("--dest",type=Path,default=STUDY/"report")
    main(parser.parse_args())
