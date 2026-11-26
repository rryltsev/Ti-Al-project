import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
ANALYSIS = HERE / "analysis"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def coords(points):
    return " ".join(f"({float(x):.6f},{float(y):.6f})" for x,y in points)


def alloy_diagram(contacts):
    liang = [r for r in rows(HERE/"calphad/liang_contacts.csv") if r["model"]=="binary10" and r["branch"]=="Ti-rich"]
    experiment = rows(HERE/"calphad/kainuma_table1.csv")
    c46 = {}
    for r in rows(HERE.parent/"author_comparison/mc/mc_same_windows.csv"):
        t = int(r["T_K"])
        if r["source"]=="author" and r["model"]=="QHA" and r["author_resolved"]=="True":
            if t not in c46 or int(r["N"])>int(c46[t]["N"]):
                c46[t]=r
    out=[r"""\begin{center}\begin{tikzpicture}
\begin{axis}[width=.93\linewidth,height=72mm,xmin=565,xmax=1470,ymin=2,ymax=29,
  xlabel={\(T\), К},ylabel={Al, ат.\%},xtick={600,800,1000,1200,1400},
  xticklabels={600,800,1000,1200,1400},ytick={5,10,15,20,25},
  tick label style={font=\sffamily\footnotesize},label style={font=\sffamily\small},
  axis x line*=bottom,axis y line*=left,ymajorgrids,y grid style={linegray!65},
  legend style={font=\sffamily\footnotesize,draw=none,fill=white,at={(.5,-.27)},anchor=north,legend columns=2,column sep=5pt}]"""]
    for branch, f in enumerate(["hcp_Al_at_pct","D019_Al_at_pct"]):
        style = ",forget plot" if branch else ""
        out.append(r"\addplot[teal,line width=1.05pt,mark=*,mark size=1.4pt"+style+"] coordinates {"
                   +coords([(r["T_K"],r[f]) for r in liang])+"};")
        if not branch:
            out.append(r"\addlegendentry{CALPHAD, Liang}")
    for branch,f in enumerate(["author_alpha","author_alpha2"]):
        style = ",forget plot" if branch else ""
        out.append(r"\addplot[muted,dashed,mark=triangle,mark size=1.9pt"+style+"] coordinates {"
                   +coords([(t,c46[t][f]) for t in sorted(c46)])+"};")
        if not branch:
            out.append(r"\addlegendentry{Квантовый C46, MC}")
    for branch,(val,err) in enumerate([("alpha_at_pct","alpha_reported_error_at_pct"),
                                       ("alpha2_at_pct","alpha2_reported_error_at_pct")]):
        style=",forget plot" if branch else ""
        points=" ".join(f"({r['T_K']},{r[val]}) +- (0,{r[err]})" for r in experiment)
        out.append(r"\addplot[ink,only marks,mark=square,mark size=2pt"+style+
                   ",error bars/.cd,y dir=both,y explicit] coordinates {"+points+"};")
        if not branch:
            out.append(r"\addlegendentry{Эксперимент, Kainuma}")
    out.append(r"\addplot[rust,only marks,mark=diamond*,mark size=2.9pt] coordinates {"
               +coords([(t,x) for t,pair in contacts for x in pair])+"};")
    out.append(r"""\addlegendentry{Прямой MTP, свободное \(c/a\)}
\end{axis}\end{tikzpicture}
\captionof{figure}{Ti-богатые ветви сольвуса. Для CALPHAD и C46 линии соединяют
рассчитанные изотермы. Каждый ромб соответствует одной фазе при
температуре, где прямой расчёт разрешил две отдельные фазы. Планки у квадратов
передают точность из экспериментальной таблицы.}
\end{center}""")
    return "\n".join(out)


def order_diagnostic(paths):
    out=[r"""\begin{center}\begin{tikzpicture}
\begin{axis}[width=.92\linewidth,height=60mm,xmin=17.5,xmax=35,ymin=0,ymax=1.05,
  xlabel={Средний состав, ат.\% Al},ylabel={\(\langle Q\rangle\)},xtick={18,20,22,24,26,28,30,32,34},
  axis x line*=bottom,axis y line*=left,ymajorgrids,y grid style={linegray!65},
  tick label style={font=\sffamily\footnotesize},label style={font=\sffamily\small},
  legend style={font=\sffamily\footnotesize,draw=none,fill=white,at={(.98,.97)},anchor=north east}]
\path[fill=teal!8] (axis cs:19.5,0) rectangle (axis cs:23,1.03);"""]
    styles=["muted,dotted","rust,dashed","black!65,dashdotted","teal,solid"]
    for i,path in enumerate(paths):
        data,=json.loads(path.read_text(encoding="utf-8"))
        assert data["cap"]==.40 and data["axial_free"] and abs(data["temperature_K"]-1273.15)<.001
        points = [(100*r["x_Al"],r["mean_Q"] if r["effective_frames_with_time_correlation"]>=10 else float("nan"))
                  for r in data["scan"]["grid"] if 17<=100*r["x_Al"]<=35.5]
        if not any(y == y for _,y in points):
            continue
        out.append(r"\addplot["+styles[i]+",line width=1.1pt,no marks,unbounded coords=jump] coordinates {"+coords(points)+"};")
        out.append(r"\addlegendentry{\(N="+str(data["N"])+r"\)}")
    out.append(r"""\end{axis}\end{tikzpicture}
\captionof{figure}{Химический порядок при 1273,15 К на разных размерах,
\(W<0{,}40\), свободное \(c/a\). Тонированная полоса отмечает
экспериментальный интервал 19,5--23,0 ат.\% Al. Показаны точки с оценкой
эффективного числа кадров после поправки на временную корреляцию не менее 10.
Положительное среднее \(Q\) в конечной ячейке само по себе совместимо с
однофазным раствором.}
\end{center}""")
    return "\n".join(out)
