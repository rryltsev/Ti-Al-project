"""Assemble a standalone XeLaTeX supplement from independently simulated wall ensembles."""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
COLORS = {.35: "tight", .40: "loose"}


def n(value, places=2):
    return "---" if value in (None, "") else f"{float(value):.{places}f}".replace(".", ",")


def interval(bounds, places=2):
    return "[" + n(bounds[0], places) + "; " + n(bounds[-1], places) + "]"


def tabular(head, rows, columns):
    return "\n".join([r"\begin{center}\small\setlength{\tabcolsep}{3.6pt}\begin{tabular}{@{}"+columns+r"@{}}\toprule",
                      head+r"\\\midrule", *(" & ".join(row)+r"\\" for row in rows),
                      r"\bottomrule\end{tabular}\end{center}"])


def pooled_table(diff):
    rows = []
    for name in diff["sources"]:
        result = json.loads((STUDY / name).read_text(encoding="utf8"))
        central = result["central"]
        bounds = result["interval_95"][result["bootstrap_columns"].index("T_K")]
        rows.append([n(result["W"]), n(central["T_K"], 1), interval(bounds, 1),
                     n(central["mu_eV"], 4), n(central["x_atpct"]),
                     n(central["jump_Q"]), n(central["latent_meV_atom"], 1)])
    return (r"\subsection*{Вершина на 2304 атомах}"+ "\n" +
            tabular(r"\(w\) & \(T_*\), К & 95\% для \(T_*\), К & \(\Delta\mu_*\), эВ & \(x_*\), ат.\% Al & \(\Delta Q\) & \(\ell\), мэВ/атом",
                    rows, "rrrrrrr")+"\n"+
            r"{\small \(\Delta Q=\langle Q\rangle_{\alpha_2}-\langle Q\rangle_\alpha\). "
            r"При совпадении составов \(\ell=\big[\langle E-\Delta\mu k\rangle_\alpha-\langle E-\Delta\mu k\rangle_{\alpha_2}\big]/N\) даёт теплоту разупорядочения при \(p=0\).}\par")


def score_plot(rows):
    present = [r for r in rows if r["Smax"] and r["S_low95"] and r["S_high95"]]
    if not present:
        raise ValueError("No direct maxima with bootstrap intervals")
    temps = [float(r["T_K"]) for r in present]
    lo, hi = min(temps), max(temps)
    margin = max(.4, (hi-lo)*.04)
    out = [r"\begin{center}\begin{tikzpicture}\begin{axis}[width=.94\linewidth,height=58mm,"
           r"axis x line*=bottom,axis y line*=left,ymajorgrids,y grid style={hair},"
           r"tick label style={font=\sffamily\small},scaled x ticks=false,"
           r"legend columns=2,legend style={font=\small,draw=none,at={(.5,1.01)},anchor=south},"
           r"xlabel={Температура, К},ylabel={\(S=\max_{\Delta\mu}\ln R\)},"
           + f"xmin={lo-margin},xmax={hi+margin}"+"]",
           r"\addplot[muted,dashed,no marks,forget plot] coordinates {"
           +f"({lo-margin},0) ({hi+margin},0)"+"};"]
    for wall, color in COLORS.items():
        observations = [r for r in present if float(r["W"]) == wall]
        for r in observations:
            t=float(r["T_K"])
            out.append(r"\addplot[no marks,forget plot,"+color+"] coordinates {"
                       +f"({t},{float(r['S_low95'])}) ({t},{float(r['S_high95'])})"+"};")
            mark = "o" if r["max_on_field_edge"].lower()=="true" else "*"
            out.append(r"\addplot[only marks,forget plot,mark="+mark+",mark size=2.2pt,"+color
                       +"] coordinates {"+f"({t},{float(r['Smax'])})"+"};")
        if observations:
            out.extend([r"\addlegendimage{only marks,mark=*,mark size=1.8pt,"+color+"}",
                        r"\addlegendentry{\(w="+n(wall)+r"\)}"])
    out.append(r"\end{axis}\end{tikzpicture}\end{center}")
    out.append(r"\captionof{figure}{Максимум логарифма отношения весов на прямой сетке полей при \(q_c=0{,}37\). "
               r"Заполненный символ отмечает внутренний максимум, полый --- достижение края проверенных полей. "
               r"Вертикальные отрезки дают 95\%-е интервалы независимого блочного анализа каждой изотермы.}")
    return "\n".join(out)


def paired_figures(rows, profiles):
    scores = score_plot(rows)
    if not profiles.is_file():
        return scores
    with profiles.open(encoding="utf8",newline="") as f:
        profile = [r for r in csv.DictReader(f) if float(r["T_K"])==1243 and r["field_kind"]=="matched_3.545"]
    if not all(any(float(r["W"])==wall for r in profile) for wall in COLORS):
        return scores
    out = [r"\begin{center}\begin{tikzpicture}\begin{axis}[width=.97\linewidth,height=53mm,"
           r"axis x line*=bottom,axis y line*=left,ymajorgrids,y grid style={hair},"
           r"tick label style={font=\sffamily\footnotesize},label style={font=\small},"
           r"legend columns=2,legend style={font=\footnotesize,draw=none,at={(.5,1.01)},anchor=south},"
           r"xmin=0,xmax=1,ymin=0,xlabel={Параметр порядка \(Q\)},ylabel={\(-\ln[P(Q)/P_{\max}]\)}]"]
    for wall, color in COLORS.items():
        points = []
        for r in sorted((r for r in profile if float(r["W"])==wall),key=lambda r:float(r["Q"])):
            y = r["F_kBT"] if r["F_kBT"] and float(r["local_time_ESS_approx"] or 0.)>=5 else "nan"
            points.append(f"({float(r['Q']):.4f},{y})")
        out.append(r"\addplot[no marks,line width=1pt,unbounded coords=jump,"+color+
                   "] coordinates {"+" ".join(points)+"};")
        out.append(r"\addlegendentry{\(w="+n(wall)+r"\)}")
    out.append(r"\end{axis}\end{tikzpicture}\end{center}")
    out.append(r"\captionof{figure}{Свободная энергия по \(Q\) при 1243~К и общем поле 3,545~эВ. "
               r"Каждая кривая отсчитана от своего минимума. Показаны значения с локальным эффективным "
               r"числом кадров от пяти при сглаживании с \(\sigma_Q=0{,}04\).}")
    scores = scores.replace(".94\\linewidth,height=58mm",".97\\linewidth,height=53mm")
    scores = scores.replace(r"tick label style={font=\sffamily\small}",
                            r"tick label style={font=\sffamily\footnotesize},label style={font=\small}")
    return (r"\begin{minipage}[t]{.49\linewidth}"+scores+
            r"\end{minipage}\hfill\begin{minipage}[t]{.49\linewidth}"+"\n".join(out)+r"\end{minipage}")


def field_curves(rows, path, require_final=True):
    with path.open(encoding="utf8",newline="") as f:
        curves=list(csv.DictReader(f))
    out=[]
    for wall,color in COLORS.items():
        data=sorted((r for r in curves if float(r["W"])==wall),key=lambda r:float(r["mu_eV"]))
        sim=[r for r in data if r["is_simulated_field"].lower() in ("true","1","1.0")]
        if len(sim)<2:
            raise ValueError(f"Dense 1243 K scores or simulated-field flags missing for {wall}")
        if require_final and not all("final" in str(r["source"]).lower() for r in data):
            raise ValueError(f"Nonfinal 1243 K field scores for {wall}")
        zero_range=(min(float(r["mu_eV"]) for r in sim),max(float(r["mu_eV"]) for r in sim))
        out.append(r"\begin{minipage}[t]{.49\linewidth}\begin{center}\begin{tikzpicture}\begin{axis}["
                   r"width=.97\linewidth,height=55mm,axis x line*=bottom,axis y line*=left,ymajorgrids,"
                   r"y grid style={hair},scaled x ticks=false,tick label style={font=\sffamily\footnotesize},"
                   r"label style={font=\small},title style={font=\sffamily\bfseries\small},"
                   r"xticklabel style={/pgf/number format/fixed,/pgf/number format/precision=3},"
                   r"xlabel={\(\Delta\mu\), эВ},ylabel={\(D(1243\,{\rm K},\Delta\mu)\)},title={\(w="+n(wall)+r"\)}]")
        out.append(r"\addplot[muted,dashed,no marks] coordinates {"+f"({zero_range[0]},0) ({zero_range[1]},0)"+"};")
        vertices=" ".join(f"({float(r['mu_eV']):.7f},{float(r['score']):.6f})" for r in data)
        out.append(r"\addplot[no marks,line width=1pt,"+color+"] coordinates {"+vertices+"};")
        points=[]
        for r in sim:
            mu=float(r["mu_eV"])
            points.append(f"({mu:.7f},{float(r['score']):.6f})")
            if r["lower95"] and r["upper95"]:
                out.append(r"\addplot[no marks,line width=.6pt,"+color+"] coordinates {"
                           +f"({mu:.7f},{float(r['lower95']):.6f}) ({mu:.7f},{float(r['upper95']):.6f})"+"};")
        out.append(r"\addplot[only marks,mark=*,mark size=2.1pt,"+color+"] coordinates {"+" ".join(points)+"};")
        r,=[r for r in rows if float(r["T_K"])==1243. and float(r["W"])==wall]
        roots=[float(r[side+"_mu"]) for side in ("Ti","Al") if r[side+"_mu"]]
        if roots:
            out.append(r"\addplot[only marks,mark=diamond*,mark options={draw=ink,fill=white},mark size=3pt] coordinates {"
                       +" ".join(f"({mu:.7f},0)" for mu in roots)+"};")
        out.append(r"\end{axis}\end{tikzpicture}\end{center}\end{minipage}")
        if wall==.35:
            out.append(r"\hfill")
    out.append(r"\captionof{figure}{Отношение суммарного веса D0$_{19}$ к четырём весам \(\alpha\), в логарифмической шкале. "
               r"Показана кривая MBAR внутри фактически исследованного интервала. Точки обозначают приложенные поля, "
               r"отрезки --- их блочные 95\%-е интервалы, светлые ромбы --- пересечения с нулём. "
               r"Панели имеют собственные масштабы осей.}")
    return "\n".join(out)


def contacts(rows, selected):
    body=[]
    for r in sorted(rows,key=lambda x:(float(x["T_K"]),float(x["W"]))):
        if selected and float(r["T_K"]) not in selected:
            continue
        for branch,title in (("Ti","Ti"),("Al","Al")):
            if not r[branch+"_mu"]:
                continue
            lo,hi=r[branch+"_gap_low95_pp"],r[branch+"_gap_high95_pp"]
            body.append([n(r["T_K"],1),n(r["W"]),title,n(r[branch+"_mu"],4),
                         n(r[branch+"_x_alpha_pct"]),n(r[branch+"_x_alpha2_pct"]),
                         n(r[branch+"_gap_pp"]),interval([lo,hi]) if lo and hi else "---"])
    if not body:
        return ""
    return (r"\subsection*{Фазовые контакты в интервале приложенных полей}"+"\n"+
            tabular(r"\(T\), К & \(w\) & ветвь & \(\Delta\mu\), эВ & \(x_{\alpha}\) & \(x_{\alpha_2}\) & \(\Delta x\) & 95\% для \(\Delta x\)",
                    body,"rrcrrrrr")+"\n"+
            r"{\small Составы в ат.\% Al, \(\Delta x=x_{\alpha_2}-x_\alpha\) в процентных пунктах. "
            r"Приведены центральные корни на прямой сетке при \(q_c=0{,}37\).}\par")


def geometry_note(path):
    data=json.loads(path.read_text(encoding="utf8"))["records"]
    if not data or any(not r["analyzed"] or r["omitted"] for r in data):
        raise ValueError("Geometry contributions require complete trajectories")
    frame_count=sum(r["frames"] for r in data)
    min_clear=min(r["guard_by_cap"]["0.40"]["recorded_min_A"] for r in data)
    min_lower=min(r["guard_by_cap"]["0.40"]["min_reachable_interval_bound_A"] for r in data)
    return (r"\subsection*{Проверка по новым траекториям}"+"\n"+
            f"Для {frame_count:,} продуктивных кадров наименьший сохранённый зазор до более строгой геометрической границы ".replace(",",r"\,")+
            r"\(h_{0{,}40}\) равен "+n(min_clear,5)+r"~\AA. Нижняя граница зазора среди всех интервалов между записями равна "+
            n(min_lower,5)+r"~\AA.")


def main(args):
    with args.direct.open(encoding="utf8",newline="") as f:
        rows=[r for r in csv.DictReader(f) if float(r["cut_Q"])==.37]
    diff=json.loads(args.difference.read_text(encoding="utf8"))
    if diff["N"]!=2304 or diff["cut_Q"]!=.37 or not rows:
        raise ValueError("Source information does not correspond to N=2304, q_c=.37")
    values = {"pooled":pooled_table(diff),"plot":paired_figures(rows,args.profiles),"fields":field_curves(rows,args.fields),
              "contacts":contacts(rows,args.contacts_at),"geometry":geometry_note(args.geometry)}
    for part in ("overview","interpretation","method","inference","support","sampling"):
        values[part]=(HERE / (part+".tex")).read_text(encoding="utf8").strip()
    tex=(HERE/"template.tex").read_text(encoding="utf8")
    for key,value in values.items():
        tex=tex.replace("@"+key.upper()+"@",value)
    args.dest.parent.mkdir(parents=True,exist_ok=True)
    args.dest.write_text(tex,encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--difference",type=Path,required=True)
    p.add_argument("--direct",type=Path,default=STUDY/"analysis/direct/final_summary.csv")
    p.add_argument("--profiles",type=Path,default=STUDY/"analysis/direct/order_profiles.csv")
    p.add_argument("--fields",type=Path,default=STUDY/"analysis/direct/field_profiles_1243.csv")
    p.add_argument("--geometry",type=Path,default=STUDY/"analysis/geometry_new.json")
    p.add_argument("--contacts-at",type=float,nargs="+")
    p.add_argument("--dest",type=Path,default=STUDY/"report/TiAl_solvus_cage2304.tex")
    main(p.parse_args())
