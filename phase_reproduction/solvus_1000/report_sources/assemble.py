import json
from pathlib import Path
from revision_plots import ce_temperature, ti_integrand

project = Path(__file__).resolve().parents[1]
parts = project / "report_sources"
output = project / "output/pdf/expanded/TiAl_solvus_1000K.tex"


def read(name):
    return (parts / (name + ".tex")).read_text(encoding="utf-8").strip()


figure = [r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{axis}[width=.91\linewidth,height=58mm,xmin=0,xmax=1,ymin=0,
 xlabel={Параметр порядка \(Q\)},ylabel={Плотность \(P(Q)\)},
 tick label style={font=\sffamily\footnotesize}, label style={font=\sffamily\small},
 legend style={font=\sffamily\footnotesize,draw=none,fill=none,
 at={(.05,.97)},anchor=north west,legend columns=2,column sep=5pt},
 axis x line*=bottom,axis y line*=left,axis line style={muted},
 ymajorgrids,y grid style={linegray!65},
 xtick={0,0.2,0.4,0.6,0.8,1.0},xticklabels={0,{0,2},{0,4},{0,6},{0,8},1}]
"""]
for size, name, style in [
    (96, "pilot96", "muted, densely dotted"),
    (288, "pilot288", "rust, dashed"),
    (768, "analysis768_free", "black!55, dashdotted"),
    (1400, "analysis1400_free", "teal, solid"),
]:
    datasets = json.loads((project / (name + ".json")).read_text(encoding="utf-8"))
    dataset = next(d for d in datasets.values() if d["cap"] == .40 and d["axial_free"])
    profile = dataset["equal_single_variant"]["order_profile"]
    coords = list(zip(profile["Q"], profile["density"], strict=True))
    figure.append(r"\addplot[" + style + r",line width=1.1pt,no marks] coordinates {")
    figure.extend(" ".join(f"({q:.5f},{density:.7f})" for q, density in coords[i:i + 5])
                  for i in range(0, len(coords), 5))
    figure.append(r"}; \addlegendentry{\(N=" + str(size) + r"\)}")
figure.append(r"""
\end{axis}
\end{tikzpicture}
\caption{Равновесная плотность химического порядка после снятия зонтика
при \(W<0{,}40\) и свободном \(c/a\). Каждая кривая построена при своём
поле сосуществования по весам на один вариант порядка и нормирована на
единицу. С ростом \(N\) уменьшается вероятность промежуточного порядка.
При определении поля сравниваются площади под двумя пиками, а не их высоты.}
\label{fig:order}
\end{figure}
""")
bulk = read("bulk_results")
anchor = r"\subsection{Проверки положения границ}"
assert bulk.count(anchor) == 1
bulk = bulk.replace(anchor, "\n".join(figure) + "\n" + anchor)
small = read("small_cells")
anchor = r"\subsection{Нелинейность свободной энергии по составу в ячейке на 16 атомов}"
assert small.count(anchor) == 1
small = small.replace(anchor, ti_integrand(project.parent) + "\n" + anchor)
ce = read("ce_phase")
assert ce.count("% FIG_CE_TEMPERATURE") == 1
ce = ce.replace("% FIG_CE_TEMPERATURE", ce_temperature(project.parent))
for _ in range(2):
    for rule in ("toprule", "midrule", "bottomrule"):
        small = small.replace(r"\hline", "\\" + rule, 1)
outlook = read("outlook")
references = read("literature")
document = "\n\n".join([read("opening"), read("ce_model"), ce, small, read("ensemble"), bulk,
                       outlook, read("provenance"), r"\begin{thebibliography}{99}", references,
                       r"\end{thebibliography}", r"\end{document}", ""])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(document, encoding="utf-8")
