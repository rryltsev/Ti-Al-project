import csv
import io
import json
import tarfile
from pathlib import Path

import numpy as np


def coords(xs, ys):
    return " ".join(f"({x:.5f},{y:.5f})" for x, y in zip(xs, ys, strict=True))


def ce_temperature(root):
    path = root / "author_comparison/mc/mc_same_windows.csv"
    selected = {}
    temperatures = list(range(600, 1201, 100))
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            temperature = int(row["T_K"])
            if row["model"] != "QHA" or row["author_resolved"] != "True" or temperature not in temperatures:
                continue
            key = (row["source"], temperature)
            if key not in selected or int(row["N"]) > int(selected[key]["N"]):
                selected[key] = row
    lines = [r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{axis}[width=.90\linewidth,height=65mm,xmin=3,xmax=26.1,ymin=570,ymax=1240,
 xlabel={Al, ат.\%},ylabel={\(T\), К},xtick={5,10,15,20,25},ytick={600,800,1000,1200},
 yticklabels={600,800,1000,1200},
 axis x line*=bottom,axis y line*=left,axis line style={muted},
 tick label style={font=\sffamily\footnotesize},label style={font=\sffamily\small},
 ymajorgrids,y grid style={linegray!65},
 legend style={font=\sffamily\footnotesize,draw=none,fill=white,at={(.02,.985)},anchor=north west}]
"""]
    for kind, label, style in [
        ("author", r"Авторский C46", "teal,solid,mark=*"),
        ("ours", r"Подгонка ATAT", "rust,dashed,mark=square"),
    ]:
        rows = [selected[kind, t] for t in temperatures]
        for i, branch in enumerate(["author_alpha", "author_alpha2"]):
            points = coords([float(row[branch]) for row in rows], temperatures)
            extra = "" if i == 0 else ",forget plot"
            lines.append(r"\addplot[" + style + ",line width=1pt,mark size=1.65pt" + extra
                         + "] coordinates {" + points + r"};")
            if i == 0:
                lines.append(r"\addlegendentry{" + label + "}")
    lines.append(r"""
\addplot[ink,only marks,mark=diamond*,mark size=2.8pt] coordinates {(16.08266,1000) (22.15918,1000)};
\addlegendentry{MTP, \(c/a\) как у C46}
\end{axis}
\end{tikzpicture}
\caption{Рассчитанные ветви \(\alpha\) (слева) и \(\alpha_2\) (справа)
в квантовом QHA с химическим MC. Точки каждого цвета соединяют результаты
одной подгонки. При 1000 и 1200 К обе подгонки представлены ячейкой
из 5120 узлов, при 600 К — 5120 для C46 и 768 для ATAT, при остальных
температурах — 768 узлов. Ромбы показывают прямой классический MTP
при 1000 К, \(N=1400,\ W<0{,}40\), с геометрическим ограничением C46.}
\label{fig:ce-temp}
\end{figure}""")
    return "\n".join(lines)


def ti_integrand(root):
    archive = root / "cluster/runs/alloy-ti-n8-01/outputs/draws.tar.gz"
    check_data = json.loads((root / "alloy_ti/cage_sensitivity.json").read_text(encoding="utf-8"))
    reference = check_data["caps"]["0.40"]["N8_gaussian"]["per_orbit"]
    beta = 1.0 / (1000 * 8.617333262145e-5)
    grid = np.linspace(0, 1, 65)
    lines = [r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{axis}[width=.89\linewidth,height=56mm,xmin=0,xmax=1,ymin=-4,ymax=7.5,
 xlabel={Параметр связи \(\lambda\)},ylabel={\(I(\lambda)\), мэВ/атом},
 xtick={0,.2,.4,.6,.8,1},xticklabels={0,{0,2},{0,4},{0,6},{0,8},1},ytick={-4,-2,0,2,4,6},
 axis x line*=bottom,axis y line*=left,axis line style={muted},
 tick label style={font=\sffamily\footnotesize},label style={font=\sffamily\small},
 ymajorgrids,y grid style={linegray!65},
 legend style={font=\sffamily\footnotesize,draw=none,fill=white,at={(.995,.98)},anchor=north east}]
\addplot[black!45,thin,forget plot] coordinates {(0,0) (1,0)};
"""]
    with tarfile.open(archive) as tar:
        for kind, label, style in [
            ("d019", r"D0$_{19}$", "teal,solid"),
            ("basal", "Базальная пара", "rust,dashed"),
            ("interlayer", "Межслойная пара", "ink,dashdotted"),
        ]:
            record, = [m for m in tar.getmembers() if Path(m.name).name == f"samples_{kind}.npz"]
            with np.load(io.BytesIO(tar.extractfile(record).read())) as draws:
                use = draws["max_wall_fraction"] < .40
                delta = draws["delta_eV_cell"][use]
                base = draws["log_harmonic_to_proposal"][use]
            logw = base[None, :] - beta * grid[:, None] * delta[None, :]
            logz = np.logaddexp.reduce(logw, axis=1)
            curve = np.exp(logw - logz[:, None]) @ delta * (1000 / 8)
            endpoint = -(logz[-1] - logz[0]) / beta
            np.testing.assert_allclose(endpoint, reference[kind]["matched_anharmonic_shift_eV_cell"], atol=1e-9)
            area = np.trapezoid(curve, grid)
            assert abs(area - endpoint * (1000 / 8)) < .0001
            print(f"N=8 {kind}: integral {area:.5f}, endpoint {endpoint * 125:.5f} meV/atom")
            lines.append(r"\addplot[" + style + ",line width=1.15pt,no marks] coordinates {"
                         + coords(grid, curve) + r"}; \addlegendentry{" + label + "}")
    lines.append(r"""
\end{axis}
\end{tikzpicture}
\caption{Подынтегральная функция
\(I(\lambda)=I_\lambda/N=\langle U_{\rm MTP}-U_{\rm h}\rangle_{\lambda,W}/N\)
по одной гауссовской смеси для каждой из трёх декораций, \(N=8\),
\(T=1000\)~К, \(W<0{,}40\). Интеграл на 65 узлах даёт
\(-0{,}228\) мэВ/атом для D0$_{19}$, \(+2{,}296\) для базальной
и \(+2{,}054\) для межслойной пары. Разность первых двух по схеме
«базальная пара минус D0$_{19}$» равна \(+2{,}524\) мэВ/атом.
Убывание следует из \(dI/d\lambda=-\beta\,\mathrm{Var}_{\lambda,W}
(U_{\rm MTP}-U_{\rm h})/N\) при фиксированной области интегрирования.}
\label{fig:ti-integrand}
\end{figure}""")
    return "\n".join(lines)
