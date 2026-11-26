import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def make_table():
    pilot = json.loads((root / "paired768_final.json").read_text())
    matched = json.loads((root / "paired5120.json").read_text())
    bulk = json.loads((root / "dense_grid.json").read_text())
    items = [(768, 13, *pilot["estimates"], pilot["paired_difference_intervals_95"])]
    items.append((5120, 13, *matched["estimates"], matched["paired_difference_intervals_95"]))
    model = bulk["primary"]["full_25"]
    items.append((5120, 25, model["quantum"], model["classical"],
                  bulk["intervals_95"]["full_25"]["classical_minus_quantum"]))
    def num(x, sign=False):
        return format(x, "+.3f" if sign else ".3f").replace(".", "{,}")
    rows = []
    for n,w,q,c,ci in items:
        part = [str(n), str(w)]
        part += ["$"+num(x)+"$" for x in [q[0], q[1], c[0], c[1]]]
        for i in range(2):
            delta = num(c[i]-q[i], True)
            part += ["$"+delta+"$"]
        rows.append(" & ".join(part) + r"\\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{8}{l}{95\%-е интервалы парного сдвига, п.п.}\\")
    for n,w,q,c,ci in items:
        intervals = [r"$[" + num(a,True) + r";\," + num(b,True) + r"]$" for a,b in ci[:2]]
        rows.append(rf"{n} & {w} & \multicolumn{{3}}{{c}}{{\(\alpha\)\enspace {intervals[0]}}}"
                    + rf" & \multicolumn{{3}}{{c}}{{\(\alpha_2\)\enspace {intervals[1]}}}\\")
    return r"""
\begin{center}\small
\begin{tabular}{@{}rr rr rr rr@{}}
\toprule
 && \multicolumn{2}{c}{Квантовые, ат.\%}
 & \multicolumn{2}{c}{Классические, ат.\%}
 & \multicolumn{2}{c}{Сдвиг, п.п.}\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}
\(N\) & Окна & \(\alpha\) & \(\alpha_2\) & \(\alpha\) & \(\alpha_2\) & \(\alpha\) & \(\alpha_2\)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}\end{center}
"""
