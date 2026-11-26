"""Assemble a single XeLaTeX source, including both numerical figures."""
import json
from pathlib import Path

from figures import alloy_diagram, order_diagnostic
from tables import contacts


PARTS = Path(__file__).resolve().parent
STUDY = PARTS.parent


def read(name):
    return (PARTS / (name + ".tex")).read_text(encoding="utf-8").strip()


def decimal(value, precision=2):
    return f"{value:.{precision}f}".replace(".","{,}")


def document(primary):
    direct = [(t["temperature_K"], t["Al_at_pct"]) for t in primary["MTP_solvus"]
              if t["Al_at_pct"] is not None]
    high_files = [STUDY/p for p in primary["high_profile_files"]]
    preamble = (STUDY.parent/"classical_c46_1000/report/C46_classical_1000K.tex").read_text(encoding="utf-8")
    preamble = preamble[:preamble.index(r"\begin{document}")]
    preamble = preamble.replace("xelatex C46_classical_1000K.tex", "xelatex TiAl_600K_1273K.tex")
    preamble = preamble.replace("Классическая статистика колебаний в C46 для Ti–Al при 1000 К",
                                "Равновесие фаз α и α₂ в Ti–Al при 600 и 1273,15 К")
    preamble = preamble.replace("Равновесные составы фаз α и α₂ при фиксированном отношении осей",
                                "Прямой MTP, эксперимент, кластерное разложение и CALPHAD")
    preamble = preamble.replace("Классические колебания в C46", "Равновесие ГПУ-фаз")
    preamble = preamble.replace(r"\(1000\) К", r"600 и 1273,15 К")
    preamble += r"""
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}
\definecolor{linegray}{HTML}{C5D0D3}
\definecolor{rust}{HTML}{A8492B}
\begin{document}
"""
    low=primary["MTP_solvus"][0]["primary"]["phase_fit"]
    commands=[]
    for i,phase in enumerate(("Alpha","Ordered")):
        v=low["Al_at_pct"][i]
        l,h=low["Al_at_pct_conditional_95"][i]
        for name,value in [(phase,v),(phase+"Lo",l),(phase+"Hi",h)]:
            commands.append(r"\newcommand{\Low"+name+"}{"+decimal(value)+"}")
    fixed, = [r["phase_fit"] for r in primary["low_checks"] if r["N"]==768 and not r["axial_free"] and r["cap"]==.4]
    for name,value in zip(("Alpha","Ordered"),fixed["Al_at_pct"]):
        commands.append(r"\newcommand{\Fixed"+name+"}{"+decimal(value)+"}")
    for name,record in zip(("Middle","Large"),primary["large_cell_high_response"]):
        commands.append(r"\newcommand{\Chi"+name+"}{"+decimal(record["max_dx_dmu_eV_inv"],3)+"}")
    low_text = read("low_results").replace("% CONTROL_TABLE", contacts(primary["low_checks"]))
    parts = [preamble,"\n".join(commands), read("overview"), alloy_diagram(direct), r"\clearpage", read("experiment"),
             read("comparison"), r"\clearpage", read("method"), r"\clearpage", low_text,
             r"\clearpage", read("high_results"), order_diagnostic(high_files), r"\clearpage", read("reproduce"),
             r"\end{document}", ""]
    return "\n\n".join(parts)


def main():
    primary = json.loads((STUDY/"results.json").read_text(encoding="utf-8"))
    dest=STUDY/"report/TiAl_600K_1273K.tex"
    dest.parent.mkdir(exist_ok=True,parents=True)
    dest.write_text(document(primary),encoding="utf-8")


if __name__ == "__main__":
    main()
