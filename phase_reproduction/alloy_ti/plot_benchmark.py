"""Plot the Ti–Al finite-cell benchmarks from the verified project outputs."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "cluster/runs"
INK = "#192E3F"
MUTED = "#4A5D6C"
TEAL = "#00796F"
ORANGE = "#AC4B12"
BLUE = "#336183"
PURPLE = "#815292"
HEIGHTS = [3.4, 2.6, 1.15, 0.35]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def local_number(value, digits=1):
    return f"{value:.{digits}f}".replace("-", "−").replace(".", ",")


def style_axis(ax, xlim, spacing):
    ax.set_xlim(*xlim)
    ax.set_ylim(-.15, 4.35)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#DFE5E8", linewidth=.75)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.xaxis.set_major_locator(MultipleLocator(spacing))
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=3)
    ax.tick_params(axis="y", length=0, pad=7, labelsize=10, colors=INK)
    ax.set_xlabel(ax.get_xlabel(), color=INK, labelpad=8)


def group_heading(ax, y, text):
    ax.text(-.018, y, text, transform=ax.get_yaxis_transform(), ha="left", va="center",
            color=MUTED, fontsize=9.4, fontweight="bold",
            bbox={"facecolor":"white", "edgecolor":"none", "pad":1.7})


def point(ax, x, y, color, marker, se=None):
    ax.errorbar(x, y, xerr=se, fmt=marker, color=color, markersize=7, zorder=3,
                elinewidth=1.5, capsize=3.4, markeredgecolor="white", markeredgewidth=.6)
    digits = 2 if se is not None and se < .99 else 1
    label = local_number(x) if se is None else f"{local_number(x, digits)} ± {local_number(se, digits)}"
    ax.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points", ha="left",
                color=color, fontweight="bold", fontsize=9)


def main(cage_summary=None):
    n8 = read(HERE / "summary_n8.json")
    static = read(RUNS / "alloy-ti-n16-01/outputs/n16_static.json")
    # A short-range collapse was sampled inside the original site domain.
    # The earlier unrestricted integrals must never enter either panel.
    path = cage_summary or HERE / "cage_sensitivity.json"
    source = read(path) if path.is_file() else None
    cage = None
    if source:
        if (source["temperature_K"] != 1000 or source["primary_cap"] != .40 or
                source["model_sha256"] != static["model_sha256"]):
            raise ValueError("The figure requires the matched 1000 K, W < 0.40 conditional benchmark")
        cage = source["caps"]["0.40"]
        for group in cage["N16_gaussian"]["compositions"].values():
            if group["orbit_issues"]:
                raise ValueError("An N16 orbit has unresolved overlap in the specified cage")
    if n8["temperature_K"] != 1000 or static["temperature_K"] != 1000:
        raise ValueError("The benchmark compares matched 1000 K runs")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.labelsize":10})
    fig = plt.figure(figsize=(12.5, 6.15), facecolor="white")
    ax_a = fig.add_axes([.107, .205, .37, .55])
    ax_b = fig.add_axes([.635, .205, .335, .55])
    fig.text(.047, .933, "Свободная энергия ГПУ Ti–Al при 1000 K",
             fontsize=21, color=INK, fontweight="bold")
    fig.text(.047, .877, "Потенциал Qi и соавт.     •     Фиксированные форма и объём ячейки, 17,288 Å³/атом",
             fontsize=11, color=MUTED)
    fig.text(.047, .79, "А    Расположение пары Al в Ti₆Al₂", fontsize=13, color=INK, fontweight="bold")
    fig.text(.555, .79, "Б    Изменение свободной энергии с составом", fontsize=13, color=INK, fontweight="bold")

    ax_a.set_xlabel("Свободная энергия относительно D0₁₉, мэВ/атом")
    style_axis(ax_a, (54, 101), 10)
    a_labels = ["C46", "Прямой MTP", "Гармоника"]
    if cage:
        a_labels.append("Полный MTP")
    ax_a.set_yticks(HEIGHTS[:len(a_labels)], labels=a_labels)
    group_heading(ax_a, 4.08, "Квантовые колебания в гармоническом приближении")
    group_heading(ax_a, 1.83, "8 узлов, ограниченные смещения, W < 0,40" if cage else
                  "Классические ядра в 8-узельной ячейке")
    models = ["C46_bulk_qHA", "direct_bulk_qHA"]
    if not cage:
        models.append("direct_N8_classical_HA")
    for row, model in enumerate(models):
        gaps = n8["gaps_and_probabilities"][model]["gaps_meV_atom"]
        for pair, color, marker in [("basal", TEAL, "o"), ("interlayer", ORANGE, "D")]:
            point(ax_a, gaps[pair], HEIGHTS[row], color, marker)
    if cage:
        conditional = cage["N8_with_student_interlayer"]["gaps_relative_to_d019"]
        for pair, color, marker in [("basal", TEAL, "o"), ("interlayer", ORANGE, "D")]:
            for i, name in enumerate(("harmonic_meV_atom","target_meV_atom"), start=2):
                data = conditional[pair][name]
                point(ax_a, data["value"], HEIGHTS[i], color, marker, data["se"])

    ax_b.set_xlabel("F(4) − 2F(3) + F(2), мэВ/ячейку")
    deriv = static["chemical_derivatives"]
    fields = ["ce_static_eV_cell", "direct_static_eV_cell"]
    if not cage:
        fields.append("direct_static_plus_classical_gamma_config_eV_cell")
    values = [1000*deriv[key]["central_second_difference_eV"] for key in fields]
    finite = []
    if cage:
        targeted = HERE / "targeted_comparison.json"
        if targeted.is_file():
            trials = read(targeted)["caps"]["0.40"]["estimates"]
        else:
            trials = {"gaussian":cage["N16_gaussian"], "student":source["independent_N16"]["0.40"]["student"]}
        for row,key in enumerate(("central_second_difference_harmonic_eV","central_second_difference_target_eV"),start=2):
            for dataset, color, marker, offset in [(trials["gaussian"],BLUE,"o",.16),
                                                   (trials["student"],PURPLE,"s",-.17)]:
                est = dataset[key]
                finite.append((1000*est["value"],1000*est["se"],HEIGHTS[row]+offset,color,marker))
    extent = values + [x+sign*se for x,se,*_ in finite for sign in (-1,1)]
    limits = (min(-135, min(extent)-12), max(10, max(extent)+22))
    style_axis(ax_b, limits, 30)
    ax_b.axvline(0, color="#8899A5", lw=1.0)
    b_labels = ["C46", "Прямой MTP", "Гармоника"]
    if finite:
        b_labels.append("Полный MTP")
    ax_b.set_yticks(HEIGHTS[:len(b_labels)], labels=b_labels)
    group_heading(ax_b, 4.08, "Статические энергии декораций")
    group_heading(ax_b, 1.83, "16 узлов, ограниченные смещения, W < 0,40" if cage else
                  "Классические ядра в 16-узельной ячейке")
    for val, y in zip(values, HEIGHTS):
        point(ax_b, val, y, INK, "o")
    for x,se,y,color,marker in finite:
        point(ax_b, x, y, color, marker, se)
    handles = [Line2D([], [], linestyle="none", marker=m, color=c, markersize=7, label=l)
               for m,c,l in [("o",TEAL,"Базальная пара Al"),("D",ORANGE,"Межслойная пара Al")]]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(.045, .077), ncol=2,
               frameon=False, fontsize=9.4, labelcolor=INK, columnspacing=1.5, handletextpad=.3)
    if cage:
        methods = [Line2D([],[],linestyle="none",marker=m,color=c,markersize=7,label=l)
                   for m,c,l in [("o",BLUE,"Гауссова смесь"),("s",PURPLE,"Смесь Гаусса и Стьюдента")]]
        fig.legend(handles=methods, loc="center left", bbox_to_anchor=(.548,.077), ncol=2,
                   frameon=False, fontsize=9.4, labelcolor=INK, columnspacing=1.2, handletextpad=.3)
        fig.text(.047, .022, "W — максимальная доля расстояния до грани Вороного. k — число Al. Погрешности ±1 SE.",
                 fontsize=8.4, color=MUTED)
        caption = ("Увеличена статистика четырёх основных конфигураций." if targeted.is_file()
                   else "Для 16 узлов показаны две независимые оценки.")
        fig.text(.555, .022, caption,
                 fontsize=8.4, color=INK, fontweight="bold")
    fig.savefig(HERE / "benchmark.png", dpi=250, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cage-summary", type=Path, help="Conditional estimates with an explicit integration domain")
    main(cli.parse_args().cage_summary)
