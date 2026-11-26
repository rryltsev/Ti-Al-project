"""Build one independent TeX source from the verified C46 analysis sections."""
from pathlib import Path
from table import make_table

stage = Path(__file__).resolve().parent
project = stage.parent
parts = [
    (stage / "header.tex").read_text(encoding="utf-8"),
    (stage / "mc_results.tex").read_text(encoding="utf-8").replace("% RESULTS_TABLE", make_table()),
    r"\clearpage",
    (stage / "mc_method.tex").read_text(encoding="utf-8"),
    r"\clearpage",
    (project / "fit_method.tex").read_text(encoding="utf-8"),
    r"""
\subsection*{Воспроизведение}
Расчётный каталог проекта \nolinkurl{phase_reproduction/classical_c46_1000/}
содержит скрипт \nolinkurl{fit_classical.py} для перевода авторских меток в
классический предел, полученные сетки \nolinkurl{quantum_grid.npz} и
\nolinkurl{classical_grid.npz}, поструктурные значения
\nolinkurl{structure_labels.csv} и проверку по прямым фононным спектрам
\nolinkurl{spectra_validation.json}. Границы среднего поля рассчитаны на этих
двух сетках с граничными условиями сплайна \emph{not-a-knot}.
Результаты Монте-Карло содержатся в \nolinkurl{paired5120.json},
\nolinkurl{paired768_final.json} и \nolinkurl{dense_grid.json};
\nolinkurl{direct_comparison.json} содержит остаток относительно прямого MTP.

Исходные взаимодействия MTP опубликованы
\href{https://doi.org/10.1103/PhysRevMaterials.7.103602}{Qi и соавт. (2023)}.
Ансамбль с ограниченной дисперсией состава для сплавов рассмотрен
\href{https://doi.org/10.1103/PhysRevB.86.134204}{Садигом и Эрхартом (2012)},
двусторонняя оценка свободных энергий~---
\href{https://doi.org/10.1016/0021-9991(76)90078-4}{Беннеттом (1976)}.
\end{document}""",
]
text = "\n".join(p.strip() for p in parts) + "\n"
(project / "report" / "C46_classical_1000K.tex").write_text(text, encoding="utf-8")
