# Фазовая диаграмма α/α₂ в Ti–Al

Файлы этого направления лежат в `phase_reproduction/`. Для работы с рукописью v16 начните с авторского C46. Для проверки влияния обучающей выборки используйте независимое разложение ATAT. Для равновесия при явном учёте тепловых смещений используйте прямые расчёты на MTP Qi. Все составы ниже даны в атомных процентах Al, в порядке α / α₂. [Описание исследования](../phase_reproduction/README.md)

## Какие результаты использовать

| Вопрос | Численный результат | Первичные таблицы и отчёт |
|---|---|---|
| Воспроизводится ли авторский C46? | На авторских данных независимый код воспроизводит 46 × 20 корреляций с максимальным отклонением 5,34 × 10⁻¹⁴, а полные ECI на девяти объёмах с расхождением до 2,84 × 10⁻¹⁴ эВ/кластер. При 1000 K авторский QHA–MC даёт 14,958 / 21,958. | [Комплект C46](../phase_reproduction/author_package/START_HERE.md), [сравнение](../phase_reproduction/author_comparison/README.md), [окна](../phase_reproduction/author_package/phase_check/MC_window_means.csv) |
| Что даёт новая выборка? | 39 обучающих и 7 контрольных декораций. Для температурного разложения устойчивы 396 из 414 точек «ячейка–объём». При 1000 K одинаковая обработка MC даёт 14,764 / 22,707 на независимом QHA. Сдвиг α₂ относительно авторского C46 равен +0,748 п.п. | [Матрица дизайна](../phase_reproduction/atomistic/atat/output/design.csv), [метки MTP](../phase_reproduction/atat_comparison/MTP_labels_414.csv), [ECI по объёмам и температурам](../phase_reproduction/atat_comparison/ECI_TV.csv), [сопоставление фазовых границ](../phase_reproduction/author_comparison/thermo/mfa_1000K.csv) |
| Насколько важна статистика колебаний C46? | При 1000 K, 5120 узлах и 25 окнах квантовый C46 даёт 14,933 / 21,922, классический 14,896 / 21,921. При том же фиксированном отношении осей прямой MTP на 1400 атомах даёт 16,083 / 22,159. | [Пара моделей, интервалы и контроль стартов](../phase_reproduction/classical_c46_1000/README.md), [отчёт](../phase_reproduction/classical_c46_1000/report/C46_classical_1000K.pdf) |
| Где лежат границы прямого MTP при свободном c/a? | При 600 K на 1400 атомах получено 5,866 / 24,265, при 1000 K — 14,98 / 22,44. Давление равно нулю, граница тепловых смещений W < 0,40. При 1273,15 K в экспериментальном диапазоне 19,5–23,0 ат.% Al разрешить два равновесных состава не удалось. | [1000 K, расширенный отчёт](../phase_reproduction/solvus_1000/output/pdf/expanded/TiAl_solvus_1000K.pdf), [600 и 1273,15 K](../phase_reproduction/solvus_temperatures/results.json), [отчёт](../phase_reproduction/solvus_temperatures/report/TiAl_600K_1273K.pdf) |
| Где замыкается сольвус MTP? | Для W < 0,40 и 2304 атомов вершина при 1243,64 K, около 27,2 ат.% Al. Условный 95%-й блочный интервал для этого размера 1242,39–1245,03 K. Разброс между размерами и экстраполяциями даёт ориентир 1240–1247 K. В завершённом исследовании 36 заданий и 738 700 производственных кадров. | [Результаты](../phase_reproduction/solvus_closure/results.json), [зависимость от размера](../phase_reproduction/solvus_closure/analysis/vertex_size_limit.json), [отчёт](../phase_reproduction/solvus_closure/report/TiAl_solvus_closure.pdf) |

Расчёт прямого MTP охватывает конденсированный ГПУ-бассейн. Верхняя изотерма и положение вершины зависят от механического ансамбля и допустимых смещений. Числа для фиксированного и свободного c/a стоят в отдельных строках таблицы. Для сопоставления с экспериментом и CALPHAD доступны оцифровка, данные диффузионных пар, версия движка и фазы базы. [Ограничение смещений](../phase_reproduction/alloy_ti/README.md) · [Эксперимент](../phase_reproduction/solvus_temperatures/experiment.md) · [CALPHAD](../phase_reproduction/solvus_temperatures/calphad/metadata.json)

## Рукопись и происхождение файлов

Снимок авторского комплекта от 25 ноября 2026 г. включает все 124 файла из исходного SHA-256-манифеста и сам манифест. В `structures/` находятся исходные и релаксированные ячейки, в `models/` — корреляции, целевые энергии, маски устойчивости, сетки коэффициентов и геометрия орбит. В `phase_check/` сохранены средние по окнам и алгоритм касательной. SHA-256 исходного авторского ZIP — `58a9d1cf5b85811c55d3caa83f9c57889c76fa02add86e8f13615be76bbcf323`. [Манифест C46](../phase_reproduction/author_package/manifest.json) · [Исходный код подгонки и MC](../phase_reproduction/author_package/source/)

Рукопись и дополнительные материалы v16 доступны в DOCX вместе с двумя методическими PDF. Октябрьский набор для рисунка описывает более ранний вариант диаграммы. Там сохранены исходные расчётные точки, оцифрованный эксперимент, иллюстративные сглаженные линии, Excel и векторный рисунок. Ссылки на две статьи из оригинального каталога `articles/` есть в библиографии. [Рукопись v16](../phase_reproduction/manuscript_v16/) · [Числа октябрьского рисунка](../phase_reproduction/phase_figure_20261016/data/) · [Библиография](../phase_reproduction/phase_figure_20261016/references.bib)

## Начать на новом компьютере

Команды выполняются из корня клона. Для численного анализа нужны Python 3.10+, NumPy и SciPy, для графиков — Matplotlib. В примерах `python` обозначает интерпретатор этого окружения. Сохранённые ранее пути к `blind_pilot/.venv-local` в лабораторных журналах можно заменить своим интерпретатором. Для чтения готовых отчётов и таблиц достаточно клона.

```sh
python phase_reproduction/author_package/check_eci.py
python phase_reproduction/author_package/phase_check/check_binodals.py
python phase_reproduction/thermo/mfa.py phase_reproduction/author_comparison/eci/author_refit_grid.npz tmp/replay_mfa --temperatures 1000 --models static HA QHA --volume-spline not-a-knot
```

Первая команда сверяет 16 020 строк CSV с массивами авторской модели. Вторая повторяет касательные по сохранённым авторским окнам. Третья создаёт отдельный результат среднего поля при 1000 K. Для собственной подгонки независимого разложения распакуйте `TiAl_ATAT_replay.zip` в рабочую папку и перейдите в неё. Комплект содержит 46 декораций, метки по девяти объёмам, матрицу дизайна и код. [Архив воспроизведения](../phase_reproduction/TiAl_ATAT_replay.zip)

```sh
python code/atomistic/fit_eci.py design labels/v100.tar.gz fitted_vref --ratios 1
python code/atomistic/atat/export_eci.py design fitted_vref/grid.npz native_HA1000 --temperature 1000
```

Файл `native_HA1000/eci.out` можно побайтово сравнить с результатом из репозитория. Для независимого получения первичных меток нужны CPU MLIP с MTP Qi, phonopy и ATAT. Для повторения приведённой подгонки достаточно сохранённых меток. [Эталонный eci.out](../phase_reproduction/atomistic/atat/eci_HA1000/eci.out) · [Команды и версии сборки на UMT](../phase_reproduction/cluster/README.md)

## Восстановить сырые данные

В [индексе сырых данных](../phase_reproduction/raw_files.csv) для каждого файла указаны имя ZIP, путь относительно корня клона, размер и SHA-256. По префиксу имени можно выбрать необходимый расчёт. Распакуйте все части нужной группы в корень клона. Восстановление атомистических серий из вложенных архивов UMT выполняет следующая команда, ей хватает стандартной библиотеки Python.

```sh
python phase_reproduction/restore_series.py solvus_1000 solvus_temperatures solvus_closure solvus_cage2304
```

Можно указать одну серию исследований, например `solvus_closure`. Скрипт сверяет SHA-256 каждого исходного архива, разворачивает CSV окон, конечные состояния и журналы и проверяет совпадение ранее восстановленных файлов. В [индексе архивов серий](../phase_reproduction/series_archives.csv) есть связь между серией и исходным архивом UMT. Для `solvus_cage2304` взяты оба набора архивов — первоначальная выгрузка и дополнение от 26 ноября 2026 г. с меткой `20261126T120144Z`.

| Префикс после `tial-phase-` | Назначение |
|---|---|
| `atomistic-raw` | Полученные на UMT прямые метки, силы, фононные результаты, конечные геометрии и малые атомистические тесты. Для пересчёта ECI из меток уже достаточно компактного архива в Git. |
| `mc-windows`, `mc-surfaces` | Сохранённые конфигурационные выборки и рассчитанные профили свободной энергии независимых решёточных MC и классического C46. |
| `solvus-1000` | 26 полученных серий при 1000 K. |
| `solvus-temperatures` | 34 полученные серии при 600 и 1273,15 K. |
| `solvus-closure` | 36 полученных серий по замыканию купола. |
| `solvus-cage2304` | Все 71 серия исследования двух стенок. Первые два ZIP содержат 20 ранее полученных серий. Три ZIP с меткой `20261126T120144Z` добавляют остальные 51 и сохранённые префиксы траекторий. |

После восстановления можно посмотреть команды каждой обработки с исходными параметрами окон и блочных повторов. Следующие вызовы только выводят список или команду.

```sh
python phase_reproduction/solvus_temperatures/analysis/reproduce.py --list
python phase_reproduction/solvus_closure/analysis/reproduce.py T1245_N2304 --show
python phase_reproduction/solvus_closure/thermal_cases.py direct_1245_n1400 --show
```

## Две стенки при N = 2304

Срез от **26 ноября 2026 г., 12:01:44 UTC** (17:01:44 по Екатеринбургу). Для всех 71 заданий Slurm в учёте записаны `COMPLETED` и код выхода `0:0`. Проверены 71 архив UMT, 944 оконные траектории и 998 400 производственных кадров. Для обоих значений W объём и c/a свободны, давление равно нулю. [Учёт Slurm](../phase_reproduction/solvus_cage2304/atomistic/slurm_accounting.csv) · [Серии и их контрольные суммы](../phase_reproduction/solvus_cage2304/atomistic/series_summary.csv) · [Сверка исходных данных](../phase_reproduction/solvus_cage2304/atomistic/provenance.json)

При разделителе фаз Q = 0,42 интерполяция прямых температурных сечений после монотонной регрессии даёт для узкой стенки W < 0,35 температуру **1262,59 K** (условный 95%-й интервал 1260,69–1263,94 K). Для широкой стенки W < 0,40 получается **1243,46 K** (1243,09–1243,88 K). Разность оценок **+19,13 K**, условный блочный интервал разности **+17,26…+20,51 K**. Обе температуры удалось ограничить прямыми изотермами в 99,45% повторных выборок. Эти оценки описывают чувствительность термодинамики ограниченного кристаллического ансамбля к выбору допустимых тепловых смещений при фиксированном размере системы. [Температурные корни и интервалы](../phase_reproduction/solvus_cage2304/analysis/direct/direct_temperature_zeros.json) · [Проверка каждого поля](../phase_reproduction/solvus_cage2304/analysis/direct/independent_brackets.json) · [Прямые составы и диагностика](../phase_reproduction/solvus_cage2304/analysis/direct/final_summary.csv)

Для проверки зависимостей от выбора разделителя или параметров MBAR сначала восстановите серию из обеих выгрузок. Производственные траектории `series/` и префиксы для выбора полей учтены отдельно. В итоговый профиль входят полные траектории. Код обработки прямых изотерм, межтемпературного объединения и сборки таблиц сохранён вместе с промежуточными профилями. [Метод анализа](../phase_reproduction/solvus_cage2304/analysis/direct/README.md) · [Профили при 1243 K](../phase_reproduction/solvus_cage2304/analysis/direct/field_profiles_1243.csv) · [Код температурной обработки](../phase_reproduction/solvus_cage2304/finish_thermal.py)

```sh
python phase_reproduction/restore_series.py solvus_cage2304
python phase_reproduction/solvus_cage2304/analysis/direct/run.py --mode final --require-all
```

На UMT исходные архивы находятся на исторической учётной записи `u3129` в `/home6/u3129/tial-cage2304-20261126`. Доступ к этой учётной записи с нового компьютера можно настроить через личный алиас `mlip-umt`. При запуске новой серии Codex должен спросить, какую из учётных записей `u3129` и `u2626` использовать. [Протокол работы с существующими архивами](../phase_reproduction/solvus_cage2304/atomistic/README.md)

## Внешние входы для новых вычислений

- **MTP Ti–Al Qi et al.** Файл `MTP_TiAl/fitted_mtps/fitted.mtp` из исходного набора, типы 0 = Ti и 1 = Al, вариант с поверхностями. Сохраните его как `phase_reproduction/inputs/fitted.mtp`. SHA-256 `704f7fd900c7446286066b741084c30ebc7fcf26e5b9b8fd2f29a18ba4ffd661`. [Паспорт с URL и путём](../phase_reproduction/inputs/source.json) · [Набор Qi et al.](https://doi.org/10.6084/m9.figshare.24079611)
- **CALPHAD Liang et al.** Файл термодинамической базы `phase_reproduction/inputs/Liang2024.tdb` берётся из материалов к публикации. SHA-256 `badf3611150a5037baf39fb270bfb4005e1e15bc67136b66947cf630cfacec1e`. В репозитории есть выходные таблицы и код расчёта при нулевом давлении для pycalphad 0.11.2. [Публикация Liang et al.](https://doi.org/10.1007/s11669-024-01124-9) · [Вычисленные границы](../phase_reproduction/solvus_temperatures/calphad/liang_contacts.csv)
- **ATAT 3.52.** Для повторного перечисления декораций, расчёта корреляций и нативной подгонки нужен оригинальный дистрибутив. SHA-256 использованного архива `8536fb078375b30096df780e02f4e0c97972f5ebaad9a84a94f22e00f50c7516`. [Сайт ATAT](https://axelvandewalle.github.io/www-avdw/atat/) · [Сборка и команды](../phase_reproduction/cluster/README.md)
