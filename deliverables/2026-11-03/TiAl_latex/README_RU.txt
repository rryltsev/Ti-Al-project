Исходники статьи Ti--Al, снимок от 3 ноября 2026.

Основной документ: main.tex. SI: supporting_materials.tex.
Каталоги sections/, figures/, supporting_data/ и четыре файла .bib сохраняют
относительные пути. Файлы .tex редактируются в UTF-8.
Сборка в TeX Live/MiKTeX с REVTeX 4.2: latexmk -pdf main.tex
Отдельная сборка SI: latexmk -pdf supporting_materials.tex
Альтернатива: tectonic main.tex и tectonic supporting_materials.tex.
Нужны обычные пакеты REVTeX, graphicx, amsmath, natbib, hyperref и lmodern.
Текущий проверенный основной PDF приложен как main.pdf.

Рисунок 1 и Table II обновлены по 49 общим проверенным структурам из Test50.
Обновлены соответствующие числа в Results и Conclusions. Без энергетического
выравнивания; PBE 20k, остальные модели и геометрии сохранены.
Сопутствующие данные всех рисунков поставляются отдельным архивом figure_data.
Исторические диагностические таблицы и рисунки SI сохранены с их прежними
выборками и чекпойнтами; их нельзя интерпретировать как текущий тест на 49 структурах.
SI в этой редакции не переписывался. В supporting_data есть исторические файлы;
актуальная тестовая метрика: test49_metrics.json.
