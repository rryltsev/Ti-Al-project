# Ti–Al FM: benchmark protocol v0.2 — 6 ноября 2026

Статус: split common992-grouped-v1-20261106-02 выпущен и проверен; runtime/physics gates остаются открытыми.
Источник постановки: CODEX_TiAl_FM_RESEARCH_PLAN.md. Первый цикл: 0 новых DFT.

- Исторические train/validation сохраняются без изменений для воспроизведения.
- Общий пул содержит 992 геометрии с тремя метками; неполные тройки сохраняются отдельно в manifest.
- Новый split:694/149/149, по консервативным группам(nTi,nAl), содержащим целиком каждую fixed-species MD/relaxation trajectory. Exact/near duplicate states одинакового числа атомов и состава не могут пересекать части. Точные liquid parent IDs не восстановлены. По решению пользователя прежние 49 вынесены из нового test и остаются архивной диагностикой; их родители не принудительно исключаются из train.
- Пары ± смещений одного harmonic parent остаются вместе. Первоначально harmonic labels используются только для контроля. Исторический PBE15k уже выбирали с учётом фононов.
- Полный train=694; N=100/200/400 строятся только внутри него. N=850 для этого split недоступно. Во всех методах и XC одинаковые ID и вложенные подвыборки.
- Одна исходная FM для каждого N; отдельно считать уникальные геометрии, geometry–XC labels, атомы, development/replay/harmonic labels и teacher evaluations.
- Энергия: полная энергия eV; метрика meV/atom, structure-weighted. Силы: Cartesian components eV/Å, pooled atom-weighted; дополнительно equal-structure и Ti/Al. Stress: ASE tensile-positive, −virial/volume, шесть независимых компонент, 160.21766208 GPa на eV/Å³.
- Исторический external benchmark: raw energy; VASP TOTEN. Train-aligned энергия — отдельный результат; fit только на train данного N, rank/condition elemental-offset matrix обязательно публиковать.
- Test49 — отдельный архивный ранее изученный benchmark, с адаптивным выбором PBE checkpoint и остановкой на 49/50. Не использовать его для нового подбора.
- Финальные интервалы: bootstrap по независимым родителям. До полного восстановления родителей не представлять component bootstrap как независимые наблюдения.
- Gate B0 требует fresh runtime checks E/F и E/strain, rotations/permutations/PBC, native/export, precision-dependent tolerances и train alignment. Пересчёт сохранённых residuals закрывает только арифметическую часть.
- Gate H0 требует исходную FM и FT на одинаковых DFT cells, 4×4×3 и ±0.01 Å. Raw/drift/symmetry/ASR результаты раздельны; mode tracking не заменять sorted-frequency RMSE.
- Численные пороги качества и устойчивости определяются до выбора финалиста на development; сейчас они не назначены. FT/KD/дорогая MD закрыты до прохождения gates.
- Один screening seed=10; финалисты 10/20/30; до двух дополнительных LR только по development-диагностике. Весь tuning cost учитывается.
