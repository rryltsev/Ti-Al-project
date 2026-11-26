# GRACE N=200 fp64: baseline → optimizer → четыре пилота

Состояние на 2026-11-08T18:07:08.449584+00:00 UTC. Все этапы отправлены в Slurm. Текущая причина ожидания baseline — Priority; зависимые этапы — Dependency. Численные baseline, alignment и optimizer gates пока не выполнены: PASS не объявлен.

| Этап | Job | Условие запуска |
|---|---|---|
| Исходная GRACE: B0, export и alignment | 20990460 | GPU priority/resources |
| Исходные фононы H0 | 20990461 | baseline COMPLETED |
| Четыре optimizer smoke-проверки | 20990464 | H0 COMPLETED |
| Общий optimizer gate | 20990465 | все smoke COMPLETED |
| Четыре пилота fp64 | 20990466 | optimizer gate COMPLETED/PASS |
| Итоговые фононы пилотов | 20990467 | все пилоты COMPLETED |

Массивы GPU выполняются с concurrency1. Имеющиеся задания других исследований не менялись.

## Исходная модель и данные

GRACE-2L-OMAT-medium-base-fp64, тот же checkpoint revision4aff666029967745f7993f348eb87a804d625cff и архив SHA763b7e5679c81ea82d39e595e701d0c9f6565064393f705cbe76663c9583d0aa. Полные 89 элементов сохранены. Источники неизменны; локальные и staged входы/код проверены SHA-256. Синтаксис Python, bash и встроенного Python gate проверен отдельно.

PBE и r²SCAN: train200 и validation149 из общего grouped split. Координаты и E/F/stress сохранены с 17 значащими цифрами; ранее проверено точное обратное чтение. Backend test_filename означает только validation149. Test149 и архивные 49 не передаются обучению.

## Численные gates

B0 включает конечность предсказаний, pure-Al/pure-Ti/alloy probes, перестановки атомов, переносы/PBC, вращения E/F/stress, force finite differences на трёх шагах и stress FD по всем шести Voigt-компонентам. Native/export предсказания сравниваются на тех же probes.

Для каждого XC рассчитываются исходные предсказания всех 200 train и 149 validation. Две поправки Al/Ti определяются равновесным по структурам least squares в E/atom строго по train200. Validation не участвует в подгонке. Поправки встраиваются штатным ConstantScaleShiftTarget; проверяется равенство E_aligned=E_source+nAl*bAl+nTi*bTi при неизменных F/stress. Сохраняются отдельные aligned model.yaml и model-only checkpoint; исходные веса не перезаписываются.

H0 заново получает силы на force000/±0.01Å для PBE/r²SCAN DFT-ячеек, затем использует общий central/symfc/ASR protocol, Γ, bands, DOS и mesh50³. Это исходная гармоническая baseline; DFT convergence этими проверками не сертифицируется.

## Optimizer и рецепт пилота

Native Adam, fp64, LR1e−3→1e−4 cosine, Huber energy/forces/stress weights16/32/128, batch1,50epochs, безEMA. Единый train/test bucket уменьшает число XLA-компиляций; задано 200 update batches на эпоху. Native scheduler key исправлен на minimum_learning_rate. Смещение энергий фиксируется до сравнения full/restricted.

Для каждой из четырёх конфигураций выполняются 5 smoke optimizer updates с тем же рецептом. Проверяется model-only restore, нулевые epoch/step/Adam iterations и свежие slots до первого шага; затем конечность результатов, изменение обучаемых тензоров, отсутствие изменения frozen tensors, фактическая fp64-маска и GPU memory/step timings. Дополнительно проверяются одинаковые первые 5 batch внутри XC для full/restricted. Smoke-веса отбрасываются: научный пилот стартует заново из aligned source checkpoint.

Restricted использует штатные I2/reducing_, rho/reducing_, I1/reducing_: ожидается 21111690 trainable scalars против 21764956 full, около 97%. Это ограничение tensor groups, а не сильное сжатие модели.

Все пилоты идут 10000 шагов/50epochs. Сохраняется фиксированный final checkpoint. Native автоматическая загрузка best-validation checkpoint после обучения отключена наблюдателем; факт фиксируется в validation-selection.json. Во время обучения проверяется frozen mask на первом шаге и каждые 1000 шагов. После обучения: final native/export numerical QC, train/validation метрики и фононы. Дополнительное train-only энергетическое выравнивание при отчётной оценке сохраняется отдельно от raw calibrated errors.

| Task index | XC | FT | N | Seed | Точность |
|---:|---|---|---:|---:|---|
| 0 | PBE | full | 200 | 10 | fp64 |
| 1 | PBE | restricted | 200 | 10 | fp64 |
| 2 | r²SCAN | full | 200 | 10 | fp64 |
| 3 | r²SCAN | restricted | 200 | 10 | fp64 |

Лимит каждого пилота 4ч/1A100/4CPU/40GiB — planning cap; измеренный training forecast появится после optimizer gate. Нельзя переносить измеренное ускорение inference fp32/fp64 на backprop/Adam.

Отслеживание: scripts/manage_grace_pilot.py status. Evidence сохраняется в baseline/, aligned/, smoke/, optimizer-gate.json и cases/. Следующие вычисления выполнятся автоматически по Slurm afterok-зависимостям, если все предыдущие проверки успешны. При сбое обучение останется заблокировано; неизменённый gate не обходится.
