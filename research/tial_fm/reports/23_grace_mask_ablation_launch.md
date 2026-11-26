# GRACE N200: restricted + MLP / restricted + radial

Снимок: 2026-11-09T07:22:27.584141+00:00. Четыре новых теста поставлены в Slurm.

| Этап | Job | Зависимость |
|---|---|---|
| Проверки optimizer, 5 шагов | 20990673 | Доступная A100 |
| Общий optimizer gate | 20990674 | Все smoke afterok |
| Обучения | 20990675 | Gate afterok |
| Фононы | 20990676 | Все обучения afterok |
| Сравнение 8 моделей | 20990677 | Все фононы afterok |

```
JOBID      STATE               NODELIST(REASON)
            20990677    PENDING                   (Dependency)
    20990676_[0-3%2]    PENDING                   (Dependency)
            20990674    PENDING                   (Dependency)
    20990673_[0-3%1]    PENDING                    (Resources)
    20990675_[0-3%1]    PENDING                   (Dependency)
```

| Индекс | XC | Маска | Дополнительные параметры | Всего trainable scalars |
|---:|---|---|---:|---:|
| 0 | PBE | restricted_mlp | 1088 | 21112778 |
| 1 | PBE | restricted_radial | 33152 | 21144842 |
| 2 | r2SCAN | restricted_mlp | 1088 | 21112778 |
| 3 | r2SCAN | restricted_radial | 33152 | 21144842 |

Базовая restricted-маска: I2/reducing_, rho/reducing_, I1/reducing_. Вариант MLP добавляет только LinMLPOut2ScalarTarget_; radial добавляет только R_Linear_ и R1_Linear_. Варианты независимы; совместного размораживания MLP+radial здесь нет.

Прочие настройки неизменны: тот же исходный GRACE-2L-OMAT-medium-base-fp64 checkpoint; train200/validation149; seed10; batch1; 50 эпох /10000 steps; Adam LR1e-3→1e-4 cosine; Huber E/F/stress16/32/128; clipnorm1; noEMA; fixed final checkpoint. Изменение только маски подтверждено configuration-audit.json.

Baseline B0/H0 и train-only alignment повторно используются с проверкой хешей. Aligned checkpoint скопированы внутри UMT в новый каталог; parent-модели и завершённые результаты не менялись. До первого шага проверяется полное совпадение initial parameter signatures, затем совпадение первых5 batch с parent-пилотами внутри XC. Frozen tensors контролируются на первом шаге, каждые1000 и на финальном шаге. Smoke-веса отбрасываются; научные пилоты начинают заново.

Сравнение выполняется CPU-job после фононов: raw/aligned validation E, F, stress; sorted-band RMSE; Γ doublet/singlet; мнимые моды mesh50³; время и peak TensorFlow allocator. Для каждой модели проверяются финальные хеши, 10000 шагов, export/numerical QC и 11 проверок frozen weights. Все восемь моделей должны иметь одинаковые train/validation IDs и первые5 batch внутри XC.

Автоматический результат будет сохранён на UMT в /home6/u3129/tial-fm-research/grace-n200-mask-ablation-20261109-01/comparison/summary.json и comparison/comparison.md. Сбор: scripts/manage_grace_pilot.py collect --run grace-n200-mask-ablation-20261109-01 --collection-name collected-final. Статус: та же команда с action status --brief.

На момент записи новые модели ещё не оценены. Текущее сравнение full/restricted seed10: [отчёт22](22_grace_n200_results.md). Новые тесты тоже используют только seed10; разброс по seeds этим запуском не измеряется. Test149 и архивные49 не используются.
