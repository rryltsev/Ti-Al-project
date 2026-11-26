# Baseline: воспроизведение сохранённых предсказаний

Независимо пересчитаны raw energy, Cartesian force и stress residuals для 49 общих структур, трёх FT и одной неизменённой FM. SHA-256 модели, POSCAR, порядок элементов и SCF-флаг каждой используемой метки проверены. MAE/RMSE/bias совпадают с сохранённой неокруглённой таблицей (rtol 1e-9, atol 1e-10). Хвосты и residuals сохранены в tables.

| XC | Energy RMSE, meV/atom | Force RMSE, eV/Å | Stress RMSE, GPa |
|---|---:|---:|---:|
| PBE | 1.467424850 | 0.062624657 | 0.072420548 |
| PBEsol | 1.765577387 | 0.062244827 | 0.069686309 |
| r2SCAN | 1.578391872 | 0.071359645 | 0.073083852 |

PBE здесь **20k**, в harmonic/physical branch **15k**; PBEsol=23100, r²SCAN=23200. Это разные baseline identities. Старые PBE5k/15k не перезаписывались.

Причина 50→49: r²SCAN для TiAl-test-025 отменён 04.11 до начала выполнения. PBE/PBEsol labels для этого ID существуют; они не удаляются. Для тройного сравнения используется общий набор49; для одиночных анализов неполная тройка сохраняется отдельно. Источник: `testset/monitor/runs/early-stop-20261104/closure.json`.

Это повторная арифметическая оценка cached predictions, **0 новых model evaluations**. Новые native/export, finite differences, equivariance и train-only alignment не выполнялись; полный Gate B0 открыт. Исторические интервалы не пересчитывались как independent-parent bootstrap. Полученное совпадение не доказывает отсутствие parent leakage.
