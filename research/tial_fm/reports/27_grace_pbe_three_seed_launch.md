# GRACE N200 PBE: повторы seeds20/30

Запуск 2026-11-09T20:06:39.955565+00:00. Восемь новых обучений: full, restricted, restricted + MLP, restricted + radial × seeds20/30. Seed10 взят из четырёх завершённых моделей. r²SCAN не включён в новые входные данные и запуски.

Обучение: массив 20991808, максимум одна A100 одновременно. Постобработка: 20991810, afterany всего массива. Если какие-либо результаты отсутствуют, итоговый summary будет INCOMPLETE с перечислением ошибок; постоянной зависимости afterok здесь нет.

## Зафиксированные настройки

GRACE-2L-OMAT-medium-base-fp64; прежний PBE checkpoint с тем же train-only выравниванием; train200/validation149, неизменный grouped split; Adam, LR1e-3→1e-4 cosine, Huber E/F/stress16/32/128, batch1, 50эпох/10000updates. Выбор checkpoint: фиксированная финальная эпоха50. Test149 и архивные49 не используются. Единственное научное изменение относительно соответствующего seed10 — seed обучения.

Все восемь 5-step smoke прошли: исходные тензоры совпадают с source, optimizer свежий, frozen tensors неизменны. Первые пять batches совпадают между четырьмя масками одного seed и различаются между seeds20/30. Production сверяет первые batches со своим smoke; smoke-веса отбрасываются.

| Index | Case | Обучаемых параметров |
|---:|---|---:|
| 0 | PBE-full-N200-seed20 | 21764956 |
| 1 | PBE-restricted-N200-seed20 | 21111690 |
| 2 | PBE-restricted_mlp-N200-seed20 | 21112778 |
| 3 | PBE-restricted_radial-N200-seed20 | 21144842 |
| 4 | PBE-full-N200-seed30 | 21764956 |
| 5 | PBE-restricted-N200-seed30 | 21111690 |
| 6 | PBE-restricted_mlp-N200-seed30 | 21112778 |
| 7 | PBE-restricted_radial-N200-seed30 | 21144842 |

## Автоматическое сравнение

После обучения: numerical/export QC, validation149 с raw и train-aligned energy, F/stress; фононы TiAl в тех же DFT-ячейках; Ti₃Al в той же архивной Cmcm-геометрии, включая total/Al/Ti DOS, δ0.005/0.01/0.02Å, mesh14/20, raw/ASR. Скрипт агрегирования объединяет восемь новых результатов с четырьмя seed10, проверяет хеши моделей и одинаковые IDs, пересчитывает метрики, формирует среднее±sampleSD и парные различия с full по seeds10/20/30.

Ti₃Al: сравнение с архивной PBE-DOS; FORCE_SETS отсутствует, поэтому DFT-ошибка ветвей не вычисляется. Все физические проверки — development diagnostics. Они не доказывают сходимость DFT, независимость от pretraining или устойчивость длительной MD.

## Хранение

До запуска освобождено 2.712GiB: промежуточные checkpoints завершённых работ перенесены в15 локальных архивов с проверкой каждого SHA-256. Финальные модели и экспорты сохранены. Перед отправкой production подтверждено 5.948GiB свободной квоты.

Best-validation checkpoints не записываются, поскольку они не участвуют в выборе. Регулярный полный Adam checkpoint сохраняется во время обучения; после успешной оценки он сжимается в optimizer-final.tar.gz, содержимое проверяется по хешам до удаления несжатых дубликатов. Архив остаётся на UMT и позволяет восстановить optimizer state. Контроль квоты выполняется перед стартом и записью.

Результаты на момент отправки ещё не готовы; новых численных выводов по seeds нет. Итог на UMT: `/home6/u3129/tial-fm-research/grace-pbe-seeds-20261110-01/comparison/summary.json` и `comparison/comparison.md`. Локальный статус/сбор: `manage_grace_pbe_seeds.py status` / `collect`.

Артефакты: `runs/grace-pbe-seeds-20261110-01/configuration-audit.json`, `launch-gates.json`, `submission-summary.json`, `storage-recovery/verified.json`.
