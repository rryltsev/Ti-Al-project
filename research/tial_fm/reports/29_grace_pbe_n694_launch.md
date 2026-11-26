# GRACE PBE: N694 full и restricted + radial, три seeds

Отправлено 2026-11-10T03:09:21.260034+00:00. Обучение 20991849, сравнение 20991850. Шесть моделей, максимум одна новая A100 одновременно. r²SCAN не включён в обучение и оценку.

## Сопоставимость с N200

Train694 — полный разрешённый grouped train; N200 строго вложен. Все200 общих структур и их метки совпадают точно. Validation149 совпадает побайтно. Группы train/validation не пересекаются; условность реконструкции родителей при утраченных жидкостных траекториях сохраняется. Test149 и архивные49 не используются.

GRACE-2L-OMAT-medium-base-fp64, прежние исходные FM weights и backend. Каждый запуск начинается с исходной модели и нового Adam, без переноса весов N200. Энергетические сдвиги Al/Ti оценены заново только по train694. Baseline, numerical/export QC и фононы исходной модели пройдены.

При submission optimizer gate: PENDING; завершены smoke: PBE-full-N694-seed10, PBE-restricted_radial-N694-seed10, PBE-full-N694-seed20. Все завершённые smoke PASS, исходные веса идентичны. Production ждёт итоговой проверки20991845 и до обучения требует PASS всех6 smoke. Проверяются frozen tensors, свежий Adam и одинаковые первые5 batches внутри seed. Smoke-веса отбрасываются.

Adam LR1e-3→1e-4 cosine; Huber E/F/stress16/32/128, delta0.01; batch1, clipnorm1, noEMA, fp64. 50 эпох =34700 updates; N200 имел50 эпох/10000 updates. Это сравнение при одинаковом числе эпох, а не при одинаковом числе обновлений или GPU-времени. Checkpoint фиксирован заранее: финальная эпоха50.

| Index | Модель | Обучаемых параметров |
|---:|---|---:|
| 0 | PBE-full-N694-seed10 | 21764956 |
| 1 | PBE-restricted_radial-N694-seed10 | 21144842 |
| 2 | PBE-full-N694-seed20 | 21764956 |
| 3 | PBE-restricted_radial-N694-seed20 | 21144842 |
| 4 | PBE-full-N694-seed30 | 21764956 |
| 5 | PBE-restricted_radial-N694-seed30 | 21144842 |

## Последующая оценка

Для каждого нового финального checkpoint: numerical/export QC; train694/validation149 E raw и train-aligned, F, stress; фононы TiAl в тех же DFT-ячейках и на том же пути; Ti₃Al total/partial DOS в неизменной архивной Cmcm-геометрии, δ0.005/0.01/0.02Å, mesh14/20 и ASR sensitivity. Ti₃Al сравнивается с сохранённой PBE-DOS, DFT FORCE_SETS отсутствует.

CPU-сравнение afterany всего массива объединяет6 новых N694 и6 завершённых N200 (full/radial × seeds10/20/30). Проверяет хеши checkpoint и геометрии, membership, frozen tensors, optimizer counters, пересчитывает метрики и выдаёт mean±sampleSD, парные различия N694−N200 и radial−full. При пропусках/ошибках сохраняет INCOMPLETE с причинами. afterany предотвращает вечное ожидание DependencyNeverSatisfied, но не скрывает ошибки.

Результаты: `/home6/u3129/tial-fm-research/grace-pbe-n694-20261110-01/comparison/summary.json` и `comparison/comparison.md`. Локальный статус/сбор: `manage_grace_n694.py status` / `collect`. Численных результатов N694 на момент отправки ещё нет.

Первый smoke20991836 и gate20991837 завершились ошибкой: baseline успел завершиться раньше архивирования, и quota guard остановил smoke до загрузки модели/создания optimizer. После освобождения квоты smoke повторён; исходные логи сохранены в recovery/smoke-quota-20991836. Порог защиты не снижался.

## Ресурсы и хранение

Production:6 × максимум90min,1A100,4CPU,32GiB RAM. Это верхний предел9 GPU-hours, не прогноз фактического расхода; baseline/smoke учитываются отдельно. Финальные модели/экспорты остаются на UMT, Adam checkpoints после успешной оценки сжимаются с проверкой SHA-256.

Перед запуском перенесены в локальные проверенные архивы только завершённые промежуточные checkpoints и optimizer archives: освобождено 2.951GiB. Перенос исторических r²SCAN-файлов относится только к хранению; новых r²SCAN-расчётов нет. Свободная квота при launch: 4.343GiB.

Артефакты: `configuration-audit.json`, `local-verification.json`, `remote-preflight.json`, `launch-gates.json`, `submission-summary.json`, `storage-recovery/verified.json` в каталоге запуска.
