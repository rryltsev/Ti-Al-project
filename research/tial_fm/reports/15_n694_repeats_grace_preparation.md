# N=694: seeds 20/30; подготовка GRACE-2L

Обновлено UTC: 2026-11-07T15:59:48.232300+00:00. Предыдущие 16 DPA4 моделей завершены; добавлены 8 новых N694 повторов.

GPU array **20989407**, CPU QC **20989409**, итоговая агрегация **20989410**. Максимум одна новая GPU-задача одновременно. Оценка 8 повторов 21.71 GPU·ч основана на фактических 10.854 GPU·ч четырёх seed 10; очередь в оценку не входит. Лимит каждого запуска 4 ч, 4 CPU, 40 GB.

| Task | XC | Метод | Seed | Последний подтверждённый шаг |
|---|---|---|---:|---:|
| 20989407_0 | PBE | lora16 | 20 | 3800/34700 |
| 20989407_1 | PBE | full | 20 | 0/34700 |
| 20989407_2 | r2SCAN | lora16 | 20 | 0/34700 |
| 20989407_3 | r2SCAN | full | 20 | 0/34700 |
| 20989407_4 | PBE | lora16 | 30 | 0/34700 |
| 20989407_5 | PBE | full | 30 | 0/34700 |
| 20989407_6 | r2SCAN | lora16 | 30 | 0/34700 |
| 20989407_7 | r2SCAN | full | 30 | 0/34700 |

Рецепт N694 seed 10 сохранён: 50 epochs, исходный DPA4-Neo, LR 1e-4→1e-6, HybridMuon, LoRA rank/alpha 16 или full FT. Отличаются training seed и пути. Все 1931 staged-файла и 8 input-verifiers проверены; завершённые seed 10 checkpoint хеши подтверждены. Начальный PBE–LoRA seed 20: zero-step PASS, 22 adapter-тензора и порядок первых 100 batch отличаются от seed 10; исходные pretrained-тензоры совпадают.

После каждого обучения автоматически выполняются validation 149, гармоники и метрики. После массива — numerical QC, физические диагностики и агрегация всех 12 N694 моделей: mean/sample SD, парные LoRA−full разности, проверка независимых seeds. Отдельно сохраняется парный seed 10 контроль размера N=200→694.

## Подготовка GRACE

Выбран **GRACE-2L-OMAT-medium-base-fp64**, revision `4aff666029967745f7993f348eb87a804d625cff`, SHA-256 архива `763b7e5679c81ea82d39e595e701d0c9f6565064393f705cbe76663c9583d0aa`. Окружение TensorPotential 0.6.1 / TensorFlow 2.20.0. Фиксируется medium/base/OMat24/fp64; выводы не переносятся на FS, SMAX или другие размеры.

Старая small GRACE-2L-OMAT была скачана при инвентаризации и сохранена как невыбранный кандидат; она не используется в pilot package. Официальная документация помечает small как deprecated. [Документация foundation models](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/).

Подготовлены 4 native draft inputs: PBE/r²SCAN × full/restricted, N=200, seed 10, 50 epochs. Ограниченная маска использует штатный substring-фильтр `I2/reducing_`, `rho/reducing_`, `I1/reducing_`; собственной схемой LoRA она не называется. Все элементы исходной модели сохранены, energy shift пока не применяется.

NPY→extxyz: 200 train и149 validation для каждого XC. 17 значащих цифр, обратное чтение ASE даёт нулевую численную разницу E/F/cell/positions/symmetric stress. Stress=−sym(virial)/volume, eV/Å³. GRACE `test_filename` указывает только на validation 149; held-out test 149 отсутствует в пакете.

CPU preflight job: 20989422; статус: PASS.
Фактические обучаемые скаляры: full **21,764,956**, restricted **21,111,690** (97.00% full). Маска содержит 12 тензоров и замораживает только около 3% исходных trainable параметров: это ограничение типов обучаемых тензоров, а не аналог экономии числа параметров DPA4 LoRA. Native mask совпадает с ожидаемым списком и не изменяет значения весов; backend успешно прочитал все 4 файла данных.
Первая CPU-проверка 20989421 завершилась ошибкой проверочного хеша строковых TensorFlow metadata: numpy object-array.tobytes() содержит адреса объектов. Попытка и лог сохранены. Во второй попытке 20989422 хешируются значения строк, все численные веса остаются неизменны; PASS за 10 с Slurm. Обучение и optimizer steps в обеих попытках отсутствуют. Foundation checkpoint не содержит optimizer slots; при будущей загрузке сохраняется требование fresh Adam и сброса epoch/step.

GRACE training ещё не отправлен. Следующие проверки: собственные B0/H0, нулевой шаг и frozen tensors после реального optimizer-step, свежий Adam без foundation optimizer state, короткое профилирование A100. Черновой GRACE Huber recipe E/F/stress 16/32/128, LR 1e-3→1e-4, batch 1 требует этого профиля; он не объявляется эквивалентным DPA4 MAE E/F/virial. Выбор фиксированного final epoch 50, без best-validation checkpoint.

Новых DFT 0. Test 149 и архивные 49 не используются для настройки. Фононная устойчивость не заменяет длинный MD-тест.

[Предыдущее сравнение 16 моделей](14_multiseed_fulltrain_comparison.md).
