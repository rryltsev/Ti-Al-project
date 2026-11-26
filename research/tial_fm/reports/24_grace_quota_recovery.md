# GRACE: восстановление после исчерпания квоты

Снимок: 2026-11-09T11:33:21.037168+00:00.

20990673_0–3 и optimizer gate20990674 завершились PASS. PBE restricted+MLP (20990675_0) прошёл10000 шагов и final validation/QC PASS. Остальные три элемента массива20990675 остановились после200 шагов при первой записи checkpoint.best_test_loss: ResourceExhaustedError / Disk quota exceeded.

Причина: квота аккаунта на /home6 —100GiB; свободно было около430MiB, тогда как один checkpoint с состоянием Adam —512–522MB. Общая свободная ёмкость файловой системы не отражает доступную пользовательскую квоту. В первоначальном запуске отсутствовали quota-aware preflight и проверка записи checkpoint в smoke.

Промежуточные checkpoint 5 завершённых GRACE-моделей перенесены в локальные .tar.gz архивы. Для каждого исходного файла проверены размер и SHA-256 внутри локального архива, перед удалением повторно проверены source SHA-256 и сохранность final checkpoint. Освобождено 5,161,176,468 bytes (4.807GiB). На UMT в папках checkpoints сохранены LOCAL_ARCHIVE.json с точным местом копии и hashes. Финальные модели, экспорт, исходные веса, данные и результаты не удалялись.

В повторах отключена только запись неиспользуемого best-validation checkpoint. Протокол изначально выбирает fixed final epoch50; best-reload уже был отключён. Periodic checkpoint с полным состоянием optimizer каждые5 эпох сохраняется. Схема Adam, LR, loss, batch order, seed, split и маски не менялись.

Добавлены проверки пользовательской квоты: минимум1.5GiB до старта и640MiB перед periodic checkpoint/final export. Отдельный CPU probe требует минимум4GiB, реально записывает512MiB и проверяет fsync; временный probe-файл удаляется. Эти проверки обнаружат новые ограничения квоты, но не резервируют место от других задач аккаунта.

Неудачные попытки целиком сохранены в recovery/disk-quota-v1/cases. Все три повтора начинают со step0 и fresh Adam, чтобы сохранить тот же протокол; успешная PBE+MLP модель не переобучается.

| Этап | Job | Условие |
|---|---|---|
| Проверка квоты и записи512MiB | 20990789 | CPU |
| Повторы индексов1–3 | 20990790 | storage probe afterok |
| Фононы всех4 моделей | 20990676 | retry afterok; успешный index0 сохранён |
| Сравнение с full/restricted | 20990677 | фононы afterok |

```
JOBID      STATE               NODELIST(REASON)
            20990677    PENDING                   (Dependency)
    20990676_[0-3%2]    PENDING                   (Dependency)
    20990790_[2-3%1]    PENDING            (JobArrayTaskLimit)
          20990790_1    RUNNING                     tesla-a101
```

Подробный manifest архивов: storage-recovery/archive-plan.json, verified-archives.json, released.json. Изменения кода и политики: storage-recovery/repair.json и new-files/. Старые код и manifest сохранены в remote recovery/disk-quota-v1/.
