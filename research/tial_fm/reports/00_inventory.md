# Первичная инвентаризация — 6 ноября 2026

Выполнен первый локальный аудит по программе. Этапы 1–3 начаты, но data gate пока не пройден.

## Проверенные данные

| XC | Всего меток | Train | Validation |
|---|---:|---:|---:|
| PBE | 999 | 850 | 149 |
| PBEsol | 998 | 849 | 149 |
| r²SCAN | 993 | 845 | 148 |

Выбранный пул: 1000 геометрий; 880×216, 104×250, 16×256 атомов. Все 2990 NPY-меток сопоставлены с выбранными геометриями при допуске 5×10⁻⁵ Å, с перестановками атомов, периодическими образами и общим переносом. Проверены конечность массивов, число записей и отсутствие exact geometry overlap внутри каждого исторического split. Cell axes сохранены; произвольные замены базиса не проверялись. Все исходные E/F/virial массивы прочитаны и хешированы.

Общий пул: **992 тройки labels**, 218346 атомов геометрий, 655038 атомов–меток. Общих исторических train-ID только **658**, validation-ID — **14**. Эти пересечения не являются новым независимым split.

Для 120 геометрий найден parent registry. Хеши всех 46 твёрдотельных траекторий совпали с архивом. Train/validation используют общие траектории: **11 PBE / 14 PBEsol / 11 r²SCAN**. **24 из 25** твёрдотельных структур внешнего benchmark имеют родителя, представленного в исходном пуле. Геометрическая неповторяемость не устраняет эту зависимость.

Для 880 геометрий жидкостного/релаксационного пула parent/frame пока не восстановлены. Целевой поиск проведён в MACE-database, MD-liquid+relax/RESULT, исходных scripts, reference_origins и trajectory_audit; также просмотрено содержимое двух известных ZIP. В 220 каталогах родителей есть initial POSCARs; локально найдены только две trajectory.xyz. ZIP AlTi_liq_configs содержит 11 начальных структур; select-200 ZIP содержит геометрии, а не полный parent registry. Это ограничение локальной доступности, не доказательство утраты данных на всех носителях.

## Артефакты и пути

- Исходный проект: `C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project`.
- Геометрии: `MACE-dataset-gen/MACE-database/TiAl-liq-relax-ss_select-1000.xyz`.
- NPY labels: `DPA4-finetune/TiAl_{PBE,PBEsol,scan}_upd/npy_format/{train,validate}`.
- Raw VASP candidates: `VASP-output/r2SCAN/RESULTS`, `VASP-output/VASP_PBEsol/RESULTS`; их SCF и связь каждого energy field с обучающей меткой требуют отдельного аудита. Полный PBE raw archive пока не установлен.
- Актуальные main/SI: `manuscript/main.tex`, `manuscript/supporting_materials.tex`; main имеет дату 03.11, sections содержат последующие правки KRP. Сохранены фактические исходники; PDF 03.11 и 06.11 не смешиваются.
- Родительская FM: `testset/monitor/models/pretrained/DPA4-Neo-OMat24-v20260805.pt`; SHA-256 fd7f34ae28f921201e4a0328ddf56179892752084f8363e7796038201471c989.
- FT checkpoints/configs/logs: `TiAl-Project/DPA4-checkpoints`, `DPA4-finetune`; frozen exports и метаданные: `testset/monitor/models`.
- Фононы: `phonons/runs/alpha-ti-xc-phonons-20261021-01`, `phonons/results/figure2-six-panels-20261101-02`; raw forces, POSCAR, Phonopy YAML, arrays, processing scripts найдены.
- MD: `TiAl-Project/MD-results`, `MD-scripts`, локальные `audit/oct25`, `audit/oct29`, `audit/nov01`, `audit/nov05`, `audit/nov06`. Это установленные locations; полный MD provenance audit ещё не выполнен.
- GRACE: подтверждено существующее TensorPotential 0.6.1 / TensorFlow 2.20.0 окружение; старый registry содержит GRACE-1L, pinned GRACE-2L пока не установлен. Не заменять 2L на 1L молча.
- DPA4C: исторические synthetic training checks в MLIP-research; Ti–Al training/export/MD qualification ещё нет.

## Snapshot и границы проверки

В `runs/initial-audit-20261106-01/snapshot` сохранены копии main/SI, sections, supporting tables/configs и плана. `source_hashes.json` фиксирует **5025** файлов. Weights/большие данные остаются в исходном хранилище; manifest не считается их резервной копией. Идентификатор geometry основан на выбранном source/frame и проверенном структурном сопоставлении, а не на равенстве файловых хешей.

Источник исторических IDs — reference_origins.json; все привязки NPY повторно проверены по структурам. Единый data_manifest и historical splits доступны в `manifests/`. Near-duplicate audit нового grouped split, полный SCF/DFT-settings audit, гармоническая исходная FM, runtime parameter counts и budget нового цикла остаются открытыми.

Изменений численных результатов рукописи, нового DFT и нового обучения: **0**.
