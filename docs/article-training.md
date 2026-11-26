# Чекпойнты и обучающие данные статьи

В `legacy/article-training/models/` лежат три исходных чекпойнта статьи, по которым считались опубликованные метрики и свойства. Исходные имена, пути архива, байты и SHA-256 сведены в [реестр файлов](../legacy/article-training/files.csv). [Реестр выбора](../research/tial_fm/manifests/models.json) фиксирует дату замены PBE 15k на 20k после просмотра результатов Test50.

| Файл | Функционал | SHA-256 исходного .pt |
|---|---|---|
| [PBE.pt](../legacy/article-training/models/PBE.pt) | PBE, 20 000 шагов | `3b0b4b69f8fca7be5bae4a5f24498aac904efad08f126f4a57de035e0379c67e` |
| [PBEsol.pt](../legacy/article-training/models/PBEsol.pt) | PBEsol | `5a5359a713848d98a7746c70b46be65313d7d990c6bf39d68b40908dd9c7c295` |
| [r2SCAN.pt](../legacy/article-training/models/r2SCAN.pt) | r²SCAN | `8b2c339bf472564aee4ff62a2b2f390349d2d8fa20617e38d09c63e3d03db309` |

В [протоколах](../legacy/article-training/protocol/) сохранены команды дообучения, конфигурации и кривые потерь. Семейство вложений `article-dataset-*.zip` хранит исходные XYZ с DFT-метками для каждого функционала, файлы выбора train/validation и готовые массивы обучения в формате DeePMD. В `article-historical-models-*.zip` собраны контрольные точки отдельных шагов, исходные настройки, результаты диагностики и исторические экспорты. Внутри ZIP сохранены пути от корня исходной папки TiAl-Project и перечень `FILES.sha256`. Каждый архив можно распаковать в одну общую рабочую папку.

Независимое исследование full FT, LoRA и масок GRACE использовало другое групповое разбиение и другие чекпойнты. Для его повторения используйте наборы N200 и N694 и версии, указанные в [исследовательском руководстве](research.md). Метрики статьи можно пересчитать из сохранённых прогнозов командой `python testset/verify_archived_metrics.py`. Новые вычисления DPA4 требуют валидации Triton2-экспорта выбранного веса по E/F/stress. [Политика и параметры экспорта](workflows/mlip-deployment/SKILL.md) · [условия распространения моделей](model-notices.md).
