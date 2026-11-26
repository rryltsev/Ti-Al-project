#!/bin/bash -l

# SLURM directives with readable formatting
#SBATCH --job-name=mace_md           # Имя задачи
#SBATCH --nodes=1                    # Количество узлов
#SBATCH --ntasks=1                   # Общее количество задач
#SBATCH --cpus-per-task=1            # Количество CPU на задачу
#SBATCH --time=1200                  # Лимит времени в минутах (1200 мин = 20 часов)
#SBATCH --mem=4G                     # Объем памяти на узел (4 ГБ)
#SBATCH --partition=apollo           # Очередь для CPU (заменяем tesla/apollo на нужную)

# Убираем переменные окружения для GPU, они не нужны
unset CUDA_VISIBLE_DEVICES

# Активируем виртуальное окружение
source /home/u2626/SOFT/anaconda3/bin/activate mace

# Запускаем скрипт молекулярной динамики
python mace_md_npt.py