#!/bin/bash -l

# SLURM directives
#SBATCH --job-name=mace_md_array    # Имя задачи
#SBATCH --nodes=1                   # Количество узлов
#SBATCH --ntasks=1                  # Общее количество задач на узел
#SBATCH --cpus-per-task=1           # Количество CPU на задачу
#SBATCH --time=30                   # Лимит времени в минутах (30 мин для подготовки массива)
#SBATCH --mem=1G                    # Объем памяти на узел
#SBATCH --partition=apollo          # Очередь для CPU
#SBATCH --output=mace_md_array_%j.out  # Лог-файл с ID задачи
#SBATCH --error=mace_md_array_%j.err   # Ошибки с ID задачи
#SBATCH --array=0-999%999            # Массив задач (до 1000, максимум 10 одновременно)

# Текущая директория
current_dir=$(pwd)
result_dir="$current_dir/RESULT"

# Проверка существования папки RESULT
if [ ! -d "$result_dir" ]; then
    echo "Ошибка: папка $result_dir не найдена!"
    exit 1
fi

# Собираем все подпапки с run_mace_md_npt.sh в массив
mapfile -t formula_dirs < <(find "$result_dir" -type f -name "run_mace_md_npt.sh" -exec dirname {} \;)

# Проверка, есть ли задачи
if [ ${#formula_dirs[@]} -eq 0 ]; then
    echo "Ошибка: не найдено ни одной папки с run_mace_md_npt.sh!"
    exit 1
fi

# Определяем текущую задачу из массива
task_index=$SLURM_ARRAY_TASK_ID

# Проверяем, что индекс в пределах массива
if [ "$task_index" -ge "${#formula_dirs[@]}" ]; then
    echo "Задача $task_index превышает количество папок (${#formula_dirs[@]}) - пропускаем."
    exit 0
fi

# Переходим в нужную папку и запускаем скрипт
formula_dir="${formula_dirs[$task_index]}"
echo "Обрабатываем папку: $formula_dir"
cd "$formula_dir" || {
    echo "Не удалось перейти в $formula_dir"
    exit 1
}
if [ -f "run_mace_md_npt.sh" ]; then
    echo "Запускаем run_mace_md_npt.sh в $formula_dir"
    sbatch run_mace_md_npt.sh
else
    echo "Ошибка: run_mace_md_npt.sh не найден в $formula_dir"
fi

echo "Задача $task_index завершена!"