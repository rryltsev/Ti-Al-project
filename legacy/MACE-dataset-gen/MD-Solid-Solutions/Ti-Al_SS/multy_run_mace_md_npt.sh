#!/bin/bash -l

# SLURM directives
#SBATCH --job-name=mace_md_batch    # Имя задачи
#SBATCH --nodes=1                   # Количество узлов
#SBATCH --ntasks=1                  # Общее количество задач
#SBATCH --cpus-per-task=1           # Количество CPU на задачу
#SBATCH --time=60                   # Лимит времени в минутах (1 час)
#SBATCH --mem=1G                    # Объем памяти на узел (1 ГБ)
#SBATCH --partition=apollo          # Очередь для CPU
#SBATCH --output=mace_md_batch_%j.out  # Лог-файл с ID задачи
#SBATCH --error=mace_md_batch_%j.err   # Ошибки с ID задачи

# Текущая директория
current_dir=$(pwd)

# Путь к папке RESULT
result_dir="$current_dir/RESULT"

# Проверка существования папки RESULT
if [ ! -d "$result_dir" ]; then
    echo "Ошибка: папка $result_dir не найдена!"
    exit 1
fi

# Обход всех папок первого уровня внутри RESULT
for poscar_dir in "$result_dir"/*; do
    if [ -d "$poscar_dir" ]; then
        echo "Обрабатываем папку: $poscar_dir"
        # Обход всех подпапок с редуцированными формулами
        for formula_dir in "$poscar_dir"/*; do
            if [ -d "$formula_dir" ]; then
                echo "  Заходим в подпапку: $formula_dir"
                cd "$formula_dir" || {
                    echo "Не удалось перейти в $formula_dir"
                    continue
                }
                # Проверка наличия run_mace_md_npt.sh
                if [ -f "run_mace_md_npt.sh" ]; then
                    echo "    Запускаем run_mace_md_npt.sh в $formula_dir"
                    sbatch run_mace_md_npt.sh
                else
                    echo "    Ошибка: run_mace_md_npt.sh не найден в $formula_dir"
                fi
                cd "$current_dir" # Возвращаемся в исходную директорию
            fi
        done
    fi
done

echo "Все задачи отправлены!"