#!/bin/bash -l

# SLURM directives
#SBATCH --job-name=mace_md_relax_batch   # Имя задачи
#SBATCH --nodes=1                        # Количество узлов
#SBATCH --ntasks=1                       # Общее количество задач
#SBATCH --cpus-per-task=1                # Количество CPU на задачу
#SBATCH --time=60                        # Лимит времени в минутах (1 час)
#SBATCH --mem=1G                         # Объем памяти на узел (1 ГБ)
#SBATCH --partition=apollo               # Очередь для CPU
#SBATCH --output=mace_md_relax_batch_%j.out  # Лог-файл с ID задачи
#SBATCH --error=mace_md_relax_batch_%j.err   # Ошибки с ID задачи

# Текущая директория
current_dir=$(pwd)

# Путь к папке RESULT
result_dir="$current_dir/RESULT"

# Проверка существования папки RESULT
if [ ! -d "$result_dir" ]; then
    echo "Ошибка: папка $result_dir не найдена!"
    exit 1
fi

# Обход всех папок первого уровня в RESULT
for outer_dir in "$result_dir"/*; do
    if [ -d "$outer_dir" ]; then
        echo "Обрабатываем папку: $outer_dir"
        # Обход всех подпапок (второй уровень)
        for inner_dir in "$outer_dir"/*; do
            if [ -d "$inner_dir" ]; then
                script_path="$inner_dir/run_mace_md_npt_relax.sh"
                if [ -f "$script_path" ]; then
                    echo "  Запускаем скрипт: $script_path"
                    cd "$inner_dir" || {
                        echo "Не удалось перейти в $inner_dir"
                        continue
                    }
                    sbatch run_mace_md_npt_relax.sh
                    cd "$current_dir"
                else
                    echo "  Скрипт run_mace_md_npt_relax.sh не найден в $inner_dir"
                fi
            fi
        done
    fi
done

echo "Все задачи отправлены!"
