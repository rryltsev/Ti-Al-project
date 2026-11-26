#!/bin/bash -l

# Имя выходного файла
output_file="TiAl-SS-mace.xyz"

# Проверяем, существует ли выходной файл, и если да, удаляем его
if [ -f "$output_file" ]; then
    rm "$output_file"
fi

# Ищем все файлы xyx.xyz в поддиректориях и объединяем их в dataset.xyz
find . -type f -name "xyz.xyz" -exec cat {} + > "$output_file"

