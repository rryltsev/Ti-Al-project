#!/bin/bash
set -eu -o pipefail
export LC_ALL=C
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" windows.tsv)
read -r model T nx ny nz c0 mu seed equil prod start coefficient_file step observables <<< "$line"
file="samples/${SLURM_ARRAY_TASK_ID}.csv"
./vcs motifs.txt "${coefficient_file:-${model}.txt}" "$nx" "$ny" "$nz" "$T" "$c0" 100 "$mu" "$equil" "$prod" 10 "$seed" "$start" "$model" "${step:-0}" "$file.tmp" ${observables:+"$observables"}
gzip -c "$file.tmp" > "$file.gz.part"
mv "$file.gz.part" "$file.gz"
rm -- "$file.tmp"
