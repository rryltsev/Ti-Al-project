#!/bin/bash
set -euo pipefail
export LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root=/home6/u3129/tial-pd-20261124-atomistic
py=/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python
lsfit="$root/atat-lsfit-build-01/bin/lsfit"
corrdump="$root/atat-build-sqs-01/atat/src/corrdump"
sha256sum -c inputs.sha256
sha256sum "$lsfit" "$corrdump"
mkdir fits
while read -r source ratio mode; do
    dest="fits/$source"
    mkdir "$dest"
    cp design.csv design.json lat.in clusters.out selected.out "$dest/"
    archive="$root/$source/labels.tar.gz"
    sha256sum "$archive" > "$dest/labels.sha256"
    opt=()
    if [[ $mode == static ]]; then opt+=(--static-only); fi
    (cd "$dest"; "$py" ../../fit_native.py --archive "$archive" --ratio "$ratio" \
      --lsfit "$lsfit" --corrdump "$corrdump" "${opt[@]}")
done < sources.txt
tar -czf fits.tar.gz fits
sha256sum fits.tar.gz
