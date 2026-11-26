#!/bin/bash
set -euo pipefail
export LC_ALL=C
src=/home6/u3129/tial-pd-20261124-atomistic/atat-build-sqs-01/atat3_52.tar.gz
echo "8536fb078375b30096df780e02f4e0c97972f5ebaad9a84a94f22e00f50c7516  $src" | sha256sum -c -
test "$(id -un)" = u3129
work=$(mktemp -d "${TMPDIR:-/tmp}/tial-lsfit-${SLURM_JOB_ID}-XXXXXX")
tar -xzf "$src" -C "$work"
dest="$PWD/bin"
mkdir "$dest"
cd "$work/atat"
./foolproof.sh "$dest" g++
make -j "${SLURM_CPUS_PER_TASK}" -C src lsfit
install -m 0755 src/lsfit "$dest"
cd "$dest"
sha256sum lsfit > ../binaries.sha256
