#!/bin/bash
set -euo pipefail
export LC_ALL=C
sha256sum -c inputs.sha256
echo "8536fb078375b30096df780e02f4e0c97972f5ebaad9a84a94f22e00f50c7516  atat3_52.tar.gz" | sha256sum -c -
test "$(id -un)" = u3129
hostname
g++ --version
work=$(mktemp -d "${TMPDIR:-/tmp}/tial-atat-${SLURM_JOB_ID}-XXXXXX")
tar -xzf atat3_52.tar.gz -C "$work"
dest="$PWD/atat/src"
mkdir -p "$dest"
cd "$work/atat"
./foolproof.sh "$dest" g++
make -C src -j "${SLURM_CPUS_PER_TASK}" genstr corrdump maps cv mcsqs gensqs
for tool in genstr corrdump maps cv mcsqs gensqs; do
    install -m 0755 "src/$tool" "$dest/$tool"
done
cd "$dest"
sha256sum genstr corrdump maps cv mcsqs gensqs > ../../binaries.sha256
./genstr -h > ../../genstr-help.txt 2>&1 || test -s ../../genstr-help.txt
./corrdump -h > ../../corrdump-help.txt 2>&1 || test -s ../../corrdump-help.txt
cat ../../binaries.sha256
