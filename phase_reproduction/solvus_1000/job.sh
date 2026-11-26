#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@90
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
test "$(id -un)" = u3129
sha256sum -c inputs.sha256
echo '704f7fd900c7446286066b741084c30ebc7fcf26e5b9b8fd2f29a18ba4ffd661  fitted.mtp' | sha256sum -c -
CXX=${CXX:-g++}
"$CXX" --version
"$CXX" -std=c++17 -O3 -march=native sgmc.cpp -o sgmc
sha256sum sgmc
py=/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python
"$py" -u run_windows.py --workers "${SLURM_CPUS_PER_TASK:-1}" &
driver=$!
trap 'kill -TERM "$driver" 2>/dev/null || true' USR1 TERM
set +e
while true; do
    wait "$driver"
    rc=$?
    kill -0 "$driver" 2>/dev/null || break
done
if test -f results.tar.gz; then sha256sum results.tar.gz window_status.json; fi
exit "$rc"
