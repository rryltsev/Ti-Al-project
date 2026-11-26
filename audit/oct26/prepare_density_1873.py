"""Prepare the PBE-15k density rerun from the archived 1873 K liquids."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'audit/oct20/deps'))
import numpy as np
from ase.io import read, write

OUT = HERE / 'pbe15k_density_1873'
OUT.mkdir(exist_ok=True)
assert not (OUT / 'protocol.json').exists(), 'Already prepared'
OLD = ROOT / 'audit/oct25/pbe_full_enthalpy_gpu'
old = json.loads((OLD / 'protocol.json').read_text())
base = Path(old['states'][0]['initial_source']).parent.parent.parent / '1873K/pbe'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')

states, baseline = [], {}
for i, previous in enumerate(sorted(old['states'], key=lambda s: s['x_Al'])):
    name, x = previous['composition'], previous['x_Al']
    source = base / (name + '_1873K.traj')
    frames = read(source, index=':')
    assert len(frames) == 301, (name, len(frames))
    atoms = frames[-1].copy()
    assert len(atoms) == 200 and atoms.get_chemical_symbols().count('Al') == round(x * 200)
    assert atoms.pbc.all() and not atoms.constraints
    atoms.calc = None
    initial_file = name + '.traj'
    write(OUT / initial_file, atoms)
    reread = read(OUT / initial_file)
    assert np.array_equal(reread.numbers, frames[-1].numbers)
    assert np.allclose(reread.positions, frames[-1].positions, rtol=0, atol=1e-12)
    assert np.allclose(reread.cell, frames[-1].cell, rtol=0, atol=1e-12)
    rows = []
    for j, a in enumerate(frames):
        volume = a.get_volume()
        rows.append(dict(step=j*100, time_ps=j*.1, T_K=a.get_temperature(),
                         V_A3_atom=volume/len(a),
                         rho_g_cm3=float(a.get_masses().sum()/volume*1.66053906660)))
    baseline[name] = dict(source=str(source), sha256=sha(source), x_Al=x, rows=rows)
    states.append(dict(composition=name, x_Al=x, initial_file=initial_file,
                       initial_sha256=sha(OUT / initial_file), initial_source=str(source),
                       initial_source_sha256=sha(source), initial_frame=300,
                       bulk_modulus_GPa=110*(1-x)+76*x, seed=20261026+i*101))

tasks = [dict(id=i, checkpoint=15000, state=i) for i in range(11)]
save('tasks.json', tasks)
save('baseline_5k.json', baseline)
cfg = {**old, 'remote': '/home6/u3129/tial-testsets/pbe15k-density-1873-20261026-01',
       'models': {'15000': old['models']['15000']}, 'states': states,
       'tasks_sha256': sha(OUT/'tasks.json'), 'T_K': 1873,
       'allocation_wall_hours': 16,
       'scope': 'Density versus Al atomic fraction at 1873 K; PBE checkpoint 15000; 11 compositions, one A100, sequential runs.',
       'initialization': 'Final archived 5k liquid configuration at 1873 K; new seeded velocities; 30 ps equilibration and 30 ps production. Velocities and NPT integrator reinitialized at the stage boundary, matching the prior 2000 K rerun.',
       'comparison': 'Archived 1873 K density baseline retained. New MD uses checkpoint 15000; no statistical pairing assumed.',
       'density': 'Time average of instantaneous total cell mass divided by volume; g/cm^3.',
       'uncertainty': 'Conservative SEM from 3, 5 and 6 ps blocks; half-production drift; save full trajectories for liquid-state and convergence assessment.',
       'validation': 'Checkpoint hash and training step, input hashes and composition, CPU/GPU energies/forces/stresses on all 11 starting configurations, timed 100-step pilot.'}
save('protocol.json', cfg)
shutil.copyfile(OLD/'md_common.py', OUT/'md_common.py')
worker = (OLD/'gpu_enthalpy_worker.py').read_text()
worker = worker.replace('One GPU, one checkpoint, all compositions without further queue submissions.',
                        'One A100, PBE 15000, density series at 1873 K.')
worker = worker.replace('projected=seconds/100*60000*11/3600',
                        "projected=seconds/100*(cfg['equil_steps']+cfg['production_steps'])*len(cfg['states'])/3600")
(OUT/'density_worker.py').write_text(worker, encoding='utf-8', newline='\n')
job = '''#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export DP_INTRA_OP_PARALLELISM_THREADS=4 DP_INTER_OP_PARALLELISM_THREADS=1
export DP_TF32_INFER=0 DP_AMP_INFER=0 DP_TRITON_INFER=0 DP_COMPILE_INFER=0
export LOCAL_RANK=0
unset DEVICE
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate lammps_deepmd_dpa4
export CUDA_HOME=/opt/cuda-13.0
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$PWD/cache"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor"
export PYTORCH_KERNEL_CACHE_PATH="$PWD/cache/torch/kernels"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$PYTORCH_KERNEL_CACHE_PATH"
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv > gpu-at-start.txt
srun --exclusive --exact -N1 -n1 -c4 --gres=gpu:a100:1 --mem=28G --cpu-bind=cores python -u density_worker.py 15000 > worker-15000.log 2>&1
python -u summarize_density.py > analysis.log 2>&1
'''
(OUT/'job.sbatch').write_text(job, encoding='utf-8', newline='\n')
(OUT/'README_RU.txt').write_text(
    'Пересчёт зависимости плотности от состава при 1873 K, PBE 15000.\n'
    '11 составов x_Al=0,0.1,...,1; 200 атомов; 1 бар; шаг 1 фс.\n'
    'На каждый состав: 30 пс уравновешивания + 30 пс production.\n'
    'Исходные структуры: последние кадры архивных жидких траекторий 5k при 1873 K.\n'
    'Одна A100, 4 CPU, 32 GiB, лимит 16 часов.\n'
    'Проверка CPU/GPU и 100-шаговый пилот выполняются до основной серии.\n'
    'Все траектории и скалярные логи сохраняются; после серии автоматически создаются density_1873K.csv и density_summary.json.\n'
    'Блочные ошибки и дрейф являются диагностикой; жидкое состояние требует проверки траекторий.\n',
    encoding='utf-8')
for name in ['md_common.py', 'density_worker.py']:
    compile((OUT/name).read_text(), str(OUT/name), 'exec')
save('preparation-verification.json', dict(status='passed', compositions=11,
     x_Al=[s['x_Al'] for s in states], atoms_per_composition=200,
     initial_frames_match_sources=True, baseline_frames_per_composition=301,
     protocol_sha256=sha(OUT/'protocol.json'), checkpoint_sha256=cfg['models']['15000']['sha256']))
print('Prepared 11 compositions at 1873 K, PBE 15000, one A100')
