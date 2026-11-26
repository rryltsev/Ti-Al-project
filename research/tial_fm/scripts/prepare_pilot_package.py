"""Create an immutable, self-contained baseline/training input snapshot."""
from pathlib import Path
import hashlib,json,shutil,tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
ID='dpa4-n200-pilot-20261106-02';OUT=ROOT/'runs'/ID/'package';REMOTE='/home6/u3129/tial-fm-research/'+ID
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    full=ROOT/'datasets/common992-grouped-v1-20261106-02';small=ROOT/'datasets/nested-train-v1-20261106'
    for xc in ['PBE','PBEsol','r2SCAN']:
        for part in ['train','validation','test']:
            src=(small if part=='train' else full)/'deepmd'/xc/part
            shutil.copytree(src,OUT/'data'/xc/part)
    for name,src in [('subsets.json',small/'subsets.json'),('splits.json',full/'splits.json'),('data_manifest.jsonl',full/'data_manifest.jsonl'),('reference_figure.npz',PROJECT/'phonons/results/figure2-six-panels-20261101-02/figure-data.npz'),('build_pbesol_phonons.py',PROJECT/'phonons/scripts/build_pbesol_phonons.py')]:shutil.copyfile(src,OUT/name)
    for xc in ['PBE','PBEsol','r2SCAN']:
        src=PROJECT/'phonons/results'/(xc+'-DFT-pair-inputs-20261101') if xc!='PBEsol' else PROJECT/'phonons/results/PBEsol-phonons-preliminary-20261031-02/inputs'
        shutil.copytree(src,OUT/'harmonic_inputs'/xc)
    for name in ['baseline_worker.py','harmonic_worker.py']:shutil.copyfile(ROOT/'scripts'/name,OUT/name)
    records={r['geometry_id']:r for r in map(json.loads,(full/'data_manifest.jsonl').read_text().splitlines())}
    order=json.loads((small/'subsets.json').read_text())['subsets']['200'];probes=[]
    selected=[next(i for i in order if records[i]['composition'].get('Ti',0)==records[i]['natoms']),next(i for i in order if records[i]['composition'].get('Al',0)==records[i]['natoms']),next(i for i in order if len(records[i]['composition'])==2 and .3<records[i]['composition']['Al']/records[i]['natoms']<.7)]
    for sys in sorted((OUT/'data/PBE/train').iterdir()):
        ids=json.loads((sys/'geometry_ids.json').read_text());types=np.loadtxt(sys/'type.raw',dtype=int,ndmin=1);names=(sys/'type_map.raw').read_text().split()
        for i,sid in enumerate(ids):
            if sid in selected:probes.append(dict(id=sid,symbols=[names[t] for t in types],positions=np.load(sys/'set.000/coord.npy')[i].reshape(-1,3).tolist(),cell=np.load(sys/'set.000/box.npy')[i].reshape(3,3).tolist()))
    assert len(probes)==3;dump(OUT/'probes.json',probes)
    cfg=dict(id=ID,remote_root=REMOTE,model=dict(path='/home6/u3129/tial-testsets/pretrained-neo-20261022-01/DPA4-Neo-OMat24-v20260805.pt',sha256='fd7f34ae28f921201e4a0328ddf56179892752084f8363e7796038201471c989'),split_sha256=sha(full/'splits.json'),subsets_sha256=sha(small/'subsets.json'),authorization=dict(current_request='2026-11-06: N200, baseline, original phonons, pilot DPA4 PBE/r2SCAN',prior_policy_source='phonons/config/protocol.json',prior_policy_sha256=sha(PROJECT/'phonons/config/protocol.json'),no_new_DFT=True,cumulative_agent_imposed_limit=None),allocation=dict(account='imet-2026-2',CPU=dict(partition='apollo',cpus=4,memory='24G',time='00:45:00'),GPU=dict(partition='v100',gpus=1,cpus=4,memory='40G',time='04:00:00'),existing_jobs_not_modified=True),gates=dict(B0='native/export, symmetry, energy-force/virial finite differences; all evaluation artifacts saved',H0='DFT force/band reconstruction, matched original FM response, diagnostics'),pilot=dict(XCs=['PBE','r2SCAN'],methods=['lora16','full'],N=200,validation=149,test=149,seed=10,epochs=50,checkpoint_selection='fixed final 50 epoch checkpoint; development may diagnose convergence; no test-driven selection'))
    dump(OUT/'protocol.json',cfg)
    common='''#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$PWD/cache/$SLURM_JOB_ID"
mkdir -p "$XDG_CACHE_HOME"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
export DP_TF32_INFER=0 DP_AMP_INFER=0 DP_TRITON_INFER=0 DP_COMPILE_INFER=0
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate lammps_deepmd_dpa4
export CUDA_HOME=/opt/cuda-13.0
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
'''
    (OUT/'baseline.sbatch').write_text(common+'export CUDA_VISIBLE_DEVICES=""\npython baseline_worker.py\nexport PYTHONPATH=/home6/u3129/tial-phonons/envs/phonopy-4.8.2\n/home6/u3129/mlip-research/envs/kplib_111/bin/python harmonic_worker.py\n',encoding='utf8',newline='\n')
    dump(OUT/'FILES_SHA256.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
    archive=OUT.parent/'package.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():tf.add(p,arcname=p.relative_to(OUT).as_posix())
    dump(OUT.parent/'package.json',dict(id=ID,remote_root=REMOTE,path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size))
    print(json.dumps(dict(path=str(archive),size=archive.stat().st_size),indent=2))
if __name__=='__main__':main()
