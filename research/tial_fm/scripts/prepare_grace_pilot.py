from pathlib import Path
import hashlib,json,shutil,tarfile
ROOT=Path(__file__).resolve().parents[1];ID='grace-n200-fp64-pilot-20261109-02';RUN=ROOT/'runs'/ID;OUT=RUN/'package'
REMOTE='/home6/u3129/tial-fm-research/'+ID;ENV='/home6/u3129/mlip-research/envs/grace_ft_a100'
OLD=ROOT/'runs/grace-2l-preparation-20261107-01';DPA=ROOT/'runs/dpa4-n694-seeds-20261107-01/package'
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    shutil.copytree(OLD/'data',OUT/'data');shutil.copytree(DPA/'harmonic_inputs',OUT/'harmonic_inputs')
    for f in ['probes.json','harmonic_worker.py','build_pbesol_phonons.py','reference_figure.npz']:shutil.copyfile(DPA/f,OUT/f)
    for f in ['grace_pilot_common.py','grace_pilot_baseline.py','grace_pilot_harmonics.py']:shutil.copyfile(ROOT/'scripts'/f,OUT/f)
    shutil.copyfile(OLD/'data-preflight.json',OUT/'data-preflight.json')
    cases=[]
    for xc in ['PBE','r2SCAN']:
        for method in ['full','restricted']:
            name=f'{xc}-{method}-N200-seed10';cfg=json.loads((OLD/'package/draft-configs'/name/'input.yaml').read_text())
            cfg['data'].update(filename=REMOTE+f'/data/{xc}/train.extxyz',test_filename=REMOTE+f'/data/{xc}/validation.extxyz',max_workers=1)
            cfg['potential'].update(filename=REMOTE+f'/aligned/{xc}/model.yaml',checkpoint_name=REMOTE+f'/aligned/{xc}/checkpoint')
            cfg['fit'].update(reset_optimizer=True,train_max_n_buckets=1,test_max_n_buckets=1)
            cfg['fit']['scheduler_params']={'minimum_learning_rate':.0001}
            dump(OUT/'cases'/name/'input.yaml',cfg);cases.append(dict(name=name,XC=xc,method=method,N=200,validation=149,seed=10,epochs=50,steps=10000))
    source='/home6/u3129/tial-fm-research/grace-2l-preparation-20261107-01/checkpoint'
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OLD/'package/checkpoint').iterdir()}
    plan=dict(id=ID,model='GRACE-2L-OMAT-medium-base-fp64',source_checkpoint=source,source_hashes=hashes,cases=cases,precision='float64',test_access=False,new_DFT=False,max_concurrent_GPU_jobs=1,
        recipe='Adam LR1e-3 ->1e-4 cosine, Huber E/F/stress16/32/128, batch1, 50epochs, noEMA, fixed final checkpoint; same XC train-only alignment across full/restricted.',
        restricted_patterns=['I2/reducing_','rho/reducing_','I1/reducing_'],training_selection='Final step10000, no best-validation reload. Smoke models are discarded.')
    dump(OUT/'plan.json',plan)
    common='''#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd REMOTE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TF_USE_LEGACY_KERAS=1 TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$PWD/cache/$SLURM_JOB_ID" GRACE_CACHE="$PWD/cache/grace"
export GRACE_CUDA_ROOT=ENV/lib/python3.11/site-packages/nvidia/cuda_nvcc
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$GRACE_CUDA_ROOT"
export PATH="$GRACE_CUDA_ROOT/bin:$PATH"
'''.replace('REMOTE',REMOTE).replace('ENV',ENV)
    (OUT/'baseline.sbatch').write_text(common+ENV+'/bin/python grace_pilot_baseline.py\n',newline='\n')
    (OUT/'harmonics.sbatch').write_text(common+'export CUDA_VISIBLE_DEVICES=""\n/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python grace_pilot_harmonics.py\n',newline='\n')
    dump(OUT/'FILES_SHA256.json',{p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()})
    arc=RUN/'package.tar.gz'
    with tarfile.open(arc,'w:gz') as t:
        for p in OUT.rglob('*'):
            if p.is_file():t.add(p,arcname=p.relative_to(OUT).as_posix())
    dump(RUN/'package.json',dict(id=ID,remote_root=REMOTE,path=str(arc),bytes=arc.stat().st_size,sha256=hashlib.sha256(arc.read_bytes()).hexdigest()))
    print('PREPARED',len(cases),arc.stat().st_size)
if __name__=='__main__':main()
