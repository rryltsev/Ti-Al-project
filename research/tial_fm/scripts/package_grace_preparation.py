"""Prepare one pinned GRACE 2L model, four draft FT inputs and CPU mask audit."""
from pathlib import Path
import ast,copy,hashlib,json,shutil,tarfile
from fetch_grace_2l import MODEL
ROOT=Path(__file__).resolve().parents[1]
ID='grace-2l-preparation-20261107-01';RUN=ROOT/'runs'/ID;OUT=RUN/'package'
REMOTE='/home6/u3129/tial-fm-research/'+ID
ENV='/home6/u3129/mlip-research/envs/grace_ft_a100'

def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
def main():
    OUT.mkdir(exist_ok=False)
    card=json.loads((RUN/'checkpoints'/MODEL/'model-card.json').read_text())
    source=(RUN/'checkpoints'/MODEL/card['model_yaml']).parent
    (OUT/'checkpoint').mkdir()
    for name in ['model.yaml','checkpoint.index','checkpoint.data-00000-of-00001']:
        shutil.copyfile(source/name,OUT/'checkpoint'/name)
    dump(OUT/'model-card.json',card)
    shutil.copytree(RUN/'data',OUT/'data')
    for name in ['data-preflight.json','backend-manifest.json']:shutil.copyfile(RUN/name,OUT/name)
    shutil.copyfile(ROOT/'scripts/grace_mask_preflight.py',OUT/'grace_mask_preflight.py')
    patterns=['I2/reducing_','rho/reducing_','I1/reducing_'];cases=[]
    for xc in ['PBE','r2SCAN']:
        for method in ['full','restricted']:
            name=f'{xc}-{method}-N200-seed10'
            config=dict(seed=10,cutoff=6.0,
                data=dict(filename=REMOTE+f'/data/{xc}/train.extxyz',test_filename=REMOTE+f'/data/{xc}/validation.extxyz',reference_energy=0,save_dataset=False,stress_units='eV/A3'),
                potential=dict(filename=REMOTE+'/checkpoint/model.yaml',checkpoint_name=REMOTE+'/checkpoint/checkpoint',param_dtype='float64',reduce_elements=False),
                fit=dict(loss=dict(energy=dict(weight=16,type='huber',delta=.01),forces=dict(weight=32,type='huber',delta=.01),stress=dict(weight=128,type='huber',delta=.01)),
                    maxiter=50,optimizer='Adam',opt_params=dict(learning_rate=.001,use_ema=False,clipnorm=1.0),scheduler='cosine_decay',scheduler_params=dict(minimal_learning_rate=.0001),
                    compute_convex_hull=False,batch_size=1,test_batch_size=1,jit_compile=True,eval_init_stats=True,reset_epoch_and_step=True,
                    train_max_n_buckets='auto',test_max_n_buckets='auto',auto_bucket_max_padding=.3,checkpoint_freq=5,train_shuffle=True))
            if method=='restricted':config['fit']['trainable_variable_names']=patterns
            # JSON is valid YAML1.2 and is accepted by yaml.safe_load.
            dump(OUT/'draft-configs'/name/'input.yaml',config)
            cases.append(dict(name=name,XC=xc,method=method,N=200,validation=149,seed=10,expected_steps=10000,epochs=50,launch_ready=False))
    plan=dict(id=ID,model=MODEL,checkpoint_sha256=card['archive_sha256'],revision=card['revision'],environment=ENV,
        cases=cases,restricted_patterns=patterns,mask_semantics='Native TensorPotential substring selection; final exact tensor list/counts must come from preflight.',
        data_note='Backend test_filename is validation149 only. The held-out test149 is not staged.',
        initial_energy_shift='None: preserve original tensors and predictions. Report raw and train-only aligned energy metrics, as for DPA4. If large energy offsets prevent optimization, diagnose after B0 before modifying the declared recipe.',
        recipe_status='Draft native GRACE recipe, not yet GPU-profiled or training-qualified. Paired full/restricted use identical seed/data/LR/loss and final epoch50. GRACE Huber E/F/stress weights16/32/128 are architecture-specific, not equivalent to DPA4 E/F/virial MAE20/20/5.',
        restore_policy='Mandatory model-only checkpoint restore and fresh Adam state in the training observer; no optimizer slots or epoch counters from foundation training. Runtime restore of all existing model objects must be asserted.',
        selection_policy='Fixed final checkpoint at epoch50; no best-validation checkpoint selection; no test access.',
        gates_before_training=['GRACE-specific B0','GRACE-specific H0','actual optimizer-step mask check and zero-step source equivalence','bounded A100 timing/memory profile'],
        no_new_DFT=True,test_access=False,training_submitted=False,preflight_only=True,
        references=['https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/',card['source_url']])
    dump(OUT/'preparation-plan.json',plan)
    batch='''#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd REMOTE
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TF_USE_LEGACY_KERAS=1
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$PWD/cache" GRACE_CACHE="$PWD/cache/grace"
ENV/bin/python grace_mask_preflight.py
'''.replace('REMOTE',REMOTE).replace('ENV',ENV)
    (OUT/'preflight.sbatch').write_text(batch,encoding='utf8',newline='\n')
    for p in OUT.rglob('*.py'):ast.parse(p.read_text())
    dump(OUT/'FILES_SHA256.json',{p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()})
    archive=RUN/'package.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():t.add(p,arcname=p.relative_to(OUT).as_posix())
    dump(RUN/'package.json',dict(id=ID,remote_root=REMOTE,path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size))
    print(json.dumps(dict(status='PREPARED',model=MODEL,cases=4,archive_bytes=archive.stat().st_size,training_submitted=False)))

if __name__=='__main__':main()
