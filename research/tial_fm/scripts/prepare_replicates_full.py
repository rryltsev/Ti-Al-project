"""Immutable inputs for seeds20/30 at N200 and seed10 controls at N694."""
from pathlib import Path
import copy, hashlib, json, shutil, tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ID='dpa4-replicates-full-20261106-02'; RUN=ROOT/'runs'/ID; OUT=RUN/'package'
PARENT=ROOT/'runs/dpa4-n200-pilot-20261106-02'
REMOTE='/home6/u3129/tial-fm-research/'+ID

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
def replace_once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    audit=json.loads((RUN/'label-audit/summary.json').read_text());rr=audit['results']['r2SCAN']
    assert rr['records']==843 and not audit['missing']
    assert not rr['non_normal'] and not rr['missing_EDIFF_marker'] and not rr['SCF_reached_NELM']
    assert rr['energy_matches']=={'e_fr_energy':843} and rr['force_max_difference_eV_A']<1e-6
    shutil.copytree(RUN/'label-audit',OUT/'label-audit')
    full=ROOT/'datasets/common992-grouped-v1-20261106-02'
    small=ROOT/'datasets/nested-train-v1-20261106'
    subsets=json.loads((small/'subsets.json').read_text());splits=json.loads((full/'splits.json').read_text())
    records={r['geometry_id']:r for r in map(json.loads,(full/'data_manifest.jsonl').read_text().splitlines())}
    tr=set(splits['split_ids']['train']);va=set(splits['split_ids']['validation']);te=set(splits['split_ids']['test'])
    assert (len(tr),len(va),len(te))==(694,149,149) and not (tr&va or tr&te or va&te)
    group_sets=[{records[i]['group_id'] for i in ids} for ids in [tr,va,te]]
    assert not (group_sets[0]&group_sets[1] or group_sets[0]&group_sets[2] or group_sets[1]&group_sets[2])
    assert set(subsets['subsets']['694'])==tr and set(subsets['subsets']['200'])<tr
    proto=json.loads((PARENT/'package/protocol.json').read_text())
    dataset_checks={}
    for n in [200,694]:
        data=OUT/'data'/f'N{n}'; checks={}
        for xc in ['PBE','r2SCAN']:
            checks[xc]={}
            for part in ['train','validation']:
                src=(small if n==200 and part=='train' else full)/'deepmd'/xc/part
                dst=data/xc/part;shutil.copytree(src,dst);ids=[];atoms=0
                for system in sorted(dst.iterdir()):
                    ii=json.loads((system/'geometry_ids.json').read_text());ids+=ii
                    types=np.loadtxt(system/'type.raw',dtype=int,ndmin=1);nat=len(types);atoms+=nat*len(ii)
                    assert nat>=128 # auto:128 must mean one structure per optimizer batch.
                    for key,width in [('coord',nat*3),('box',9),('energy',1),('force',nat*3),('virial',9)]:
                        ar=np.load(system/'set.000'/f'{key}.npy');assert ar.shape[0]==len(ii)
                        assert ar.size==len(ii)*width and np.isfinite(ar).all(),(system,key)
                    if n==200 and part=='train':
                        parent=full/'deepmd'/xc/part/system.name;pi=json.loads((parent/'geometry_ids.json').read_text())
                        idx=[pi.index(i) for i in ii]
                        for key in ['coord','box','energy','force','virial']:
                            assert np.array_equal(np.load(system/'set.000'/f'{key}.npy'),np.load(parent/'set.000'/f'{key}.npy')[idx])
                expected=set(subsets['subsets'][str(n)]) if part=='train' else va
                assert set(ids)==expected and len(ids)==len(expected)
                checks[xc][part]=dict(structures=len(ids),atoms=atoms,groups=len({records[i]['group_id'] for i in ids}))
        hashes={p.relative_to(OUT).as_posix():sha(p) for p in sorted(data.rglob('*')) if p.is_file()}
        dump(OUT/f'dataset-files-N{n}.json',hashes);dataset_checks[str(n)]=checks
    # Keep split IDs for membership checks, but stage no test coordinates or labels.
    (OUT/'data_manifest.jsonl').write_text(''.join(json.dumps(records[i])+'\n' for i in sorted(tr|va)))
    for name,src in [('subsets.json',small/'subsets.json'),('splits.json',full/'splits.json'),
        ('probes.json',PARENT/'package/probes.json'),('reference_figure.npz',PARENT/'package/reference_figure.npz'),
        ('build_pbesol_phonons.py',PARENT/'package/build_pbesol_phonons.py')]:shutil.copyfile(src,OUT/name)
    for xc in ['PBE','r2SCAN']:
        shutil.copytree(PARENT/'package/harmonic_inputs'/xc,OUT/'harmonic_inputs'/xc)
        shutil.copytree(PARENT/'collected/harmonics'/xc,OUT/'harmonics'/xc)
    for gate in ['baseline','harmonics']:
        dest=OUT/gate;dest.mkdir(exist_ok=True)
        shutil.copyfile(PARENT/'collected'/gate/'summary.json',dest/'summary.json')
        assert json.loads((dest/'summary.json').read_text())['status']=='PASS'
    shutil.copytree(PARENT/'collected/runtime-mask-preflight',OUT/'runtime-mask-preflight')
    for name in ['baseline_worker.py','harmonic_worker.py','trained_numerical_qc.py']:
        shutil.copyfile(ROOT/'scripts'/name,OUT/name)
    (OUT/'protocol.json').write_bytes((PARENT/'package/protocol.json').read_bytes())
    # baseline_worker imports protocol only for its standalone main; shared helpers
    # are unchanged. This protocol is replaced below with the expansion identity.
    plan_dir=OUT/'training-plan01';plan_dir.mkdir();cases=[];groups={'N200':[],'N694':[]}
    for n,seeds in [(200,[20,30]),(694,[10])]:
        for seed in seeds:
            for xc in ['PBE','r2SCAN']:
                for method in ['lora16','full']:
                    name=f'{xc}-{method}-N{n}-seed{seed}';dest=plan_dir/'cases'/name;dest.mkdir(parents=True)
                    old=PARENT/'collected-pilots/training-plan01/cases'/f'{xc}-{method}'
                    cfg=json.loads((old/'input.json').read_text());oldcase=json.loads((old/'case.json').read_text())
                    cfg['training']['seed']=seed
                    for part,key in [('train','training_data'),('validation','validation_data')]:
                        cfg['training'][key]['systems']=[REMOTE+f'/data/N{n}/{xc}/{part}/'+p.name for p in sorted((OUT/'data'/f'N{n}'/xc/part).iterdir())]
                    dump(dest/'input.json',cfg)
                    case=dict(name=name,XC=xc,method=method,N=n,validation=149,seed=seed,epochs=50,expected_steps=50*n,
                        source=proto['model'],split_sha256=sha(full/'splits.json'),subsets_sha256=sha(small/'subsets.json'),
                        source_recipe_sha256=oldcase['source_recipe_sha256'],configuration_sha256=sha(dest/'input.json'),
                        data_root=REMOTE+f'/data/N{n}',dataset_manifest=f'dataset-files-N{n}.json',
                        dataset_manifest_sha256=sha(OUT/f'dataset-files-N{n}.json'),
                        intent='Fixed final 50-epoch checkpoint. Same original weights, recipe and validation; training seed varied, architecture seeds kept at42. Statistics recomputed only from this case train. No test input.')
                    dump(dest/'case.json',case);cases.append(case);groups[f'N{n}'].append(name)
    obs=(PARENT/'collected-pilots/training-plan01/train_observer_v3.py').read_text()
    obs=replace_once(obs,'seed=10))',"seed=CFG['seed'],torch_initial_seed=torch.initial_seed()))")
    obs=replace_once(obs,"original_init(self,*a,**kw);holders.append(self)","original_init(self,*a,**kw);holders.append(self)\n    assert self.num_steps==CFG['expected_steps'],(self.num_steps,CFG['expected_steps'])")
    obs=replace_once(obs,"assert json.loads((ROOT/'trainer-zero-preflight/summary.json').read_text())['status']=='PASS'",
        "assert json.loads((ROOT/'preflight.json').read_text())['status']=='PASS'\n    assert json.loads((WORK/'input-verification.json').read_text())['status']=='PASS'")
    obs=obs.replace('on the N200 train only','on the case train only')
    (plan_dir/'train_observer.py').write_text(obs,newline='\n')
    ev=(ROOT/'scripts/evaluate_pilot.py').read_text()
    ev=replace_once(ev,"for part in ['train','validation']:","counts={}\n    for part in ['train','validation']:\n        counts[part]=0")
    ev=replace_once(ev,"(ROOT/'data'/xc/part).iterdir()","(Path(s['case']['data_root'])/xc/part).iterdir()")
    ev=replace_once(ev,"ids=json.loads((src/'geometry_ids.json').read_text());types=","ids=json.loads((src/'geometry_ids.json').read_text());counts[part]+=len(ids);types=")
    ev=replace_once(ev,"train_evaluated=200,validation_evaluated=149","train_evaluated=counts['train'],validation_evaluated=counts['validation'],seed=s['case']['seed'],N=s['case']['N']")
    ev=replace_once(ev,"for key in ['force000','force001','force002']:","assert counts==dict(train=s['case']['N'],validation=149),counts\n    for key in ['force000','force001','force002']:")
    (plan_dir/'evaluate_pilot.py').write_text(ev,newline='\n')
    shutil.copyfile(ROOT/'scripts/evaluate_pilot_harmonics.py',plan_dir/'evaluate_pilot_harmonics.py')
    analyzer=(ROOT/'scripts/analyze_predictions.py').read_text()
    analyzer=replace_once(analyzer,"a=ap.parse_args()","ap.add_argument('--train-size',type=int,default=200);ap.add_argument('--data-root',type=Path);a=ap.parse_args()")
    analyzer=replace_once(analyzer,"a.package/'data'/xc/part/predfile.stem","(a.data_root if a.data_root else a.package/'data')/xc/part/predfile.stem")
    analyzer=replace_once(analyzer,"assert len(train)==200","assert len(train)==a.train_size")
    analyzer=replace_once(analyzer,"fit='Equal-structure least squares in energy/atom using N200 train only; prediction E += nAl*bAl + nTi*bTi.'","fit=f'Equal-structure least squares in energy/atom using N{a.train_size} train only; prediction E += nAl*bAl + nTi*bTi.'")
    (OUT/'analyze_predictions.py').write_text(analyzer,newline='\n')
    shutil.copyfile(ROOT/'scripts/verify_expansion_case.py',OUT/'verify_case.py')
    shutil.copyfile(ROOT/'scripts/aggregate_expansion.py',OUT/'aggregate_expansion.py')
    for filename in ['analyze_pilot_physics.py','expansion_case_physics.py']:
        shutil.copyfile(ROOT/'scripts'/filename,OUT/filename)
    common=(PARENT/'collected-pilots/training-plan01/train-v3.sbatch').read_text().split('RUN=')[0]
    pick='RUN='+REMOTE+'\nGROUP="$1"\nif [ "$GROUP" = N200 ]; then\nCASES=('+ ' '.join(groups['N200']) +')\nelse\ntest "$GROUP" = N694\nCASES=('+ ' '.join(groups['N694']) +')\nfi\nCASE_NAME=${CASES[$SLURM_ARRAY_TASK_ID]}\ncd "$RUN/training-plan01/cases/$CASE_NAME"\n'
    launch=common+pick+'''test ! -e attempt.json
python -c 'import json,os,socket; open("attempt.json","x").write(json.dumps(dict(job=os.environ["SLURM_JOB_ID"],array=os.environ["SLURM_ARRAY_JOB_ID"],task=os.environ["SLURM_ARRAY_TASK_ID"],host=socket.gethostname())))'
python "$RUN/verify_case.py"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv > hardware.txt
python ../../train_observer.py --pt train input.json --finetune MODEL > train.log 2>&1
python ../../evaluate_pilot.py > evaluate.log 2>&1
unset PYTHONPATH
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python ../../evaluate_pilot_harmonics.py > harmonics.log 2>&1
N=${GROUP#N}
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python "$RUN/analyze_predictions.py" --package "$RUN" --data-root "$RUN/data/$GROUP" --train-size "$N" --predictions evaluation/predictions --output evaluation/metrics > metrics.log 2>&1
'''.replace('MODEL',proto['model']['path'])
    (plan_dir/'train.sbatch').write_text(launch,newline='\n')
    qc=common+'export CUDA_VISIBLE_DEVICES=""\n'+pick+'test -f evaluation/summary.json\npython "$RUN/trained_numerical_qc.py" --case "$CASE_NAME" > numerical-QC.log 2>&1\n/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python "$RUN/expansion_case_physics.py" --case "$CASE_NAME" > physical-diagnostics.log 2>&1\n'
    (plan_dir/'qc.sbatch').write_text(qc,newline='\n')
    aggregate='''#!/bin/bash
set -eo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cd REMOTE
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python aggregate_expansion.py --group "$1"
'''.replace('REMOTE',REMOTE)
    (OUT/'aggregate.sbatch').write_text(aggregate,newline='\n')
    prev=json.loads((PARENT/'comparison-v1/comparison.json').read_text())
    existing={name:dict(remote_path=json.loads((PARENT/'package.json').read_text())['remote_root']+'/training-plan01/cases/'+name,
                       model_sha256=v['model_sha256'],seed=10,N=200) for name,v in prev['cases'].items()}
    plan=dict(id=ID,remote_root=REMOTE,parent_run=str(PARENT),cases=cases,groups=groups,existing_seed10=existing,
        no_new_DFT=True,test_access=False,max_concurrent_new_GPUs=1,
        seeds_N200=[10,20,30],seeds_N694=[10],epochs=50,
        estimated_GPU_hours=dict(N200_8_additional=5.51,N694_4_controls=8.55,total=14.06),
        estimate_qualification='Linear scaling of measured A100 training time, plus observed export/evaluation overhead. Queue delay and larger-train statistics are excluded.',
        resources=dict(account='imet-2026-2',GPU_partition='v100',CPU_partition='apollo',GPU_tasks=dict(gpus=1,cpus=4,memory='40G',time='04:00:00')),
        followup='Automatic per-case validation/harmonics/QC and seed aggregate after each batch. N694 is a one-seed paired size control, not a three-seed estimate.',
        seed_policy='Only training.seed changes from10 to20/30. Descriptor/fitting seeds42 and initial pretrained tensors remain fixed; LoRA A initialization uses the backend PyTorch RNG. Per-parameter initial hashes and exposure traces are retained.')
    dump(OUT/'plan.json',plan)
    proto.update(id=ID,remote_root=REMOTE,pilot=dict(N=[200,694],validation=149,seeds_N200=[10,20,30],seeds_N694=[10],epochs=50),
                 authorization=dict(current_request='2026-11-06: extend N200 to three seeds and start full train N694 controls',no_new_DFT=True,existing_jobs_not_modified=True))
    dump(OUT/'protocol.json',proto)
    dump(OUT/'preflight.json',dict(status='PASS',datasets=dataset_checks,group_overlap=0,train200_nested_in694=True,
        test_data_staged=False,configuration_count=len(cases),zero_step_policy='Every GPU case performs exact initialized live/merged/source and frozen-mask checks before its first optimizer step; no separate duplicated CPU initialization.'))
    dump(OUT/'FILES_SHA256.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
    archive=RUN/'package.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():t.add(p,arcname=p.relative_to(OUT).as_posix())
    dump(RUN/'package.json',dict(id=ID,remote_root=REMOTE,path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size))
    print(json.dumps(dict(cases=len(cases),archive_bytes=archive.stat().st_size,preflight='PASS'),indent=2))

if __name__=='__main__':main()
