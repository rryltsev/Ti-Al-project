"""Freeze eight PBE-only GRACE replicates; only training seed differs from seed10."""
from pathlib import Path
import ast,json,shutil,tarfile
from prepare_grace_pilot import ROOT,RUN as PARENT,REMOTE as PARENT_REMOTE,ENV,OLD
from prepare_grace_mask_ablation import RUN as MASK,REMOTE as MASK_REMOTE
from prepare_grace_mask_ablation import load,dump,sha
ID='grace-pbe-seeds-20261110-01';RUN=ROOT/'runs'/ID;OUT=RUN/'package';REMOTE='/home6/u3129/tial-fm-research/'+ID
TI3=ROOT/'runs/ti3al-pbe-phonons-20261109-02';TI3_REMOTE='/home6/u3129/tial-fm-research/'+TI3.name
METHODS=['full','restricted','restricted_mlp','restricted_radial']
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    for folder in ['data','harmonic_inputs']:
        shutil.copytree(PARENT/'package'/folder/'PBE',OUT/folder/'PBE')
    shutil.copytree(PARENT/'collected-final/aligned/PBE',OUT/'aligned/PBE')
    for name in ['summary.json','numerical-QC.json','harmonics/summary.json']:
        target=OUT/'baseline'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/'collected-final/baseline'/name,target)
    for name in ['probes.json','harmonic_worker.py','build_pbesol_phonons.py','reference_figure.npz','data-preflight.json']:
        shutil.copyfile(PARENT/'package'/name,OUT/name)
    shutil.copyfile(OLD/'backend-manifest.json',OUT/'backend-manifest.json')
    parent_plan=load(PARENT/'package/plan.json');cases=[];parents=[];audit={}
    for method in METHODS:
        run,remote=(PARENT,PARENT_REMOTE) if method in ['full','restricted'] else (MASK,MASK_REMOTE)
        c=next(c for c in load(run/'package/plan.json')['cases'] if c['XC']=='PBE' and c['method']==method)
        w=run/'collected-final/cases'/c['name'];inv=load(w/'parameter-inventory.json')
        parents.append(dict(root=remote,case=c,ti3al_name='GRACE-'+method+'-N200-seed10'))
        for seed in [20,30]:
            case=dict(c,name=f'PBE-{method}-N200-seed{seed}',seed=seed,expected_trainable=inv['trainable'])
            cfg=load(w/'input.yaml');original=load(w/'input.yaml');cfg['seed']=seed
            check=json.loads(json.dumps(cfg));check['seed']=10;assert check==original
            cfg=json.loads(json.dumps(cfg).replace(remote,REMOTE));dump(OUT/'cases'/case['name']/'input.yaml',cfg)
            cases.append(case);audit[case['name']]=dict(parent=c['name'],source_run=run.name,only_scientific_change='seed',input_sha256=sha(OUT/'cases'/case['name']/'input.yaml'))
    # Group each seed contiguously for paired comparisons and serial resource use.
    cases.sort(key=lambda c:(c['seed'],METHODS.index(c['method'])))
    inv=load(PARENT/'collected-final/cases/PBE-full-N200-seed10/parameter-inventory.json')
    plan=dict(parent_plan,id=ID,cases=cases,parents=parents,ti3al_parent=TI3_REMOTE,initial_parameter_signatures={'PBE':sorted([[v['name'],v['shape'],v['initial_sha256']] for v in inv['parameters']])},active_XC=['PBE'],paused_XC=['r2SCAN'],matched_batches='Within XC and seed; production must match own discarded smoke.',controlled_changes=audit)
    dump(OUT/'plan.json',plan)
    subsets=load(ROOT/'datasets/nested-train-v1-20261106/subsets.json')
    dump(OUT/'membership.json',dict(train=sorted(subsets['subsets']['200']),validation=sorted(subsets['validation_ids'])) )
    dump(OUT/'STORAGE_POLICY.json',dict(skip_unused_best_checkpoint=True,minimum_start_bytes=int(1.5*2**30),minimum_checkpoint_bytes=640*2**20,minimum_final_export_bytes=640*2**20,post_success='Lossless verified gzip of final native Adam checkpoint; final model and export retained.'))
    names=['grace_pilot_common.py','grace_pilot_train.py','grace_pilot_evaluate.py','grace_pilot_harmonics.py','grace_optimizer_gate.py','grace_storage_guard.py','grace_pack_optimizer.py','grace_mask_aggregate.py','grace_pbe_seed_post.py','grace_pbe_seed_aggregate.py']
    for name in names:shutil.copyfile(ROOT/'scripts'/name,OUT/name);ast.parse((OUT/name).read_text())
    ti=OUT/'ti3al';shutil.copytree(TI3/'package/source',ti/'source');shutil.copytree(TI3/'collected/prepared',ti/'prepared')
    shutil.copyfile(TI3/'package/overlap-candidates.json',ti/'overlap-candidates.json')
    text=(ROOT/'scripts/ti3al_phonon_worker.py').read_text().replace("len(cases)==16", "len(cases)==len(json.loads((ROOT/'models.json').read_text()))").replace("seed_counts={'DPA4':3,'GRACE':1}","seed_counts={'GRACE':2}")
    (ti/'ti3al_phonon_worker.py').write_text(text,encoding='utf8')
    shutil.copyfile(ROOT/'scripts/ti3al_asr_sensitivity.py',ti/'ti3al_asr_sensitivity.py')
    dump(ti/'models.json',[dict(name=c['name'],backend='GRACE',method=c['method'],N=200,seed=c['seed'],XC='PBE',model=REMOTE+'/cases/'+c['name']) for c in cases])
    dump(ti/'ASR_FILES_SHA256.json',{n:sha(ti/n) for n in ['ti3al_asr_sensitivity.py','ti3al_phonon_worker.py']})
    dump(ti/'FILES_SHA256.json',{p.relative_to(ti).as_posix():sha(p) for p in ti.rglob('*') if p.is_file()})
    common=(MASK/'package/train.sbatch').read_text().split('RUN=')[0].replace(MASK_REMOTE,REMOTE)
    pick='RUN='+REMOTE+'\nCASES=('+' '.join(c['name'] for c in cases)+')\nCASE_NAME=${CASES[$SLURM_ARRAY_TASK_ID]}\n'
    gate=ENV+'''/bin/python -c 'import json; assert json.load(open("optimizer-gate.json"))["status"] == "PASS"'\n'''
    scripts={
      'smoke.sbatch':common+pick+'mkdir -p "$RUN/smoke/$CASE_NAME"\n'+ENV+'/bin/python "$RUN/grace_pilot_train.py" --case "$CASE_NAME" --smoke > "$RUN/smoke/$CASE_NAME/train.log" 2>&1\n',
      'gate.sbatch':common+'export CUDA_VISIBLE_DEVICES=""\n/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python grace_optimizer_gate.py\n',
      'train.sbatch':common+pick+gate+ENV+'/bin/python "$RUN/grace_pilot_train.py" --case "$CASE_NAME" > "$RUN/cases/$CASE_NAME/train.log" 2>&1\n'+ENV+'/bin/python "$RUN/grace_pilot_evaluate.py" --case "$CASE_NAME" > "$RUN/cases/$CASE_NAME/evaluate.log" 2>&1\n'+ENV+'/bin/python "$RUN/grace_pack_optimizer.py" "$CASE_NAME"\n',
      'post.sbatch':common+'export CUDA_VISIBLE_DEVICES=""\n/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python grace_pbe_seed_post.py\n'}
    for name,text in scripts.items():(OUT/name).write_text(text,encoding='utf8',newline='\n')
    dump(OUT/'TRAINING_FILES_SHA256.json',{n:sha(OUT/n) for n in names+list(scripts)+['backend-manifest.json','STORAGE_POLICY.json']})
    dump(OUT/'FILES_SHA256.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()})
    arc=RUN/'package.tar.gz'
    with tarfile.open(arc,'w:gz') as tar:
        for p in OUT.rglob('*'):
            if p.is_file():tar.add(p,arcname=p.relative_to(OUT).as_posix())
    dump(RUN/'package.json',dict(id=ID,remote_root=REMOTE,path=str(arc),sha256=sha(arc)))
    dump(RUN/'configuration-audit.json',dict(status='PASS',cases=audit,train200_validation149_unchanged=True,seed10_baseline_reused=True,r2SCAN_new_runs=0))
    print('PREPARED',len(cases),arc.stat().st_size)
if __name__=='__main__':main()
