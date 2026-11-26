"""PBE full and radial, N694, same 50-epoch recipe and fixed validation149."""
import ast,json,shutil,tarfile
from prepare_grace_pbe_seeds import ROOT,ENV,TI3,TI3_REMOTE,load,dump,sha
from prepare_grace_data import main as prepare_data
ID='grace-pbe-n694-20261110-01';RUN=ROOT/'runs'/ID;OUT=RUN/'package';REMOTE='/home6/u3129/tial-fm-research/'+ID
PREVIOUS=ROOT/'runs/grace-pbe-seeds-20261110-01';PREVIOUS_REMOTE='/home6/u3129/tial-fm-research/'+PREVIOUS.name
METHODS=['full','restricted_radial']

def main():
    OUT.mkdir(parents=True,exist_ok=False);prepare_data(OUT,694,('PBE',))
    old=PREVIOUS/'package';plan=load(old/'plan.json')
    for folder in ['harmonic_inputs','ti3al']:
        shutil.copytree(old/folder,OUT/folder)
    for name in ['probes.json','harmonic_worker.py','build_pbesol_phonons.py','reference_figure.npz','backend-manifest.json','STORAGE_POLICY.json']:
        shutil.copyfile(old/name,OUT/name)
    assert sha(OUT/'data/PBE/validation.extxyz')==sha(old/'data/PBE/validation.extxyz')
    data=load(OUT/'data-preflight.json')['data']['PBE'];membership={p:sorted(data[p]['ids']) for p in ['train','validation']}
    old_membership=load(old/'membership.json');assert set(old_membership['train'])<set(membership['train'])
    dump(OUT/'membership.json',membership);dump(OUT/'membership-N200.json',old_membership)
    parents=[p for p in plan['parents'] if p['case']['method'] in METHODS]
    for c in plan['cases']:
        if c['method'] in METHODS:parents.append(dict(root=PREVIOUS_REMOTE,case=c,ti3al_root=PREVIOUS_REMOTE+'/ti3al',ti3al_name=c['name']))
    for p in parents:p.setdefault('ti3al_root',TI3_REMOTE)
    cases=[];audit={}
    for seed in [10,20,30]:
        for method in METHODS:
            src=next(c for c in plan['cases'] if c['seed']==20 and c['method']==method)
            c=dict(src,name=f'PBE-{method}-N694-seed{seed}',seed=seed,N=694,steps=34700)
            cfg=load(old/'cases'/src['name']/'input.yaml');cfg['seed']=seed
            cfg=json.loads(json.dumps(cfg).replace(PREVIOUS_REMOTE,REMOTE));dump(OUT/'cases'/c['name']/'input.yaml',cfg)
            cases.append(c);audit[c['name']]=dict(recipe_identical_except_seed_and_data_paths=True,input_sha256=sha(OUT/'cases'/c['name']/'input.yaml'))
    for k in ['initial_parameter_signatures','controlled_changes','reference_first5_batch_hashes']:plan.pop(k,None)
    plan.update(id=ID,cases=cases,parents=parents,training_selection='Fixed final epoch50 / step34700; fresh foundation weights, discarded smoke',comparison_design='N200 versus N694 at equal epochs, not equal updates; train-only offsets fitted separately.',controlled_changes=audit)
    dump(OUT/'plan.json',plan)
    names=['grace_pilot_common.py','grace_pilot_baseline.py','grace_pilot_train.py','grace_pilot_evaluate.py','grace_pilot_harmonics.py','grace_optimizer_gate.py','grace_storage_guard.py','grace_pack_optimizer.py','grace_mask_aggregate.py','grace_n694_aggregate.py','grace_pbe_seed_post.py']
    for name in names:shutil.copyfile(ROOT/'scripts'/name,OUT/name)
    post=(OUT/'grace_pbe_seed_post.py').read_text().replace('grace_pbe_seed_aggregate.py','grace_n694_aggregate.py');(OUT/'grace_pbe_seed_post.py').write_text(post)
    ti=OUT/'ti3al';dump(ti/'models.json',[dict(name=c['name'],backend='GRACE',method=c['method'],N=694,seed=c['seed'],XC='PBE',model=REMOTE+'/cases/'+c['name']) for c in cases])
    text=(ti/'ti3al_phonon_worker.py').read_text().replace("seed_counts={'GRACE':2}","seed_counts={'GRACE':3}")
    (ti/'ti3al_phonon_worker.py').write_text(text)
    dump(ti/'ASR_FILES_SHA256.json',{n:sha(ti/n) for n in ['ti3al_asr_sensitivity.py','ti3al_phonon_worker.py']})
    dump(ti/'FILES_SHA256.json',{p.relative_to(ti).as_posix():sha(p) for p in ti.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'})
    common=(old/'train.sbatch').read_text().split('RUN=')[0].replace(PREVIOUS_REMOTE,REMOTE)
    pick='RUN='+REMOTE+'\nCASES=('+' '.join(c['name'] for c in cases)+')\nCASE_NAME=${CASES[$SLURM_ARRAY_TASK_ID]}\n'
    gate=ENV+'''/bin/python -c 'import json; assert json.load(open("optimizer-gate.json"))["status"] == "PASS"'\n'''
    scripts={
      'baseline.sbatch':common+ENV+'/bin/python grace_pilot_baseline.py\n',
      'baseline-harmonics.sbatch':common+'export CUDA_VISIBLE_DEVICES=""\n/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python grace_pilot_harmonics.py\n',
      'smoke.sbatch':common+pick+'mkdir -p "$RUN/smoke/$CASE_NAME"\n'+ENV+'/bin/python "$RUN/grace_pilot_train.py" --case "$CASE_NAME" --smoke > "$RUN/smoke/$CASE_NAME/train.log" 2>&1\n',
      'gate.sbatch':common+'export CUDA_VISIBLE_DEVICES=""\n/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python grace_optimizer_gate.py\n',
      'train.sbatch':common+pick+gate+ENV+'/bin/python "$RUN/grace_pilot_train.py" --case "$CASE_NAME" > "$RUN/cases/$CASE_NAME/train.log" 2>&1\n'+ENV+'/bin/python "$RUN/grace_pilot_evaluate.py" --case "$CASE_NAME" > "$RUN/cases/$CASE_NAME/evaluate.log" 2>&1\n'+ENV+'/bin/python "$RUN/grace_pack_optimizer.py" "$CASE_NAME"\n',
      'post.sbatch':common+'export CUDA_VISIBLE_DEVICES=""\n/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python grace_pbe_seed_post.py\n'}
    for name,text in scripts.items():(OUT/name).write_text(text,encoding='utf8',newline='\n')
    for p in OUT.rglob('*.py'):ast.parse(p.read_text())
    dump(OUT/'TRAINING_FILES_SHA256.json',{n:sha(OUT/n) for n in names+list(scripts)+['backend-manifest.json','STORAGE_POLICY.json']})
    dump(OUT/'FILES_SHA256.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()})
    arc=RUN/'package.tar.gz'
    with tarfile.open(arc,'w:gz') as tar:
        for p in OUT.rglob('*'):
            if p.is_file():tar.add(p,arcname=p.relative_to(OUT).as_posix())
    dump(RUN/'package.json',dict(id=ID,remote_root=REMOTE,path=str(arc),sha256=sha(arc)))
    dump(RUN/'configuration-audit.json',dict(status='PASS',cases=audit,validation_identical=True,nested_train=True,epochs=50,steps=34700,alignment='new fit on train694 only',PBE_only=True))
    print('PREPARED',len(cases),arc.stat().st_size)
if __name__=='__main__':main()
