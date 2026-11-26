"""Four controlled mask ablations reusing the qualified N200 fp64 baseline."""
from pathlib import Path
import ast,hashlib,json,shutil,tarfile
from prepare_grace_pilot import ROOT,RUN as PARENT,REMOTE as PARENT_REMOTE,ENV,OLD

ID='grace-n200-mask-ablation-20261109-01';RUN=ROOT/'runs'/ID;OUT=RUN/'package';REMOTE='/home6/u3129/tial-fm-research/'+ID
def load(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    for folder in ['data','harmonic_inputs']:shutil.copytree(PARENT/'package'/folder,OUT/folder)
    shutil.copytree(PARENT/'collected-final/baseline',OUT/'baseline')
    shutil.copytree(PARENT/'collected-final/aligned',OUT/'aligned')
    for name in ['probes.json','harmonic_worker.py','build_pbesol_phonons.py','reference_figure.npz','data-preflight.json']:shutil.copyfile(PARENT/'package'/name,OUT/name)
    names=['grace_pilot_common.py','grace_pilot_train.py','grace_pilot_evaluate.py','grace_pilot_harmonics.py','grace_optimizer_gate.py']
    for name in names:shutil.copyfile(ROOT/'scripts'/name,OUT/name);ast.parse((OUT/name).read_text())
    shutil.copyfile(OLD/'backend-manifest.json',OUT/'backend-manifest.json')
    parent_plan=load(PARENT/'package/plan.json');base=parent_plan['restricted_patterns'];cases=[];signatures={};batch_hashes={};changes={}
    variants={'restricted_mlp':['LinMLPOut2ScalarTarget_'],'restricted_radial':['R_Linear_','R1_Linear_']}
    for xc in ['PBE','r2SCAN']:
        parent=PARENT/f'collected-final/cases/{xc}-restricted-N200-seed10';inventory=load(parent/'parameter-inventory.json')
        signatures[xc]=sorted([[v['name'],v['shape'],v['initial_sha256']] for v in inventory['parameters']])
        batch_hashes[xc]=load(parent/'checks/10000.json')['first5_batch_hashes']
        for method,extra in variants.items():
            case=f'{xc}-{method}-N200-seed10';cfg=load(parent/'input.yaml');original=json.loads(json.dumps(cfg));patterns=base+extra
            cfg['fit']['trainable_variable_names']=patterns
            # All scientific settings remain byte-for-byte equal apart from mask.
            comparison=json.loads(json.dumps(cfg));comparison['fit']['trainable_variable_names']=original['fit']['trainable_variable_names'];assert comparison==original
            cfg=json.loads(json.dumps(cfg).replace(PARENT_REMOTE,REMOTE))
            import math
            selected=[v for v in inventory['parameters'] if any(p in v['name'] for p in patterns)]
            count=sum(math.prod(v['shape']) for v in selected)
            assert count==21111690+(1088 if method=='restricted_mlp' else 33152)
            dump(OUT/'cases'/case/'input.yaml',cfg)
            cases.append(dict(name=case,XC=xc,method=method,N=200,validation=149,seed=10,epochs=50,steps=10000,trainable_patterns=patterns,expected_trainable=count,expected_trainable_tensors=len(selected)))
            changes[case]=dict(parent=f'{xc}-restricted-N200-seed10',extra_patterns=extra,extra_parameters=count-21111690,only_scientific_change='trainable_variable_names',input_sha256=sha(OUT/'cases'/case/'input.yaml'))
    plan=dict(parent_plan,id=ID,cases=cases,parent_run=PARENT_REMOTE,initial_parameter_signatures=signatures,reference_first5_batch_hashes=batch_hashes,controlled_changes=changes)
    dump(OUT/'plan.json',plan)
    common=(PARENT/'training-package/train.sbatch').read_text().split('RUN=')[0].replace(PARENT_REMOTE,REMOTE)
    pick='RUN='+REMOTE+'\nCASES=('+' '.join(c['name'] for c in cases)+')\nCASE_NAME=${CASES[$SLURM_ARRAY_TASK_ID]}\n'
    gate=ENV+'''/bin/python -c 'import json; assert json.load(open("optimizer-gate.json"))["status"] == "PASS"'
'''
    scripts={
        'smoke.sbatch':common+pick+'mkdir -p "$RUN/smoke/$CASE_NAME"\n'+ENV+'/bin/python "$RUN/grace_pilot_train.py" --case "$CASE_NAME" --smoke > "$RUN/smoke/$CASE_NAME/train.log" 2>&1\n',
        'optimizer-gate.sbatch':common+'export CUDA_VISIBLE_DEVICES=""\n/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python grace_optimizer_gate.py\n',
        'train.sbatch':common+pick+gate+ENV+'/bin/python "$RUN/grace_pilot_train.py" --case "$CASE_NAME" > "$RUN/cases/$CASE_NAME/train.log" 2>&1\n'+ENV+'/bin/python "$RUN/grace_pilot_evaluate.py" --case "$CASE_NAME" > "$RUN/cases/$CASE_NAME/evaluate.log" 2>&1\n',
        'trained-harmonics.sbatch':common+pick+'export CUDA_VISIBLE_DEVICES=""\n/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python "$RUN/grace_pilot_harmonics.py" --case "$CASE_NAME"\n'}
    for name,text in scripts.items():(OUT/name).write_text(text,encoding='utf8',newline='\n')
    dump(OUT/'TRAINING_FILES_SHA256.json',{n:sha(OUT/n) for n in names+list(scripts)+['backend-manifest.json']})
    dump(OUT/'FILES_SHA256.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()})
    arc=RUN/'package.tar.gz'
    with tarfile.open(arc,'w:gz') as t:
        for p in OUT.rglob('*'):
            if p.is_file():t.add(p,arcname=p.relative_to(OUT).as_posix())
    dump(RUN/'package.json',dict(id=ID,remote_root=REMOTE,path=str(arc),bytes=arc.stat().st_size,sha256=sha(arc)))
    dump(RUN/'configuration-audit.json',dict(status='PASS',cases=changes,source_baseline='Reused B0/H0 PASS; identical source checkpoint, alignment and data; no new DFT.',test_access=False))
    print('PREPARED',ID,len(cases),'cases',arc.stat().st_size,'bytes')

if __name__=='__main__':main()
