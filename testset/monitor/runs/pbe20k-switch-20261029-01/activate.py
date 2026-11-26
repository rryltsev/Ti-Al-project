"""Activate already verified PBE20k cache at the user's explicit request."""
from pathlib import Path
import copy
import hashlib
import json
import math
import shutil
import time

RUN=Path(__file__).resolve().parent
MON=RUN.parents[1]
WORK=MON.parents[1]
SOURCE=MON/'runs/pbe20k-comparison-20261029-01'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

assert not (RUN/'activation.json').exists(), 'Already activated; do not repeat'
registry=read(MON/'model_registry.json')
assert registry==read(RUN/'before/model_registry.json')
previous=copy.deepcopy(registry)
old=copy.deepcopy(registry['models']['PBE'])
assert old['checkpoint_step']==15000
verified=read(SOURCE/'collected/verification.json')
source_manifest=read(WORK/'audit/oct25/pbe_checkpoint_sweep/input-manifest.json')['models']['20000']
model=copy.deepcopy(read(SOURCE/'registry-template.json')['models']['PBE'])
assert verified['status']=='passed' and verified['checkpoint_step']==20000
assert verified['stored_train_step']==19999
assert read(SOURCE/'collected/checkpoint-metadata.json')['train_infos']['step']==19999
assert verified['checkpoint_sha256']==model['sha256']==source_manifest['sha256']==sha(Path(source_manifest['local_path']))
assert model['sha256']=='3b0b4b69f8fca7be5bae4a5f24498aac904efad08f126f4a57de035e0379c67e'
model.update(user_selected_date='2026-10-29',prediction_cache='PBE-20000')
geometry=read(MON/'data/all50-geometries.json')
rows={r['metadata']['structure_id']:r for r in geometry['records']}
outputs=read(SOURCE/'collected/outputs.json')
assert all(sha(SOURCE/'collected'/n)==h for n,h in outputs.items())
cache=read(MON/'prediction-manifest.json');before_cache=copy.deepcopy(cache)
assert len(cache)==250 and all(sha(MON/n)==h for n,h in cache.items())
files=list((SOURCE/'collected/predictions/PBE').glob('*.json'))
assert len(files)==50 and {f.stem for f in files}==set(rows)
dest=MON/'predictions/PBE-20000';dest.mkdir(exist_ok=False)
for f in files:
    pred=read(f);row=rows[f.stem]
    assert pred['structure_id']==row['metadata']['structure_id']
    assert pred['functional']=='PBE' and pred['model_sha256']==model['sha256']
    assert pred['poscar_sha256']==row['metadata']['poscar_sha256'] and pred['symbols']==row['symbols']
    assert pred['energy_offset_applied_eV']==0
    assert len(pred['forces_eV_A'])==len(row['symbols']) and all(len(x)==3 for x in pred['forces_eV_A'])
    assert len(pred['virial_eV'])==3 and all(len(x)==3 for x in pred['virial_eV'])
    assert all(math.isfinite(x) for x in [pred['energy_eV']]+[v for a in pred['forces_eV_A'] for v in a]+[v for a in pred['virial_eV'] for v in a])
    shutil.copy2(f,dest/f.name)
    cache[(dest/f.name).relative_to(MON).as_posix()]=sha(dest/f.name)
assert len(cache)==300 and all(sha(MON/n)==h for n,h in before_cache.items())
registry['models']['PBE']=model
registry['policy']='Fixed user-selected models. PBE switched5k->15k on2026-10-26 and15k->20k on2026-10-29 by explicit user requests after Test50 inspection. Preserve separate histories; no training or energy shifts on Test50.'
selection={'change':'PBE15000 ->20000','date':'2026-10-29','authorization':'User explicitly requested PBE20k for further monitoring, rebuilt graphs/metrics and separate15k history in this chat.',
 'qualification':'Replacement after viewing Test50 comparison. Preserve adaptive checkpoint-selection provenance; not a blind preselected checkpoint. PBEsol/r2SCAN/pretrained unchanged.',
 'previous_PBE':old,'previous_registry_sha256':sha(RUN/'before/model_registry.json'),'previous_segment':'results/model-history/PBE-15000/history.json'}
registry['model_selection_history']=registry.get('model_selection_history',[])+[registry['model_selection'],selection]
registry['model_selection']=selection
registry['frozen_unix']=time.time()
registry['active_source_manifest']='runs/pbe20k-switch-20261029-01/active-source-manifest.json'
active={'models':registry['models'],'source':source_manifest,'inference_job':'20964816','verification':'models/PBE-20000/verification.json','previous_active_registry_sha256':sha(RUN/'before/model_registry.json')}
save(RUN/'active-source-manifest.json',active)
registry['active_source_manifest_sha256']=sha(RUN/'active-source-manifest.json')
registry['source_manifest_scope']='Original source_manifest_sha256 describes initial exports; active_source_manifest describes current PBE20k and unchanged other models.'
modeldir=MON/'models/PBE-20000';modeldir.mkdir(exist_ok=False)
for n in ['verification.json','checkpoint-metadata.json']:shutil.copy2(SOURCE/'collected'/n,modeldir/n)
save(modeldir/'source-manifest.json',{'model':model,'source':source_manifest,'inference_run':'runs/pbe20k-comparison-20261029-01','inference_job':'20964816'})
save(MON/'prediction-manifest.json',cache)
save(MON/'model_registry.json',registry)
for xc in ['PBEsol','r2SCAN']:assert registry['models'][xc]==previous['models'][xc]
save(RUN/'activation.json',{'unix':time.time(),'predictions':50,'old_cache_files_unchanged':250,'registry_sha256':sha(MON/'model_registry.json'),'prediction_manifest_sha256':sha(MON/'prediction-manifest.json'),'previous_cache':'predictions/PBE-15000','active_cache':'predictions/PBE-20000','history15k':'results/model-history/PBE-15000/history.json','inference_reused':True})
print('Activated20k; all250 old cache files preserved; current200 comparison files among300 archived+active.')
