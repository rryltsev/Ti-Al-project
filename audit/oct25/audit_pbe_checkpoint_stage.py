from pathlib import Path
import json, hashlib, statistics
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];MON=ROOT/'testset/monitor'
OUT=HERE/'pbe_train_comparison'
SOURCE=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
registry=read(MON/'model_registry.json'); source_manifest=read(MON/'models/source-manifest.json')
audit={'models':{},'training_log':{},'author_clarification':read(OUT/'author_checkpoint_clarification.json')}
for xc in ['PBE','PBEsol','r2SCAN']:
    mpath=MON/'models'/xc/'checkpoint-metadata.json'; meta=read(mpath)
    source=Path(source_manifest['models'][xc]['path'])
    assert sha(source)==registry['models'][xc]['checkpoint_sha256']
    assert sha(MON/'models'/xc/'model_frozen.pt2')==registry['models'][xc]['sha256']
    audit['models'][xc]={'train_infos':meta['train_infos'],'source_checkpoint':str(source),
                        'checkpoint_sha256':sha(source),'metadata_sha256':sha(mpath),
                        'frozen_sha256':registry['models'][xc]['sha256']}
lp=SOURCE/'DPA4-finetune/lcurve_pbe.out'
lines=[line for line in lp.read_text().splitlines() if line.strip() and not line.startswith('#')]
samples={}
for line in lines:
    a=line.split()
    if int(a[0]) in [4900,5000,5100,23400]:
        samples[a[0]]={'energy_mae_validation_meV_atom':1000*float(a[1]),
                       'energy_mae_training_meV_atom':1000*float(a[2]),'learning_rate':float(a[9]),'line':line}
audit['training_log']={'path':str(lp),'sha256':sha(lp),'last_step':int(lines[-1].split()[0]),'samples':samples,
    'caveat':'Logged rolling/batch metrics are not identical to full-dataset inference. Author confirms deliberate early-checkpoint selection based on Ti phonons.'}
windows=[]
for lo,hi in [(3000,5000),(5001,10000),(10001,15000),(15001,20000),(20001,23400)]:
    rr=[[float(x) for x in line.split()] for line in lines if lo<=int(line.split()[0])<=hi]
    windows.append({'steps':[lo,hi],'n_logged_rows':len(rr),
        'energy_mae_val_meV_atom':1000*statistics.mean(r[1] for r in rr),
        'energy_mae_train_meV_atom':1000*statistics.mean(r[2] for r in rr),
        'force_mae_val_eV_A':statistics.mean(r[3] for r in rr),
        'force_mae_train_eV_A':statistics.mean(r[4] for r in rr)})
audit['training_log']['descriptive_window_means']=windows
cp=SOURCE/'DPA4-finetune/TiAl_tune_pbe_DPA4-shard-Neo-OMat24-v20260805.json';cfg=read(cp)
audit['config']={'path':str(cp),'sha256':sha(cp),'loss':cfg['loss']}
(OUT/'checkpoint_stage_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps({xc:a['train_infos'] for xc,a in audit['models'].items()},indent=2))
print(json.dumps(samples,indent=2))
print(json.dumps(windows,indent=2))
