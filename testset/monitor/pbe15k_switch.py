"""Stage/submit and collect the user-authorized, provenance-preserving PBE 15k switch."""
from pathlib import Path
import base64
import hashlib
import json
import math
import shlex
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/pbe15k-switch-20261026-01'
REMOTE='/home6/u3129/tial-testsets/pbe15k-test50-20261026-01'
sys.path.insert(0,str(ROOT.parent/'dft'))
import cluster_test50 as c

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

action=sys.argv[1]
if action=='submit':
    RUN.mkdir(exist_ok=True)
    assert not (RUN/'submission-intent.json').exists(),'Reconcile prior intent before resubmitting'
    source=read(ROOT.parents[1]/'audit/oct25/pbe_checkpoint_sweep/input-manifest.json')['models']['15000']
    checkpoint=read(ROOT.parents[1]/'audit/oct25/pbe_checkpoint_sweep/remote-input.json')['models']['15000']
    assert source['sha256']==checkpoint['sha256']==sha(source['local_path'])
    for name in ('model_registry.json','prediction-manifest.json','analyze.py','README.txt','implementation-manifest.json','automation-prompt.txt'):
        shutil.copy2(ROOT/name,RUN/('before-'+name))
    shutil.copy2(ROOT/'results/latest.json',RUN/'before-metrics.json')
    reg=read(ROOT/'model_registry.json');old=reg['models']['PBE']
    assert old['checkpoint_sha256']=='59c9a637757b6d3bacc2ba87e78cc8861375ef2b0e75c37384e2e2c9211471ac'
    reg['models']['PBE']={**old, 'path':checkpoint['path'],'sha256':checkpoint['sha256'],
       'checkpoint_path':checkpoint['path'],'checkpoint_sha256':checkpoint['sha256'],
       'checkpoint_step':15000,'artifact_format':'original PyTorch checkpoint; direct CPU DeepPot inference',
       'prediction_cache':'PBE-15000','user_selected_date':'2026-10-26'}
    reg['policy']='Fixed user-selected models. PBE changed from 5k to 15k on explicit user request after earlier Test50 inspection; preserve both versions. No fitting, training or further checkpoint selection on Test50.'
    reg['model_selection']={'change':'PBE 5000 -> 15000','date':'2026-10-26',
      'authorization':'User explicitly requested 15k in this chat, referring to Analysis of TiAl article materials chat.',
      'source_thread':'01a12c64-36f9-7850-92d2-127c677cbe5f',
      'qualification':'Model replacement after earlier Test50 results were available. Report this history; do not claim a blind preselected checkpoint. PBEsol/r2SCAN/pretrained unchanged.',
      'previous_PBE':old,'previous_registry_sha256':sha(RUN/'before-model_registry.json')}
    reg['frozen_unix']=time.time()
    files={'inference-input.json':json.dumps({'checkpoint':checkpoint,'geometry_sha256':sha(ROOT/'data/all50-geometries.json')}).encode(),
       'registry-template.json':json.dumps(reg,indent=2).encode(),
       'all50-geometries.json':(ROOT/'data/all50-geometries.json').read_bytes(),
       'predict.py':(ROOT/'predict.py').read_bytes(),'pbe15k_worker.py':(ROOT/'pbe15k_worker.py').read_bytes()}
    job=(ROOT/'alignment/train-offsets-v1/training_job.sbatch').read_text().replace('$SLURM_ARRAY_TASK_ID','${SLURM_ARRAY_TASK_ID:-0}').replace('python predict_training.py','python pbe15k_worker.py')
    files['job.sbatch']=job.encode()
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    save(RUN/'staging-intent.json',{'remote':REMOTE,'files_sha256':hashes,'source':source,'unix':time.time()})
    c.cluster.run('umask 077; mkdir '+shlex.quote(REMOTE))
    for n,b in files.items():
        c.upload(b,REMOTE+'/'+n);(RUN/n).write_bytes(b)
    code='import pathlib,hashlib; p=pathlib.Path('+repr(REMOTE)+'); h='+repr(hashes)+'; assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in h.items()); assert hashlib.sha256(pathlib.Path('+repr(checkpoint['path'])+').read_bytes()).hexdigest()=='+repr(checkpoint['sha256'])+'; print("Verified staging and source checkpoint")'
    print(c.cluster.run('python3 -c '+shlex.quote(code)),flush=True)
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal',
      '--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=16G','--time=00:20:00','--no-requeue',
      '--job-name=test50-PBE15k','--chdir='+REMOTE,'--output='+REMOTE+'/inference-%j.out',REMOTE+'/job.sbatch']
    save(RUN/'submission-intent.json',{'command':args,'unix':time.time()})
    jid=c.cluster.run(shlex.join(args)).strip().split(';')[0];assert jid.isdigit(),jid
    save(RUN/'submission.json',{'job_id':jid,'remote':REMOTE,'unix':time.time()});print('Submitted',jid)
elif action=='status':
    jid=read(RUN/'submission.json')['job_id']
    acct=c.cluster.run('sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,NodeList -Pn')
    (RUN/'accounting.txt').write_text(acct,encoding='utf-8');print(acct)
    print(c.cluster.run('tail -n 5 '+shlex.quote(REMOTE+'/inference-'+jid+'.out')))
elif action=='collect':
    jid=read(RUN/'submission.json')['job_id']
    acct=c.cluster.run('sacct -X -j '+jid+' --format=JobID,State,ExitCode,Elapsed -Pn')
    (RUN/'accounting.txt').write_text(acct,encoding='utf-8')
    assert any(x.split('|')[:3]==[jid,'COMPLETED','0:0'] for x in acct.splitlines()),acct
    code='import pathlib,json,base64; p=pathlib.Path('+repr(REMOTE)+'); h=json.loads((p/"outputs.json").read_text()); print(json.dumps({n:{"sha256":digest,"base64":base64.b64encode((p/n).read_bytes()).decode()} for n,digest in h.items()}))'
    data=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)))
    for n,item in data.items():
        dest=RUN/'collected'/n;assert (RUN/'collected').resolve() in dest.resolve().parents
        payload=base64.b64decode(item['base64']);assert hashlib.sha256(payload).hexdigest()==item['sha256']
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert dest.read_bytes()==payload
        else:dest.write_bytes(payload)
    save(RUN/'collection.json',{'job_id':jid,'files_sha256':{n:v['sha256'] for n,v in data.items()},'unix':time.time()})
    print('Collected and verified',len(data),'files')
elif action=='activate':
    collected=RUN/'collected'
    verified=read(collected/'verification.json')
    reg=read(collected/'model_registry.json')
    model=reg['models']['PBE']
    assert verified['status']=='passed' and verified['checkpoint_step']==15000
    assert verified['checkpoint_sha256']==model['sha256']==model['checkpoint_sha256']
    assert read(collected/'checkpoint-metadata.json')['train_infos']['step']==14999
    old=read(RUN/'before-model_registry.json')
    assert read(ROOT/'model_registry.json')==old,'Active registry changed concurrently'
    for xc in ('PBEsol','r2SCAN'):assert reg['models'][xc]==old['models'][xc]
    cache=read(ROOT/'prediction-manifest.json')
    assert cache==read(RUN/'before-prediction-manifest.json')
    assert all(sha(ROOT/n)==digest for n,digest in cache.items())
    geometry=read(ROOT/'data/all50-geometries.json')
    rows={r['metadata']['structure_id']:r for r in geometry['records']}
    files=list((collected/'predictions/PBE').glob('*.json'))
    assert len(files)==50 and {p.stem for p in files}==set(rows)
    dest=ROOT/'predictions/PBE-15000';dest.mkdir(exist_ok=True)
    collection=read(RUN/'collection.json')['files_sha256']
    for path in files:
        assert sha(path)==collection[path.relative_to(collected).as_posix()]
        pred=read(path);r=rows[path.stem]
        assert pred['model_sha256']==model['sha256'] and pred['functional']=='PBE'
        assert pred['structure_id']==r['metadata']['structure_id'] and pred['symbols']==r['symbols']
        assert pred['poscar_sha256']==r['metadata']['poscar_sha256'] and pred['energy_offset_applied_eV']==0
        assert len(pred['forces_eV_A'])==len(r['symbols']) and all(len(f)==3 for f in pred['forces_eV_A'])
        assert len(pred['virial_eV'])==3 and all(len(v)==3 for v in pred['virial_eV'])
        assert all(math.isfinite(v) for v in [pred['energy_eV']]+[x for f in pred['forces_eV_A'] for x in f]+[x for f in pred['virial_eV'] for x in f])
        target=dest/path.name
        if target.exists():assert target.read_bytes()==path.read_bytes()
        else:shutil.copy2(path,target)
        cache[target.relative_to(ROOT).as_posix()]=sha(target)
    modeldir=ROOT/'models/PBE-15000';modeldir.mkdir(exist_ok=True)
    shutil.copy2(collected/'checkpoint-metadata.json',modeldir/'checkpoint-metadata.json')
    shutil.copy2(collected/'verification.json',modeldir/'verification.json')
    save(modeldir/'source-manifest.json',{'model':model,'source':read(RUN/'staging-intent.json')['source']})
    save(ROOT/'prediction-manifest.json',cache)
    save(ROOT/'model_registry.json',reg)
    save(RUN/'activation.json',{'unix':time.time(),'predictions':50,'registry_sha256':sha(ROOT/'model_registry.json'),
       'prediction_manifest_sha256':sha(ROOT/'prediction-manifest.json'),'previous_files_unchanged':True,
       'previous_PBE_cache':'predictions/PBE','active_PBE_cache':'predictions/PBE-15000'})
    print('Activated PBE 15k; all 200 previous predictions and old registry archive preserved')
