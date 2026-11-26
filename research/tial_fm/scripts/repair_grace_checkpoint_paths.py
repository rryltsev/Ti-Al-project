"""Audited recovery of failed GRACE smoke jobs; preserve existing downstream IDs."""
import argparse,ast,datetime,hashlib,json,shlex
from pathlib import Path
import cluster_pilot as c
import manage_expansion as m
from prepare_grace_pilot import ROOT,RUN,REMOTE

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--revision',default='checkpoint-path-v1');ap.add_argument('--failed-job',default='20990595');args=ap.parse_args()
    assert args.revision.replace('-','').isalnum() and args.failed_job.isdigit()
    source=(ROOT/'scripts/grace_pilot_train.py').read_text(encoding='utf8')
    ast.parse(source)
    payload=dict(source=source,sha256=hashlib.sha256(source.encode()).hexdigest())
    code=r'''import ast,hashlib,json,subprocess,sys
from pathlib import Path
p=Path(REMOTE).resolve();data=json.load(sys.stdin);archive=p/'recovery'/REVISION
marker=archive/'repair.json'
if marker.exists():
 assert json.loads(marker.read_text())['new_sha256']==data['sha256']
 print(marker.read_text());sys.exit(0)
accounting=subprocess.check_output(['sacct','-X','-j',FAILED_JOB,'-P','-n','--format=JobID,State'],text=True)
assert all(f'{FAILED_JOB}_{i}|FAILED' in accounting for i in range(4)),accounting
states={}
for job in ['20990596','20990597','20990598']:
 states[job]=subprocess.check_output(['scontrol','show','job',job],text=True)
 assert 'JobState=PENDING' in states[job]
assert not list((p/'cases').glob('*/attempt.json'))
assert json.loads((p/'baseline/summary.json').read_text())['status']=='PASS'
assert json.loads((p/'baseline/harmonics/summary.json').read_text())['status']=='PASS'
manifest=json.loads((p/'TRAINING_FILES_SHA256.json').read_text())
for name,h in manifest.items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h,name
ast.parse(data['source']);assert hashlib.sha256(data['source'].encode()).hexdigest()==data['sha256']
path_checks=[]
for xc in ['PBE','r2SCAN']:
 prefix=p/'aligned'/xc/'checkpoint';alias=Path(REMOTE)/'aligned'/xc/'checkpoint'
 assert prefix.resolve()==alias.resolve() and Path(str(prefix)+'.index').is_file()
 wrong=p/'aligned'/('r2SCAN' if xc=='PBE' else 'PBE')/'checkpoint'
 assert alias.resolve()!=wrong.resolve()
 alignment=json.loads((p/'aligned'/xc/'alignment.json').read_text())
 for name,h in alignment['checkpoint_hashes'].items():assert hashlib.sha256((prefix.parent/name).read_bytes()).hexdigest()==h
 path_checks.append(dict(XC=xc,alias=str(alias),canonical=str(prefix),same=True,other_XC_rejected=True))
archive.mkdir(parents=True,exist_ok=False)
for name in ['grace_pilot_train.py','TRAINING_FILES_SHA256.json']:(archive/name).write_bytes((p/name).read_bytes())
(archive/'scheduler-before.json').write_text(json.dumps(states,indent=2))
smoke=(p/'smoke').resolve();destination=(archive/'smoke').resolve()
assert smoke.is_relative_to(p) and destination.is_relative_to(p) and smoke!=p
smoke.rename(destination)
(p/'grace_pilot_train.py').write_text(data['source'])
manifest['grace_pilot_train.py']=data['sha256']
(p/'TRAINING_FILES_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name,h in manifest.items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h,name
record=dict(status='STAGED',old_smoke_job=FAILED_JOB,new_sha256=data['sha256'],path_checks=path_checks,archive=str(archive),preserved_gates=['baseline','H0'],accounting=accounting)
marker.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''.replace('REMOTE',repr(REMOTE)).replace('REVISION',repr(args.revision)).replace('FAILED_JOB',repr(args.failed_job))
    result=json.loads(c.ssh(m.PY+' -c '+shlex.quote(code),json.dumps(payload).encode()))
    folder=RUN/'recovery'/args.revision;folder.mkdir(parents=True,exist_ok=True)
    (folder/'repair.json').write_text(json.dumps(result,indent=2)+'\n')
    (folder/'grace_pilot_train.py').write_text(source,encoding='utf8',newline='\n')
    print(json.dumps(result,indent=2))
    m.RUN=RUN
    job=m.submit(REMOTE,'smoke-'+args.revision,['sbatch','--parsable','--account=imet-2026-2','--nodes=1','--ntasks=1','--partition=v100','--gres=gpu:a100:1','--cpus-per-task=4','--mem=40G','--time=01:00:00','--array=0-3%1','--job-name=TiAl-GRACE-smoke-retry','--output=smoke-%A_%a.log','smoke.sbatch'])
    code=r'''import subprocess,json
from pathlib import Path
p=Path(REMOTE);job=JOB
before=subprocess.check_output(['scontrol','show','job','20990596'],text=True)
assert 'JobState=PENDING' in before
subprocess.run(['scontrol','update','JobId=20990596','Dependency=afterok:'+job],check=True)
states={j:subprocess.check_output(['scontrol','show','job',j],text=True) for j in ['20990596','20990597','20990598']}
assert 'Dependency=afterok:'+job in states['20990596']
assert 'Dependency=afterok:20990596' in states['20990597']
assert 'Dependency=afterok:20990597' in states['20990598']
record=dict(smoke=job,optimizer_gate='20990596',train='20990597',harmonics='20990598',states=states)
(p/'recovery'/REVISION/'dependencies.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''.replace('REMOTE',repr(REMOTE)).replace('JOB',repr(job)).replace('REVISION',repr(args.revision))
    record=json.loads(m.remote_python(code));(folder/'dependencies.json').write_text(json.dumps(record,indent=2)+'\n')
    (RUN/'active-recovery.json').write_text(json.dumps(dict(revision=args.revision,**{k:v for k,v in record.items() if k!='states'}),indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='states'},indent=2))

if __name__=='__main__':main()
