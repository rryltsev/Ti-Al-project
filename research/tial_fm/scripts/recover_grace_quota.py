"""Preserve failed epoch-one attempts, then retry only array indices 1,2,3."""
import argparse,ast,hashlib,json,shlex
from pathlib import Path
import cluster_pilot as c
import manage_expansion as m
from prepare_grace_mask_ablation import ROOT,RUN,OUT,REMOTE

REC=RUN/'storage-recovery'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['stage','submit']);a=ap.parse_args();m.RUN=RUN
    if a.action=='stage':
        # The CPU probe below checks the actual account-wide headroom (4 GiB).
        released=json.loads((REC/'released.json').read_text());assert released['status']=='PASS' and released['bytes_released']>3*2**30
        files={n:(ROOT/'scripts'/n).read_text(encoding='utf8') for n in ['grace_pilot_train.py','grace_storage_guard.py']}
        for s in files.values():ast.parse(s)
        policy=dict(skip_unused_best_checkpoint=True,minimum_start_bytes=1536*2**20,minimum_checkpoint_bytes=640*2**20,minimum_final_export_bytes=640*2**20,reason='Fixed-final selection does not use best-validation checkpoints; regular optimizer checkpoints remain enabled every5 epochs.')
        files['STORAGE_POLICY.json']=json.dumps(policy,indent=2)+'\n'
        common=(OUT/'optimizer-gate.sbatch').read_text();files['storage-probe.sbatch']=common.replace('grace_optimizer_gate.py','grace_storage_guard.py --probe')
        folder=REC/'new-files';folder.mkdir(exist_ok=True)
        for n,s in files.items():(folder/n).write_text(s,encoding='utf8',newline='\n')
        code=r'''from pathlib import Path
import ast,hashlib,json,sys,shutil,subprocess
p=Path(REMOTE).resolve();files=json.load(sys.stdin);archive=p/'recovery/disk-quota-v1';marker=archive/'repair.json'
if marker.exists():
 d=json.loads(marker.read_text());assert d['new_files']=={n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()};print(marker.read_text());sys.exit(0)
accounting=subprocess.check_output(['sacct','-X','-j','20990675','-P','-n','--format=JobID,State,ExitCode'],text=True)
assert '20990675_0|COMPLETED|0:0' in accounting
assert all(f'20990675_{i}|FAILED|1:0' in accounting for i in [1,2,3])
for job in ['20990676','20990677']:
 state=subprocess.check_output(['scontrol','show','job',job],text=True);assert 'JobState=PENDING' in state
for manifest in ['FILES_SHA256.json','TRAINING_FILES_SHA256.json']:
 for n,h in json.loads((p/manifest).read_text()).items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
assert json.loads((p/'optimizer-gate.json').read_text())['status']=='PASS'
plan=json.loads((p/'plan.json').read_text());first=p/'cases'/plan['cases'][0]['name'];assert json.loads((first/'evaluation-summary.json').read_text())['status']=='PASS'
archive.mkdir(parents=True,exist_ok=False)
for name in ['FILES_SHA256.json','TRAINING_FILES_SHA256.json','grace_pilot_train.py']:(archive/name).write_bytes((p/name).read_bytes())
for case in plan['cases'][1:]:
 work=(p/'cases'/case['name']).resolve();dest=(archive/'cases'/case['name']).resolve()
 assert work.is_relative_to(p/'cases') and dest.is_relative_to(archive) and not (work/'training-summary.json').exists()
 assert 'Disk quota exceeded' in (work/'train.log').read_text()
 original=(work/'input.yaml').read_bytes();dest.parent.mkdir(exist_ok=True);work.rename(dest);work.mkdir();(work/'input.yaml').write_bytes(original)
for n,s in files.items():(p/n).write_text(s)
for manifest in ['FILES_SHA256.json','TRAINING_FILES_SHA256.json']:
 d=json.loads((p/manifest).read_text())
 for n,s in files.items():d[n]=hashlib.sha256(s.encode()).hexdigest()
 # The general manifest also records the updated training manifest.
 if manifest=='TRAINING_FILES_SHA256.json':
  (p/manifest).write_text(json.dumps(d,indent=2)+'\n')
 else:general=d
general['TRAINING_FILES_SHA256.json']=hashlib.sha256((p/'TRAINING_FILES_SHA256.json').read_bytes()).hexdigest()
(p/'FILES_SHA256.json').write_text(json.dumps(general,indent=2)+'\n')
for manifest in ['FILES_SHA256.json','TRAINING_FILES_SHA256.json']:
 for n,h in json.loads((p/manifest).read_text()).items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
for n in ['grace_pilot_train.py','grace_storage_guard.py']:ast.parse((p/n).read_text())
subprocess.run(['bash','-n',str(p/'storage-probe.sbatch')],check=True)
record=dict(status='PASS',failed_job='20990675',preserved_success_index=0,restarted_indices=[1,2,3],restart='fresh aligned checkpoint and fresh Adam; failed epoch-one attempts archived',new_files={n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()},accounting=accounting)
marker.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''.replace('REMOTE',repr(REMOTE))
        d=json.loads(c.ssh(m.PY+' -c '+shlex.quote(code),json.dumps(files).encode()));(REC/'repair.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2));return
    assert json.loads((REC/'repair.json').read_text())['status']=='PASS'
    common=['sbatch','--parsable','--account=imet-2026-2','--nodes=1','--ntasks=1']
    probe=m.submit(REMOTE,'storage-probe',common+['--partition=apollo','--cpus-per-task=1','--mem=2G','--time=00:05:00','--job-name=TiAl-GRACE-storage','--output=storage-%j.log','storage-probe.sbatch'])
    retry=m.submit(REMOTE,'train-quota-retry',common+['--partition=v100','--gres=gpu:a100:1','--cpus-per-task=4','--mem=40G','--time=04:00:00','--array=1-3%1','--dependency=afterok:'+probe,'--job-name=TiAl-GRACE-mask-retry','--output=train-retry-%A_%a.log','train.sbatch'])
    code=r'''from pathlib import Path
import json,subprocess
p=Path(REMOTE)
for job in ['20990676','20990677']:assert 'JobState=PENDING' in subprocess.check_output(['scontrol','show','job',job],text=True)
subprocess.run(['scontrol','update','JobId=20990676','Dependency=afterok:'+RETRY],check=True)
states={j:subprocess.check_output(['scontrol','show','job',j],text=True) for j in ['20990676','20990677']}
assert 'Dependency=afterok:'+RETRY in states['20990676'] and 'Dependency=afterok:20990676' in states['20990677']
d=dict(storage_probe=PROBE,retry=RETRY,harmonics='20990676',comparison='20990677',states=states)
(p/'recovery/disk-quota-v1/dependencies.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
'''.replace('REMOTE',repr(REMOTE)).replace('RETRY',repr(retry)).replace('PROBE',repr(probe))
    d=json.loads(m.remote_python(code));(REC/'dependencies.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:v for k,v in d.items() if k!='states'},indent=2))

if __name__=='__main__':main()
