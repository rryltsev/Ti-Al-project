"""Submit, inspect and collect the authorized multi-seed/full-train expansion."""
from pathlib import Path
import argparse,datetime,json,shlex
import cluster_pilot as c

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-replicates-full-20261106-02'
PY='/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python'

def remote_python(code):return c.ssh(PY+' -c '+shlex.quote(code))

def submit(remote,key,args):
    code=r'''from pathlib import Path
import datetime,json,subprocess
root=Path(REMOTE);folder=root/'submissions';folder.mkdir(exist_ok=True)
record=folder/(KEY+'.json')
if record.exists():print(record.read_text())
else:
    lock=folder/(KEY+'.lock');lock.mkdir()
    try:
        command=COMMAND
        result=subprocess.run(command,cwd=root,text=True,capture_output=True,check=True)
        job=result.stdout.strip().split(';')[0];assert job.isdigit(),result.stdout
        data=dict(key=KEY,job_id=job,command=command,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        record.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
    finally:lock.rmdir()
'''.replace('REMOTE',repr(remote)).replace('KEY',repr(key)).replace('COMMAND',repr(args))
    record=json.loads(remote_python(code));folder=RUN/'submissions';folder.mkdir(exist_ok=True)
    (folder/(key+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(key,record['job_id']);return record['job_id']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','submit','status']);a=ap.parse_args()
    remote=json.loads((RUN/'package.json').read_text())['remote_root']
    if a.action=='preflight':
        code=r'''from pathlib import Path
import ast,hashlib,importlib.util,json,subprocess,sys
root=Path(REMOTE)
manifest=json.loads((root/'FILES_SHA256.json').read_text())
assert all(hashlib.sha256((root/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
for p in root.rglob('*.py'):ast.parse(p.read_text())
for p in root.rglob('*.sbatch'):subprocess.run(['bash','-n',str(p)],check=True)
assert not any(p.name=='test' for p in (root/'data').rglob('*'))
plan=json.loads((root/'plan.json').read_text());assert len(plan['cases'])==12
for case in plan['cases']:
    work=root/'training-plan01/cases'/case['name']
    subprocess.run([sys.executable,str(root/'verify_case.py')],cwd=work,capture_output=True,check=True)
sys.path.insert(0,str(root));import aggregate_expansion as agg
parent_checks=[]
for name,expected in plan['existing_seed10'].items():
    result,rows=agg.collect(Path(expected['remote_path']),expected)
    parent_checks.append(dict(name=name,model_sha256=result['model_sha256'],structures=len(rows)))
for module,expected in [('deepmd.pt.train.training','2ee355877da3ba10bf213b5653b3b858caa7795d846528c47ab1d7881dc3d090'),('deepmd.pt.model.descriptor.sezm_nn.lora','03591f701e266b9db0e395c53765391183dda38e9c8786d1d0a304a14c5a1644')]:
    # Resolve source paths without constructing models or allocating GPU memory.
    suffix=module.replace('.','/')+'.py';p=Path('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/lib/python3.12/site-packages')/suffix
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
print(json.dumps(dict(status='PASS',files=len(manifest),case_count=len(plan['cases']),all_case_input_verifiers_executed=True,parent_checks=parent_checks)))
'''.replace('REMOTE',repr(remote))
        result=json.loads(remote_python(code));(RUN/'remote-preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    elif a.action=='submit':
        assert json.loads((RUN/'remote-preflight.json').read_text())['status']=='PASS'
        common=['sbatch','--parsable','--account=imet-2026-2','--nodes=1','--ntasks=1']
        gpu=['--partition=v100','--gres=gpu:1','--cpus-per-task=4','--mem=40G','--time=04:00:00']
        j200=submit(remote,'N200-training',common+gpu+['--array=0-7%1','--job-name=TiAl-N200-seeds','--output=training-plan01/N200-%A_%a.log','training-plan01/train.sbatch','N200'])
        q200=submit(remote,'N200-QC',common+['--partition=apollo','--cpus-per-task=4','--mem=12G','--time=00:10:00','--array=0-7%4','--dependency=afterany:'+j200,'--job-name=TiAl-N200-seeds-QC','--output=training-plan01/N200-QC-%A_%a.log','training-plan01/qc.sbatch','N200'])
        a200=submit(remote,'N200-aggregate',common+['--partition=apollo','--cpus-per-task=1','--mem=8G','--time=00:10:00','--dependency=afterany:'+q200,'--job-name=TiAl-N200-3seeds','--output=N200-aggregate-%j.log','aggregate.sbatch','N200'])
        j694=submit(remote,'N694-training',common+gpu+['--array=0-3%1','--dependency=afterany:'+j200,'--job-name=TiAl-N694-control','--output=training-plan01/N694-%A_%a.log','training-plan01/train.sbatch','N694'])
        q694=submit(remote,'N694-QC',common+['--partition=apollo','--cpus-per-task=4','--mem=12G','--time=00:10:00','--array=0-3%4','--dependency=afterany:'+j694,'--job-name=TiAl-N694-QC','--output=training-plan01/N694-QC-%A_%a.log','training-plan01/qc.sbatch','N694'])
        a694=submit(remote,'N694-aggregate',common+['--partition=apollo','--cpus-per-task=1','--mem=8G','--time=00:10:00','--dependency=afterany:'+q694,'--job-name=TiAl-N694-compare','--output=N694-aggregate-%j.log','aggregate.sbatch','N694'])
        (RUN/'submission-summary.json').write_text(json.dumps(dict(N200_training=j200,N200_QC=q200,N200_aggregate=a200,N694_training=j694,N694_QC=q694,N694_aggregate=a694,max_new_GPU_concurrency=1,dependency_reason='Full train controls wait for completion of N200 GPU array. CPU QC and failure-visible aggregation use afterany so failed tasks cannot leave a permanent dependency wait.'),indent=2)+'\n')
    else:
        code=r'''from pathlib import Path
import datetime,json,subprocess
root=Path(REMOTE);plan=json.loads((root/'plan.json').read_text());cases={}
jobs=[json.loads(p.read_text())['job_id'] for p in (root/'submissions').glob('*.json')]
for case in plan['cases']:
    p=root/'training-plan01/cases'/case['name'];item={}
    for f in ['attempt.json','input-verification.json','zero-step.json','progress.json','training-summary.json','evaluation/summary.json','evaluation/harmonics/summary.json','evaluation/numerical-QC/summary.json','evaluation/physical-diagnostics/summary.json']:
        if (p/f).exists():
            try:item[f]=json.loads((p/f).read_text())
            except json.JSONDecodeError:pass
    if (p/'parameter-inventory.json').exists():
        inv=json.loads((p/'parameter-inventory.json').read_text());item['inventory']={k:v for k,v in inv.items() if k!='parameters'}
    if (p/'train.log').exists():item['train_tail']=(p/'train.log').read_text(errors='replace').splitlines()[-6:]
    cases[case['name']]=item
queue=subprocess.check_output(['squeue','-r','-j',','.join(jobs),'-o','%.22i %.10T %.12M %.35R'],text=True)
accounting=subprocess.check_output(['sacct','-j',','.join(jobs),'--format=JobID,State,ElapsedRaw,AllocCPUS,ExitCode','-P'],text=True)
aggregates={p.parent.name:json.loads(p.read_text()) for p in (root/'aggregates').glob('*/summary.json')}
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue=queue,accounting=accounting,cases=cases,aggregates=aggregates)))
'''.replace('REMOTE',repr(remote))
        result=json.loads(remote_python(code));folder=RUN/'progress-snapshots';folder.mkdir(exist_ok=True)
        path=folder/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json');path.write_text(json.dumps(result,indent=2)+'\n')
        (RUN/'latest-progress.json').write_text(json.dumps(dict(snapshot=str(path),utc=result['utc']),indent=2)+'\n')
        print(result['queue']);print(json.dumps({k:dict(steps=v.get('progress.json',{}).get('steps',0),zero_step=v.get('zero-step.json',{}).get('status'),train=v.get('training-summary.json',{}).get('status'),inventory=v.get('inventory'),tail=v.get('train_tail',[])) for k,v in result['cases'].items() if v},indent=2))

if __name__=='__main__':main()
