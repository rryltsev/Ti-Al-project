from pathlib import Path
import datetime,json,shlex
import cluster_pilot as c
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02';remote=json.loads((RUN/'package.json').read_text())['remote_root']
code='''from pathlib import Path
import json,subprocess,datetime
root=Path(REMOTE);cases={}
for name in ['PBE-lora16','PBE-full','r2SCAN-lora16','r2SCAN-full']:
    p=root/'training-plan01/cases'/name;item={}
    for filename in ['attempt.json','progress.json','parameter-inventory.json','zero-step.json','training-summary.json','parameter-update-check.json','numerical-check-step01000/summary.json','evaluation/summary.json','evaluation/numerical-QC/summary.json']:
        if (p/filename).exists():
            try:item[filename]=json.loads((p/filename).read_text())
            except json.JSONDecodeError:pass
    item['checkpoint_checks']={x.name:json.loads(x.read_text()) for x in (p/'checks').glob('*.json')}
    if (p/'train.log').exists():item['train_log_tail']=(p/'train.log').read_text(errors='replace').splitlines()[-18:]
    cases[name]=item
queue=subprocess.check_output(['squeue','-r','-j','20988454,20988460,20988469','-o','%.20i %.10T %.12M %.35R'],text=True)
acct=subprocess.check_output(['sacct','-j','20988437,20988440,20988446,20988445,20988447,20988449,20988452,20988473','--format=JobID,JobName,State,ElapsedRaw,AllocCPUS,TotalCPU,MaxRSS,ExitCode','-P'],text=True)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,queue=queue,completed_CPU_accounting=acct)))
'''.replace('REMOTE',repr(remote))
data=json.loads(c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code)))
folder=RUN/'progress-snapshots';folder.mkdir(exist_ok=True);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');path=folder/(stamp+'.json');path.write_text(json.dumps(data,indent=2)+'\n')
(RUN/'latest-progress.json').write_text(json.dumps(dict(snapshot=str(path),utc=data['utc']),indent=2)+'\n')
print(data['queue']);print(json.dumps({name:dict(progress=v.get('progress.json'),zero_step=v.get('zero-step.json',{}).get('status'),checkpoint_checks=list(v['checkpoint_checks'])) for name,v in data['cases'].items()},indent=2))
