"""Submit and collect only the bounded CPU preparation audit (no GRACE training)."""
from pathlib import Path
import argparse,json,hashlib,shlex
import manage_expansion as m
from package_grace_preparation import RUN,REMOTE,ROOT,ENV

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['submit','status','repair-hash']);a=ap.parse_args();m.RUN=RUN
    if a.action=='repair-hash':
        old=RUN/'latest-preflight-status.json';failure=RUN/'preflight-v1-failure.json'
        if not failure.exists():failure.write_bytes(old.read_bytes())
        assert 'FAILED' in json.loads(failure.read_text())['accounting']
        oldsub=RUN/'preflight-submission-v1.json'
        if not oldsub.exists():oldsub.write_bytes((RUN/'preflight-submission.json').read_bytes())
        worker=(ROOT/'scripts/grace_mask_preflight.py').read_text(encoding='utf8')
        digest=hashlib.sha256(worker.encode()).hexdigest()
        batch=(RUN/'package/preflight.sbatch').read_text().replace('grace_mask_preflight.py','grace_mask_preflight_v2.py')
        batch=batch.replace(ENV+'/bin/python grace_mask_preflight_v2.py','sha256sum -c preflight-v2.sha256\n'+ENV+'/bin/python grace_mask_preflight_v2.py')
        payload={'grace_mask_preflight_v2.py':worker,'preflight-v2.sbatch':batch,'preflight-v2.sha256':digest+'  grace_mask_preflight_v2.py\n'}
        local=RUN/'repair-v2';local.mkdir(exist_ok=True)
        for name,s in payload.items():(local/name).write_text(s,encoding='utf8',newline='\n')
        code='''import json,sys
from pathlib import Path
root=Path(REMOTE)
for name,body in json.load(sys.stdin).items():
    p=root/name
    if p.exists():assert p.read_text()==body
    else:p.write_text(body)
print('STAGED_REPAIR_V2')
'''.replace('REMOTE',repr(REMOTE))
        print(m.c.ssh(m.PY+' -c '+shlex.quote(code),json.dumps(payload).encode()).decode())
        job=m.submit(REMOTE,'GRACE-mask-data-preflight-v2',['sbatch','--parsable','--account=imet-2026-2','--partition=apollo','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=16G','--time=00:10:00','--job-name=TiAl-GRACE-prep-v2','--output=preflight-v2-%j.log','preflight-v2.sbatch'])
        (RUN/'preflight-submission.json').write_text(json.dumps(dict(job_id=job,resources=dict(cpus=4,memory='16G',time='00:10:00',gpus=0),training=False,audit_subdir='preflight-v2',log_prefix='preflight-v2-'),indent=2)+'\n')
        return
    if a.action=='submit':
        assert (RUN/'stage.json').exists()
        job=m.submit(REMOTE,'GRACE-mask-data-preflight',['sbatch','--parsable','--account=imet-2026-2','--partition=apollo','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=16G','--time=00:10:00','--job-name=TiAl-GRACE-prep','--output=preflight-%j.log','preflight.sbatch'])
        (RUN/'preflight-submission.json').write_text(json.dumps(dict(job_id=job,resources=dict(cpus=4,memory='16G',time='00:10:00',gpus=0),training=False),indent=2)+'\n')
        return
    sub=json.loads((RUN/'preflight-submission.json').read_text());job=sub['job_id']
    code='''from pathlib import Path
import datetime,json,subprocess
root=Path(REMOTE);job=JOB
files={p.name:json.loads(p.read_text()) for p in (root/AUDIT).glob('*.json')}
log=root/(PREFIX+job+'.log')
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,accounting=subprocess.check_output(['sacct','-j',job,'-X','--format=JobID,State,Elapsed,ExitCode','-P'],text=True),log_tail=log.read_text(errors='replace').splitlines()[-24:] if log.exists() else [])))
'''.replace('REMOTE',repr(REMOTE)).replace('JOB',repr(job)).replace('AUDIT',repr(sub.get('audit_subdir','preflight'))).replace('PREFIX',repr(sub.get('log_prefix','preflight-')))
    data=json.loads(m.remote_python(code));(RUN/'latest-preflight-status.json').write_text(json.dumps(data,indent=2)+'\n')
    for name,value in data['files'].items():
        out=RUN/'collected-preflight';out.mkdir(exist_ok=True)
        p=out/name
        if p.exists():assert json.loads(p.read_text())==value
        else:p.write_text(json.dumps(value,indent=2)+'\n')
    print(data['accounting'])
    if 'summary.json' in data['files']:
        s=data['files']['summary.json'];print(json.dumps({k:v for k,v in s.items() if k not in ['parameters','checkpoint_optimizer_keys']}))
    else:print('\n'.join(data['log_tail']))

if __name__=='__main__':main()
