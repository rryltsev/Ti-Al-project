"""Recover only the known XML GGA-parser failures, without invoking VASP."""
from pathlib import Path
import json,os,time
import workflow as w
pre=json.loads((w.ROOT/'preflight.json').read_text())
rows={r['structure_id']:r for r in pre['records']}
report=[]
for p in sorted((w.ROOT/'calculations').glob('*/*/status.json')):
    status=json.loads(p.read_text())
    if status.get('state')!='failed' or status.get('error')!="'GGA'":continue
    d=p.parent;inputs=json.loads((d/'input-manifest.json').read_text())
    assert all(w.sha(d/n)==h for n,h in inputs['input_sha256'].items())
    w.validate(d,inputs,rows[status['structure_id']])
    backup=d/'status-before-gga-validator-fix.json'
    with backup.open('x') as f:json.dump(status,f,indent=2)
    w.save(p,dict(state='validated',structure_id=status['structure_id'],functional=status['functional'],job_id=status['job_id'],finished_unix=time.time(),revalidated_without_DFT_rerun=True,original_slurm_exit_retained=True,validator_sha256=w.sha(w.ROOT/'workflow.py')))
    report.append(dict(structure_id=status['structure_id'],functional=status['functional'],revalidated=True))
result=dict(unix=time.time(),repair_job_id=os.environ.get('SLURM_JOB_ID'),records=report,DFT_rerun=False)
w.save(w.ROOT/('validation-repair-'+str(os.environ.get('SLURM_JOB_ID','initial'))+'.json'),result)
print(json.dumps(result,indent=2))
