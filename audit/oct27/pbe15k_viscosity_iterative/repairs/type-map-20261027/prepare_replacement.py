import json,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
submission=json.loads((ROOT/'submission.json').read_text())
assert submission['job_id']=='20961934'
cfg=json.loads((ROOT/'protocol.json').read_text())
code=f'''
import hashlib,json,pathlib,subprocess,time
p=pathlib.Path({cfg['remote']!r})
r=p/'repairs/type-map-20261027'
receipt=r/'replacement-submission.json'
if receipt.exists():
    print(receipt.read_text())
    raise SystemExit(0)
queue=subprocess.check_output(['squeue','--me','-h','-o','%i|%j'],text=True)
assert not any(line.split('|')[0]=='20961934' or line.split('|')[1]=='TiAl-visc15k-iter' for line in queue.splitlines()),queue
acct=subprocess.check_output(['sacct','-j','20961934','--format=JobID,State,ExitCode','-Pn'],text=True)
assert '20961934|FAILED|1:0' in acct,acct
repair=json.loads((r/'repair.json').read_text())
assert hashlib.sha256((p/'workflow.py').read_bytes()).hexdigest()==repair['new_sha256']
assert not (p/'final.json').exists()
args={submission['command']!r}
with (r/'replacement-intent.json').open('x') as fp:
    json.dump(dict(command=args,previous_job_id='20961934',reason='Failed job was purged; scontrol requeue returned Invalid job id',unix=time.time(),queue=queue,accounting=acct),fp,indent=2)
result=subprocess.run(args,capture_output=True,text=True)
(r/'replacement-sbatch-result.json').write_text(json.dumps(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2))
assert result.returncode==0,result.stderr
jid=result.stdout.strip().split(';')[0]
assert jid.isdigit(),result.stdout
record=dict(job_id=jid,command=args,unix=time.time(),previous_job_id='20961934',reason='Replacement after corrected preflight assertion; original job purged')
receipt.write_text(json.dumps(record,indent=2)+'\\n')
print(receipt.read_text())
'''
(HERE/'submit_replacement.sh').write_text('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -B -c '+shlex.quote(code)+'\n',encoding='utf-8')
print('Guarded one-time replacement command prepared; no submission performed.')
