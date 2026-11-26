"""Recover the interrupted pre-submission upload, preserving every existing file."""
import cluster as c
import base64,json,shlex,subprocess
from datetime import datetime,timezone

p=c.RECORD/'DPA4-PBE-20k-matched-20261102-01'
r=c.REMOTE+'/'+p.name
assert not (p/'submission-intent.json').exists() and not (p/'submission.json').exists()
intent=json.loads((p/'staging-intent.json').read_text())
(p/'staging-failure.json').write_text(json.dumps({'reason':'SSH banner exchange timeout while uploading worker.py; no sbatch attempted','utc':datetime.now(timezone.utc).isoformat()},indent=2))
data={n:base64.b64encode((p/n).read_bytes()).decode() for n in intent['files']}
code='''import pathlib,base64,hashlib
p=pathlib.Path(REMOTE)
for n,b64 in DATA.items():
 f=p/n; b=base64.b64decode(b64)
 if f.exists(): assert f.read_bytes()==b, str(f)
 else:
  with f.open('xb') as handle: handle.write(b)
 assert hashlib.sha256(f.read_bytes()).hexdigest()==HASHES[n]
print('All remote input hashes verified')
'''.replace('REMOTE',repr(r)).replace('DATA',repr(data)).replace('HASHES',repr(intent['files']))
uploaded=subprocess.run(c.ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt','python3 -'],input=code,text=True,capture_output=True)
assert uploaded.returncode==0, uploaded.stdout+uploaded.stderr
print(uploaded.stdout,flush=True)
queue=c.run("squeue -h -u u3129 -o '%i|%Z|%T'")
assert r not in queue
(p/'pre-submit-queue.txt').write_text(queue)
args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=12G','--time=00:15:00','--no-requeue','--job-name=ti-DPA4-PBE20k','--chdir='+r,'--output='+r+'/run-%j.out',r+'/job.sbatch']
(p/'submission-intent.json').write_text(json.dumps({'command':args,'utc':datetime.now(timezone.utc).isoformat()},indent=2))
job=c.run(shlex.join(args)).strip().split(';')[0];assert job.isdigit(),job
(p/'submission.json').write_text(json.dumps({'job_id':job,'remote':r},indent=2))
print('Submitted',job,flush=True)
