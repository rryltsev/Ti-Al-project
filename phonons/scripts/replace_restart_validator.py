"""Replace only pending restart jobs after a proven CHGCAR rounding mismatch."""
import base64
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from cluster import ROOT, RECORD, REMOTE, CFG, run, save, ssh

payload={p.name:base64.b64encode(p.read_bytes()).decode() for p in
         [ROOT/'scripts'/n for n in ['restart_step_v2.py','restart-step-v2.sbatch','checkpoint_guard_v2.py']]}
code='''
import json, base64, sys, hashlib
from pathlib import Path
root=Path(REMOTE_ROOT)
data=json.load(sys.stdin)
for name,encoded in data.items():
    raw=base64.b64decode(encoded); p=root/name
    if p.exists():
        assert p.read_bytes()==raw, 'Immutable file differs: '+name
    else:
        with p.open('xb') as f: f.write(raw)
print(json.dumps({n:hashlib.sha256(base64.b64decode(b)).hexdigest() for n,b in data.items()}))
'''.replace('REMOTE_ROOT',repr(REMOTE))
out=subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',CFG['python_base']+' -c '+shlex.quote(code)],
                   input=json.dumps(payload).encode(),capture_output=True,check=True)
hashes=json.loads(out.stdout)
save('restart-validator-v2-manifest.json',dict(files=hashes,reason='CHGCAR six-decimal coordinates; max position tolerance 3e-5 A, cell 1e-6 A, rtol=0'))
for xc,old,parent in [('PBE','20958709','20958089'),('r2SCAN','20958711','20958091')]:
    name=xc+'-production02-force-001-restart02'
    if (RECORD/(name+'-intent.json')).exists(): raise RuntimeError('Reconcile existing intent')
    state=run('squeue -h -j '+old+' -o %T').strip()
    if state!='PENDING': raise RuntimeError('Old job not pending: '+old+' '+state)
    save(name+'-replacement-intent.json',dict(old_job=old,reason='Rounding tolerance fix plus 8h checkpoint reserve',state=state))
    run('scancel '+old)
    terminal=run('sacct -X -nP -j '+old+' --format=JobIDRaw,State')
    if 'CANCELLED' not in terminal: raise RuntimeError('Cancellation not confirmed: '+terminal)
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal',
          '--nodes=1','--ntasks=32','--ntasks-per-node=32','--cpus-per-task=1','--mem=300G','--time=20:00:00',
          '--dependency=afterany:'+parent,'--kill-on-invalid-dep=yes','--job-name=tir2-'+xc+'-f001',
          '--chdir='+REMOTE,'--output='+REMOTE+'/'+name+'-%j.out',REMOTE+'/restart-step-v2.sbatch',xc,'force-001',parent]
    intent=dict(utc=datetime.now(timezone.utc).isoformat(),command=args,parent_job=parent,replaces=old,source_hashes=hashes)
    save(name+'-intent.json',intent)
    result=run('set -o pipefail; '+shlex.join(args)+' | tee '+shlex.quote(REMOTE+'/'+name+'.jobid')).strip()
    job=result.split(';')[0]
    if not job.isdigit(): raise RuntimeError('Unknown submission outcome: '+result)
    save(name+'-submission.json',dict(intent,job_id=job,state='submitted'))
    print(xc,job,flush=True)
