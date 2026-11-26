"""Reconcile failed launchers and replace them with an absolute-path wrapper."""
import hashlib
import shlex
import subprocess
from datetime import datetime, timezone
from cluster import ROOT, RECORD, REMOTE, run, save, ssh

script=ROOT/'scripts/launch_v4.sbatch'
raw=script.read_bytes().replace(b'\r\n',b'\n')
script.write_bytes(raw)
sha=hashlib.sha256(raw).hexdigest()
dest=REMOTE+'/launch_v4.sbatch'
known=run('if test -f '+shlex.quote(dest)+'; then sha256sum '+shlex.quote(dest)+'; fi').strip()
if known:
    if known.split()[0]!=sha: raise RuntimeError('Immutable wrapper differs')
else:
    subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt','set -e; set -o noclobber; umask 077; cat > '+shlex.quote(dest)],input=raw,check=True)
if run('sha256sum '+shlex.quote(dest)).split()[0]!=sha: raise RuntimeError('Deployment mismatch')
check=run('bash -n '+shlex.quote(dest)+' && cd /home/u3129 && TIAL_LAUNCH_CHECK=1 bash '+shlex.quote(dest)+' restart_step_v2.py PBE force-001 20958089 && TIAL_LAUNCH_CHECK=1 bash '+shlex.quote(dest)+' production_step_v3.py PBEsol force-001')
if REMOTE+'/restart_step_v2.py' not in check or REMOTE+'/production_step_v3.py' not in check: raise RuntimeError(check)
save('launch-v4-preflight.json',dict(utc=datetime.now(timezone.utc).isoformat(),sha256=sha,output=check))
print(check,flush=True)
cases=[('PBE','20959188','20958089','restart_step_v2.py',1,'PBE-production02-force-001-restart03'),
       ('r2SCAN','20959189','20958091','restart_step_v2.py',1,'r2SCAN-production02-force-001-restart03'),
       ('PBEsol','20959187',None,'production_step_v3.py',2,'PBEsol-production03-force-001-submit03')]
for xc,old,parent,driver,nodes,name in cases:
    if (RECORD/(name+'-intent.json')).exists(): raise RuntimeError('Existing intent: reconcile before retry')
    accounting=run('sacct -X -nP -j '+old+' --format=JobIDRaw,State,Elapsed,ExitCode')
    if parent:
        if '|FAILED|00:00:00|2:0' not in accounting: raise RuntimeError(accounting)
        oldname=xc+'-production02-force-001-restart02-'+old+'.out'
        log=run('cat '+shlex.quote(REMOTE+'/'+oldname))
        if "can't open file '/misc/home4/u3129/restart_step_v2.py'" not in log: raise RuntimeError(log)
        save(name+'-failure-audit.json',dict(old_job=old,accounting=accounting,log=log,remedy='Absolute driver path; cd explicit run root after environment setup'))
    else:
        state=run('squeue -h -j '+old+' -o %T').strip()
        if state!='PENDING': raise RuntimeError('PBEsol no longer pending: '+state)
        save(name+'-replacement-audit.json',dict(old_job=old,state=state,reason='Same launcher cwd bug, fix before VASP starts'))
        run('scancel '+old)
        if 'CANCELLED' not in run('sacct -X -nP -j '+old+' --format=State'): raise RuntimeError('Cancellation not confirmed')
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal',
          '--nodes='+str(nodes),'--ntasks='+str(32*nodes),'--ntasks-per-node=32','--cpus-per-task=1',
          '--mem=300G','--time=20:00:00','--job-name=ti4-'+xc+'-f001','--chdir='+REMOTE,
          '--output='+REMOTE+'/'+name+'-%j.out',dest,driver,xc,'force-001']
    if parent: args.append(parent)
    intent=dict(utc=datetime.now(timezone.utc).isoformat(),command=args,replaces=old,parent_job=parent,wrapper_sha256=sha)
    save(name+'-intent.json',intent)
    response=run('set -o pipefail; cd '+shlex.quote(REMOTE)+'; '+shlex.join(args)+' | tee '+shlex.quote(name+'.jobid')).strip()
    job=response.split(';')[0]
    if not job.isdigit(): raise RuntimeError('Unknown submission outcome: '+response)
    save(name+'-submission.json',dict(intent,job_id=job,state='submitted'))
    print(xc,job,flush=True)
