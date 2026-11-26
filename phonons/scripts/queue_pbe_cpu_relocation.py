"""Queue one immutable Apollo continuation after the A100 CPU job exits cleanly."""
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess
from cluster import ROOT, RECORD, REMOTE, run, save, ssh

NAME = 'PBE-production02-force-001-relocate-apollo-20261024'
JOB = '20959336'
if __name__ == '__main__':
    if (RECORD/(NAME+'-intent.json')).exists():
        raise RuntimeError('Reconcile existing submission intent; no blind retries')
    parent = run('scontrol show job -o '+JOB)
    if not ('JobState=RUNNING ' in parent or 'JobState=COMPLETED ' in parent):
        raise RuntimeError('Unexpected parent state: '+parent)
    folder = REMOTE+'/PBE/production-02/force-001-restart-01'
    run('test -f '+folder+'/checkpoint-request.json && test -f '+folder+'/STOPCAR && test ! -e '+REMOTE+'/PBE/production-02/force-001-restart-02')
    queue = run('squeue -h -u u3129 -o "%i|%j|%T|%E"')
    if 'ti-PBE-cpu-return' in queue:
        raise RuntimeError('Matching job already queued; reconcile: '+queue)
    files = [ROOT/'scripts'/n for n in ['restart_step_v3.py','launch_v5.sbatch']]
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for p in files:
        dest = REMOTE+'/'+p.name
        known = run('if test -f '+shlex.quote(dest)+'; then sha256sum '+shlex.quote(dest)+'; fi').strip()
        if known:
            if known.split()[0] != hashes[p.name]:
                raise RuntimeError('Immutable remote script differs: '+p.name)
        else:
            subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',
                'set -e; set -o noclobber; umask 077; cat > '+shlex.quote(dest)], input=p.read_bytes(),check=True)
        if run('sha256sum '+shlex.quote(dest)).split()[0] != hashes[p.name]:
            raise RuntimeError('Source transfer checksum mismatch')
    preflight = run('bash -n '+REMOTE+'/launch_v5.sbatch && cd /home/u3129 && TIAL_LAUNCH_CHECK=1 bash '+REMOTE+'/launch_v5.sbatch restart_step_v3.py PBE force-001 '+JOB)
    save(NAME+'-preflight.json',dict(utc=datetime.now(timezone.utc).isoformat(),source_hashes=hashes,result=preflight))
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal',
          '--nodes=1','--ntasks=32','--ntasks-per-node=32','--cpus-per-task=1','--mem=240G',
          '--time=2-00:00:00','--dependency=afterany:'+JOB,'--kill-on-invalid-dep=yes',
          '--job-name=ti-PBE-cpu-return','--chdir='+REMOTE,'--output='+REMOTE+'/'+NAME+'-%j.out',
          REMOTE+'/launch_v5.sbatch','restart_step_v3.py','PBE','force-001',JOB]
    intent=dict(utc=datetime.now(timezone.utc).isoformat(),command=args,parent_job=JOB,
                parent_scheduler=parent,queue_before=queue,source_hashes=hashes,scientific_changes=[],
                reason='Release scarce A100 host for GPU work; preserve density on clean soft stop')
    save(NAME+'-intent.json',intent)
    response=run('set -o pipefail; '+shlex.join(args)+' | tee '+shlex.quote(REMOTE+'/'+NAME+'.jobid')).strip()
    newjob=response.split(';')[0]
    if not newjob.isdigit():
        raise RuntimeError('Ambiguous submission outcome: '+response)
    save(NAME+'-submission.json',dict(intent,job_id=newjob,state='submitted'))
    print(json.dumps(dict(job_id=newjob,parent_job=JOB,partition='apollo',preflight=preflight)))
