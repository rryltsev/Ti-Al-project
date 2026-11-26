"""Queue Apollo PBE from the last durable density after explicit forced cancellation."""
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess
from cluster import ROOT, RECORD, REMOTE, run, save, ssh

NAME = 'PBE-production02-force-001-apollo-forced-stop-20261024'
CHECKPOINT_SHA = 'bfb82d8a9eadfc0dcea5eda5ef9402bc7b8a2fbd8bd0bd00c05bc777d505731a'

if __name__ == '__main__':
    if (RECORD/(NAME+'-intent.json')).exists():
        raise RuntimeError('Reconcile existing intent before any repeated submission')
    old = run('squeue -h -j 20959336,20959747 -o "%i|%T|%R"')
    if old.strip():
        raise RuntimeError('Cancellation still in progress; do not submit yet: '+old)
    accounting = run('sacct -X -nP -j 20959336,20959747 --format=JobIDRaw,State,Elapsed,ExitCode')
    if not all(job+'|CANCELLED' in accounting for job in ['20959336','20959747']):
        raise RuntimeError('Cancellation not reconciled: '+accounting)
    source = REMOTE+'/PBE/production-02/force-001'
    run('test ! -e '+REMOTE+'/PBE/production-02/force-001-restart-03')
    observed = run('sha256sum '+source+'/CHGCAR').split()[0]
    if observed != CHECKPOINT_SHA:
        raise RuntimeError('Durable original checkpoint changed')
    files = [ROOT/'scripts'/n for n in ['restart_step_v4.py','launch_v6.sbatch']]
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for p in files:
        dest = REMOTE+'/'+p.name
        known = run('if test -f '+shlex.quote(dest)+'; then sha256sum '+shlex.quote(dest)+'; fi').strip()
        if known:
            if known.split()[0] != hashes[p.name]:
                raise RuntimeError('Immutable remote file differs: '+p.name)
        else:
            subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',
                'set -e; set -o noclobber; umask 077; cat > '+shlex.quote(dest)],input=p.read_bytes(),check=True)
        if run('sha256sum '+shlex.quote(dest)).split()[0] != hashes[p.name]:
            raise RuntimeError('Source transfer hash mismatch')
    launcher = REMOTE+'/launch_v6.sbatch'
    preflight = run('bash -n '+launcher+' && cd /home/u3129 && TIAL_LAUNCH_CHECK=1 bash '+launcher+' restart_step_v4.py PBE force-001 20958089')
    rejection = run('SLURM_JOB_PARTITION=v100 bash '+launcher+' restart_step_v4.py PBE force-001 20958089 2>&1; result=$?; test "$result" = 65')
    save(NAME+'-preflight.json',dict(preflight=preflight,a100_refusal=rejection,source_hashes=hashes,checkpoint_sha256=observed))
    args=['sbatch','--parsable','--partition=apollo','--exclude=tesla-a101',
          '--account=imet-2026-2','--qos=normal','--nodes=1','--ntasks=32',
          '--ntasks-per-node=32','--cpus-per-task=1','--mem=240G','--time=2-00:00:00',
          '--job-name=ti-PBE-apollo','--chdir='+REMOTE,'--output='+REMOTE+'/'+NAME+'-%j.out',
          launcher,'restart_step_v4.py','PBE','force-001','20958089']
    intent=dict(utc=datetime.now(timezone.utc).isoformat(),command=args,source_hashes=hashes,
                parent_job='20958089',checkpoint_source=source,checkpoint_sha256=observed,
                cancelled_jobs=accounting,scientific_changes=[],
                note='User explicitly requested immediate release of A100. Unsaved iterations of 20959336 are discarded; clean original checkpoint is retained.')
    save(NAME+'-intent.json',intent)
    response=run('set -o pipefail; '+shlex.join(args)+' | tee '+shlex.quote(REMOTE+'/'+NAME+'.jobid')).strip()
    job=response.split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Ambiguous submission: '+response)
    save(NAME+'-submission.json',dict(intent,job_id=job,state='submitted'))
    print(json.dumps(dict(job_id=job,partition='apollo',source=source,preflight=preflight,a100_refusal=rejection)))
