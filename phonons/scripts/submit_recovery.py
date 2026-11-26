"""Bounded OOM correction within the original aggregate CPU-hour ceiling."""
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess
from cluster import ROOT, RECORD, REMOTE, CFG, run, save, ssh

jobs = [json.loads(p.read_text())['job_id'] for p in RECORD.glob('*-submission.json')]
accounting = run('sacct -X -nP -j ' + ','.join(jobs) + ' --format=JobIDRaw,State,CPUTimeRAW')
rows = [r.split('|') for r in accounting.splitlines() if r.strip()]
if len(rows) != len(jobs) or any(r[1].split()[0] not in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY'] for r in rows):
    raise RuntimeError('Prior allocations must be terminal and fully reconciled: ' + accounting)
used = sum(int(r[2]) for r in rows)
reserved = 3 * 16 * 80 * 60
if used + reserved > CFG['resources']['maximum_reserved_CPU_hours'] * 3600:
    raise RuntimeError('Exceeds original aggregate CPU-hour ceiling')
files = [ROOT / 'scripts/recover_forces.py', ROOT / 'scripts/recovery.sbatch']
record = {'utc': datetime.now(timezone.utc).isoformat(), 'reason': 'Confirmed OOM in all force-000 steps; MPI ranks remained alive until explicit cancellation.',
          'prior_accounting': accounting, 'used_CPU_seconds': used, 'reserved_CPU_seconds': reserved,
          'total_worst_case_CPU_hours': (used+reserved)/3600,
          'memory_GiB_per_job': 192, 'walltime': '01:20:00', 'ranks':16,
          'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
save('recovery-plan.json', record)
for p in files:
    subprocess.run(ssh.ssh_args() + ['-o','BatchMode=yes','mlip-umt',
        'set -e; set -o noclobber; cat > ' + shlex.quote(REMOTE+'/'+p.name)], input=p.read_bytes(),check=True)
for xc in CFG['functionals']:
    name = xc + '-recovery01'
    if (RECORD / (name+'-intent.json')).exists(): raise RuntimeError('Reconcile prior submission intent')
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal',
          '--nodes=1','--ntasks=16','--cpus-per-task=1','--mem=192G','--time=01:20:00',
          '--job-name=ti-ph-'+name,'--chdir='+REMOTE,'--output='+REMOTE+'/'+name+'-%j.out',REMOTE+'/recovery.sbatch',xc]
    intent={'utc':datetime.now(timezone.utc).isoformat(),'command':args,'state':'submission_intent'}
    save(name+'-intent.json',intent)
    result=run('set -o pipefail; cd '+shlex.quote(REMOTE)+'; '+shlex.join(args)+' | tee '+shlex.quote(name+'.jobid')).strip()
    job=result.split(';')[0]
    if not job.isdigit(): raise RuntimeError('Ambiguous submission: '+result)
    save(name+'-submission.json',dict(intent,job_id=job,state='submitted'))
    print(xc,job,flush=True)
