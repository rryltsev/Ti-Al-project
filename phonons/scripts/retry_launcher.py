"""One corrected submission after confirmed pre-VASP shell-argument failure."""
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess
from cluster import ROOT, RECORD, REMOTE, CFG, run, save, ssh

path = ROOT / 'scripts/production-v2.sbatch'
jobs = {xc: json.loads((RECORD / (xc + '-submission.json')).read_text())['job_id']
        for xc in CFG['functionals']}
status = run('sacct -X -nP -j ' + ','.join(jobs.values()) +
             ' --format=JobIDRaw,State,ExitCode,CPUTimeRAW')
rows = [line.split('|') for line in status.splitlines() if line.strip()]
if len(rows) != 3 or any(r[1:3] != ['FAILED', '1:0'] for r in rows):
    raise RuntimeError('Initial failures are not fully reconciled: ' + status)
for xc in CFG['functionals']:
    run('test ! -f ' + shlex.quote(REMOTE + '/' + xc + '/relax-01/OUTCAR'))
save('launcher-correction.json', {
    'utc': datetime.now(timezone.utc).isoformat(), 'prior_jobs': jobs, 'prior_accounting': status,
    'reason': 'Intel environment initialization changed $1; XC was empty before VASP started.',
    'correction': 'Capture and validate XC_NAME before sourcing Intel environment.',
    'script_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'max_corrected_submissions': 3})
subprocess.run(ssh.ssh_args() + ['-o', 'BatchMode=yes', 'mlip-umt',
    'set -e; set -o noclobber; cat > ' + shlex.quote(REMOTE + '/production-v2.sbatch')],
    input=path.read_bytes(), check=True)
for xc in CFG['functionals']:
    name = xc + '-v2'
    if (RECORD / (name + '-intent.json')).exists(): raise RuntimeError('Reconcile existing retry intent')
    args = ['sbatch', '--parsable', '--partition=apollo', '--account=imet-2026-2', '--qos=normal',
            '--nodes=1', '--ntasks=16', '--cpus-per-task=1', '--mem=32G', '--time=12:00:00',
            '--job-name=ti-ph-' + name, '--chdir=' + REMOTE,
            '--output=' + REMOTE + '/' + name + '-%j.out', REMOTE + '/production-v2.sbatch', xc]
    intent = {'utc': datetime.now(timezone.utc).isoformat(), 'command': args, 'state': 'submission_intent'}
    save(name + '-intent.json', intent)
    response = run('set -o pipefail; cd ' + shlex.quote(REMOTE) + '; ' + shlex.join(args) +
                   ' | tee ' + shlex.quote(name + '.jobid')).strip()
    job = response.split(';')[0]
    if not job.isdigit(): raise RuntimeError('Ambiguous submission: ' + response)
    save(name + '-submission.json', dict(intent, job_id=job, state='submitted'))
    print(xc, job, flush=True)
