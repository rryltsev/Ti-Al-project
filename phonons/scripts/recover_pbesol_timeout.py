"""One corrected PBEsol attempt; never overwrite or blindly resubmit."""
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from cluster import ROOT, RECORD, REMOTE, CFG, run, save, ssh


def main():
    name = 'PBEsol-production03-force-001-submit02'
    if (RECORD/(name+'-intent.json')).exists():
        raise RuntimeError('Existing intent; reconcile instead of resubmitting')
    evidence = run('sacct -X -nP -j 20958090,20958710 --format=JobIDRaw,State,Elapsed,ExitCode')
    if '20958090|TIMEOUT|' not in evidence or '20958710|CANCELLED' not in evidence:
        raise RuntimeError('Unexpected scheduler state: '+evidence)
    if run('squeue -h -j 20958710').strip():
        raise RuntimeError('Prior continuation still in queue')
    save('PBEsol-production02-timeout-audit.json',dict(utc=datetime.now(timezone.utc).isoformat(),
         scheduler=evidence,source=REMOTE+'/PBEsol/production-02/force-001',
         checkpoint_absent=True,remedy='64 ranks, KPAR8; 8h minimum reserve includes force finalization',
         superseded_pending_job='20958710'))
    source=RECORD/'snapshot/PBEsol/recovery-01/force-001'
    hashes=json.loads((source/'input-hashes.json').read_text())
    expected=''.join(h+'  '+REMOTE+'/PBEsol/recovery-01/force-001/'+n+'\n' for n,h in hashes.items())
    subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt','sha256sum -c -'],input=expected.encode(),capture_output=True,check=True)
    files=[ROOT/'config/execution-policy-03.json',*[ROOT/'scripts'/n for n in ['checkpoint_guard_v2.py','production_step_v3.py','production-step-v3-unix.sbatch']]]
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for p in files:
        dest=REMOTE+'/'+p.name
        known=run('if test -f '+shlex.quote(dest)+'; then sha256sum '+shlex.quote(dest)+'; fi').strip()
        if known:
            if known.split()[0] != manifest[p.name]: raise RuntimeError('Immutable remote file differs')
        else:
            subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',
                'set -e; set -o noclobber; umask 077; cat > '+shlex.quote(dest)],input=p.read_bytes(),check=True)
        if run('sha256sum '+shlex.quote(dest)).split()[0] != manifest[p.name]: raise RuntimeError('Transfer mismatch')
    save('production03-source-manifest.json',dict(files=manifest,source_input_hashes=hashes))
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal',
          '--nodes=2','--ntasks=64','--ntasks-per-node=32','--cpus-per-task=1','--mem=300G','--time=20:00:00',
          '--job-name=ti3-PBEsol-f001','--chdir='+REMOTE,'--output='+REMOTE+'/'+name+'-%j.out',
          REMOTE+'/production-step-v3-unix.sbatch','PBEsol','force-001']
    intent=dict(utc=datetime.now(timezone.utc).isoformat(),command=args,source_hashes=manifest,source_input_hashes=hashes)
    save(name+'-intent.json',intent)
    response=run('set -o pipefail; '+shlex.join(args)+' | tee '+shlex.quote(REMOTE+'/'+name+'.jobid')).strip()
    job=response.split(';')[0]
    if not job.isdigit(): raise RuntimeError('Ambiguous submission: '+response)
    save(name+'-submission.json',dict(intent,job_id=job,state='submitted'))
    print('PBEsol corrected attempt',job,flush=True)


if __name__=='__main__': main()
