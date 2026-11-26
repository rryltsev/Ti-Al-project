"""Deploy immutable restart code, arm in-allocation guards and queue afterany jobs."""
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess
from cluster import ROOT, RECORD, REMOTE, CFG, run, save, ssh


def main():
    files = [ROOT/'scripts'/n for n in ['checkpoint_guard.py','restart_step.py','restart-step.sbatch']]
    manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for p in files:
        dest = REMOTE + '/' + p.name
        known = run('if test -f '+shlex.quote(dest)+'; then sha256sum '+shlex.quote(dest)+'; fi').strip()
        if known:
            if known.split()[0] != manifest[p.name]:
                raise RuntimeError('Remote immutable script differs: ' + p.name)
        else:
            subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',
                'set -e; set -o noclobber; umask 077; cat > '+shlex.quote(dest)],input=p.read_bytes(),check=True)
        if run('sha256sum '+shlex.quote(dest)).split()[0] != manifest[p.name]:
            raise RuntimeError('Deployment hash mismatch')
    save('checkpoint-restart-source-manifest.json',dict(utc=datetime.now(timezone.utc).isoformat(),files=manifest))
    for xc in CFG['functionals']:
        original = json.loads((RECORD/(xc+'-production02-force-001-submission.json')).read_text())
        job = original['job_id']
        folder = REMOTE+'/'+xc+'/production-02/force-001'
        name = xc+'-production02-force-001-restart01'
        if not (RECORD/(name+'-guard-intent.json')).exists():
            # Convert scheduler local timestamps on the cluster itself.
            import re
            info = run('scontrol show job -o '+job)
            if 'JobState=RUNNING ' not in info:
                raise RuntimeError('Parent not running; reconcile before arming: '+info)
            deadline = run('date -d '+shlex.quote(re.search(r'EndTime=(\S+)',info).group(1))+' +%s').strip()
            args = ['srun','--jobid='+job,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=64M',
                    CFG['python_base'],REMOTE+'/checkpoint_guard.py',folder,job,deadline]
            save(name+'-guard-intent.json',dict(command=args,source_hashes=manifest,deadline_epoch=deadline))
            response = run('nohup '+shlex.join(args)+' > '+shlex.quote(REMOTE+'/'+name+'-guard.log')+' 2>&1 < /dev/null & echo $!')
            save(name+'-guard-launch.json',dict(launcher_pid=response.strip(),parent_job=job))
            print(xc,'guard launcher',response.strip(),flush=True)
        if (RECORD/(name+'-intent.json')).exists():
            print(xc,'existing restart intent: do not resubmit',flush=True)
            continue
        args = ['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal',
                '--nodes=1','--ntasks=32','--ntasks-per-node=32','--cpus-per-task=1','--mem=300G',
                '--time=20:00:00','--dependency=afterany:'+job,'--kill-on-invalid-dep=yes',
                '--job-name=tir-'+xc+'-f001','--chdir='+REMOTE,'--output='+REMOTE+'/'+name+'-%j.out',
                REMOTE+'/restart-step.sbatch',xc,'force-001',job]
        intent = dict(utc=datetime.now(timezone.utc).isoformat(),command=args,parent_job=job,source_hashes=manifest)
        save(name+'-intent.json',intent)
        response = run('set -o pipefail; '+shlex.join(args)+' | tee '+shlex.quote(REMOTE+'/'+name+'.jobid')).strip()
        newjob = response.split(';')[0]
        if not newjob.isdigit():
            raise RuntimeError('Unknown submission outcome: '+response)
        save(name+'-submission.json',dict(intent,job_id=newjob,state='submitted'))
        print(xc,'dependent restart',newjob,flush=True)


if __name__ == '__main__':
    main()
