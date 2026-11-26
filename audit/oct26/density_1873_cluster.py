"""Stage and submit exactly one private A100 density calculation."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'pbe15k_density_1873'
sys.path.insert(0, str(ROOT/'testset/dft'))
import cluster_test50 as c
cfg = json.loads((OUT/'protocol.json').read_text())
remote = cfg['remote']

def run(command):
    result = subprocess.run(c.cluster.ssh.ssh_args()+['-o', 'BatchMode=yes', 'mlip-umt', command],
                            capture_output=True, encoding='utf-8', errors='replace', timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return result.stdout

def save(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')

action = sys.argv[1]
if action == 'stage':
    assert not (OUT/'staging-intent.json').exists()
    names = ['protocol.json', 'tasks.json', 'baseline_5k.json', 'md_common.py',
             'density_worker.py', 'summarize_density.py', 'job.sbatch', 'README_RU.txt',
             'preparation-verification.json']+[s['initial_file'] for s in cfg['states']]
    files = {name: (OUT/name).read_bytes() for name in names}
    hashes = {name: hashlib.sha256(b).hexdigest() for name, b in files.items()}
    save('staging-intent.json', dict(remote=remote, files=hashes, unix=time.time()))
    run('umask 077; mkdir '+shlex.quote(remote))
    for name, content in files.items():
        c.upload(content, remote+'/'+name)
    code = ('import pathlib,hashlib; p=pathlib.Path('+repr(remote)+'); h='+repr(hashes)+'; '
            'assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); '
            'm='+repr(cfg['models']['15000'])+'; '
            'assert hashlib.sha256(pathlib.Path(m["path"]).read_bytes()).hexdigest()==m["sha256"]; '
            'print("Staging and checkpoint verified")')
    print(run('python3 -c '+shlex.quote(code)))
    print(run('bash -n '+shlex.quote(remote+'/job.sbatch')))
    save('staging-complete.json', dict(files=hashes, checkpoint=cfg['models']['15000'], unix=time.time()))
elif action == 'submit':
    assert (OUT/'staging-complete.json').exists() and not (OUT/'submission-intent.json').exists()
    availability = run('scontrol show node tesla-a101 -o')
    (OUT/'availability-before-submit.txt').write_text(availability)
    assert 'Gres=gpu:a100:2' in availability
    args = ['sbatch', '--parsable', '--partition=v100', '--nodelist=tesla-a101',
            '--account=imet-2026-2', '--qos=normal', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=4', '--gres=gpu:a100:1', '--mem=32G', '--time=16:00:00',
            '--no-requeue', '--job-name=TiAl-rho1873-PBE15k', '--chdir='+remote,
            '--output='+remote+'/allocation-%j.log', remote+'/job.sbatch']
    save('submission-intent.json', dict(command=args, unix=time.time()))
    jid = run(shlex.join(args)).strip().split(';')[0]
    assert jid.isdigit(), jid
    save('submission.json', dict(job_id=jid, command=args, unix=time.time()))
    print('Submitted', jid)
elif action == 'status':
    jid = json.loads((OUT/'submission.json').read_text())['job_id']
    scheduler = run('scontrol show job '+jid+' -o; sacct -j '+jid+
                    ' --format=JobID,State,ExitCode,Elapsed,AllocTRES,NodeList -Pn')
    (OUT/'scheduler-latest.txt').write_text(scheduler)
    print(scheduler)
    code = '''import pathlib,json
p=pathlib.Path(REMOTE)
d={"metadata":{},"logs":{}}
for n in ["15000/preflight.json","15000/progress.json","15000/summary.json","density_summary.json"]:
 f=p/n
 if f.exists():d["metadata"][n]=json.loads(f.read_text())
for f in p.glob("*.log"):
 with f.open("rb") as fp:
  fp.seek(max(0,f.stat().st_size-3000));d["logs"][f.name]=fp.read().decode(errors="replace")
print(json.dumps(d))
'''.replace('REMOTE', repr(remote))
    data = json.loads(run('python3 -c '+shlex.quote(code)))
    save('status.json', data)
    print(json.dumps(data, indent=2))
else:
    raise ValueError(action)
