"""Stage an immutable continuation bundle and submit exactly one A100 job."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'r2scan_enthalpy_extend30k'
sys.path.insert(0, str(ROOT/'testset/dft'))
import cluster_test50 as c
cfg = json.loads((OUT/'protocol.json').read_text())
REMOTE = cfg['remote']

def run(command):
    p = subprocess.run(c.cluster.ssh.ssh_args()+['-o', 'BatchMode=yes', 'mlip-umt', command],
                       capture_output=True, encoding='utf-8', errors='replace', timeout=120)
    if p.returncode:
        raise RuntimeError(p.stdout+p.stderr)
    return p.stdout

def save(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')

action = sys.argv[1]
if action == 'stage':
    assert not (OUT/'staging-intent.json').exists()
    names = ['protocol.json', 'baseline.json', 'worker.py', 'summarize.py', 'job.sbatch',
             'README_RU.txt', 'preparation-verification.json']+[s['initial_file'] for s in cfg['states']]
    files = {n: (OUT/n).read_bytes() for n in names}
    hashes = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
    save('staging-intent.json', dict(remote=REMOTE, files=hashes, unix=time.time()))
    run('umask 077; mkdir '+shlex.quote(REMOTE))
    for name, content in files.items():
        c.upload(content, REMOTE+'/'+name)
    code = ('import pathlib,hashlib,ast; p=pathlib.Path('+repr(REMOTE)+'); h='+repr(hashes)+'; '
            'assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); '
            'm='+repr(cfg['model'])+'; '
            'assert hashlib.sha256(pathlib.Path(m["path"]).read_bytes()).hexdigest()==m["sha256"]; '
            '[ast.parse((p/n).read_text()) for n in ["worker.py","summarize.py"]]; '
            'print("Staging, model checksum and Python syntax verified")')
    print(run('python3 -c '+shlex.quote(code)))
    run('bash -n '+shlex.quote(REMOTE+'/job.sbatch'))
    save('staging-complete.json', dict(files=hashes, model=cfg['model'], unix=time.time()))
elif action == 'submit':
    assert (OUT/'staging-complete.json').exists() and not (OUT/'submission-intent.json').exists()
    node = run('scontrol show node tesla-a101 -o')
    (OUT/'availability-before-submit.txt').write_text(node)
    assert 'Gres=gpu:a100:2' in node
    args = ['sbatch', '--parsable', '--partition=v100', '--nodelist=tesla-a101',
            '--account=imet-2026-2', '--qos=normal', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=4', '--gres=gpu:a100:1', '--mem=32G', '--time=12:00:00',
            '--no-requeue', '--job-name=TiAl-r2SCAN-H-plus30k', '--chdir='+REMOTE,
            '--output='+REMOTE+'/allocation-%j.log', REMOTE+'/job.sbatch']
    save('submission-intent.json', dict(command=args, unix=time.time()))
    job = run(shlex.join(args)).strip().split(';')[0]
    assert job.isdigit(), job
    save('submission.json', dict(job_id=job, command=args, unix=time.time()))
    print('Submitted', job)
elif action == 'status':
    job = json.loads((OUT/'submission.json').read_text())['job_id']
    scheduler = run('sacct -j '+job+
                    ' --format=JobID,State,ExitCode,Elapsed,Start,End,AllocTRES,NodeList -Pn')
    queue = run('squeue -h -u u3129 -o "%i|%T|%R|%S"')
    live = [line for line in queue.splitlines() if line.split('|')[0] == job]
    scheduler += '\nLive queue: '+ ('\n'.join(live) if live else 'job absent (see accounting above)')+'\n'
    (OUT/'scheduler-latest.txt').write_text(scheduler)
    print(scheduler)
    code = '''import pathlib,json
p=pathlib.Path(REMOTE)
d={"metadata":{},"logs":{}}
for name in ["preflight.json","timing.json","progress.json","summary.json","enthalpy_summary.json"]:
 f=p/name
 if f.exists():d["metadata"][name]=json.loads(f.read_text())
for f in p.glob("*.log"):
 with f.open("rb") as fp:
  fp.seek(max(0,f.stat().st_size-2500));d["logs"][f.name]=fp.read().decode(errors="replace")
print(json.dumps(d))
'''.replace('REMOTE', repr(REMOTE))
    data = json.loads(run('python3 -c '+shlex.quote(code)))
    save('status.json', data)
    compact = {**data, 'metadata': {k: v for k, v in data['metadata'].items()
                                  if k != 'enthalpy_summary.json'}}
    if 'enthalpy_summary.json' in data['metadata']:
        summary = data['metadata']['enthalpy_summary.json']
        compact['enthalpy_summary'] = dict(status=summary['status'],
             compositions=len(summary['compositions']),
             equiatomic=summary['compositions']['Al0.5Ti0.5'])
    print(json.dumps(compact, indent=2))
else:
    raise ValueError(action)
