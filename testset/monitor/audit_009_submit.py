"""One traceable, small CPU job checks Test009 against immutable PBE models."""
from pathlib import Path
import hashlib
import json
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent
DEST = HERE / 'audits' / 'TiAl-test-009-20261023'
REMOTE = '/home6/u3129/tial-testsets/audit-test009-20261023-01'
sys.path.insert(0, str(HERE.parent / 'dft'))
import cluster_test50 as cluster

def save(name, value):
    (DEST / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

assert not (DEST / 'submission-intent.json').exists(), 'Reconcile existing intent; never submit twice'
job = (HERE / 'runs/model-verify-infer-20261022-01/model_job.sbatch').read_text(encoding='utf-8')
job = job[:job.index('python verify_exports.py')] + 'python model_audit.py\n'
files = {'model_audit.py': (DEST / 'model_audit.py').read_bytes(),
         'model_job.sbatch': job.encode(), 'POSCAR': (DEST / 'PBE/POSCAR').read_bytes(),
         'model_registry.json': (HERE / 'model_registry.json').read_bytes(),
         'cached.json': (HERE / 'predictions/PBE/TiAl-test-009.json').read_bytes()}
save('staging-intent.json', {'remote_root': REMOTE, 'unix': time.time(),
     'files': {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}})
cluster.cluster.run('umask 077; mkdir ' + shlex.quote(REMOTE))
for name, data in files.items():
    cluster.upload(data, REMOTE + '/' + name)
code = 'import pathlib,hashlib,json; p=pathlib.Path(' + repr(REMOTE) + '); h=' + repr({k: hashlib.sha256(v).hexdigest() for k, v in files.items()}) + '; assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in h.items()); print(json.dumps({"verified":len(h)}))'
save('staging-result.json', json.loads(cluster.cluster.run('python3 -c ' + shlex.quote(code))))
args = ['sbatch', '--parsable', '--partition=apollo', '--account=imet-2026-2', '--qos=normal',
        '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=16G', '--time=00:20:00',
        '--no-requeue', '--job-name=t50-audit009', '--chdir=' + REMOTE,
        '--output=' + REMOTE + '/audit-%j.out', REMOTE + '/model_job.sbatch']
save('submission-intent.json', {'unix': time.time(), 'command': args,
     'purpose': 'Re-evaluate original PBE checkpoint and frozen export on unchanged Test009; no training, freezing or cache replacement.'})
job_id = cluster.cluster.run(shlex.join(args)).strip()
assert job_id.split(';')[0].isdigit(), job_id
save('submission.json', {'job_id': job_id.split(';')[0], 'unix': time.time(), 'remote_root': REMOTE})
print(job_id)
