"""Stage and execute the one-time crystal-priority operation with durable traces."""
from pathlib import Path
import hashlib
import json
import shlex
import time
import cluster_test50 as cluster

HERE = Path(__file__).resolve().parent
LOCAL = HERE / 'crystal-priority-20261023'
REMOTE = '/home6/u3129/tial-testsets/crystal-priority-20261023-01'
assert not (LOCAL / 'execution-intent.json').exists(), 'Reconcile previous execution before any retry'
manifest = HERE.parent / 'TiAl-test50-v1/manifest.json'
authorization = {'unix': time.time(), 'request': 'User requested crystal configurations to run ahead of liquids on 2026-10-23.',
                 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 'selection_basis': 'Family and frozen structure ID only, independent of observed model errors.',
                 'scope': 'Pending jobs of arrays 20957366/20957369/20957372 only; retain all 50 geometries and throttle 5 each.'}
files = {'crystal_priority_remote.py': (HERE / 'crystal_priority_remote.py').read_bytes(),
         'authorization.json': (json.dumps(authorization, indent=2) + '\n').encode()}
intent = {'unix': time.time(), 'remote_root': REMOTE, 'authorization': authorization,
          'files': {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
(LOCAL / 'execution-intent.json').write_text(json.dumps(intent, indent=2) + '\n', encoding='utf-8')
cluster.cluster.run('umask 077; mkdir ' + shlex.quote(REMOTE))
for name, data in files.items():
    cluster.upload(data, REMOTE + '/' + name)
verify = 'import pathlib,hashlib; p=pathlib.Path(' + repr(REMOTE) + '); h=' + repr(intent['files']) + '; assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in h.items())'
cluster.cluster.run('python3 -c ' + shlex.quote(verify))
error = None
try:
    print(cluster.cluster.run('cd ' + shlex.quote(REMOTE) + ' && python3 crystal_priority_remote.py'))
except Exception as exc:
    error = str(exc)
finally:
    code = 'import pathlib,json,hashlib; p=pathlib.Path(' + repr(REMOTE) + '); names=["change-intent.json","change-result.json"]; print(json.dumps({n:{"text":(p/n).read_text(),"sha256":hashlib.sha256((p/n).read_bytes()).hexdigest()} for n in names if (p/n).exists()}))'
    records = json.loads(cluster.cluster.run('python3 -c ' + shlex.quote(code)))
    for name, record in records.items():
        data = record['text'].encode()
        assert hashlib.sha256(data).hexdigest() == record['sha256']
        (LOCAL / name).write_bytes(data)
    (LOCAL / 'collection.json').write_text(json.dumps({'unix': time.time(), 'remote_root': REMOTE,
        'files': {k: v['sha256'] for k, v in records.items()}, 'execution_error': error}, indent=2) + '\n', encoding='utf-8')
if error:
    raise RuntimeError(error)
