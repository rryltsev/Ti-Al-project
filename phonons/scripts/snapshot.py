"""Collect small, non-licensed inputs and status with verified transfer hashes."""
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
from datetime import datetime, timezone
from cluster import CFG, RECORD, REMOTE, run

code = '''
import base64, hashlib, json
from pathlib import Path
root = Path(REMOTE_ROOT)
allowed = {'INCAR', 'KPOINTS', 'POSCAR', 'POSCAR-relaxed', 'SPOSCAR',
           'phonopy_disp.yaml', 'potcar-provenance.json', 'input-hashes.json',
           'status.json', 'validation.json', 'geometry.json', 'electronic-diagnostics.json',
           'software.json', 'environment.lock.txt', 'setup.success', 'protocol.json',
           'workflow.py', 'production.sbatch', 'production-v2.sbatch', 'setup.sbatch',
           'input-manifest.json', 'execution-policy-02.json', 'production_step.py',
           'production-step.sbatch', 'provenance.json'}
result = {}
for p in root.rglob('*'):
    if p.name not in allowed or not p.is_file() or 'pip-cache' in p.parts or 'tmp' in p.parts:
        continue
    if p.stat().st_size > 2000000: raise RuntimeError('Unexpected metadata size')
    data = p.read_bytes()
    result[str(p.relative_to(root))] = {'sha256': hashlib.sha256(data).hexdigest(),
                                      'base64': base64.b64encode(data).decode()}
print(json.dumps(result))
'''.replace('REMOTE_ROOT', repr(REMOTE))
payload = json.loads(run(CFG['python_base'] + ' -c ' + shlex.quote(code)))
target = RECORD / 'snapshot'
hashes = {}
for name, record in payload.items():
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts: raise ValueError(name)
    data = base64.b64decode(record['base64'])
    if hashlib.sha256(data).hexdigest() != record['sha256']: raise ValueError(name)
    dest = target.joinpath(*rel.parts)
    dest.parent.mkdir(exist_ok=True, parents=True)
    dest.write_bytes(data)
    hashes[name] = record['sha256']
jobs = [json.loads(p.read_text())['job_id'] for p in RECORD.glob('*-submission.json')]
status = run('sacct -nP -j ' + ','.join(jobs) +
             ' --format=JobIDRaw,JobName,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW,NodeList; '
             'squeue -j ' + ','.join(jobs) + ' -o "%.18i %.24j %.10T %.10M %R"')
(RECORD / 'scheduler-latest.txt').write_text(status)
(RECORD / 'snapshot-manifest.json').write_text(json.dumps({
    'utc': datetime.now(timezone.utc).isoformat(), 'remote': REMOTE, 'files': hashes,
    'scope': 'Small inputs and live metadata only; not a raw-output archive'}, indent=2) + '\n')
print(f'Collected and hash-verified {len(hashes)} metadata/input files')
print(status)
