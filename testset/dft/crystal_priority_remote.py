"""One-time user-authorized Nice changes for pending Test50 tasks only."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import subprocess
import time

ROOT = Path('/home6/u3129/tial-testsets/test50-dft-20261021-01')
ARRAYS = {'PBE': '20957366', 'PBEsol': '20957369', 'r2SCAN': '20957372'}
TRACE = Path.cwd()
assert TRACE.name == 'crystal-priority-20261023-01'

def save(name, value):
    path = TRACE / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def run(*args):
    return subprocess.check_output(args, universal_newlines=True)

def records(job):
    return [dict(re.findall(r'(\w+)=(\S*)', line)) for line in run('scontrol', 'show', 'job', job, '-o').splitlines() if line.startswith('JobId=')]

def inspect():
    return {xc: records(job) for xc, job in ARRAYS.items()}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

assert not (TRACE / 'change-intent.json').exists(), 'Existing intent: reconcile, never repeat blindly'
cfg = json.loads((ROOT / 'protocol.json').read_text())
manifest_path = Path(cfg['dataset_root']) / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
expected = json.loads((TRACE / 'authorization.json').read_text())
assert sha(manifest_path) == expected['manifest_sha256']
assert len(manifest) == 50
families = ['POSCAR_alpha-Ti-hcp_orthogonal', 'POSCAR_beta-Ti-bcc', 'POSCAR_gamma_AlTi_L1_0']
groups = {family: sorted([r['structure_id'] for r in manifest if r['family'] == family]) for family in families}
nice = {}
order = []
for layer in range(max(map(len, groups.values()))):
    for family in families:
        if layer < len(groups[family]):
            sid = groups[family][layer]
            nice[sid] = layer
            order.append(sid)
for row in manifest:
    if row['family'] == 'liquid':
        nice[row['structure_id']] = 20 + (int(row['structure_id'].rsplit('-', 1)[1]) - 1) // 5
assert len(nice) == 50
before = inspect()
assert all(r.get('ArrayTaskThrottle') == '5' and r.get('Dependency') == '(null)' for rows in before.values() for r in rows)
protocol_hash = sha(ROOT / 'protocol.json')
inputs = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'calculations').glob('*/*/input-manifest.json')}
actions = []
for xc, rows in before.items():
    for row in rows:
        if row['JobState'] != 'PENDING':
            continue
        index = int(row['ArrayTaskId'])
        sid = manifest[index]['structure_id']
        assert sid == 'TiAl-test-' + str(index + 1).zfill(3)
        actions.append({'functional': xc, 'job': ARRAYS[xc] + '_' + str(index), 'structure_id': sid,
                        'old_nice': int(row['Nice']), 'nice': nice[sid]})
# Lower liquid priority first, then prefer common crystal IDs in family rounds.
actions.sort(key=lambda a: (-a['nice'], a['structure_id'], a['functional']))
intent = {'unix': time.time(), 'authorization': expected, 'arrays': ARRAYS, 'before': before,
          'policy': 'All pending solids precede pending liquids. Nice 0 for 026/034/043, 1 for 027/035/044, etc.; liquids 20..24. Equal Nice across functionals for the same ID. Existing running tasks unchanged. No holds, dependencies, throttle changes, cancellations, requeues, submissions or DFT input changes.',
          'crystal_order': order, 'actions': actions, 'protocol_sha256': protocol_hash, 'input_manifests': inputs}
save('change-intent.json', intent)
result = {'started_unix': time.time(), 'state': 'applying', 'updates': []}
save('change-result.json', result)
try:
    for action in actions:
        rows = records(action['job'])
        assert len(rows) == 1
        row = rows[0]
        update = dict(action)
        result['updates'].append(update)
        if row['JobState'] != 'PENDING':
            update.update(state='skipped', observed_state=row['JobState'])
            save('change-result.json', result)
            continue
        assert row['ArrayTaskThrottle'] == '5' and row['Dependency'] == '(null)'
        update['state'] = 'intent'
        save('change-result.json', result)
        if int(row['Nice']) != action['nice']:
            run('scontrol', 'update', 'JobId=' + action['job'], 'Nice=' + str(action['nice']))
        update['state'] = 'applied'
        save('change-result.json', result)
    after = inspect()
    result['after'] = after
    for xc, rows in after.items():
        for row in rows:
            assert row['ArrayTaskThrottle'] == '5' and row['Dependency'] == '(null)'
            if row['JobState'] == 'PENDING':
                sid = manifest[int(row['ArrayTaskId'])]['structure_id']
                assert int(row['Nice']) == nice[sid], row
        new = {r['JobId']: r for r in rows}
        for old in before[xc]:
            if old['JobState'] not in ['RUNNING', 'COMPLETING']:
                continue
            if old['JobId'] not in new:
                continue  # Normal scheduler cleanup; accounting is recorded below.
            for key in ['StartTime', 'NodeList', 'Restarts', 'NumCPUs', 'NumTasks', 'TimeLimit', 'Nice']:
                assert old[key] == new[old['JobId']][key], (old['JobId'], key)
    assert sha(ROOT / 'protocol.json') == protocol_hash
    assert all(sha(ROOT / name) == digest for name, digest in inputs.items())
    result['queue_after'] = run('squeue', '-h', '-r', '-j', ','.join(ARRAYS.values()), '-o', '%F|%K|%T|%R')
    result['accounting_after'] = run('sacct', '-X', '-j', ','.join(ARRAYS.values()), '--format=JobID,State,ExitCode,Elapsed', '-P')
    result.update(state='applied_and_verified', finished_unix=time.time(),
                  pending_nice_counts={xc: dict(Counter(r['Nice'] for r in rows if r['JobState'] == 'PENDING')) for xc, rows in after.items()})
except Exception as exc:
    result.update(state='partial_changes_require_reconciliation', error=repr(exc))
    raise
finally:
    save('change-result.json', result)
print(json.dumps({'state': result['state'], 'updates': len(result['updates']), 'remote_trace': str(TRACE)}))
