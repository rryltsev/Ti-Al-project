"""One-time, reversible Slurm dependency updates; run on the login node."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import time

ROOT = Path('/home6/u3129/tial-testsets/test50-dft-20261021-01')
ARRAYS = {'PBE': '20957366', 'PBEsol': '20957369', 'r2SCAN': '20957372'}
WIDTH = 5


def run(*args):
    return subprocess.check_output(args, text=True)


def queue():
    raw = run('squeue', '-h', '-r', '-j', ','.join(ARRAYS.values()),
              '-o', '%F|%K|%T')
    return {f'{array}_{index}': state for array, index, state in
            (line.strip().split('|') for line in raw.splitlines())}


def job_records(job):
    raw = run('scontrol', 'show', 'job', job, '-o')
    return [dict(re.findall(r'(\w+)=(\S*)', line)) for line in raw.splitlines()
            if line.startswith('JobId=')]


def inspect():
    return {xc: job_records(job) for xc, job in ARRAYS.items()}


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def make_plan():
    states = queue()
    before = inspect()
    for records in before.values():
        assert all(r['ArrayTaskThrottle'] == '5' for r in records)
        assert all(r['Dependency'] == '(null)' for r in records), 'Existing dependencies: inspect before changing'
    pending = {job for job, state in states.items() if state == 'PENDING'}
    actions = []
    # Increasing wave order ensures dependencies refer to earlier waves only.
    for index in range(50):
        first = index // WIDTH * WIDTH
        predecessors = [f'{a}_{i}' for a in ARRAYS.values()
                        for i in range(max(0, first - WIDTH), first)]
        for xc, array in ARRAYS.items():
            job = f'{array}_{index}'
            if job not in pending:
                continue
            needs = [j for j in predecessors if j in pending]
            assert all(int(j.split('_')[1]) // WIDTH < index // WIDTH for j in needs)
            if needs:
                actions.append({'job': job, 'functional': xc, 'depends_on': needs,
                                'dependency': 'after:' + ':'.join(needs)})
    return {'unix': time.time(), 'width': WIDTH, 'arrays': ARRAYS,
            'policy': 'Each group of five may start after the previous group has started for all three functionals; after dependencies do not wait for completion.',
            'before': before, 'queue_before': states, 'actions': actions}


def apply(plan_path):
    plan = json.loads(plan_path.read_text())
    assert plan['arrays'] == ARRAYS and plan['width'] == WIDTH
    result_path = plan_path.with_name(plan_path.stem + '-result.json')
    assert not result_path.exists(), 'Existing result: reconcile before another attempt'
    result = {'unix': time.time(), 'plan': str(plan_path), 'updates': []}
    save(result_path, result)
    try:
        for action in plan['actions']:
            record = job_records(action['job'])
            assert len(record) == 1
            record = record[0]
            if record['JobState'] != 'PENDING':
                result['updates'].append({'job': action['job'], 'skipped_state': record['JobState']})
                continue
            assert record['Dependency'] == '(null)', (action['job'], record['Dependency'])
            update = {'job': action['job'], 'dependency': action['dependency'], 'state': 'intent'}
            result['updates'].append(update)
            save(result_path, result)
            run('scontrol', 'update', 'JobId=' + action['job'],
                'Dependency=' + action['dependency'])
            update['state'] = 'applied'
            save(result_path, result)
        result['after'] = inspect()
        result['queue_after'] = queue()
        for xc, old_records in plan['before'].items():
            new_by_id = {r['JobId']: r for r in result['after'][xc]}
            for old in old_records:
                if old['JobState'] != 'RUNNING':
                    continue
                new = new_by_id.get(old['JobId'])
                if new:
                    for key in ['StartTime', 'NodeList', 'NumCPUs', 'NumTasks', 'TimeLimit', 'Restarts']:
                        assert old[key] == new[key], (old['JobId'], key)
        assert all(r['ArrayTaskThrottle'] == '5' for rs in result['after'].values() for r in rs)
        result['state'] = 'applied_and_verified'
    except Exception as exc:
        result['state'] = 'error_partial_changes_require_reconciliation'
        result['error'] = str(exc)
        raise
    finally:
        save(result_path, result)
    print(json.dumps({'result': str(result_path), 'updates': len(result['updates']), 'state': result['state']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'apply'])
    parser.add_argument('plan_path', nargs='?')
    args = parser.parse_args()
    if args.action == 'plan':
        plan = make_plan()
        path = ROOT / ('launch-waves-' + str(int(plan['unix'])) + '.json')
        assert not path.exists()
        save(path, plan)
        print(json.dumps({'plan': str(path), 'updates': len(plan['actions']),
                          'first_actions': plan['actions'][:3], 'acyclic': True}))
    else:
        apply(Path(args.plan_path))
