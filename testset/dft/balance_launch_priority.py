"""Prefer earlier five-structure groups through pending-job Nice values."""
from pathlib import Path
from collections import Counter
import argparse
import json
import time
from balance_launch_waves import ARRAYS, ROOT, queue, inspect, job_records, run, save


def plan():
    states = queue()
    before = inspect()
    assert all(r['Dependency'] == '(null)' and r['ArrayTaskThrottle'] == '5'
               for rs in before.values() for r in rs)
    pending = sorted(j for j, state in states.items() if state == 'PENDING')
    base_wave = min(int(j.split('_')[1]) // 5 for j in pending)
    actions = [{'job': j, 'nice': int(j.split('_')[1]) // 5 - base_wave} for j in pending]
    data = {'unix': time.time(), 'arrays': ARRAYS, 'before': before,
            'base_wave': base_wave, 'actions': actions,
            'policy': 'Pending tasks in each five-structure group share the same Nice. Earlier groups have higher priority. No hard dependencies or background controller; array throttles remain 5.'}
    path = ROOT / ('launch-priority-' + str(int(data['unix'])) + '.json')
    assert not path.exists()
    save(path, data)
    print(json.dumps({'plan': str(path), 'pending': len(actions),
                      'nice_counts': dict(Counter(a['nice'] for a in actions))}))


def apply(path):
    data = json.loads(path.read_text())
    assert data['arrays'] == ARRAYS
    result_path = path.with_name(path.stem + '-result.json')
    assert not result_path.exists(), 'Reconcile existing result first'
    result = {'unix': time.time(), 'updates': [], 'plan': str(path)}
    save(result_path, result)
    try:
        for action in data['actions']:
            rows = job_records(action['job'])
            assert len(rows) == 1
            row = rows[0]
            if row['JobState'] != 'PENDING':
                result['updates'].append(dict(action, skipped_state=row['JobState']))
                continue
            assert row['Dependency'] == '(null)'
            update = dict(action, old_nice=row['Nice'], state='intent')
            result['updates'].append(update)
            save(result_path, result)
            run('scontrol', 'update', 'JobId=' + action['job'], 'Nice=' + str(action['nice']))
            update['state'] = 'applied'
            save(result_path, result)
        result['after'] = inspect()
        result['queue_after'] = queue()
        for xc, rows in result['after'].items():
            for row in rows:
                assert row['ArrayTaskThrottle'] == '5'
                assert row['Dependency'] == '(null)'
                if row['JobState'] == 'PENDING':
                    assert int(row['Nice']) == int(row['ArrayTaskId']) // 5 - data['base_wave'], row
            new_by_id = {r['JobId']: r for r in rows}
            for old in data['before'][xc]:
                if old['JobState'] != 'RUNNING':
                    continue
                new = new_by_id.get(old['JobId'])
                if new:
                    for key in ['StartTime', 'NodeList', 'NumCPUs', 'NumTasks', 'TimeLimit', 'Restarts', 'Nice']:
                        assert old[key] == new[key], (old['JobId'], key)
        result['state'] = 'applied_and_verified'
    except Exception as exc:
        result['state'] = 'partial_changes_require_reconciliation'
        result['error'] = str(exc)
        raise
    finally:
        save(result_path, result)
    print(json.dumps({'result': str(result_path), 'state': result['state'], 'updates': len(result['updates'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'apply'])
    parser.add_argument('path', nargs='?')
    args = parser.parse_args()
    if args.action == 'plan':
        plan()
    else:
        apply(Path(args.path))
