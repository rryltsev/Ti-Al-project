"""Read-only queue evidence and one explicitly authorized same-limit refresh."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time

import cluster_test50 as c

OUT = Path(__file__).resolve().parent / 'queue-repair-20261024'
ARRAYS = {'PBE': '20957366', 'PBEsol': '20957369', 'r2SCAN': '20957372'}

INSPECT = r'''
import hashlib,json,pathlib,re,subprocess,time
root=pathlib.Path('/home6/u3129/tial-testsets/test50-dft-20261021-01')
arrays={'PBE':'20957366','PBEsol':'20957369','r2SCAN':'20957372'}
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def records(s):
 return [dict(re.findall(r'(\w+)=(\S*)',line)) for line in s.splitlines() if line.startswith('JobId=')]
raw={xc:run(['scontrol','show','job',job,'-o']) for xc,job in arrays.items()}
result={'unix':time.time(),'arrays':{xc:records(v['stdout']) for xc,v in raw.items()},'array_commands':raw}
commands={
 'queue':['squeue','-h','-r','-j',','.join(arrays.values()),'-o','%F|%K|%T|%R'],
 'accounting':['sacct','-X','-j',','.join(arrays.values()),'--format=JobID,State,ExitCode,Elapsed,Start,End','-P'],
 'start_estimates':['squeue','--start','-r','-j',','.join(arrays.values()),'-o','%i|%T|%S|%R'],
 'priorities':['sprio','-l','-j',','.join(arrays.values())],
 'nodes':['scontrol','show','nodes','-o'],
 'partition':['scontrol','show','partition','apollo','-o'],
 'cluster_queue':['squeue','-h','-r','-p','apollo','-o','%T|%Q|%a|%u|%l|%C|%m|%R'],
 'sdiag':['sdiag'],
 'config':['scontrol','show','config'],
 'reservations':['scontrol','show','reservations','-o'],
 'association':['sacctmgr','-nP','show','assoc','where','user=u3129','account=imet-2026-2','format=Cluster,Account,User,Partition,QOS,GrpTRES,MaxJobs,MaxSubmitJobs'],
}
result.update({k:run(v) for k,v in commands.items()})
result['protocol_sha256']=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
result['input_manifests']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'calculations').glob('*/*/input-manifest.json')}
print(json.dumps(result))
'''


def save(name, value):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def inspect():
    return json.loads(c.cluster.run('python3 -c ' + shlex.quote(INSPECT)))


def summary(data):
    from collections import Counter
    result = {'unix': data['unix'], 'arrays': {}}
    for xc, rows in data['arrays'].items():
        pending = [r for r in rows if r['JobState'] == 'PENDING']
        active = [r for r in rows if r['JobState'] in ('RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED')]
        result['arrays'][xc] = {
            'pending': len(pending), 'active': len(active),
            'reasons': dict(Counter(r['Reason'] for r in pending)),
            'throttles': sorted(set(r.get('ArrayTaskThrottle', '') for r in rows)),
            'dependencies': sorted(set(r.get('Dependency', '') for r in pending)),
            'first_pending': [{k: r.get(k) for k in ('JobId','ArrayTaskId','Priority','Nice','Reason','LastSchedEval','StartTime')}
                              for r in sorted(pending, key=lambda r: (-int(r['Priority']),int(r['ArrayTaskId'])))[:6]],
        }
    return result


def verify_preservation(before, after):
    assert after['protocol_sha256'] == before['protocol_sha256']
    assert len(after['input_manifests']) == 150
    assert after['input_manifests'] == before['input_manifests']
    checked = 0
    for xc, rows in before['arrays'].items():
        new = {r['JobId']: r for r in after['arrays'][xc]}
        assert all(r.get('ArrayTaskThrottle') == '5' for r in new.values())
        for r in rows:
            if r['JobId'] not in new:
                assert r['JobState'] in ('COMPLETED','FAILED','CANCELLED'), r
                continue
            n = new[r['JobId']]
            for key in ('Nice','Dependency','Restarts','NumCPUs','NumTasks','TimeLimit','ReqTRES','Command','WorkDir'):
                assert n.get(key) == r.get(key), (xc,r['JobId'],key)
            if r['JobState'] in ('RUNNING','COMPLETING'):
                for key in ('StartTime','NodeList'):
                    assert n.get(key) == r.get(key), (xc,r['JobId'],key)
            checked += 1
    return {'preserved_job_records':checked,'input_manifests_verified':150,'all_throttles':5,
            'nice_dependencies_restarts_resources_inputs_preserved':True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['inspect', 'refresh', 'verify'])
    a = ap.parse_args()
    if a.action == 'inspect':
        data = inspect()
        save('before.json', data)
    elif a.action == 'refresh':
        assert not (OUT/'refresh-intent.json').exists(), 'Reconcile existing intent before any repetition'
        data = inspect()
        # Abort before mutation if the fixed crystal-first policy or limit changed.
        for xc, rows in data['arrays'].items():
            assert data['array_commands'][xc]['returncode'] == 0
            assert rows and all(r.get('ArrayTaskThrottle') == '5' for r in rows)
            for r in rows:
                if r['JobState'] != 'PENDING':
                    continue
                i = int(r['ArrayTaskId'])
                desired = 20+i//5 if i<25 else i-25 if i<33 else i-33 if i<42 else i-42
                assert int(r['Nice']) == desired, (xc, i, r['Nice'], desired)
                assert r['Dependency'] == '(null)' and int(r['Priority']) > 0
        targets = [ARRAYS[xc] for xc, rows in data['arrays'].items()
                   if any(r['JobState']=='PENDING' and r['Reason']=='JobArrayTaskLimit' for r in rows)]
        commands = [['scontrol','update','JobId='+j,'ArrayTaskThrottle=5'] for j in targets]
        save('refresh-intent.json', {'unix':time.time(),'before':data,'commands':commands,
             'scope':'One-time explicit user authorization: reapply existing throttle 5 only to arrays with stale JobArrayTaskLimit. Preserve all Nice values, running tasks and DFT inputs.'})
        updates=[]
        for command in commands:
            value=c.cluster.run(shlex.join(command))
            updates.append({'command':command,'output':value,'unix':time.time()})
            save('refresh-result.json',{'updates':updates,'state':'refreshing'})
        data=inspect()
        checks=verify_preservation(json.loads((OUT/'refresh-intent.json').read_text(encoding='utf-8'))['before'],data)
        save('refresh-result.json',{'updates':updates,'state':'applied_and_verified','after':data,'checks':checks})
    else:
        data=inspect()
        if (OUT/'refresh-intent.json').exists():
            data['preservation_checks']=verify_preservation(json.loads((OUT/'refresh-intent.json').read_text(encoding='utf-8'))['before'],data)
        save('verification-'+str(int(data['unix']))+'.json',data)
    print(json.dumps(summary(data),indent=2))


if __name__ == '__main__':
    main()
