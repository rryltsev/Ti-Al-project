"""Request a durable soft stop of our CPU PBE job on the scarce A100 host."""
import json
import subprocess
from datetime import datetime, timezone
from cluster import RECORD, REMOTE, run, save, ssh

JOB = '20959336'
FOLDER = REMOTE + '/PBE/production-02/force-001-restart-01'
NAME = 'PBE-release-a100-20261024-attempt02'

if __name__ == '__main__':
    if (RECORD / (NAME + '-intent.json')).exists():
        raise RuntimeError('Existing intent: reconcile remote STOPCAR and scheduler before retry')
    info = run('scontrol show job -o ' + JOB)
    if not all(x in info for x in ('JobState=RUNNING', 'Partition=v100', 'NodeList=tesla-a101')):
        raise RuntimeError('Allocation changed; inspect before action: ' + info)
    run('test ! -e ' + FOLDER + '/STOPCAR && test ! -e ' + FOLDER + '/checkpoint-request.json')
    intent = dict(utc=datetime.now(timezone.utc).isoformat(), job_id=JOB, folder=FOLDER,
                  reason='CPU-only VASP occupies 240 GiB host RAM on the two-A100 node; release for GPU work',
                  scheduler_before=info, action='STOPCAR LABORT; preserve density; continue on apollo')
    save(NAME + '-intent.json', intent)
    code = '''import json, pathlib, subprocess, time
p = pathlib.Path(FOLDER)
s = json.loads((p/'status.json').read_text())
assert s['job_id'] == JOB and s['state'] == 'running', s
assert subprocess.check_output(['squeue','-h','-j',JOB,'-o','%T'],universal_newlines=True).strip() == 'RUNNING'
text = (p/'OUTCAR').read_text(errors='replace')
assert 'General timing and accounting informations' not in text, 'Already finished'
record = dict(job_id=JOB, requested_epoch=time.time(), state='stop_requested', reason='Release A100 host memory; relocate CPU-only VASP to apollo', manual=True)
with (p/'checkpoint-request.json').open('x') as f: json.dump(record,f,indent=2)
with (p/'STOPCAR').open('x') as f: f.write('LABORT = .TRUE.\\n')
print(json.dumps(record))
'''
    code = 'FOLDER=' + repr(FOLDER) + '\nJOB=' + repr(JOB) + '\n' + code
    result = subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt','python3 -'],
                            input=code, text=True, capture_output=True)
    save(NAME + '-result.json', dict(intent, returncode=result.returncode,
                                    stdout=result.stdout, stderr=result.stderr))
    print(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stderr)
