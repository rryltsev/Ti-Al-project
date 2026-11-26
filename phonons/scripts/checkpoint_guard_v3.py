"""Six-hour checkpoint reserve for the measured PBEsol force-002 calculation only."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def stop_due(now, deadline, text):
    loops = [float(x) for x in re.findall(r'LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)', text)]
    # Six hours minimum; measured prior PBEsol finalization was 2002 seconds.
    reserve = max(21600, 2 * max(loops[-8:], default=0) + 10800)
    return now >= deadline - reserve, reserve


def guard(folder, job, deadline):
    folder = Path(folder)
    if job != '20962927' or not str(folder).endswith('/PBEsol/production-04/force-002'):
        raise RuntimeError('This measured reserve applies only to PBEsol job 20962927')
    lock = folder / 'checkpoint-guard-v3.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    while True:
        text = (folder / 'OUTCAR').read_text(errors='replace') if (folder / 'OUTCAR').exists() else ''
        if 'General timing and accounting informations' in text:
            return
        state = subprocess.check_output(['squeue', '-h', '-j', job, '-o', '%T'], text=True).strip()
        if state not in ('RUNNING', 'COMPLETING'):
            return
        due, reserve = stop_due(time.time(), deadline, text)
        record = dict(job_id=job, checked_epoch=time.time(), deadline_epoch=deadline,
                      reserve_seconds=reserve, state='armed', guard_revision='PBEsol-measured-v3', minimum_reserve_seconds=21600)
        if due:
            record['state'] = 'stop_requested'
            # Record intent first. A stopped SCF is never a converged force result.
            (folder / 'checkpoint-request.json').write_text(json.dumps(record, indent=2))
            with (folder / 'STOPCAR').open('x') as f:
                f.write('LABORT = .TRUE.\n')
        temp = folder / 'checkpoint-guard.json.tmp'
        temp.write_text(json.dumps(record, indent=2))
        temp.replace(folder / 'checkpoint-guard.json')
        if due:
            return
        time.sleep(300)


if __name__ == '__main__':
    guard(sys.argv[1], sys.argv[2], float(sys.argv[3]))
